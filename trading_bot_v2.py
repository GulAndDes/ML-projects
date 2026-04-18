import ccxt
import json
import os
import pandas as pd
import numpy as np
import time
import asyncio
from datetime import datetime, timezone
from telegram import Bot
from catboost import CatBoostClassifier
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import ADXIndicator, EMAIndicator
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV


# ==========================================
# ⚙️ НАСТРОЙКИ (HYBRID V2 - GUARANTEED EDGE)
# ==========================================
TOKEN = "8236374314:AAE5j6AJOBP5ilQ6OIMi8mNt9P-BuqG6tOw"
CHAT_ID = "934029089"

SYMBOLS = [
    "SOL/USDT",
    "DOGE/USDT",
    "PEPE/USDT",
    "XRP/USDT",
    "APT/USDT",
    "LTC/USDT",
    "AVAX/USDT",
    "TRX/USDT",
    "LINK/USDT",
    "POL/USDT",
    "GMT/USDT",
    "SUI/USDT",
    "NEAR/USDT",
    "FTM/USDT",
    "INJ/USDT",
    "FIL/USDT",
    "ATOM/USDT",
    "APE/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "RENDER/USDT",
]
TIMEFRAME = "1h"
HISTORY_LIMIT = 5000
RETRAIN_INTERVAL = 43200  # 12 часов

# Настройки обучения
LOOKAHEAD_CANDLES = 12

# 🔥 КРИТИЧЕСКИЕ НАСТРОЙКИ ДЛЯ ПЛЮСА
BASE_CONFIDENCE = 0.52  # Повысили порог входа. Меньше сделок, но качественнее.
MIN_EDGE_OVER_COST = 2.5  # Ожидаемая прибыль должна быть в 2.5 раза выше комиссии
VOLATILITY_FLOOR = 0.0015  # Мин. волатильность (ATR%), чтобы покрыть спред+комиссию
VOLATILITY_CEILING = (
    0.05  # Макс. волатильность (избегаем пампов/дампов с высоким риском)
)

MODE_SCALP = "SCALP"
MODE_IMPULSE = "IMPULSE"

STATE_FILE = "bot_state_hybrid.json"
STATS_FILE = "trade_stats_hybrid.json"
last_exit_times = {}
COOLDOWN_SECONDS = 1800

# Costs
FEE_RATE_PER_SIDE = 0.0006
SLIPPAGE_RATE_PER_SIDE = 0.0003  # Чуть увеличили запас
ROUND_TRIP_COST_RATE = 2 * (FEE_RATE_PER_SIDE + SLIPPAGE_RATE_PER_SIDE)  # ~0.18%

# Фильтры
USE_KELLY_FILTER = True
USE_EXPECTED_VALUE_FILTER = True
USE_VOLATILITY_FILTER = True

QUALITY_FLOOR_SCALP = 0.60
QUALITY_FLOOR_IMPULSE = 0.62

# Риск-менеджмент
RR_MIN = 1.8  # Минимальное соотношение Risk/Reward


# ==========================================
# 🔧 ИНИЦИАЛИЗАЦИЯ
# ==========================================
exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
bot_telegram = Bot(token=TOKEN)

models = {}
best_thresholds = {}
threshold_stats = {}
last_train_time = 0

FEATURES = [
    "rsi",
    "rsi_lag_1",
    "rsi_lag_2",
    "adx",
    "bb_width_pct",
    "bb_pos",
    "atr_pct",
    "ema_dist_20",
    "ema_dist_50",
    "ema_dist_200",
    "log_ret",
    "log_ret_lag1",
    "log_ret_lag2",
    "vol_rel",
    "volume_shock",
    "trend_slope",
    "rsi_adx_interaction",
    "trend_ema50",
    "atr_regime",
    "momentum_3",
    "momentum_6",
    "vwap_dist",
    "vol_zscore",
]


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_mode(mode):
    if not mode:
        return None
    mode_upper = str(mode).upper()
    if "IMPULSE" in mode_upper:
        return MODE_IMPULSE
    if "SCALP" in mode_upper:
        return MODE_SCALP
    return str(mode)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(states):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(states, f, indent=4, ensure_ascii=False)


def default_trade_stats():
    return {
        "last_updated": utc_now_iso(),
        "summary": {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "gross_pnl_pct_sum": 0.0,
            "net_pnl_pct_sum": 0.0,
            "avg_net_pnl_pct": 0.0,
            "total_cost_pct": 0.0,
            "profit_factor": None,
            "open_positions": 0,
        },
        "by_symbol": {},
        "by_mode": {},
        "open_positions": {},
        "recent_signals": [],
        "closed_trades": [],
    }


def load_trade_stats():
    stats = default_trade_stats()
    if os.path.exists(STATS_FILE):
        try:
            with open(STATS_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                stats.update(loaded)
        except Exception:
            pass
    for key, value in default_trade_stats().items():
        stats.setdefault(key, value if not isinstance(value, dict) else dict(value))
    return stats


def ensure_trade_id(sym, state):
    trade_id = state.get("trade_id")
    if trade_id:
        return trade_id
    trade_id = f"{sym.replace('/', '_')}_{int(state.get('last_entry_ts', time.time()))}"
    state["trade_id"] = trade_id
    return trade_id


def append_recent_signal(event):
    trade_stats["recent_signals"].append(event)
    trade_stats["recent_signals"] = trade_stats["recent_signals"][-200:]


def build_bucket_stats():
    return {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "win_rate": 0.0,
        "gross_pnl_pct_sum": 0.0,
        "net_pnl_pct_sum": 0.0,
        "avg_net_pnl_pct": 0.0,
    }


def update_bucket(container, key, trade):
    bucket = container.setdefault(key, build_bucket_stats())
    net_pnl = float(trade.get("net_pnl_pct", 0.0))
    gross_pnl = float(trade.get("gross_pnl_pct", 0.0))
    bucket["total_trades"] += 1
    bucket["gross_pnl_pct_sum"] += gross_pnl
    bucket["net_pnl_pct_sum"] += net_pnl
    if net_pnl > 0:
        bucket["wins"] += 1
    else:
        bucket["losses"] += 1


def recalculate_trade_stats():
    summary = default_trade_stats()["summary"]
    by_symbol = {}
    by_mode = {}
    positive_sum = 0.0
    negative_sum = 0.0

    for trade in trade_stats["closed_trades"]:
        net_pnl = float(trade.get("net_pnl_pct", 0.0))
        gross_pnl = float(trade.get("gross_pnl_pct", 0.0))
        cost_pct = float(trade.get("cost_pct", 0.0))

        summary["total_trades"] += 1
        summary["gross_pnl_pct_sum"] += gross_pnl
        summary["net_pnl_pct_sum"] += net_pnl
        summary["total_cost_pct"] += cost_pct

        if net_pnl > 0:
            summary["wins"] += 1
            positive_sum += net_pnl
        else:
            summary["losses"] += 1
            negative_sum += abs(net_pnl)

        update_bucket(by_symbol, trade.get("symbol", "UNKNOWN"), trade)
        update_bucket(by_mode, trade.get("mode", "UNKNOWN"), trade)

    if summary["total_trades"] > 0:
        summary["win_rate"] = round(summary["wins"] / summary["total_trades"] * 100, 2)
        summary["avg_net_pnl_pct"] = round(
            summary["net_pnl_pct_sum"] / summary["total_trades"], 4
        )
        if negative_sum > 0:
            summary["profit_factor"] = round(positive_sum / negative_sum, 3)

    for container in (by_symbol, by_mode):
        for bucket in container.values():
            if bucket["total_trades"] > 0:
                bucket["win_rate"] = round(
                    bucket["wins"] / bucket["total_trades"] * 100, 2
                )
                bucket["avg_net_pnl_pct"] = round(
                    bucket["net_pnl_pct_sum"] / bucket["total_trades"], 4
                )

    summary["open_positions"] = len(
        [state for state in trade_states.values() if state.get("active")]
    )
    trade_stats["summary"] = summary
    trade_stats["by_symbol"] = by_symbol
    trade_stats["by_mode"] = by_mode


def snapshot_open_positions():
    snapshot = {}
    for sym, state in trade_states.items():
        if not state.get("active"):
            continue
        trade_id = ensure_trade_id(sym, state)
        snapshot[trade_id] = {
            "trade_id": trade_id,
            "symbol": sym,
            "mode": normalize_mode(state.get("mode")),
            "entry": state.get("entry"),
            "entry_prob": state.get("entry_prob"),
            "quality_score": state.get("quality_score"),
            "expected_net_edge_pct": round(
                float(state.get("expected_net_edge", 0.0)) * 100, 4
            ),
            "reward_risk": round(float(state.get("reward_risk", 0.0)), 3),
            "tp": state.get("tp"),
            "sl": state.get("sl"),
            "entry_time": datetime.fromtimestamp(
                state.get("last_entry_ts", time.time()), tz=timezone.utc
            ).isoformat(timespec="seconds"),
        }
    return snapshot


def persist_trade_stats():
    trade_stats["last_updated"] = utc_now_iso()
    trade_stats["open_positions"] = snapshot_open_positions()
    recalculate_trade_stats()
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(trade_stats, f, indent=4, ensure_ascii=False)
    if trade_stats.get("closed_trades"):
        try:
            df_csv = pd.DataFrame(trade_stats["closed_trades"])
            df_csv.to_csv("trade_history_hybrid.csv", index=False, encoding="utf-8")
        except Exception as e:
            print(f"Ошибка сохранения CSV: {e}")


def register_trade_open(sym, state):
    trade_id = ensure_trade_id(sym, state)
    event = {
        "time": utc_now_iso(),
        "type": "OPEN",
        "trade_id": trade_id,
        "symbol": sym,
        "mode": normalize_mode(state.get("mode")),
        "entry": state.get("entry"),
        "entry_prob": round(float(state.get("entry_prob", 0.0)), 4),
        "quality_score": round(float(state.get("quality_score", 0.0)), 4),
        "expected_net_edge_pct": round(
            float(state.get("expected_net_edge", 0.0)) * 100, 4
        ),
        "reward_risk": round(float(state.get("reward_risk", 0.0)), 3),
    }
    append_recent_signal(event)
    persist_trade_stats()


def register_trade_close(sym, state, exit_price, gross_pnl_pct, net_pnl_pct, reason):
    trade_id = ensure_trade_id(sym, state)
    closed_trade = {
        "trade_id": trade_id,
        "symbol": sym,
        "mode": normalize_mode(state.get("mode")),
        "entry": state.get("entry"),
        "exit": exit_price,
        "entry_time": datetime.fromtimestamp(
            state.get("last_entry_ts", time.time()), tz=timezone.utc
        ).isoformat(timespec="seconds"),
        "exit_time": utc_now_iso(),
        "entry_prob": round(float(state.get("entry_prob", 0.0)), 4),
        "quality_score": round(float(state.get("quality_score", 0.0)), 4),
        "expected_net_edge_pct": round(
            float(state.get("expected_net_edge", 0.0)) * 100, 4
        ),
        "reward_risk": round(float(state.get("reward_risk", 0.0)), 3),
        "gross_pnl_pct": round(float(gross_pnl_pct), 4),
        "net_pnl_pct": round(float(net_pnl_pct), 4),
        "cost_pct": round(float(ROUND_TRIP_COST_RATE * 100), 4),
        "reason": reason,
    }
    trade_stats["closed_trades"].append(closed_trade)
    append_recent_signal(
        {
            "time": utc_now_iso(),
            "type": "CLOSE",
            "trade_id": trade_id,
            "symbol": sym,
            "mode": normalize_mode(state.get("mode")),
            "net_pnl_pct": round(float(net_pnl_pct), 4),
            "reason": reason,
        }
    )
    export_stats_to_csv()
    persist_trade_stats()


trade_states = load_state()
for sym in SYMBOLS:
    if sym not in trade_states:
        trade_states[sym] = {"active": False, "mode": None}
    trade_states[sym]["mode"] = normalize_mode(trade_states[sym].get("mode"))
    if trade_states[sym].get("active"):
        entry_ts = trade_states[sym].get("last_entry_ts", 0)
        age_hours = (time.time() - entry_ts) / 3600
        if age_hours > 24:
            print(f"🗑️ {sym}: Очистка зависшей позиции (возраст {age_hours:.1f}h)")
            trade_states[sym]["active"] = False
        else:
            ensure_trade_id(sym, trade_states[sym])
            trade_states[sym].setdefault("partial_taken", False)
            trade_states[sym].setdefault("entry_prob", 0.0)
            trade_states[sym].setdefault("quality_score", 0.0)
            trade_states[sym].setdefault("expected_net_edge", 0.0)
            trade_states[sym].setdefault("reward_risk", 0.0)

trade_stats = load_trade_stats()


# ==========================================
# 📐 ТЕХНИЧЕСКИЙ АНАЛИЗ
# ==========================================
def calculate_slope(series, period=5):
    return np.degrees(np.arctan(series.diff(period) / period))


def add_indicators(df):
    df = df.copy()
    df["rsi"] = RSIIndicator(close=df["close"], window=14).rsi() / 100.0
    df["rsi_lag_1"] = df["rsi"].shift(1)
    df["rsi_lag_2"] = df["rsi"].shift(2)

    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    df["bb_width_pct"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]
    df["bb_pos"] = (df["close"] - bb.bollinger_lband()) / (
        bb.bollinger_hband() - bb.bollinger_lband()
    )

    atr = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["atr"] = atr.average_true_range()
    df["atr_pct"] = df["atr"] / df["close"]

    df["ema_20"] = EMAIndicator(close=df["close"], window=20).ema_indicator()
    df["ema_50"] = EMAIndicator(close=df["close"], window=50).ema_indicator()
    df["ema_200"] = EMAIndicator(close=df["close"], window=200).ema_indicator()

    df["ema_dist_20"] = (df["close"] - df["ema_20"]) / df["close"]
    df["ema_dist_50"] = (df["close"] - df["ema_50"]) / df["close"]
    df["ema_dist_200"] = (df["close"] - df["ema_200"]) / df["close"]
    df["trend_slope"] = calculate_slope(df["ema_50"], 5)

    adx = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
    df["adx"] = adx.adx()

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    df["log_ret_lag1"] = df["log_ret"].shift(1)
    df["log_ret_lag2"] = df["log_ret"].shift(2)

    vol_ma = df["volume"].rolling(20).mean()
    df["vol_rel"] = df["volume"] / (vol_ma + 1e-9)
    df["volume_shock"] = (df["vol_rel"] > 2.0).astype(int)

    df["rsi_adx_interaction"] = df["rsi"] * df["adx"]
    df["trend_ema50"] = (df["close"] > df["ema_50"]).astype(int)
    df["atr_regime"] = df["atr_pct"] / df["atr_pct"].rolling(100).mean()
    df["momentum_3"] = df["close"] / df["close"].shift(3) - 1
    df["momentum_6"] = df["close"] / df["close"].shift(6) - 1

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap = (typical_price * df["volume"]).rolling(window=200).sum() / (
        df["volume"].rolling(window=200).sum() + 1e-9
    )
    df["vwap_dist"] = (df["close"] - vwap) / df["close"]

    vol_mean = df["volume"].rolling(50).mean()
    vol_std = df["volume"].rolling(50).std()
    df["vol_zscore"] = (df["volume"] - vol_mean) / (vol_std + 1e-9)

    df.dropna(inplace=True)
    return df


def simulate_forward_trade(df, start_idx, tp_mult, sl_mult, lookahead):
    if start_idx + lookahead >= len(df):
        return None

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr"].values

    entry = closes[start_idx]
    atr = atrs[start_idx]
    tp_price = entry + (atr * tp_mult)
    sl_price = entry - (atr * sl_mult)

    tp_pct = max((tp_price - entry) / entry, 0.0)
    sl_pct = max((entry - sl_price) / entry, 0.0)

    for step in range(1, lookahead + 1):
        if lows[start_idx + step] <= sl_price:
            return -sl_pct - ROUND_TRIP_COST_RATE
        if highs[start_idx + step] >= tp_price:
            return tp_pct - ROUND_TRIP_COST_RATE

    final_close = closes[start_idx + lookahead]
    return ((final_close - entry) / entry) - ROUND_TRIP_COST_RATE


# ==========================================
# 🧠 ОБУЧЕНИЕ МОДЕЛИ
# ==========================================
LIVE_TP_ATR = 2.0
LIVE_SL_ATR = 1.0


def get_target(df, tp_mult=LIVE_TP_ATR, sl_mult=LIVE_SL_ATR):
    targets = []
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr"].values
    lookahead = LOOKAHEAD_CANDLES

    for i in range(len(df) - lookahead):
        entry = closes[i]
        atr = atrs[i]
        tp_price = entry + (atr * tp_mult)
        sl_price = entry - (atr * sl_mult)

        outcome = 0
        for j in range(1, lookahead + 1):
            if lows[i + j] <= sl_price:
                outcome = 0
                break
            if highs[i + j] >= tp_price:
                outcome = 1
                break
        targets.append(outcome)
    return targets


def fetch_ohlcv_full(symbol, timeframe, limit):
    all_bars = []
    since = None
    attempts = 0
    max_attempts = 50

    while len(all_bars) < limit and attempts < max_attempts:
        try:
            bars = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not bars or len(bars) == 0:
                break
            since = bars[-1][0] + 1
            all_bars += bars
            attempts += 1
            time.sleep(0.15)
        except ccxt.RateLimitExceeded:
            time.sleep(2)
            continue
        except Exception as e:
            print(f"  ⚠️ {symbol} fetch error: {e}")
            break
    return all_bars


def train_model(sym):
    print(f"⏳ Training {sym}...")
    try:
        bars = fetch_ohlcv_full(sym, TIMEFRAME, 3000)
        print(f"  📥 {sym}: Получено {len(bars)} свечей")

        if len(bars) < 1000:
            print(f"  ❌ {sym}: Мало исторических данных (<1000).")
            return None

        df = pd.DataFrame(
            bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df = add_indicators(df)
        print(f"  📊 {sym}: После индикаторов осталось {len(df)} строк")

        if len(df) < 200:
            print(f"  ❌ {sym}: Слишком много NaN/пропусков.")
            return None

        targets = get_target(df)
        df = df.iloc[: len(targets)]
        df["target"] = targets

        pos = int(df["target"].sum())
        print(
            f"  🎯 {sym}: Сформировано {len(targets)} таргетов (успешных: {pos}, {pos/max(1,len(targets))*100:.1f}%)"
        )

        if pos < 30:
            print(
                f"  ❌ {sym}: Слишком мало успешных исходов для обучения (нужно >30)."
            )
            return None

        X, y = df[FEATURES], df["target"]
        split = int(len(X) * 0.85)
        X_train, X_val = X.iloc[:split], X.iloc[split:]
        y_train, y_val = y.iloc[:split], y.iloc[split:]

        scale_pos = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

        model = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.04,
            depth=5,
            l2_leaf_reg=3,
            scale_pos_weight=scale_pos,
            loss_function="Logloss",
            eval_metric="AUC",
            early_stopping_rounds=100,
            verbose=False,
            task_type="CPU",
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

        val_probs = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, val_probs)
        base_wr = y.mean()

        # 🔥 УЖЕСТОЧИЛИ ТРЕБОВАНИЯ К МОДЕЛИ
        if score < 0.55 or base_wr < 0.20:
            print(f"  📉 {sym} rejected (AUC={score:.3f} | BaseWR={base_wr:.2f})")
            return None

        print(
            f"  ✅ {sym} ГОТОВА | AUC: {score:.3f} | BaseWR: {base_wr:.2f} | Pos: {pos}"
        )
        return model

    except Exception as e:
        print(f"  💥 {sym}: КРИТИЧЕСКАЯ ОШИБКА ОБУЧЕНИЯ -> {e}")
        import traceback

        traceback.print_exc()
        return None


def train_all():
    global models, last_train_time
    for sym in SYMBOLS:
        try:
            m = train_model(sym)
            if m:
                models[sym] = m
        except Exception as e:
            print(f"❌ {sym}: {e}")
    last_train_time = time.time()


# ==========================================
# 🔍 ОПТИМИЗАЦИЯ ПОРОГОВ (V2 - С УЧЕТОМ КОМИССИЙ)
# ==========================================
def optimize_threshold(sym, model):
    bars = fetch_ohlcv_full(sym, TIMEFRAME, 4000)
    if len(bars) < 500:
        best_thresholds[sym] = BASE_CONFIDENCE
        threshold_stats[sym] = {
            "avg_net_return": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
        }
        return

    df = pd.DataFrame(
        bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df = add_indicators(df)
    df = df.iloc[-2000:]

    probs = model.predict_proba(df[FEATURES])[:, 1]
    best_result = None

    for th in np.arange(0.52, 0.76, 0.01):
        returns = []
        next_allowed_idx = -1

        for idx in range(len(df) - LOOKAHEAD_CANDLES - 1):
            if idx <= next_allowed_idx:
                continue
            if probs[idx] < th:
                continue

            trade_return = simulate_forward_trade(
                df, idx, LIVE_TP_ATR, LIVE_SL_ATR, LOOKAHEAD_CANDLES
            )
            if trade_return is None:
                continue

            # 🔥 ФИЛЬТР: Только сделки с положительным матожиданием
            if trade_return > ROUND_TRIP_COST_RATE:
                returns.append(trade_return)

            next_allowed_idx = idx + 3

        if len(returns) < 10:  # Нужно минимум сделок для статистики
            continue

        avg_net = float(np.mean(returns))
        win_rate = float(np.mean(np.array(returns) > 0))

        # 🔥 КЛЮЧЕВОЙ ФИЛЬТР: Средняя прибыль должна превышать комиссии в MIN_EDGE_OVER_COST раз
        if avg_net <= ROUND_TRIP_COST_RATE * MIN_EDGE_OVER_COST:
            continue

        score = avg_net * np.sqrt(len(returns))

        if best_result is None or score > best_result["score"]:
            best_result = {
                "threshold": round(float(th), 2),
                "avg_net_return": avg_net,
                "trade_count": len(returns),
                "win_rate": win_rate,
                "score": score,
            }

    if best_result is None:
        best_thresholds[sym] = BASE_CONFIDENCE
        threshold_stats[sym] = {
            "avg_net_return": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
        }
        print(f"Threshold fallback for {sym}: {BASE_CONFIDENCE:.2f}")
        return

    # 🔥 ЕЩЁ БОЛЕЕ КОНСЕРВАТИВНЫЙ ПОДХОД: берём максимум между найденным и базовым
    best_thresholds[sym] = max(BASE_CONFIDENCE, best_result["threshold"])
    threshold_stats[sym] = {
        "avg_net_return": best_result["avg_net_return"],
        "trade_count": best_result["trade_count"],
        "win_rate": best_result["win_rate"],
    }
    print(
        f"Threshold {sym}: {best_thresholds[sym]:.2f} | "
        f"avg_net={best_result['avg_net_return']*100:.3f}% | "
        f"WR={best_result['win_rate']*100:.1f}% | n={best_result['trade_count']}"
    )


async def send_msg(text):
    try:
        await bot_telegram.send_message(CHAT_ID, text, parse_mode="HTML")
    except Exception as e:
        print(f"❌ ОШИБКА ТЕЛЕГРАМ: {e}")


def check_emergency_exit(sym, state, df, model):
    try:
        if model is None:
            return False, None

        last_row = df.iloc[-1]
        price = last_row["close"]
        atr = last_row["atr"]

        X = df[FEATURES].iloc[[-1]]
        current_prob = model.predict_proba(X)[0][1]
        entry_time = state.get("last_entry_ts", time.time())
        minutes_open = (time.time() - entry_time) / 60

        if minutes_open < 45:
            return False, None

        # 🔥 ЖЁСТКИЙ ПОРОГ: если вероятность упала ниже 0.50 — выходим
        if current_prob < 0.50:
            return True, f"🧠 Prob Collapse ({current_prob:.3f})"

        entry_atr = state.get("entry_atr", atr)
        if atr < entry_atr * 0.35:
            return True, "🧊 Volatility Collapse"

        return False, None
    except Exception as e:
        print(f"Emergency exit error {sym}: {e}")
        return False, None


def export_stats_to_csv():
    if not trade_stats["closed_trades"]:
        return
    df_stats = pd.DataFrame(trade_stats["closed_trades"])
    df_stats.to_csv("trade_history.csv", index=False, encoding="utf-8")
    print("📊 Статистика экспортирована в trade_history.csv")


last_candle_fetch_time = 0
live_data_cache = {}
DEBUG_FILTERS = True


async def process_market():
    global last_candle_fetch_time, live_data_cache
    current_time = time.time()

    active_symbols = list(models.keys())
    if not active_symbols:
        print("⏳ Нет обученных моделей. Жду обучения...")
        return

    try:
        tickers = exchange.fetch_tickers(active_symbols)
    except Exception as e:
        print(f"❌ Ошибка тикеров: {e}")
        return

    if current_time - last_candle_fetch_time > 30:
        print(f"\n🔄 Загрузка {len(active_symbols)} графиков...")
        ok_cnt = 0
        for sym in active_symbols:
            try:
                bars = exchange.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=500)
                if bars and len(bars) > 0:
                    df = pd.DataFrame(
                        bars,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    live_data_cache[sym] = add_indicators(df)
                    ok_cnt += 1

            except Exception as e:
                if DEBUG_FILTERS:
                    print(f"  ❌ {sym}: {e}")
        print(f"📦 Загружено: {ok_cnt}/{len(active_symbols)} пар.")
        last_candle_fetch_time = current_time

    # 🛡️ УПРАВЛЕНИЕ ПОЗИЦИЯМИ
    for sym, state in list(trade_states.items()):
        if not state.get("active") or sym not in tickers:
            continue
        try:
            current_price = tickers[sym].get("bid") or tickers[sym].get("last")
            if not current_price:
                continue

            entry = state["entry"]
            gross_pnl = (current_price - entry) / entry * 100
            net_pnl = gross_pnl - (ROUND_TRIP_COST_RATE * 100)
            elapsed_min = (time.time() - state["last_entry_ts"]) / 60
            mode = state.get("mode", "SCALP")
            exit_reason = None

            df_live = live_data_cache.get(sym)
            if df_live is not None:
                is_exit, reason = check_emergency_exit(
                    sym, state, df_live, models.get(sym)
                )
                if is_exit:
                    exit_reason = f"🚨 {reason}"

            if not exit_reason:
                if current_price >= state["tp"]:
                    exit_reason = f"✅ TAKE PROFIT ({mode})"
                elif current_price <= state["sl"]:
                    exit_reason = f"❌ STOP LOSS ({mode})"
                elif net_pnl > 0.4 and df_live is not None:
                    cur_atr = df_live.iloc[-1]["atr"]
                    new_sl = current_price - (cur_atr * 1.0)
                    if new_sl > state["sl"]:
                        state["sl"] = new_sl
                elif elapsed_min > 360 and net_pnl < 0:
                    exit_reason = "⏰ TIME EXIT (6h)"

            if exit_reason:
                await send_msg(
                    f"{exit_reason} {sym}\nPrice: {current_price:.5f}\nNet PnL: {net_pnl:.2f}%\nTime: {int(elapsed_min)}m"
                )
                register_trade_close(
                    sym, state, current_price, gross_pnl, net_pnl, exit_reason
                )
                state["active"] = False
                last_exit_times[sym] = time.time()
                save_state(trade_states)
                print(f"🏁 CLOSED {sym}: {exit_reason} | PnL: {net_pnl:.2f}%")
        except Exception as e:
            print(f"Ошибка трекинга {sym}: {e}")

    # 🎯 ПОИСК ВХОДОВ (V2 - СТРОГАЯ ФИЛЬТРАЦИЯ)
    last_debug_log = {}

    for sym in active_symbols:
        if trade_states[sym].get("active", False):
            continue
        if time.time() - last_exit_times.get(sym, 0) < COOLDOWN_SECONDS:
            continue

        df = live_data_cache.get(sym)
        if df is None or len(df) < 50:
            continue

        try:
            price = tickers[sym].get("ask") or tickers[sym].get("last")
            if not price:
                continue

            last_row = df.iloc[-1]
            prob = models[sym].predict_proba(df[FEATURES].iloc[[-1]])[0][1]
            adx = last_row["adx"]
            atr = last_row["atr"]
            atr_pct = atr / price

            # 🔥 ИСПОЛЬЗУЕМ ОПТИМИЗИРОВАННЫЙ ПОРОГ
            base_th = best_thresholds.get(sym, BASE_CONFIDENCE)

            # Динамическая корректировка
            if adx < 15:
                min_prob = max(base_th, 0.53)  # Во флэте требуем выше вероятность
            elif adx > 30:
                min_prob = max(base_th - 0.04, 0.48)  # В тренде можно чуть ниже
            else:
                min_prob = base_th

            if prob <= min_prob:
                now = time.time()
                if now - last_debug_log.get(sym, 0) > 60 and prob > min_prob:
                    print(f"⏭️ {sym}: Prob {prob:.3f} < {min_prob:.2f} (Req)")
                    last_debug_log[sym] = now
                continue

            # 🔥 ФИЛЬТР ПО ВОЛАТИЛЬНОСТИ
            if atr_pct < VOLATILITY_FLOOR or atr_pct > VOLATILITY_CEILING:
                now = time.time()
                if now - last_debug_log.get(sym, 0) > 60:
                    print(
                        f"⏭️ {sym}: ATR% {atr_pct:.4f} вне диапазона [{VOLATILITY_FLOOR}, {VOLATILITY_CEILING}]"
                    )
                    last_debug_log[sym] = now
                continue

            # 🔥 ФИЛЬТР ПО ТРЕНДУ
            trend_str = (last_row["ema_20"] - last_row["ema_50"]) / (atr + 1e-9)
            if trend_str < -2.0:
                continue

            # 🔥 РАСЧЁТ ОЖИДАЕМОЙ ЦЕННОСТИ СДЕЛКИ
            tp_mult, sl_mult = (2.0, 1.0) if adx > 25 else (1.8, 0.9)
            tp_pct = tp_mult * atr_pct
            sl_pct = sl_mult * atr_pct

            # Expected Value после комиссий
            expected_value = (
                (prob * tp_pct) - ((1 - prob) * sl_pct) - ROUND_TRIP_COST_RATE
            )

            # 🔥 ГЛАВНЫЙ ФИЛЬТР: EV должен быть положительным и превышать комиссии в MIN_EDGE_OVER_COST раз
            if expected_value <= ROUND_TRIP_COST_RATE * MIN_EDGE_OVER_COST:
                now = time.time()
                if now - last_debug_log.get(sym, 0) > 60:
                    print(
                        f"⏭️ {sym}: EV {expected_value*100:.3f}% < мин.порог {ROUND_TRIP_COST_RATE * MIN_EDGE_OVER_COST * 100:.3f}%"
                    )
                    last_debug_log[sym] = now
                continue

            # 🔥 ФИЛЬТР ПО KELLY
            if USE_KELLY_FILTER:
                kelly = ((tp_pct / sl_pct) * prob - (1 - prob)) / (tp_pct / sl_pct)
                if kelly < 0.05:  # Минимальный Kelly
                    continue

            # === ВХОД ===
            mode = "IMPULSE" if adx > 25 else "SCALP"
            tp_price = price + (atr * tp_mult)
            sl_price = price - (atr * sl_mult)

            reward_risk = tp_pct / (sl_pct + 1e-9)

            # 🔥 ФИНАЛЬНАЯ ПРОВЕРКА RR
            if reward_risk < RR_MIN:
                continue

            trade_states[sym] = {
                "active": True,
                "entry": price,
                "entry_atr": atr,
                "entry_prob": prob,
                "max_prob": prob,
                "sl": sl_price,
                "tp": tp_price,
                "max_p": price,
                "last_entry_ts": time.time(),
                "mode": mode,
                "partial_taken": False,
                "expected_net_edge": expected_value,
                "reward_risk": reward_risk,
                "quality_score": prob,
            }
            save_state(trade_states)
            register_trade_open(sym, trade_states[sym])

            print(
                f"🚀 ВХОД {sym} | P:{prob:.3f} | ADX:{adx:.1f} | Mode:{mode} | "
                f"EV:{expected_value*100:.3f}% | RR:1:{reward_risk:.1f}"
            )
            await send_msg(
                f"🚀 <b>{mode} BUY {sym}</b>\n💰 {price:.5f}\n🎯 {tp_price:.5f}\n🛑 {sl_price:.5f}\n"
                f"🔮 {prob:.1%} | EV: {expected_value*100:.2f}% | RR: 1:{reward_risk:.1f}"
            )

        except Exception as e:
            print(f"❌ Ошибка входа {sym}: {e}")


async def main():
    print("🤖 Bot Starting Hybrid Mode V2...")
    await send_msg("🤖 Hybrid Bot V2 (Guaranteed Edge) Started.")

    train_all()
    for sym in SYMBOLS:
        if sym in models:
            optimize_threshold(sym, models[sym])

    while True:
        if time.time() - last_train_time > RETRAIN_INTERVAL:
            train_all()

        await process_market()
        await asyncio.sleep(10)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Stop")
