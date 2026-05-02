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
from scipy.stats import zscore
from sklearn.model_selection import TimeSeriesSplit

# НАСТРОЙКИ
TOKEN = "###"
CHAT_ID = "934029089"

SYMBOLS = [
    "BTC/USDT",
    "SOL/USDT",
    "DOGE/USDT",
    "XRP/USDT",
    "APT/USDT",
    "LTC/USDT",
    "AVAX/USDT",
    "TRX/USDT",
    "LINK/USDT",
    "FTM/USDT",
    "INJ/USDT",
    "ETH/USDT",
    "BNB/USDT",
]
TIMEFRAME = "1h"
HISTORY_LIMIT = 5000
RETRAIN_INTERVAL = 43200  # 12 часов

# Настройки обучения
LOOKAHEAD_CANDLES = 12

# КРИТИЧЕСКИЕ НАСТРОЙКИ
BASE_CONFIDENCE = 0.55  # порог для ML-модели
MIN_EDGE_OVER_COST = 1.0  # Достаточно, чтобы EV был просто положительным
VOLATILITY_FLOOR = 0.0005  # Ниже комиссия съест профит
VOLATILITY_CEILING = 0.08

MODE_SCALP = "SCALP"
MODE_IMPULSE = "IMPULSE"

STATE_FILE = "bot_state_hybrid.json"
STATS_FILE = "trade_stats_hybrid.json"
last_exit_times = {}
COOLDOWN_SECONDS = 600

# Costs
FEE_RATE_PER_SIDE = 0.0006
SLIPPAGE_RATE_PER_SIDE = 0.0003
ROUND_TRIP_COST_RATE = 0.0012

# Фильтры
USE_KELLY_FILTER = True
USE_EXPECTED_VALUE_FILTER = True
USE_VOLATILITY_FILTER = True

QUALITY_FLOOR_SCALP = 0.60
QUALITY_FLOOR_IMPULSE = 0.62

RR_MIN = 1.5  # Минимальное соотношение Risk/Reward

# БЭКТЕСТ-НАСТРОЙКИ

BACKTEST_LOOKBACK = 150  # 6 дней на 1H
BACKTEST_MIN_TRADES = 8  # Мин. сделок для статистики
BACKTEST_MIN_WIN_RATE = 0.28  # при RR 1.5 - математический плюс
BACKTEST_MIN_PROFIT_FACTOR = 0.9
BACKTEST_MAX_DRAWDOWN = 10
BACKTEST_MIN_EXPECTANCY = 0.0001
DEBUG_BACKTEST = True
BACKTEST_SOFT_MODE = False
BACKTEST_LOG_FILE = "backtest_results.csv"

# ИНИЦИАЛИЗАЦИЯ

exchange = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
bot_telegram = Bot(token=TOKEN)
best_features = {"LONG": {}, "SHORT": {}}
models_long = {}
models_short = {}
best_thresholds = {}
threshold_stats = {}
last_train_time = 0
empirical_stats = {"LONG": {}, "SHORT": {}}
FEATURES = [
    "rsi",
    "adx",
    "atr_pct",
    "ema_dist_50",
    "trend_slope",
    "momentum_3",
    "momentum_6",
    "vol_rel",
    "bb_width_pct",
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


# ЛИМИТЫ И МОНИТОРИНГ
MAX_CONCURRENT_POSITIONS = 8  # Максимум открытых сделок одновременно
MAX_ENTRIES_PER_CYCLE = 4  # Максимум новых входов за 1 цикл сканирования
MIN_EV_THRESHOLD = 0.0012  # отсекает "мусорные" сделки
PROB_FLOOR = 0.50  # Жёсткий минимум вероятности ML-модели

last_status_print = 0


def print_portfolio_status(tickers):
    global last_status_print
    if time.time() - last_status_print < 45:
        return
    last_status_print = time.time()

    active = [sym for sym, st in trade_states.items() if st.get("active")]
    if not active:
        print("Портфель: Пусто | Ожидание высококачественных сигналов...")
        return

    print(f"\nАКТИВНЫЕ ПОЗИЦИИ ({len(active)}/{MAX_CONCURRENT_POSITIONS}):")
    for sym in active:
        st = trade_states[sym]
        entry = st["entry"]
        price = tickers.get(sym, {}).get("last", entry)
        side = st.get(
            "side", "LONG"
        )  # Узнаем реальное направление сделки из сохраненного состояния

        # Считаем PnL в зависимости от направления
        if side == "LONG":
            pnl = (price - entry) / entry * 100
        else:  # SHORT
            pnl = (entry - price) / entry * 100

        elapsed = int((time.time() - st["last_entry_ts"]) / 60)

        # Текущая вероятность от ML берется для активного направления
        prob_str = "N/A"
        df = live_data_cache.get(sym)
        if df is not None:
            try:
                if side == "LONG" and sym in models_long:
                    p = models_long[sym].predict_proba(df[FEATURES].iloc[[-1]])[0][1]
                    prob_str = f"{p:.1%}"
                elif side == "SHORT" and sym in models_short:
                    p = models_short[sym].predict_proba(df[FEATURES].iloc[[-1]])[0][1]
                    prob_str = f"{p:.1%}"
            except:
                pass

        print(
            f"  {side:5s} 🔹 {sym} | {st['mode']} | PnL: {pnl:+.2f}% | ⏱ {elapsed}m | 🧠 {prob_str} | SL: {st['sl']:.9f} | TP: {st['tp']:.9f}"
        )
    print("─" * 85)


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
            print(f"{sym}: Очистка зависшей позиции (возраст {age_hours:.1f}h)")
            trade_states[sym]["active"] = False
        else:
            ensure_trade_id(sym, trade_states[sym])
            trade_states[sym].setdefault("partial_taken", False)
            trade_states[sym].setdefault("entry_prob", 0.0)
            trade_states[sym].setdefault("quality_score", 0.0)
            trade_states[sym].setdefault("expected_net_edge", 0.0)
            trade_states[sym].setdefault("reward_risk", 0.0)

trade_stats = load_trade_stats()


# ТЕХНИЧЕСКИЙ АНАЛИЗ
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
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    body_ratio = body / rng
    direction = np.where(df["close"] > df["open"], 1, -1)
    vol_score = df["volume"] / (df["volume"].rolling(50).max() + 1e-9)

    # Z-score волатильности
    range_z = (
        (df["high"] - df["low"]) - (df["high"] - df["low"]).rolling(50).mean()
    ) / ((df["high"] - df["low"]).rolling(50).std() + 1e-9)
    range_z = range_z.clip(-3, 3)

    # Истинный диапазон для знаменателя
    tr = pd.DataFrame(
        {
            "hl": df["high"] - df["low"],
            "hc": (df["high"] - df["close"].shift(1)).abs(),
            "lc": (df["low"] - df["close"].shift(1)).abs(),
        }
    ).max(axis=1)

    atr_14 = tr.rolling(14).mean().bfill()

    # Итоговая формула CSI
    df["csi"] = (
        direction
        * (0.5 * body_ratio + 0.3 * vol_score + 0.2 * range_z)
        / (atr_14 + 1e-9)
    )
    df["csi_lag1"] = df["csi"].shift(1)  # Динамика CSI

    df.dropna(inplace=True)
    return df


# ОБУЧЕНИЕ МОДЕЛИ
LIVE_TP_ATR = 1.3
LIVE_SL_ATR = 1.0


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
            print(f"  {symbol} fetch error: {e}")
            break
    return all_bars


def get_target(df, lookahead=LOOKAHEAD_CANDLES, direction="LONG"):
    """
    Мы предсказываем, пройдет ли цена расстояние минимум в 1 ATR в нашу сторону.
    """
    future_close = df["close"].shift(-lookahead)
    returns = (future_close - df["close"]) / df["close"]
    atr_pct = df["atr_pct"]

    if direction == "LONG":
        target = (returns > (atr_pct * 0.3)).astype(int)
    else:
        target = (returns < -(atr_pct * 0.3)).astype(int)

    return target


def calculate_empirical_ev(oos_probs, oos_targets, oos_returns):
    # Считает Quarter Kelly с жестким капом на основе Walk-Forward OOS данных
    results = pd.DataFrame(
        {"prob": oos_probs, "target": oos_targets, "return": oos_returns}
    )
    bins = np.linspace(0.45, 0.75, 8)
    results["bucket"] = pd.cut(results["prob"], bins)

    empirical_stats = {}
    for b, group in results.groupby("bucket"):
        if len(group) < 8:
            continue

        wr = group["target"].mean()
        avg_ret = group["return"].mean()

        avg_win = (
            group[group["return"] > 0]["return"].mean()
            if sum(group["return"] > 0) > 0
            else 0
        )
        avg_loss = (
            abs(group[group["return"] <= 0]["return"].mean())
            if sum(group["return"] <= 0) > 0
            else 0.001
        )

        if avg_win > 0 and avg_loss > 0 and wr > 0:
            kelly = wr - ((1 - wr) / (avg_win / avg_loss))
            kelly = max(0, kelly)
        else:
            kelly = 0.0

        # QUARTER KELLY
        safe_kelly = min(kelly * 0.25, 0.02)

        empirical_stats[b] = {"ev": avg_ret, "kelly": safe_kelly, "wr": wr}
        if not empirical_stats:
            empirical_stats[(0.5, 0.6)] = {
                "ev": np.mean(oos_returns),
                "kelly": 0.005,
                "wr": np.mean(oos_targets),
            }

    return empirical_stats


def is_tradeable_pair(df, sym):
    try:
        if len(df) < 400:
            print(f"{sym}: мало данных")
            return False

        # 1. Волатильность
        avg_atr_pct = df["atr_pct"].mean()
        if avg_atr_pct < 0.002:
            print(f" {sym}: низкая волатильность ({avg_atr_pct:.4f})")
            return False

        # 2. Трендовость
        slope = np.abs(df["trend_slope"]).mean()
        if slope < 0.5:
            print(f" {sym}: нет тренда (slope {slope:.2f})")
            return False

        # 3. Баланс target
        target_long = get_target(df, direction="LONG")
        target_short = get_target(df, direction="SHORT")

        if target_long.sum() < 15:
            print(f" {sym}: мало LONG сигналов ({target_long.sum()})")
            return False

        if target_short.sum() < 15:
            print(f" {sym}: мало SHORT сигналов ({target_short.sum()})")
            return False

        print(f"{sym}: OK для торговли")
        return True

    except Exception as e:
        print(f"{sym}: ошибка фильтра {e}")
        return False


def walk_forward_train(X, y, returns):
    n_splits = 3 if y.sum() < 50 else 4
    tscv = TimeSeriesSplit(n_splits=n_splits)
    oos_probs = np.zeros(len(X))
    if y.sum() < 150:
        return None, {}, []

    # STABLE FEATURE SELECTION
    fold_importances = []

    for train_idx, _ in tscv.split(X):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]

        # защита от one-class fold
        if y_tr.nunique() < 2:
            continue

        scale = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)

        cb = CatBoostClassifier(
            iterations=200,
            depth=4,
            scale_pos_weight=scale,
            verbose=False,
            task_type="CPU",
        )

        cb.fit(X_tr, y_tr)
        fold_importances.append(cb.get_feature_importance())

    if len(fold_importances) < 2:
        return None, {}, []

    avg_imp = np.mean(fold_importances, axis=0)
    top_features_idx = np.argsort(avg_imp)[-12:]
    X_pruned = X.iloc[:, top_features_idx]

    # WALK FORWARD
    valid_folds = 0

    for train_idx, test_idx in tscv.split(X_pruned):
        tr_len = len(train_idx)

        pure_train_idx = train_idx[: int(tr_len * 0.8)]
        calib_idx = train_idx[int(tr_len * 0.8) :]

        X_pure = X_pruned.iloc[pure_train_idx]
        y_pure = y.iloc[pure_train_idx]

        X_calib = X_pruned.iloc[calib_idx]
        y_calib = y.iloc[calib_idx]

        X_test = X_pruned.iloc[test_idx]

        # двойная защита
        if y_pure.nunique() < 2:
            continue

        if y_calib.nunique() < 2:
            continue

        scale = (len(y_pure) - y_pure.sum()) / max(y_pure.sum(), 1)

        cb = CatBoostClassifier(
            iterations=400,
            depth=4,
            scale_pos_weight=scale,
            verbose=False,
            l2_leaf_reg=5,
        )

        cb.fit(X_pure, y_pure)

        calibrated = CalibratedClassifierCV(cb, method="sigmoid", cv="prefit")

        calibrated.fit(X_calib, y_calib)

        oos_probs[test_idx] = calibrated.predict_proba(X_test)[:, 1]
        valid_folds += 1

    if valid_folds < 2:
        return None, {}, []

    valid_idx = oos_probs > 0

    if valid_idx.sum() < 30:
        return None, {}, []

    emp_stats = calculate_empirical_ev(
        oos_probs[valid_idx], y.iloc[valid_idx], returns.iloc[valid_idx]
    )

    # финальная модель
    split = int(len(X_pruned) * 0.8)

    X_fin_tr = X_pruned.iloc[:split]
    y_fin_tr = y.iloc[:split]

    X_fin_cal = X_pruned.iloc[split:]
    y_fin_cal = y.iloc[split:]

    if y_fin_tr.nunique() < 2 or y_fin_cal.nunique() < 2:
        return None, {}, []

    final_scale = (len(y_fin_tr) - y_fin_tr.sum()) / max(y_fin_tr.sum(), 1)

    cb_final = CatBoostClassifier(
        iterations=600,
        depth=4,
        scale_pos_weight=final_scale,
        verbose=False,
        l2_leaf_reg=5,
    )

    cb_final.fit(X_fin_tr, y_fin_tr)

    final_calibrated = CalibratedClassifierCV(cb_final, method="sigmoid", cv="prefit")

    final_calibrated.fit(X_fin_cal, y_fin_cal)

    return final_calibrated, emp_stats, X_pruned.columns.tolist()


def calculate_execution_cost(df):
    base_fee = 0.0012
    # Proxy спреда и микроструктурного шума
    spread_proxy = ((df["high"] - df["low"]) / df["close"]) * 0.03

    # Ограничиваем максимальное проскальзывание
    total_cost = base_fee + spread_proxy.clip(0.0001, 0.0030)
    return total_cost


def train_model(sym):
    print(f"⏳ Walk-Forward Training {sym}...")
    try:
        bars = fetch_ohlcv_full(sym, TIMEFRAME, 6000)
        if len(bars) < 1000:
            return None

        df = pd.DataFrame(
            bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df = add_indicators(df)
        df.dropna(inplace=True)
        # ФИЛЬТР ТОРГУЕМЫХ ПАР
        if not is_tradeable_pair(df, sym):
            return None
        # Используем ТОЧНО ТАКУЮ ЖЕ функцию для истории
        hist_costs = calculate_execution_cost(df)
        future_returns_long = (
            df["close"].shift(-LOOKAHEAD_CANDLES) - df["close"]
        ) / df["close"] - hist_costs
        future_returns_short = (
            df["close"] - df["close"].shift(-LOOKAHEAD_CANDLES)
        ) / df["close"] - hist_costs

        targets_long = get_target(df, direction="LONG")
        targets_short = get_target(df, direction="SHORT")

        valid_len = len(df) - LOOKAHEAD_CANDLES
        df = df.iloc[:valid_len]

        X = df[FEATURES]
        y_long = targets_long.iloc[:valid_len]
        y_short = targets_short.iloc[:valid_len]

        result = {"LONG": None, "SHORT": None, "STATS": {}, "FEATURES": {}}

        if y_long.sum() > 40:
            model_l, stats_l, feats_l = walk_forward_train(
                X, y_long, future_returns_long.iloc[:valid_len]
            )
            result["LONG"], result["STATS"]["LONG"], result["FEATURES"]["LONG"] = (
                model_l,
                stats_l,
                feats_l,
            )
            print(f"   {sym} LONG  | Bins: {len(stats_l)} | Top Feats: {len(feats_l)}")

        if y_short.sum() > 40:
            model_s, stats_s, feats_s = walk_forward_train(
                X, y_short, future_returns_short.iloc[:valid_len]
            )
            result["SHORT"], result["STATS"]["SHORT"], result["FEATURES"]["SHORT"] = (
                model_s,
                stats_s,
                feats_s,
            )
            print(f"   {sym} SHORT  | Bins: {len(stats_s)} | Top Feats: {len(feats_s)}")

        return result
    except Exception as e:
        print(f"  {sym}: ОШИБКА -> {e}")
        return None


def train_all():
    global models_long, models_short, empirical_stats, last_train_time, best_features

    for sym in SYMBOLS:
        try:
            m_dict = train_model(sym)

            if not m_dict:
                print(f" {sym}: skipped (insufficient class diversity)")
                continue
            if m_dict:
                if m_dict.get("LONG"):
                    models_long[sym] = m_dict["LONG"]
                    empirical_stats["LONG"][sym] = m_dict["STATS"]["LONG"]
                    best_features["LONG"][sym] = m_dict["FEATURES"]["LONG"]

                if m_dict.get("SHORT"):
                    models_short[sym] = m_dict["SHORT"]
                    empirical_stats["SHORT"][sym] = m_dict["STATS"]["SHORT"]
                    best_features["SHORT"][sym] = m_dict["FEATURES"]["SHORT"]

        except Exception as e:
            print(f"❌ {sym}: {e}")

    last_train_time = time.time()


async def send_msg(text):
    try:
        await bot_telegram.send_message(CHAT_ID, text, parse_mode="HTML")
    except Exception as e:
        print(f" ОШИБКА ТЕЛЕГРАМ: {e}")


def check_emergency_exit(sym, state, df, model):
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


def check_portfolio_kill_switch():
    """Считает Peak-to-Trough Drawdown и управляет Persistent состоянием"""
    # 1. Проверяем не находимся ли мы уже в отключке
    kill_until = trade_stats.get("kill_switch_until", 0)
    if time.time() < kill_until:
        remain_hours = (kill_until - time.time()) / 3600
        print(f" Отключение бота. Осталось: {remain_hours:.1f} часов.")
        return True

    # 2. Считаем Drawdown
    trades = trade_stats.get("closed_trades", [])
    if len(trades) < 10:
        return False  # Недостаточно данных

    equity = 100.0
    peak = 100.0
    current_dd = 0.0

    # Берем последние 50 сделок для Rolling Drawdown
    for t in trades[-50:]:
        net_pnl = float(t.get("net_pnl_pct", 0))
        equity *= 1 + (net_pnl / 100)

        if equity > peak:
            peak = equity

        dd = (peak - equity) / peak * 100
        if dd > current_dd:
            current_dd = dd

    # 3. Если просадка > 5%, уходим в persistent sleep на 72 часа
    if current_dd >= 5.0:
        freeze_time = time.time() + (86400 * 3)  # 72 часа
        trade_stats["kill_switch_until"] = freeze_time
        persist_trade_stats()

        msg = f"<b>KILL SWITCH ACTIVATED</b>\nPeak-to-Trough DD достиг {current_dd:.2f}%.\nТорговля заморожена на 72 часа. Состояние сохранено в JSON."
        print(msg)
        asyncio.create_task(send_msg(msg))
        return True

    return False


async def process_market():
    global last_candle_fetch_time, live_data_cache
    current_time = time.time()

    if check_portfolio_kill_switch():
        await asyncio.sleep(60)  # ждем минуту и выходим из цикла
        return

    active_symbols = list(set(list(models_long.keys()) + list(models_short.keys())))
    if not active_symbols:
        print("Нет обученных моделей. Жду обучения...")
        return

    try:
        tickers = exchange.fetch_tickers(active_symbols)
    except Exception as e:
        print(f" Ошибка тикеров: {e}")
        return

    # Загрузка графиков
    if current_time - last_candle_fetch_time > 30:
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
                print(f"   {sym}: {e}")
        print(f"Загружено: {ok_cnt}/{len(active_symbols)} пар.\n")
        last_candle_fetch_time = current_time

    # ДАШБОРД
    print_portfolio_status(tickers)

    #  УПРАВЛЕНИЕ ПОЗИЦИЯМИ
    for sym, state in list(trade_states.items()):
        if not state.get("active") or sym not in tickers:
            continue
        try:
            current_price = tickers[sym].get("bid") or tickers[sym].get("last")
            if not current_price:
                continue

            entry = state["entry"]
            side = state.get("side", "LONG")  # Узнаем направление

            # PnL считается
            if side == "LONG":
                gross_pnl = (current_price - entry) / entry * 100
            else:
                gross_pnl = (entry - current_price) / entry * 100

            net_pnl = gross_pnl - (ROUND_TRIP_COST_RATE * 100)
            elapsed_min = (time.time() - state["last_entry_ts"]) / 60
            mode = state.get("mode", "SCALP")
            exit_reason = None

            # БЕЗУБЫТОК
            atr_pct_entry = state["entry_atr"] / entry * 100
            if net_pnl > atr_pct_entry and not state.get("breakeven_hit"):
                if side == "LONG":
                    state["sl"] = entry + (entry * ROUND_TRIP_COST_RATE)
                else:
                    state["sl"] = entry - (entry * ROUND_TRIP_COST_RATE)
                state["breakeven_hit"] = True
                be_msg = f"<b>БЕЗУБЫТОК ({side})</b> {sym}\nТекущий профит: +{net_pnl:.2f}%\nСтоп переведен в Б/У."
                print(be_msg)
                await send_msg(be_msg)

            # ВЫХОДЫ
            if side == "LONG":
                if current_price >= state["tp"]:
                    exit_reason = f" TAKE PROFIT ({mode} LONG)"
                elif current_price <= state["sl"]:
                    exit_reason = f" STOP LOSS ({mode} LONG)"
            else:
                if current_price <= state["tp"]:
                    exit_reason = f" TAKE PROFIT ({mode} SHORT)"
                elif current_price >= state["sl"]:
                    exit_reason = f" STOP LOSS ({mode} SHORT)"

            if elapsed_min > (LOOKAHEAD_CANDLES * 60):
                exit_reason = f" TIME EXIT (ML Forecast Expired)"

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
                print(f" CLOSED {sym}: {exit_reason} | PnL: {net_pnl:.2f}%")
        except Exception as e:
            print(f"Ошибка трекинга {sym}: {e}")

    #  ПОИСК ВХОДОВ
    active_count = sum(1 for s in trade_states.values() if s.get("active"))

    active_longs = sum(
        1 for s in trade_states.values() if s.get("active") and s.get("side") == "LONG"
    )
    active_shorts = sum(
        1 for s in trade_states.values() if s.get("active") and s.get("side") == "SHORT"
    )
    MAX_DIRECTIONAL_EXPOSURE = 4

    entries_this_cycle = 0
    scan_log = []

    # НАСТОЯЩИЙ ДВИЖОК РЕЖИМА БИТКОИНА
    try:
        btc_bars = exchange.fetch_ohlcv("BTC/USDT", timeframe=TIMEFRAME, limit=300)
        btc_df = pd.DataFrame(
            btc_bars, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        btc_df["ema_50"] = EMAIndicator(
            close=btc_df["close"], window=50
        ).ema_indicator()
        btc_df["ema_200"] = EMAIndicator(
            close=btc_df["close"], window=200
        ).ema_indicator()

        btc_close = btc_df["close"].iloc[-1]
        btc_ema50 = btc_df["ema_50"].iloc[-1]
        btc_ema200 = btc_df["ema_200"].iloc[-1]

        # Режим Биткоина (Market Regime)
        btc_bullish = (btc_close > btc_ema50) and (btc_ema50 > btc_ema200)
        btc_bearish = (btc_close < btc_ema50) and (btc_ema50 < btc_ema200)
    except Exception as e:
        print(f" Ошибка BTC Regime: {e}")
        btc_bullish, btc_bearish = False, False

    active_symbols = list(set(list(models_long.keys()) + list(models_short.keys())))

    for sym in active_symbols:
        if (
            entries_this_cycle >= MAX_ENTRIES_PER_CYCLE
            or active_count >= MAX_CONCURRENT_POSITIONS
        ):
            break
        if trade_states[sym].get("active"):
            continue
        if time.time() - last_exit_times.get(sym, 0) < COOLDOWN_SECONDS:
            continue

        df = live_data_cache.get(sym)
        if df is None or len(df) < BACKTEST_LOOKBACK + 50:
            continue

        try:
            price = tickers[sym].get("ask") or tickers[sym].get("last")
            last_row = df.iloc[-1]
            features_row = df[FEATURES].iloc[[-1]]

            adx = last_row["adx"]
            atr = last_row["atr"]
            atr_pct = atr / price

            # Предсказываем только на ТОП фичах, которые модель отобрала при обучении
            if sym in models_long and sym in best_features["LONG"]:
                feats_l = best_features["LONG"][sym]
                prob_long = models_long[sym].predict_proba(df[feats_l].iloc[[-1]])[0][1]
            else:
                prob_long = 0.0

            if sym in models_short and sym in best_features["SHORT"]:
                feats_s = best_features["SHORT"][sym]
                prob_short = models_short[sym].predict_proba(df[feats_s].iloc[[-1]])[0][
                    1
                ]
            else:
                prob_short = 0.0

            # Выбор направления
            side, prob = (
                ("LONG", prob_long) if prob_long > prob_short else ("SHORT", prob_short)
            )

            # 1. PORTFOLIO CORRELATION FILTER
            if side == "LONG" and active_longs >= MAX_DIRECTIONAL_EXPOSURE:
                scan_log.append(
                    f" {sym} | Лонг отменен (Лимит: {active_longs}/{MAX_DIRECTIONAL_EXPOSURE})"
                )
                continue
            if side == "SHORT" and active_shorts >= MAX_DIRECTIONAL_EXPOSURE:
                scan_log.append(
                    f" {sym} | Шорт отменен (Лимит: {active_shorts}/{MAX_DIRECTIONAL_EXPOSURE})"
                )
                continue

            # 2. BTC REGIME FILTER
            if side == "LONG" and btc_bearish:
                scan_log.append(f" {sym} | Лонг отменен (Рынок падает, BTC Bearish)")
                continue
            if side == "SHORT" and btc_bullish:
                scan_log.append(f" {sym} | Шорт отменен (Рынок растет, BTC Bullish)")
                continue

            # 3. VOLATILITY FILTER
            if not (VOLATILITY_FLOOR <= atr_pct <= VOLATILITY_CEILING):
                scan_log.append(f" {sym} | Волатильность {atr_pct*100:.1f}% вне рамок")
                continue

            # 4. ЖЕСТКИЙ ФИЛЬТР УВЕРЕННОСТИ МОДЕЛИ (Который мы случайно удалили)
            min_prob = 0.55 if adx > 25 else 0.53
            if prob < min_prob:
                scan_log.append(
                    f"{sym} | Prob {prob:.3f} < {min_prob:.3f} (Слишком низкая уверенность)"
                )
                continue

            # Получаем Quarter Kelly + EV из эмпирической OOS статистики
            emp_dict = empirical_stats[side].get(sym, {})
            target_bucket = None
            for b in emp_dict.keys():
                if b.left < prob <= b.right:
                    target_bucket = b
                    break

            if not target_bucket:
                continue

            stats = emp_dict[target_bucket]
            emp_ev, kelly_size = stats["ev"], stats["kelly"]

            # Жесткий фильтр по ЭМПИРИЧЕСКОМУ Матожиданию
            if emp_ev <= 0.0003:  # Ожидаемая реальная прибыль
                scan_log.append(f" {sym} | Низкий реальный EV ({emp_ev*100:.2f}%)")
                continue

            # Режим и Множители
            mode = "IMPULSE" if adx > 25 else "SCALP"
            tp_mult, sl_mult = (2.0, 1.0) if mode == "IMPULSE" else (1.5, 1.0)

            # РАСЧЕТ УРОВНЕЙ ДЛЯ ВХОДА
            if side == "LONG":
                tp_price = price + (atr * tp_mult)
                sl_price = price - (atr * sl_mult)
            else:
                tp_price = price - (atr * tp_mult)
                sl_price = price + (atr * sl_mult)

            # РАСЧЕТ RISK/REWARD
            tp_pct, sl_pct = tp_mult * atr_pct, sl_mult * atr_pct
            rr = tp_pct / sl_pct
            if rr < 1.3:
                scan_log.append(f" {sym} | Низкий риск/прибыль ({rr:.2f})")
                continue
            #  БЛОК БЭКТЕСТА УДАЛЕН (ошибка утечек данных)

            # ВХОД
            trade_states[sym] = {
                "active": True,
                "side": side,
                "entry": price,
                "entry_atr": atr,
                "entry_prob": prob,
                "sl": sl_price,
                "tp": tp_price,
                "last_entry_ts": time.time(),
                "mode": mode,
                "expected_net_edge": emp_ev,
                "reward_risk": rr,
                "breakeven_hit": False,
            }
            save_state(trade_states)
            register_trade_open(sym, trade_states[sym])

            rec_size = kelly_size * 100
            dir_emoji = "📈" if side == "LONG" else "📉"

            msg = (
                f"{dir_emoji} <b>[SHADOW] {mode} {side} {sym}</b>\n"
                f" Вход: <b>{price:.5f}</b>\n"
                f" TP: {tp_price:.5f} |  SL: {sl_price:.5f}\n"
                f" Эмпирический EV: <b>{emp_ev*100:.2f}%</b>\n"
                f" Риск (Q-Kelly): <b>{rec_size:.1f}% депо</b>"
            )

            print(f" ВХОД {side} {sym} | P:{prob:.3f} | OOS EV:{emp_ev*100:.2f}%")
            await send_msg(msg)

            active_count += 1
            entries_this_cycle += 1

            if side == "LONG":
                active_longs += 1
            else:
                active_shorts += 1

        except Exception as e:
            print(f" Ошибка входа {sym}: {e}")

    print("\n ИТОГ ЦИКЛА:")
    if scan_log:
        for line in scan_log[:8]:
            print(f"  {line}")
        if len(scan_log) > 8:
            print(f"  ... и ещё {len(scan_log)-8} пропусков")
    else:
        print("   Все пары проверены, фильтры сработали штатно.")
    print("─" * 60)


async def main():
    print(" Bot Starting ")
    await send_msg("Bot Started.")

    train_all()

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
