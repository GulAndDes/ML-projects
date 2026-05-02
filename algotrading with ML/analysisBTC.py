import ccxt
import pandas as pd
import numpy as np
import time
from datetime import datetime
from catboost import CatBoostClassifier
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score

# НАСТРОЙКИ

SYMBOL = "BTC/USDT"
TIMEFRAME = "1m"
LOG_FILE = "predictions_log.csv"
CANDLES_TO_TRAIN = 50000

CONFIDENCE_UP = 0.535
CONFIDENCE_DOWN = 0.465

FEATURES = [
    "log_ret",
    "atr_pct",
    "bb_width",
    "rsi",
    "vol_rel",
    "lower_wick_ratio",
    "vwap_dist",
    "ret_3m",
    "ret_5m",
    "ret_15m",
    "3m_wick_ratio",
    "15m_high_dist",
    "15m_low_dist",
    "ret_1",
    "ret_2",
    "ret_acc",
    "ema_diff",
    "volatility_regime",
    "up_streak",
    "intra_candle_momentum",
]

exchange = ccxt.binance({"enableRateLimit": True})
model = None

in_trade = False
active_trade = {}


# ЛОГИ


def log_result(data):
    df = pd.DataFrame([data])
    try:
        df.to_csv(
            LOG_FILE,
            mode="a",
            header=not pd.io.common.file_exists(LOG_FILE),
            index=False,
        )
    except:
        pass


# ДАННЫЕ
def fetch_historical_data(total_candles):
    print(f" Скачиваем {total_candles} свечей для обучения...")
    all_bars = []
    now = exchange.milliseconds()
    since = now - (total_candles * 60 * 1000)

    while len(all_bars) < total_candles:
        bars = exchange.fetch_ohlcv(
            SYMBOL, timeframe=TIMEFRAME, since=since, limit=1000
        )
        if not bars:
            break
        since = bars[-1][0] + 1
        all_bars.extend(bars)
        print(f"   Скачано {len(all_bars)} / {total_candles}...")
        time.sleep(0.1)

    df = pd.DataFrame(
        all_bars, columns=["ts", "open", "high", "low", "close", "volume"]
    )
    df.drop_duplicates(subset=["ts"], keep="last", inplace=True)
    df["time"] = pd.to_datetime(df["ts"], unit="ms")
    return df.tail(total_candles).reset_index(drop=True)


def fetch_recent_data(limit=200):
    bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["time"] = pd.to_datetime(df["ts"], unit="ms")
    return df


# ФИЧИ
def add_features(df):
    df = df.copy()

    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))

    candle_body = abs(df["open"] - df["close"]) + 1e-9
    df["lower_wick"] = np.minimum(df["open"], df["close"]) - df["low"]
    df["lower_wick_ratio"] = df["lower_wick"] / candle_body

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    rolling_vol = df["volume"].rolling(60).sum() + 1e-9
    df["vwap"] = (typical_price * df["volume"]).rolling(60).sum() / rolling_vol
    df["vwap_dist"] = (df["close"] - df["vwap"]) / df["close"]

    df["atr"] = AverageTrueRange(
        df["high"], df["low"], df["close"], window=14
    ).average_true_range()
    df["atr_pct"] = df["atr"] / df["close"]

    bb = BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / df["close"]

    df["rsi"] = RSIIndicator(df["close"], window=14).rsi() / 100
    df["vol_rel"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-9)

    # MULTI TF
    df["ret_3m"] = df["close"] / df["close"].shift(3) - 1
    df["ret_5m"] = df["close"] / df["close"].shift(5) - 1
    df["ret_15m"] = df["close"] / df["close"].shift(15) - 1

    df["3m_high"] = df["high"].rolling(3).max()
    df["3m_low"] = df["low"].rolling(3).min()
    df["3m_open"] = df["open"].shift(2)

    body_3m = abs(df["3m_open"] - df["close"]) + 1e-9
    wick_3m = np.minimum(df["3m_open"], df["close"]) - df["3m_low"]
    df["3m_wick_ratio"] = wick_3m / body_3m

    df["15m_high"] = df["high"].rolling(15).max()
    df["15m_low"] = df["low"].rolling(15).min()

    df["15m_high_dist"] = (df["15m_high"] - df["close"]) / df["close"]
    df["15m_low_dist"] = (df["close"] - df["15m_low"]) / df["close"]

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_2"] = df["close"].pct_change(2)
    df["ret_acc"] = df["ret_1"] - df["ret_2"]

    ema_fast = df["close"].ewm(span=5).mean()
    ema_slow = df["close"].ewm(span=20).mean()
    df["ema_diff"] = (ema_fast - ema_slow) / df["close"]

    df["volatility"] = df["log_ret"].rolling(20).std()
    df["volatility_regime"] = df["volatility"] / df["volatility"].rolling(100).mean()

    df["up_streak"] = (df["log_ret"] > 0).astype(int).rolling(5).sum()

    # Насколько текущая цена ушла от цены открытия этой минуты
    df["intra_candle_momentum"] = (df["close"] - df["open"]) / df["open"]

    df.dropna(inplace=True)
    return df


# TARGET (1 минута)
def make_target(df):
    df = df.copy()

    # Возвращаем предсказание ровно на 1 свечу (1 минуту) вперед
    future_price = df["close"].shift(-1)
    future_return = (future_price - df["close"]) / df["close"]

    # порог шума
    noise_threshold = df["atr_pct"] * 0.15

    df["target"] = np.where(
        future_return > noise_threshold,
        1,
        np.where(future_return < -noise_threshold, 0, np.nan),
    )

    df.dropna(subset=["target"], inplace=True)
    return df


# ОБУЧЕНИЕ
def train_model():
    global model

    df = fetch_historical_data(CANDLES_TO_TRAIN)
    df = df.iloc[:-1]

    df = add_features(df)
    df = make_target(df)

    X = df[FEATURES]
    y = df["target"]

    t1 = int(len(df) * 0.7)
    t2 = int(len(df) * 0.85)

    X_train, y_train = X.iloc[:t1], y.iloc[:t1]
    X_val, y_val = X.iloc[t1:t2], y.iloc[t1:t2]
    X_test, y_test = X.iloc[t2:], y.iloc[t2:]

    print(" Обучаем базовую модель...")
    base_model = CatBoostClassifier(
        iterations=800,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=5,
        auto_class_weights="Balanced",
        eval_metric="AUC",
        verbose=False,
    )

    base_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50)

    print(" Калибруем вероятности...")
    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    calibrated.fit(X_val, y_val)

    model = calibrated

    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)

    print(f" ROC-AUC: {auc:.4f}")
    print(" Запуск сканирования...\n")


# ВХОД
def try_to_enter():
    global in_trade, active_trade

    df = fetch_recent_data(200)

    # Берем 100% живую цену последней сделки в эту секунду
    try:
        ticker = exchange.fetch_ticker(SYMBOL)
        live_price = ticker["last"]

        # Вшиваем эту живую цену в нашу формирующуюся свечу
        df.at[df.index[-1], "close"] = live_price
        # Расширяем тени свечи, если живая цена вышла за их пределы
        df.at[df.index[-1], "high"] = max(df.iloc[-1]["high"], live_price)
        df.at[df.index[-1], "low"] = min(df.iloc[-1]["low"], live_price)
    except Exception as e:
        pass  # Если запрос тикера сорвется, используем обычную цену

    df_features = add_features(df)
    last = df_features.iloc[-1]

    current_price = last["close"]
    X = df_features[FEATURES].iloc[[-1]]
    prob = model.predict_proba(X)[0][1]

    time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if last["atr_pct"] < 0.0003:
        return

    direction = "SKIP"
    if prob >= CONFIDENCE_UP:
        direction = "UP"
    elif prob <= CONFIDENCE_DOWN:
        direction = "DOWN"

    if direction == "SKIP":
        print(f"[{time_str}] Цена: {current_price:.2f} |  Жду (prob={prob:.3f})")
    else:
        print("-" * 40)
        print(f" [{time_str}] СИГНАЛ ВХОДА!")
        print(
            f" Направление: {direction} | Цена: {current_price:.2f} | Вероятность: {prob:.3f}"
        )
        print("⏳ Ждем 1 минуту для результата...")
        print("-" * 40)

        in_trade = True
        active_trade = {
            "entry_time": datetime.now(),
            "entry_time_str": time_str,
            "entry_price": current_price,
            "direction": direction,
            "prob": prob,
        }


# РЕЗУЛЬТАТ (1 МИНУТА)
def check_trade_result():
    global active_trade

    df = fetch_recent_data(5)
    close_price = df.iloc[-1]["close"]

    prev_dir = active_trade["direction"]
    prev_price = active_trade["entry_price"]

    real_dir = "UP" if close_price > prev_price else "DOWN"
    result = "WIN " if real_dir == prev_dir else "LOSS "

    log_data = {
        "time": active_trade["entry_time_str"],
        "entry_price": prev_price,
        "exit_price": close_price,
        "predicted": prev_dir,
        "real": real_dir,
        "prob": active_trade["prob"],
        "result": result,
    }
    log_result(log_data)

    print(
        f"\nИТОГ: {result} | Вход: {prev_price:.2f} -> Выход: {close_price:.2f} (Факт: {real_dir})\n"
    )
    print("Сканирования рынка...")


def run_bot():
    global in_trade, active_trade

    while True:
        now = datetime.now()

        if in_trade:
            delta = (now - active_trade["entry_time"]).seconds

            # ЖДЕМ 1 МИНУТУ
            if delta >= 60:
                check_trade_result()
                in_trade = False

        else:
            if now.second % 10 == 0:
                try_to_enter()
                time.sleep(1)

        time.sleep(0.2)


if __name__ == "__main__":
    train_model()
    run_bot()
