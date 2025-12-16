import numpy as np
import pandas as pd
from utils import get_logger

LOGGER = get_logger(__name__)

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=window).mean()
    return atr

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce stationary-ish, ML-friendly features based only on info available at time t.
    """
    LOGGER.info("Building features...")

    out = df.copy()

    # Log return (stationary-ish)
    out["log_ret_1"] = np.log(out["Close"]).diff()

    # Lags
    for k in [1, 2, 5, 10, 20]:
        out[f"log_ret_lag_{k}"] = out["log_ret_1"].shift(k)

    # Rolling stats on returns
    for w in [5, 10, 20, 50]:
        out[f"ret_mean_{w}"] = out["log_ret_1"].rolling(w, min_periods=w).mean()
        out[f"ret_vol_{w}"] = out["log_ret_1"].rolling(w, min_periods=w).std()

    # Momentum (cumulative return over window)
    for w in [5, 10, 20, 60]:
        out[f"mom_{w}"] = out["log_ret_1"].rolling(w, min_periods=w).sum()

    # Price vs moving averages (if present)
    for ma in ["smavg_50", "smavg_100", "smavg_240"]:
        if ma in out.columns:
            out[f"close_to_{ma}"] = (out["Close"] / out[ma]) - 1.0

    # RSI (normalized)
    out["rsi_14"] = _rsi(out["Close"], window=14) / 100.0

    # Bollinger %B (20)
    w = 20
    mid = out["Close"].rolling(w, min_periods=w).mean()
    sd = out["Close"].rolling(w, min_periods=w).std()
    upper = mid + 2 * sd
    lower = mid - 2 * sd
    out["bollinger_pctb_20"] = (out["Close"] - lower) / (upper - lower)

    # ATR relative (normalized by price)
    out["atr_rel_14"] = _atr(out, window=14) / out["Close"]

    # Optional: range features
    out["hl_range_rel"] = (out["High"] - out["Low"]) / out["Close"]
    out["oc_change_rel"] = (out["Close"] - out["Open"]) / out["Open"]

    LOGGER.info("Features built. Total columns=%d", out.shape[1])
    return out
