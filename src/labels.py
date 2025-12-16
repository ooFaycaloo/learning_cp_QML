import numpy as np
import pandas as pd
from utils import get_logger

LOGGER = get_logger(__name__)

def add_target_20d_score(df: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """
    Compute future return over `horizon` days and a score in [-1, 1].
    NOTE: scaling to [-1,1] should be calibrated on train only. Here we only compute raw future return.
    """
    out = df.copy()
    out[f"fut_ret_{horizon}"] = (out["Close"].shift(-horizon) - out["Close"]) / out["Close"]
    return out

def fit_score_scaler(train_future_returns: pd.Series, std_mult: float = 2.0) -> float:
    """
    Scale factor so that score = clip(fut_ret / scale, -1, 1).
    We use scale = std_mult * std(train_future_returns).
    """
    std = float(train_future_returns.dropna().std())
    scale = max(std_mult * std, 1e-8)
    LOGGER.info("Fitted score scale=%.6f (std_mult=%.2f, std=%.6f)", scale, std_mult, std)
    return scale

def apply_score(df: pd.DataFrame, horizon: int, scale: float) -> pd.DataFrame:
    out = df.copy()
    fr = out[f"fut_ret_{horizon}"]
    out["y_score"] = (fr / scale).clip(-1.0, 1.0)
    return out
