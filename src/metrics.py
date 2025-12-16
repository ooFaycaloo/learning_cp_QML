import numpy as np
from scipy.stats import spearmanr
from utils import get_logger

LOGGER = get_logger(__name__)

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.sign(y_true)
    yp = np.sign(y_pred)
    mask = ~np.isnan(yt) & ~np.isnan(yp)
    if mask.sum() == 0:
        return float("nan")
    return float((yt[mask] == yp[mask]).mean())

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() < 5:
        return float("nan")
    return float(spearmanr(y_true[mask], y_pred[mask]).correlation)

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((y_true[mask] - y_pred[mask]) ** 2))
