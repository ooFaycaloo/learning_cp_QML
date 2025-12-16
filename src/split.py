import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from utils import get_logger

LOGGER = get_logger(__name__)

def time_series_splits(n_samples: int, n_splits: int = 5, embargo: int = 0):
    """
    Generator yielding (train_idx, test_idx) for time series.
    Optional embargo to remove last `embargo` points of train before each test.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_idx, test_idx in tscv.split(np.arange(n_samples)):
        if embargo > 0:
            train_idx = train_idx[:-embargo] if len(train_idx) > embargo else train_idx
        yield train_idx, test_idx
