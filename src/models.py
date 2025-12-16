import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils import get_logger

LOGGER = get_logger(__name__)

@dataclass
class ModelSpec:
    name: str
    model: object

def get_baselines(random_state: int = 42):
    """
    Solid baselines for tabular time-series features.
    """
    ridge = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("model", Ridge(alpha=1.0, random_state=random_state))
    ])

    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )

    gbr = GradientBoostingRegressor(
        random_state=random_state,
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8
    )

    return [
        ModelSpec("ridge", ridge),
        ModelSpec("random_forest", rf),
        ModelSpec("gbrt", gbr),
    ]
