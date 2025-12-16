from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    xlsx_path: str = "dataset_train.xlsx"
    sheet_name: str = "Gold"
    horizon_days: int = 20

    # Target scoring
    score_clip: float = 1.0  # final score clipped to [-1, +1]
    score_scale_std_mult: float = 2.0  # scale by (mult * std) computed on train

    # Splitting
    n_splits: int = 5
    embargo: int = 0  # optional gap between train/test to reduce leakage
