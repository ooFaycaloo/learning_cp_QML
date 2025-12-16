import pandas as pd
from utils import get_logger

LOGGER = get_logger(__name__)

def load_ohlc_from_xlsx(xlsx_path: str, sheet_name: str) -> pd.DataFrame:
    """
    Dataset format (observed on the provided xlsx):
    - First row contains the real column names (Date, Open, High, Low, Close, SMAVG(50), SMAVG(100), SMAVG(240))
    - Column names in Excel are messy ("Tableau 1", "Unnamed: 1", ...)
    """
    LOGGER.info("Loading sheet=%s from %s", sheet_name, xlsx_path)
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")
    # First row is header row with actual names
    header = raw.iloc[0].tolist()
    df = raw.iloc[1:].copy()
    df.columns = header

    # Clean names
    rename = {
        "SMAVG (50)  on Close": "smavg_50",
        "SMAVG (100)  on Close": "smavg_100",
        "SMAVG (240)  on Close": "smavg_240",
    }
    df = df.rename(columns=rename)

    # Types
    df["Date"] = pd.to_datetime(df["Date"])
    num_cols = [c for c in df.columns if c != "Date"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Sort ascending time
    df = df.sort_values("Date").reset_index(drop=True)

    # Drop fully empty rows
    df = df.dropna(subset=["Close"]).reset_index(drop=True)

    LOGGER.info("Loaded %d rows, columns=%s", len(df), list(df.columns))
    return df
