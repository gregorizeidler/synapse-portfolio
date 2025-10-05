from __future__ import annotations
import os, pathlib, yaml, numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_config():
    with open(ROOT / "src" / "config.yaml", "r") as f:
        return yaml.safe_load(f)

def business_dates(start: str, end: str):
    return pd.bdate_range(start=start, end=end)

def drawdown_series(nav: pd.Series) -> pd.Series:
    peak = nav.cummax()
    return nav / peak - 1.0

def ann_factor(freq: str = "1d") -> int:
    return 252 if freq == "1d" else 52 if freq == "1w" else 12

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
