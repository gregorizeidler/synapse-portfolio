from __future__ import annotations
import pandas as pd, numpy as np
from .utils import DATA_DIR, OUT_DIR, load_config

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1/window, adjust=False).mean()
    roll_down = down.ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))

def make_features(close: pd.DataFrame) -> pd.DataFrame:
    feats = {}
    for col in close.columns:
        px = close[col].ffill().dropna()
        ret = px.pct_change()
        feats[(col, "ret_1")] = ret
        feats[(col, "ret_5")] = px.pct_change(5)
        feats[(col, "ret_20")] = px.pct_change(20)
        feats[(col, "mom_20")] = px.pct_change(20)
        feats[(col, "vol_20")] = ret.rolling(20).std()
        feats[(col, "vol_60")] = ret.rolling(60).std()
        feats[(col, "rsi_14")] = rsi(px, 14)
    feat_df = pd.concat(feats, axis=1).dropna().astype(float)
    return feat_df

def main():
    close = pd.read_csv(DATA_DIR / "prices.csv", index_col=0, parse_dates=True)
    feats = make_features(close)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUT_DIR / "features.csv")
    print(f"[features] Salvo: {OUT_DIR / 'features.csv'}  shape={feats.shape}")

if __name__ == "__main__":
    main()
