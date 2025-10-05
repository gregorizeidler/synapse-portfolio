from __future__ import annotations
import pandas as pd, numpy as np
from .utils import DATA_DIR, OUT_DIR, load_config

def _portfolio_nav(close: pd.DataFrame, weights: pd.Series) -> pd.Series:
    rets = close.pct_change().fillna(0.0)
    port_ret = rets.mul(weights, axis=1).sum(axis=1)
    nav = (1.0 + port_ret).cumprod()
    return nav

def main():
    cfg = load_config()
    close = pd.read_csv(DATA_DIR / "prices.csv", index_col=0, parse_dates=True)
    start, end = cfg["walk_forward"]["test_start"], cfg["walk_forward"]["test_end"]
    close = close.loc[start:end]
    rets = close.pct_change().fillna(0.0)
    assets = list(close.columns)
    cash_sym = cfg["universe"].get("cash_symbol", "CASH")

    # Equal-weight (exceto CASH)
    cols_nc = [c for c in assets if c != cash_sym]
    ew = pd.Series(1.0/len(cols_nc), index=cols_nc)
    ew = ew.reindex(assets).fillna(0.0)

    # MPT-only (pesos fixos)
    w_mpt = pd.read_csv(OUT_DIR / "mpt_weights.csv", index_col=0).iloc[:,0].reindex(assets).fillna(0.0)

    # SPY buy&hold, se existir
    bench = {}
    nav_ew = _portfolio_nav(close, ew)
    nav_mpt = _portfolio_nav(close, w_mpt)
    bench["EW"] = nav_ew
    bench["MPT_only"] = nav_mpt
    if "SPY" in assets:
        spy_nav = (close["SPY"].pct_change().fillna(0.0) + 1.0).cumprod()
        bench["SPY"] = spy_nav

    df = pd.DataFrame(bench)
    df.index.name = "date"
    df.to_csv(OUT_DIR / "benchmarks.csv")
    print(f"[benchmark] Salvo: {OUT_DIR / 'benchmarks.csv'}  cols={list(df.columns)}")

if __name__ == "__main__":
    main()
