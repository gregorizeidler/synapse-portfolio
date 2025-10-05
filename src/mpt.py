from __future__ import annotations
import pandas as pd, numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns, objective_functions
from .utils import DATA_DIR, OUT_DIR, load_config

def mpt_initial_weights(close: pd.DataFrame, cfg: dict) -> pd.Series:
    rf = cfg["risk"]["risk_free_rate"]
    max_w = cfg["risk"]["max_weight"]
    min_w = cfg["risk"]["min_weight"]
    profile = cfg["mpt"]["profile"]
    target_vol_map = cfg["mpt"]["target_vol"]
    target_vol = target_vol_map.get(profile, 0.10)
    l2_reg = float(cfg["mpt"].get("l2_reg", 0.001))

    mu = expected_returns.mean_historical_return(close, compounding=True)
    S = risk_models.sample_cov(close)
    ef = EfficientFrontier(mu, S, weight_bounds=(min_w, max_w))
    ef.add_objective(objective_functions.L2_reg, gamma=l2_reg)
    try:
        ef.efficient_risk(target_volatility=target_vol)
    except Exception:
        ef.max_sharpe(risk_free_rate=rf)
    w = ef.clean_weights()
    w = pd.Series(w).reindex(close.columns).fillna(0.0)
    # normaliza e reclipa
    w = (w / w.sum()).clip(lower=min_w, upper=max_w)
    w = w / w.sum()
    return w

def main():
    cfg = load_config()
    close = pd.read_csv(DATA_DIR / "prices.csv", index_col=0, parse_dates=True)
    # remove CASH da otimização MPT
    cash_sym = cfg["universe"].get("cash_symbol", "CASH")
    cols = [c for c in close.columns if c != cash_sym]
    w0 = mpt_initial_weights(close[cols], cfg)
    # adiciona CASH com peso zero
    w0 = w0.reindex(close.columns).fillna(0.0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w0.to_csv(OUT_DIR / "mpt_weights.csv", header=["weight"])
    print("[mpt] Pesos iniciais MPT:")
    print(w0.to_string())
    print(f"[mpt] Salvo: {OUT_DIR / 'mpt_weights.csv'}")

if __name__ == "__main__":
    main()
