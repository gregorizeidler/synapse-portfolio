from __future__ import annotations
import numpy as np, pandas as pd, json, matplotlib.pyplot as plt
from .utils import OUT_DIR, DATA_DIR, load_config, drawdown_series

def max_drawdown(equity: pd.Series) -> float:
    dd = drawdown_series(equity)
    return float(dd.min())

def calmar_ratio(equity: pd.Series, period_per_year: int = 252) -> float:
    ret = equity.pct_change().fillna(0.0)
    ann_ret = (1 + ret.mean())**period_per_year - 1
    mdd = abs(max_drawdown(equity))
    return float(ann_ret / mdd) if mdd > 1e-12 else float("nan")

def sharpe_ratio(returns: pd.Series, rf: float = 0.0, period_per_year: int = 252) -> float:
    excess = returns - rf / period_per_year
    mu = excess.mean() * period_per_year
    sigma = excess.std() * np.sqrt(period_per_year)
    return float(mu / sigma) if sigma > 1e-12 else float("nan")

def sortino_ratio(returns: pd.Series, rf: float = 0.0, period_per_year: int = 252) -> float:
    excess = returns - rf / period_per_year
    downside = excess.clip(upper=0)
    dd_std = downside.std() * np.sqrt(period_per_year)
    mu = excess.mean() * period_per_year
    return float(mu / dd_std) if dd_std > 1e-12 else float("nan")

def alpha_beta(strategy_ret: pd.Series, bench_ret: pd.Series, rf: float = 0.0, period_per_year: int = 252):
    # CAPM OLS: r_s - r_f = alpha + beta (r_b - r_f) + eps
    y = strategy_ret - rf / period_per_year
    x = bench_ret - rf / period_per_year
    x = x.reindex_like(y).fillna(0.0)
    X = np.vstack([np.ones(len(x)), x.values]).T
    beta_hat = np.linalg.lstsq(X, y.values, rcond=None)[0]
    alpha, beta = float(beta_hat[0] * period_per_year), float(beta_hat[1])
    return alpha, beta

def make_plots(equity_rl: pd.Series, equity_bench: pd.DataFrame):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Equity curves
    fig1, ax1 = plt.subplots()
    equity_rl.plot(ax=ax1, label="RL")
    if equity_bench is not None and not equity_bench.empty:
        equity_bench.plot(ax=ax1)
    ax1.set_title("Equity Curve (Teste)")
    ax1.set_xlabel("Data"); ax1.set_ylabel("NAV")
    ax1.legend()
    fig1.savefig(OUT_DIR / "equity_curve.png", bbox_inches="tight")
    plt.close(fig1)

    # Drawdowns
    fig2, ax2 = plt.subplots()
    drawdown_series(equity_rl).plot(ax=ax2, label="RL")
    if equity_bench is not None and not equity_bench.empty:
        for c in equity_bench.columns:
            drawdown_series(equity_bench[c]).plot(ax=ax2, label=f"{c}")
    ax2.set_title("Drawdown (Teste)")
    ax2.set_xlabel("Data"); ax2.set_ylabel("Drawdown")
    ax2.legend()
    fig2.savefig(OUT_DIR / "drawdown.png", bbox_inches="tight")
    plt.close(fig2)

def main():
    cfg = load_config()
    rf = float(cfg["risk"]["risk_free_rate"])
    # Carrega RL
    eq = pd.read_csv(OUT_DIR / "test_equity_curve.csv", parse_dates=["date"]).set_index("date")
    equity_rl = eq["nav"]
    ret_rl = eq["ret"].fillna(0.0)

    # Benchmarks
    bench_path = OUT_DIR / "benchmarks.csv"
    bench = pd.read_csv(bench_path, index_col=0, parse_dates=True) if bench_path.exists() else None

    # Métricas RL
    metrics = {
        "CAGR": float((equity_rl.iloc[-1] / equity_rl.iloc[0])**(252/len(equity_rl)) - 1.0),
        "Sharpe": sharpe_ratio(ret_rl, rf),
        "Sortino": sortino_ratio(ret_rl, rf),
        "MaxDrawdown": max_drawdown(equity_rl),
        "Calmar": calmar_ratio(equity_rl),
        "AvgTurnover": float(eq["turnover"].mean()),
    }

    # Alfa/Beta vs SPY se existir
    if bench is not None and "SPY" in bench.columns:
        spy_ret = bench["SPY"].pct_change().fillna(0.0)
        alpha, beta = alpha_beta(ret_rl, spy_ret, rf)
        metrics["Alpha_vs_SPY"] = alpha
        metrics["Beta_vs_SPY"] = beta

    # Plots
    make_plots(equity_rl, bench if bench is not None else None)

    # Salva métricas e report
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Report simples em Markdown
    lines = ["# Synapse Portfolio – Report (Teste)",
             "",
             "## Métricas",
             "```",
             json.dumps(metrics, indent=2),
             "```",
             "",
             "## Gráficos",
             "- ![Equity](equity_curve.png)",
             "- ![Drawdown](drawdown.png)"]
    with open(OUT_DIR / "report.md", "w") as f:
        f.write("\n".join(lines))

    print(f"[evaluate] Métricas salvas em {OUT_DIR / 'metrics.json'} e relatório em {OUT_DIR / 'report.md'}")

if __name__ == "__main__":
    main()
