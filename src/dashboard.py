import streamlit as st
import pandas as pd, numpy as np, pathlib, os, matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"

st.set_page_config(page_title="Synapse Portfolio (Advanced)", layout="wide")
st.title("Synapse Portfolio — Dashboard (Advanced)")
st.caption("MPT + DRL (PPO) com overlay de risco (drawdown) e benchmarks.")

eq_path = OUT_DIR / "test_equity_curve.csv"
w_path = OUT_DIR / "test_weights.csv"
mpt_path = OUT_DIR / "mpt_weights.csv"
bench_path = OUT_DIR / "benchmarks.csv"
metrics_path = OUT_DIR / "metrics.json"

if not eq_path.exists():
    st.warning("Execute o pipeline: data → features → mpt → train_rl → benchmark → evaluate")
else:
    eq = pd.read_csv(eq_path, parse_dates=["date"]).set_index("date")
    st.subheader("Equity Curve (Teste) — RL vs Benchmarks")
    fig, ax = plt.subplots()
    eq["nav"].plot(ax=ax, label="RL")
    if bench_path.exists():
        bench = pd.read_csv(bench_path, index_col=0, parse_dates=True)
        for c in bench.columns:
            bench[c].plot(ax=ax, label=c)
    ax.set_xlabel("Data"); ax.set_ylabel("NAV"); ax.legend()
    st.pyplot(fig)

    st.subheader("Drawdown (Teste)")
    fig2, ax2 = plt.subplots()
    nav = eq["nav"]
    dd = nav / nav.cummax() - 1.0
    dd.plot(ax=ax2, label="RL")
    if bench_path.exists():
        for c in bench.columns:
            bdd = bench[c] / bench[c].cummax() - 1.0
            bdd.plot(ax=ax2, label=f"{c}")
    ax2.set_xlabel("Data"); ax2.set_ylabel("Drawdown"); ax2.legend()
    st.pyplot(fig2)

    st.subheader("Retornos e Turnover (Teste)")
    c1, c2 = st.columns(2)
    with c1:
        fig3, ax3 = plt.subplots()
        eq["ret"].plot(ax=ax3)
        ax3.set_xlabel("Data"); ax3.set_ylabel("Retorno")
        st.pyplot(fig3)
    with c2:
        fig4, ax4 = plt.subplots()
        eq["turnover"].plot(ax=ax4)
        ax4.set_xlabel("Data"); ax4.set_ylabel("Turnover")
        st.pyplot(fig4)

    if w_path.exists():
        w = pd.read_csv(w_path, parse_dates=["date"]).set_index("date")
        st.subheader("Pesos (últimos 30 dias de teste)")
        st.dataframe(w.tail(30))

    if mpt_path.exists():
        mpt_w = pd.read_csv(mpt_path, index_col=0)
        st.subheader("Pesos MPT iniciais")
        st.bar_chart(mpt_w)

    if metrics_path.exists():
        import json
        metrics = json.load(open(metrics_path))
        st.subheader("Métricas (Teste)")
        st.json(metrics)

st.info("Ajuste `config.yaml` para mudar o universo, datas, custos e hiperparâmetros.")
