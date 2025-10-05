from __future__ import annotations
import pandas as pd, numpy as np
from stable_baselines3 import PPO
from .utils import DATA_DIR, OUT_DIR, MODELS_DIR, load_config
from .env import PortfolioEnv
from .train_rl import build_env

def main():
    cfg = load_config()
    env = build_env(cfg, split="test")

    model_path = MODELS_DIR / "ppo_synapse.zip"
    if not model_path.exists():
        raise FileNotFoundError("Modelo PPO não encontrado. Execute: python -m src.train_rl")
    model = PPO.load(model_path)

    obs, info = env.reset()
    navs, dates, rets, turns, weights = [], [], [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        dates.append(env.idx[env.t-1])
        navs.append(info["nav"])
        rets.append(info["return"])
        turns.append(info["turnover"])
        weights.append(info["weights"])

    df = pd.DataFrame({"date": dates, "nav": navs, "ret": rets, "turnover": turns}).set_index("date")
    df.to_csv(OUT_DIR / "backtest_equity_curve.csv")
    wdf = pd.DataFrame(weights, index=df.index, columns=env.assets)
    wdf.to_csv(OUT_DIR / "backtest_weights.csv")
    print(f"[backtest] Salvos: {OUT_DIR / 'backtest_equity_curve.csv'}, {OUT_DIR / 'backtest_weights.csv'}")

if __name__ == "__main__":
    main()
