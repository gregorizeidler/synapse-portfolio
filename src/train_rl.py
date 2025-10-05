from __future__ import annotations
import pandas as pd, numpy as np, pathlib, os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from .utils import ROOT, DATA_DIR, OUT_DIR, MODELS_DIR, load_config, ensure_dirs
from .env import PortfolioEnv

def build_env(cfg, split="train"):
    close = pd.read_csv(DATA_DIR / "prices.csv", index_col=0, parse_dates=True)
    feats = pd.read_csv(OUT_DIR / "features.csv", index_col=0, parse_dates=True)
    # recorte temporal
    if split == "train":
        start, end = cfg["walk_forward"]["train_start"], cfg["walk_forward"]["train_end"]
    else:
        start, end = cfg["walk_forward"]["test_start"], cfg["walk_forward"]["test_end"]
    close = close.loc[start:end].dropna(how="all").dropna(axis=1, how="any")
    feats = feats.loc[close.index]
    env = PortfolioEnv(prices=close, features=feats, cfg=cfg, train=(split=="train"))
    return env

def main():
    ensure_dirs()
    cfg = load_config()

    def make_train(): return build_env(cfg, split="train")
    def make_test():  return build_env(cfg, split="test")

    vec_env = DummyVecEnv([make_train])
    eval_env = DummyVecEnv([make_test])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        learning_rate=cfg["ppo"]["learning_rate"],
        n_steps=cfg["ppo"]["n_steps"],
        batch_size=cfg["ppo"]["batch_size"],
        gamma=cfg["ppo"]["gamma"],
        gae_lambda=cfg["ppo"]["gae_lambda"],
        clip_range=cfg["ppo"]["clip_range"],
        ent_coef=cfg["ppo"]["ent_coef"],
        vf_coef=cfg["ppo"]["vf_coef"],
        seed=cfg["seed"],
        tensorboard_log=str(OUT_DIR / "tb"),
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(MODELS_DIR / "best"),
        log_path=str(OUT_DIR / "eval"),
        eval_freq=max(1, cfg["ppo"]["n_steps"]),
        deterministic=True,
        render=False,
        n_eval_episodes=1,
    )

    model.learn(total_timesteps=int(cfg["ppo"]["total_timesteps"]), callback=eval_callback)
    model.save(MODELS_DIR / "ppo_synapse.zip")

    # Avaliação rápida
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1, deterministic=True)
    print(f"[train_rl] Avaliação – recompensa média: {mean_reward:.6f} ± {std_reward:.6f}")

    # Trajetória no conjunto de teste
    env = make_test()
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"date": dates, "nav": navs, "ret": rets, "turnover": turns}).set_index("date")
    df.to_csv(OUT_DIR / "test_equity_curve.csv")
    wdf = pd.DataFrame(weights, index=df.index, columns=env.assets)
    wdf.to_csv(OUT_DIR / "test_weights.csv")
    print(f"[train_rl] Salvos: {OUT_DIR / 'test_equity_curve.csv'}, {OUT_DIR / 'test_weights.csv'}")

if __name__ == "__main__":
    main()
