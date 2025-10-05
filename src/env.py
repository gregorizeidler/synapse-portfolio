from __future__ import annotations
import numpy as np, pandas as pd, gymnasium as gym
from gymnasium import spaces
from .utils import OUT_DIR, load_config

def project_capped_simplex(v, l, u, s=1.0, iters=100):
    v = np.asarray(v, dtype=float)
    n = v.size
    lvec = np.full(n, l, dtype=float)
    uvec = np.full(n, u, dtype=float)
    low = np.min(v - uvec)
    high = np.max(v - lvec)

    def S(tau):
        return float(np.minimum(np.maximum(v - tau, lvec), uvec).sum())

    for _ in range(iters):
        mid = 0.5 * (low + high)
        if S(mid) > s:
            low = mid
        else:
            high = mid
    tau = 0.5 * (low + high)
    w = np.minimum(np.maximum(v - tau, lvec), uvec)
    # correção numérica final
    if abs(w.sum() - s) > 1e-9:
        free = (w > l + 1e-9) & (w < u - 1e-9)
        if free.any():
            w[free] += (s - w.sum()) / free.sum()
            w = np.minimum(np.maximum(w, l), u)
    # renormaliza se necessário
    total = w.sum()
    if total <= 0:
        w = np.ones_like(w) / n
    else:
        w *= s / total
    return w

class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prices: pd.DataFrame, features: pd.DataFrame, cfg: dict, train: bool = True):
        super().__init__()
        self.prices = prices.copy()
        self.returns = self.prices.pct_change().fillna(0.0)
        self.features = features.copy().loc[self.returns.index]
        self.cfg = cfg
        self.train = train

        self.window = int(cfg["env"]["window_size"])
        self.temp = float(cfg["env"]["action_temperature"])
        self.step_scale = float(cfg["env"]["step_scale"])
        self.min_w = float(cfg["risk"]["min_weight"])
        self.max_w = float(cfg["risk"]["max_weight"])
        self.tx_bps = float(cfg["risk"]["transaction_cost_bps"]) / 1e4
        self.slp_bps = float(cfg["risk"]["slippage_bps"]) / 1e4
        self.turnover_pen = float(cfg["risk"]["turnover_penalty"])
        self.dev_pen = float(cfg["risk"]["deviation_penalty"])

        self.include_weights = bool(cfg["env"]["include_weights"])

        self.dd_trigger = float(cfg["risk_overlay"]["dd_trigger"])
        self.dd_hard = float(cfg["risk_overlay"]["dd_hard"])
        self.max_cash = float(cfg["risk_overlay"]["max_cash"])
        self.smoothing = float(cfg["risk_overlay"]["smoothing"])

        self.assets = list(self.prices.columns)
        self.n = len(self.assets)
        self.cash_sym = cfg["universe"].get("cash_symbol", "CASH")
        self.cash_idx = self.assets.index(self.cash_sym) if self.cash_sym in self.assets else -1

        # Obs: todas as features da data t-1 concat por ativo + pesos (opcional)
        if isinstance(self.features.columns, pd.MultiIndex):
            feature_names = sorted(set([c[1] for c in self.features.columns]))
            self.fdim = len(feature_names)
        else:
            self.fdim = self.features.shape[1] // self.n
        obs_dim = self.n * self.fdim + (self.n if self.include_weights else 0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n,), dtype=np.float32)

        # Pesos MPT iniciais (se existir)
        try:
            import pandas as pd
            w0 = pd.read_csv(OUT_DIR / "mpt_weights.csv", index_col=0).iloc[:,0].reindex(self.assets).fillna(0.0).values
        except Exception:
            w0 = np.ones(self.n) / self.n
        self.w_mpt = w0 / np.sum(w0)

        self._reset_state()

    def _reset_state(self):
        self.idx = self.prices.index
        self.t0 = self.window
        self.t = self.t0
        self.done = False
        self.nav = 1.0
        self.max_nav = 1.0
        self.w = self.w_mpt.copy()

    def _get_obs(self):
        t_idx = self.idx[self.t-1]
        row = self.features.loc[t_idx]
        if isinstance(row.index, pd.MultiIndex):
            arr = []
            for a in self.assets:
                sub = row.loc[a]
                arr.extend(sub.values.astype(np.float32))
            feat_vec = np.array(arr, dtype=np.float32)
        else:
            feat_vec = row.values.astype(np.float32).ravel()
        if self.include_weights:
            obs = np.concatenate([feat_vec, self.w.astype(np.float32)], axis=0)
        else:
            obs = feat_vec
        return obs

    def _apply_overlay(self, w_target: np.ndarray) -> np.ndarray:
        if self.cash_idx < 0:
            return w_target
        # drawdown atual
        dd = self.nav / (self.max_nav + 1e-12) - 1.0
        if dd >= self.dd_trigger:
            return w_target
        # Severidade contínua entre [0,1] conforme dd_trigger → dd_hard
        span = max(1e-6, abs(self.dd_hard) - abs(self.dd_trigger))
        sev = np.clip((abs(dd) - abs(self.dd_trigger)) / span, 0.0, 1.0)
        k = sev * self.max_cash  # quanto mover para cash
        # aplica mistura suave: reduz não-CASH por (1-k), aumenta CASH
        non_cash = np.ones_like(w_target, dtype=float)
        non_cash[self.cash_idx] = 0.0
        scale = (1.0 - k)
        w_nc = w_target * non_cash
        sum_nc = w_nc.sum()
        if sum_nc > 0:
            w_nc *= scale / sum_nc
        w_cash = 1.0 - w_nc.sum()
        w_new = w_nc.copy()
        w_new[self.cash_idx] = w_cash
        # suavização (EMA) p/ evitar saltos
        w_smooth = self.smoothing * w_target + (1 - self.smoothing) * w_new
        # reproject bounds + simplex
        w_proj = project_capped_simplex(w_smooth, self.min_w, self.max_w, s=1.0)
        return w_proj

    def reset(self, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._reset_state()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        if self.done:
            raise RuntimeError("Episode already done. Call reset().")
        action = np.asarray(action).reshape(-1)
        # Proposta: w_prev + step_scale * tanh(action)
        proposal = self.w + self.step_scale * np.tanh(action)
        w_target = project_capped_simplex(proposal, self.min_w, self.max_w, s=1.0)
        # Overlay de risco (drawdown → mais CASH)
        w_target = self._apply_overlay(w_target)

        # custos de transação por turnover
        turnover = float(np.sum(np.abs(w_target - self.w)))
        cost = (self.tx_bps + self.slp_bps) * turnover

        # retorno do passo
        r_t = float(np.dot(self.returns.iloc[self.t].values, w_target))

        # NAV
        gross = (1.0 + r_t)
        net = gross * (1.0 - cost)
        self.nav *= net
        self.max_nav = max(self.max_nav, self.nav)

        # penalizações adicionais
        dev = float(np.linalg.norm(w_target - self.w_mpt, ord=2))
        reward = np.log(max(1e-8, net)) - self.turnover_pen * turnover - self.dev_pen * dev

        # avançar
        self.w = w_target
        self.t += 1
        terminated = (self.t >= len(self.idx))
        truncated = False
        self.done = terminated or truncated
        obs = self._get_obs()
        info = {
            "nav": float(self.nav),
            "turnover": float(turnover),
            "return": float(r_t),
            "cost": float(cost),
            "weights": self.w.copy(),
        }
        return obs, float(reward), terminated, truncated, info
