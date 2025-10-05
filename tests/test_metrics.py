import pandas as pd, numpy as np
from src.evaluate import sharpe_ratio, sortino_ratio, max_drawdown

def test_metrics():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, 252))
    eq = (1 + r).cumprod()
    assert isinstance(sharpe_ratio(r), float)
    assert isinstance(sortino_ratio(r), float)
    assert max_drawdown(eq) <= 0
