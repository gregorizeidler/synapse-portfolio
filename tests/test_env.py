import numpy as np, pandas as pd
from src.env import project_capped_simplex

def test_projection_basic():
    v = np.array([0.2, 0.2, 0.6])
    w = project_capped_simplex(v, l=0.0, u=0.5, s=1.0)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all() and (w <= 0.5 + 1e-9).all()

def test_projection_edge():
    v = np.array([10.0, -10.0, 0.3])
    w = project_capped_simplex(v, l=0.0, u=0.7, s=1.0)
    assert abs(w.sum() - 1.0) < 1e-6
    assert (w >= -1e-9).all() and (w <= 0.7 + 1e-9).all()
