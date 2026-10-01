"""The one place that loads trained models for interactive use (MCP server, notebook).

Stage 2 wide-box models (surrogate/models/), loaded once and cached:

    wide_ensemble()        5-member MLP ensemble -> t_arrive, v_max (mean, std)
    arrival_classifier()   5-member classifier   -> P(pig reaches the outlet before 2000 s)
    torch_surrogate()      differentiable wrapper of both, used by the inverse design

They were trained on arrived runs inside WIDE_RANGES (regressor) and on all runs
there (classifier); see surrogate/RESULTS.md sections 9-10.
"""

import os
from functools import lru_cache

import numpy as np

from surrogate.generate_wide_data import WIDE_RANGES
from surrogate.pipeline import INPUTS, P_RES, STATIC_OVER_DYNAMIC, D

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
AREA = np.pi * D ** 2 / 4
UNITS = {"p_in": "Pa", "mass": "kg", "F_fric": "N", "L": "m", "t_arrive": "s", "v_max": "m/s"}


@lru_cache(maxsize=None)
def wide_ensemble():
    from surrogate.train_surrogate import load_ensemble
    return load_ensemble(os.path.join(MODELS, "wide_ensemble.pt"))


@lru_cache(maxsize=None)
def arrival_classifier():
    from surrogate.regime import load_classifier
    return load_classifier(os.path.join(MODELS, "arrival_classifier.pt"))


@lru_cache(maxsize=None)
def torch_surrogate():
    from surrogate.inverse_design import TorchSurrogate
    return TorchSurrogate(wide_ensemble(), arrival_classifier())


def p_start(F_fric):
    """Lowest inlet pressure at which the pig breaks free [Pa] (analytic)."""
    return P_RES + STATIC_OVER_DYNAMIC * np.asarray(F_fric) / AREA


def predict(X):
    """X (n, 4) in SI units, column order INPUTS -> dict of (n,) arrays."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu, sd, _ = wide_ensemble().predict(X)
    p_arr, _ = arrival_classifier().predict_proba(X)
    return {"t_arrive": mu[:, 0], "t_arrive_std": sd[:, 0], "v_max": mu[:, 1],
            "v_max_std": sd[:, 1], "p_arrive": p_arr,
            "start_margin": X[:, INPUTS.index("p_in")] - p_start(X[:, INPUTS.index("F_fric")])}
