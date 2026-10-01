"""The one place that loads trained models for interactive use (MCP server, notebook).

Stage 2 wide-box models (surrogate/models/), loaded once and cached:

    wide_ensemble()        5-member MLP ensemble -> t_arrive, v_max (mean, std)
    arrival_classifier()   5-member classifier   -> P(pig reaches the outlet before 2000 s)
    torch_surrogate()      differentiable wrapper of both, used by the inverse design
    deeponet()             stage-3 curve model: inputs -> pig speed V(xi) on 128 points

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


class _CurveEnsemble:
    """Stage-3 DeepONet ensemble: inputs -> pig speed on the 128-point xi grid."""

    def __init__(self, ck):
        import torch
        from surrogate.stage3.models import DeepONet, xi_features
        self.xi = np.asarray(ck["xi"])
        self._xif = xi_features(self.xi)
        self._mean, self._std, self._scale = ck["x_mean"], ck["x_std"], ck["v_scale"]
        self._nets = []
        for sd in ck["state_dicts"]:
            m = DeepONet(p=ck["p"], width=ck["width"])
            m.load_state_dict(sd)
            m.eval()
            self._nets.append(m)
        self._torch = torch

    def predict(self, X):
        """X (n, 4) SI -> (mean, std), each (n, 128) in m/s, on the grid self.xi."""
        X = np.atleast_2d(np.asarray(X, dtype=float))
        z = self._torch.tensor((np.log(X) - self._mean) / self._std, dtype=self._torch.float32)
        with self._torch.no_grad():
            P = np.stack([m(z, self._xif).numpy() for m in self._nets]) * self._scale
        return P.mean(0), P.std(0)


@lru_cache(maxsize=None)
def deeponet():
    """Stage-3 curve model (stage-1 input box, arrived runs only; see RESULTS_stage3.md)."""
    import torch
    ck = torch.load(os.path.join(HERE, "stage3", "models", "deeponet.pt"), weights_only=False)
    return _CurveEnsemble(ck)
