"""Fast checks for the stage-3 curve models (no pigsim runs).

    python -m pytest tests/test_stage3.py      # ~5 s
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from surrogate.stage3 import run_stage3 as R  # noqa: E402
from surrogate.stage3.models import DeepONet, xi_features  # noqa: E402

D = np.load(os.path.join(ROOT, "surrogate", "stage3", "curves.npz"))


def test_true_curve_integrates_to_pigsim_arrival_time():
    X, V, xi = D["X"], D["V"].astype(float), D["xi"]
    t = D["t_break"] + R.travel_time(V, X, xi)
    assert np.mean(np.abs(t - D["t_arrive"]) / D["t_arrive"]) < 0.005


def test_saved_deeponet_predicts_test_curves():
    ck = torch.load(os.path.join(ROOT, "surrogate", "stage3", "models", "deeponet.pt"),
                    weights_only=False)
    xif = xi_features(ck["xi"])
    nets = []
    for sd in ck["state_dicts"]:
        m = DeepONet(p=ck["p"], width=ck["width"])
        m.load_state_dict(sd)
        m.eval()
        nets.append(m)
    _, _, te = R.split(len(D["X"]))
    z = torch.tensor((np.log(D["X"][te]) - ck["x_mean"]) / ck["x_std"], dtype=torch.float32)
    with torch.no_grad():
        P = np.stack([m(z, xif).numpy() for m in nets]) * ck["v_scale"]
    V = D["V"][te]
    err = np.sqrt(np.sum((P.mean(0) - V) ** 2, 1) / np.sum(V ** 2, 1))
    assert P.shape == (5, len(te), 128)
    assert np.mean(err) < 0.02          # metrics.json: ~1.2 % (unweighted grid here)
