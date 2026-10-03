"""Tune the two baselines on the lumped model, for the same reward the RL policy gets.

    python control/tune_baselines.py        # ~2 min on 4 cores

(a) fixed opening: the single u that maximises the return for the mean pig
    (110 kg, 2.5 kN) -- grid of step 0.005.
(b) PI: grid over (V_ref, Kp, Ki, u_min), mean return over 32 training-range
    pigs (seed 1; the evaluation uses other seeds).

Writes control/baselines.json.
"""

import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control import scenario as S  # noqa: E402
from control.controllers import PI, FixedOpening  # noqa: E402
from control.lumped import LumpedPig  # noqa: E402
from control.rollout import run  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "baselines.json")
TUNE_SEED, N_TUNE = 1, 32

GRID = {
    "v_ref": [6.5, 7.0, 7.25, 7.5, 7.75, 8.0],
    "kp": [0.0, 0.01, 0.02, 0.04, 0.08],
    "ki": [0.001, 0.002, 0.004, 0.007, 0.01, 0.02, 0.04],
    "u_min": [0.0, 0.1, 0.2, 0.25, 0.3],
}


def load():
    with open(OUT) as fh:
        return json.load(fh)


def _score_pi(cfg):
    plant = LumpedPig(record=False)
    conds = S.sample_conditions(N_TUNE, TUNE_SEED)
    rets = [run(plant, PI(**cfg), m, F)[0]["ret"] for m, F in conds]
    return float(np.mean(rets))


def main():
    t0 = time.perf_counter()
    plant = LumpedPig(record=False)
    m0, F0 = S.MEAN_CONDITION["mass"], S.MEAN_CONDITION["F_fric"]
    us = np.round(np.arange(0.05, 1.0001, 0.005), 3)
    rets = [run(plant, FixedOpening(u), m0, F0)[0]["ret"] for u in us]
    u_best = float(us[int(np.argmax(rets))])
    print(f"fixed opening: u = {u_best:.3f} (return {max(rets):.2f} for the mean pig)")

    cfgs = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    with Pool(4) as pool:
        scores = pool.map(_score_pi, cfgs)
    order = np.argsort(scores)[::-1]
    best = cfgs[order[0]]
    print(f"PI: {len(cfgs)} settings, best {best}, mean return {scores[order[0]]:.2f}")
    for i in order[:5]:
        print("   ", cfgs[i], round(scores[i], 2))

    out = {
        "fixed": {"u": u_best, "return_mean_pig": float(max(rets))},
        "pi": {**best, "mean_return_tuning_set": float(scores[order[0]])},
        "tuning": {"seed": TUNE_SEED, "n_conditions": N_TUNE, "grid": GRID,
                   "top5": [{**cfgs[i], "ret": float(scores[i])} for i in order[:5]],
                   "wall_s": time.perf_counter() - t0},
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
