"""Post-training checks quoted in RESULTS.md.  Needs models/ensemble.pt.

    python surrogate/checks.py          # ~2 min on 4 cores (10 pigsim runs)

1. extrapolation: pigsim vs surrogate at 4 points outside the training box
2. label noise:   v_max of the 6 worst-predicted runs, re-run with a 4x finer
                  time step (cfl_pig 0.02 -> 0.005, dt_max 1 -> 0.25 s)
3. sensitivity:   output change when one input goes low -> high, averaged over
                  2000 random points in the box (surrogate only)
"""

import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate import pipeline as P  # noqa: E402
from surrogate.train_surrogate import HERE, INPUTS, OUTPUTS, RANGES, load_ensemble  # noqa: E402

MID = {k: 0.5 * (RANGES[k][0] + RANGES[k][1]) for k in INPUTS}
EXTRAP = [dict(MID, L=4500.0), dict(MID, L=250.0),
          dict(MID, p_in=2.0e5), dict(MID, p_in=11.0e5)]


def _sim(x):
    return P.simulate(**x)


def _v_max_fine(x):
    sol = P.build(**x)
    sol.run(2000.0, dt0=0.005, dt_max=0.25, cfl_pig=0.005, s_stop=0.999 * x["L"],
            verbose=False)
    return float(np.max(sol.history()["V_pig"]))


def main():
    ens = load_ensemble()
    pool = Pool(4)

    print("1. extrapolation (other inputs at mid-range)")
    res = pool.map(_sim, EXTRAP)
    mu, sd, _ = ens.predict(np.array([[x[k] for k in INPUTS] for x in EXTRAP]))
    for x, r, m, s in zip(EXTRAP, res, mu, sd):
        tag = f"p_in={x['p_in']/1e5:4.1f} bar, L={x['L']:5.0f} m"
        if not r["arrived"]:
            print(f"  {tag}: pigsim -> pig never arrives (v_max {r['v_max']:.2f} m/s); "
                  f"surrogate -> t_arrive {m[0]:.1f} +- {s[0]:.2f} s, "
                  f"v_max {m[1]:.2f} +- {s[1]:.2f} m/s")
            continue
        for j, o in enumerate(OUTPUTS):
            err = m[j] - r[o]
            print(f"  {tag}  {o:8s} pigsim {r[o]:7.2f}  surrogate {m[j]:7.2f}  "
                  f"err {100*err/r[o]:+6.2f} %  std {s[j]:6.3f}  |err|/std {abs(err)/s[j]:5.1f}")

    print("\n2. v_max label noise (6 worst-predicted runs in data.csv)")
    d = pd.read_csv(os.path.join(HERE, "data.csv"))
    m_all, _, _ = ens.predict(d[INPUTS].to_numpy())
    d["pred"] = m_all[:, 1]
    worst = d.loc[(d["pred"] - d["v_max"]).abs().sort_values(ascending=False).index[:6]]
    fine = pool.map(_v_max_fine, [{k: r[k] for k in INPUTS} for _, r in worst.iterrows()])
    for (_, r), vf in zip(worst.iterrows(), fine):
        print(f"  id {int(r['id']):3d}: pigsim default dt {r['v_max']:6.2f}  "
              f"fine dt {vf:6.2f}  (shift {vf - r['v_max']:+5.2f})  surrogate {r['pred']:6.2f}")
    pool.close()

    print("\n3. main effects: input low -> high, mean change in % of the output")
    rng = np.random.default_rng(1)
    lo = np.array([RANGES[k][0] for k in INPUTS])
    hi = np.array([RANGES[k][1] for k in INPUTS])
    base = lo + (hi - lo) * rng.random((2000, len(INPUTS)))
    mu0, _, _ = ens.predict(base)
    for i, k in enumerate(INPUTS):
        a, b = base.copy(), base.copy()
        a[:, i], b[:, i] = lo[i], hi[i]
        rel = 100 * (ens.predict(b)[0] - ens.predict(a)[0]) / mu0
        print(f"  {k:7s} t_arrive {rel[:, 0].mean():+7.1f} %   v_max {rel[:, 1].mean():+6.1f} %")


if __name__ == "__main__":
    main()
