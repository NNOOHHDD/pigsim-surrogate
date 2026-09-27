"""Stage 2b data: widen the input box so that some pigs never start or never arrive.

    python surrogate/generate_wide_data.py          # ~20 min on 4 cores, 540 runs

Stage 1 kept p_in >= 3 bar on purpose, so every pig reached the outlet.  Here the
box is widened and the low-pressure band is sampled densely:

    set "wide":  LHS over the whole wide box (WIDE_RANGES)
    set "band":  LHS with p_in limited to 1.6 - 3.2 bar, where the pig starts
                 only if p_in > 1.5 bar + 1.2 F_fric / A (1.67 - 2.86 bar here)
    set "edge":  pigs that start but may be too slow: low friction (1 - 2 kN),
                 long pipe (2 - 3 km), p_in from 0.01 bar below to 0.05 bar
                 above the start threshold.  Added after the first 500 runs
                 showed no "started but not arrived" case at all (see RESULTS.md)

"Arrived" means the pig reached s = 0.999 L before t_end = 2000 s (the same
t_end as stage 1).  A pig that does not arrive either never starts (friction
wins at t = 0) or starts but is too slow to arrive in time.  Output:
data_wide.csv, one row per run.  The stage-1 data.csv is not touched; the
training script combines the two files.
"""

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate.pipeline import D, INPUTS, S0, build  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
T_END = 2000.0

WIDE_RANGES = {
    "p_in":   (1.6e5, 8.0e5),
    "mass":   (20.0, 200.0),
    "F_fric": (1.0e3, 8.0e3),
    "L":      (500.0, 3000.0),
}
BAND_P_IN = (1.6e5, 3.2e5)
EDGE = {"mass": (20.0, 200.0), "F_fric": (1.0e3, 2.0e3), "L": (2000.0, 3000.0),
        "margin": (-0.01e5, 0.05e5)}
AREA = np.pi * D ** 2 / 4
COLUMNS = (["id", "set"] + INPUTS
           + ["t_arrive", "v_max", "arrived", "started", "s_end", "n_steps", "wall_s"])


def lhs(ranges, n, seed):
    unit = qmc.LatinHypercube(d=len(INPUTS), seed=seed).random(n)
    lo = np.array([ranges[k][0] for k in INPUTS])
    hi = np.array([ranges[k][1] for k in INPUTS])
    return qmc.scale(unit, lo, hi)


def edge_points(n, seed):
    """LHS in (margin, mass, F_fric, L); p_in = start threshold + margin."""
    keys = ["margin", "mass", "F_fric", "L"]
    unit = qmc.LatinHypercube(d=4, seed=seed).random(n)
    v = qmc.scale(unit, [EDGE[k][0] for k in keys], [EDGE[k][1] for k in keys])
    p_in = 1.5e5 + 1.2 * v[:, 2] / AREA + v[:, 0]
    return np.column_stack([p_in, v[:, 1], v[:, 2], v[:, 3]])


def simulate_ext(p_in, mass, F_fric, L):
    """pipeline.simulate() plus where the pig ended up (needed for non-arrivals)."""
    t0 = time.perf_counter()
    sol = build(p_in, mass, F_fric, L)
    s_stop = 0.999 * L
    sol.run(T_END, dt0=0.02, dt_max=1.0, cfl_pig=0.02, s_stop=s_stop, verbose=False)
    h = sol.history()
    arrived = bool(h["s_pig"][-1] >= s_stop)
    t_arr = float(np.interp(s_stop, h["s_pig"][-2:], h["t"][-2:])) if arrived else float("nan")
    s_end = float(h["s_pig"][-1])
    return {"t_arrive": t_arr, "v_max": float(np.max(h["V_pig"])), "arrived": arrived,
            "started": bool(s_end > S0 + 0.5), "s_end": s_end, "n_steps": len(h["t"]),
            "wall_s": time.perf_counter() - t0}


def _run(job):
    i, tag, x = job
    try:
        r = simulate_ext(*x)
    except Exception as exc:  # a crash is recorded, not hidden
        print(f"  run {i} failed: {exc!r}", flush=True)
        r = {"t_arrive": float("nan"), "v_max": float("nan"), "arrived": False,
             "started": False, "s_end": float("nan"), "n_steps": 0, "wall_s": float("nan")}
    return i, tag, x, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wide", type=int, default=300)
    ap.add_argument("--n-band", type=int, default=200)
    ap.add_argument("--n-edge", type=int, default=40)
    ap.add_argument("--append", action="store_true",
                    help="keep the rows already in --out and run only the missing ids")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(HERE, "data_wide.csv"))
    args = ap.parse_args()

    Xw = lhs(WIDE_RANGES, args.n_wide, seed=7)
    Xb = lhs(dict(WIDE_RANGES, p_in=BAND_P_IN), args.n_band, seed=8)
    Xe = edge_points(args.n_edge, seed=9)
    nb = args.n_wide + args.n_band
    jobs = ([(i, "wide", x) for i, x in enumerate(Xw)]
            + [(args.n_wide + i, "band", x) for i, x in enumerate(Xb)]
            + [(nb + i, "edge", x) for i, x in enumerate(Xe)])
    old = []
    if args.append and os.path.exists(args.out):
        with open(args.out) as f:
            old = [r for r in csv.reader(f)][1:]
        done = {int(r[0]) for r in old}
        jobs = [j for j in jobs if j[0] not in done]
    n = len(jobs)
    print(f"running {n} pigsim cases on {args.workers} workers -> {args.out}")
    t0 = time.perf_counter()
    rows = [[int(r[0]), r[1]] + r[2:] for r in old]
    with Pool(args.workers) as pool, open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
        for k, (i, tag, x, r) in enumerate(pool.imap_unordered(_run, jobs), 1):
            row = [i, tag, *x, r["t_arrive"], r["v_max"], int(r["arrived"]),
                   int(r["started"]), r["s_end"], r["n_steps"], r["wall_s"]]
            w.writerow(row)
            f.flush()
            rows.append(row)
            if k % 25 == 0 or k == n:
                el = time.perf_counter() - t0
                print(f"  {k}/{n} done, {el/60:.1f} min, ~{el/k*(n-k)/60:.1f} min left",
                      flush=True)

    rows.sort(key=lambda r: r[0])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        w.writerows(rows)
    n = len(rows)
    arr = sum(int(r[COLUMNS.index("arrived")]) for r in rows)
    st = sum(int(r[COLUMNS.index("started")]) for r in rows)
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min: {arr} arrived, "
          f"{st - arr} started but did not arrive, {n - st} never started")


if __name__ == "__main__":
    main()
