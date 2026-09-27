"""Step 2: design of experiments -> run pigsim many times -> surrogate/data.csv.

    python surrogate/generate_data.py                 # pilot timing, then full run
    python surrogate/generate_data.py --n 200         # skip the budget calculation

1. Pilot: a few runs are timed on the same process pool that the real
   campaign uses, so the measured time already includes the slow-down from
   running 4 solvers side by side.
2. Budget: n = budget / (time per run / workers), with a safety margin,
   rounded down to a multiple of 10.
3. Latin hypercube sample of the 4 inputs, pigsim run in parallel, one CSV
   row written per finished run (a crash keeps everything done so far).
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

from surrogate.pipeline import INPUTS, OUTPUTS, RANGES, simulate  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
COLUMNS = ["id"] + INPUTS + OUTPUTS + ["arrived", "n_steps", "wall_s"]


def latin_hypercube(n, seed):
    """n points in the 4-D input box, one per row.

    LHS splits every input range into n equal slices and puts exactly one
    point in each slice, so each input is covered evenly even with few runs
    (plain random sampling leaves gaps and clumps).
    """
    unit = qmc.LatinHypercube(d=len(INPUTS), seed=seed).random(n)
    lo = np.array([RANGES[k][0] for k in INPUTS])
    hi = np.array([RANGES[k][1] for k in INPUTS])
    return qmc.scale(unit, lo, hi)


def _run(job):
    i, x = job
    try:
        r = simulate(**dict(zip(INPUTS, x)))
    except Exception as exc:  # keep the campaign going; the row is flagged
        print(f"  run {i} failed: {exc!r}", flush=True)
        r = {"t_arrive": float("nan"), "v_max": float("nan"), "arrived": False,
             "n_steps": 0, "wall_s": float("nan")}
    return i, x, r


def pilot(pool, workers, n_pilot, seed):
    """Time n_pilot runs on the real pool; return wall seconds per run (per worker)."""
    X = latin_hypercube(n_pilot, seed + 1000)   # separate points, not in data.csv
    t0 = time.perf_counter()
    res = list(pool.imap_unordered(_run, list(enumerate(X))))
    elapsed = time.perf_counter() - t0
    walls = np.array([r["wall_s"] for _, _, r in res])
    print(f"pilot: {n_pilot} runs on {workers} workers in {elapsed:.1f} s  "
          f"(per run: mean {np.nanmean(walls):.1f} s, max {np.nanmax(walls):.1f} s)")
    return elapsed / n_pilot * workers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="number of samples; default = derived from the time budget")
    ap.add_argument("--budget-min", type=float, default=40.0,
                    help="wall-clock budget for the main campaign (the pilot, "
                         "training and plots are not included)")
    ap.add_argument("--safety", type=float, default=0.8,
                    help="use only this fraction of the budget")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=os.path.join(HERE, "data.csv"))
    args = ap.parse_args()

    with Pool(args.workers) as pool:
        n = args.n
        if n is None:
            t_run = pilot(pool, args.workers, 2 * args.workers, args.seed)
            n_fit = args.safety * args.budget_min * 60.0 * args.workers / t_run
            n = int(n_fit // 10 * 10)
            print(f"budget {args.budget_min:.0f} min x {args.safety:.0%} safety, "
                  f"{t_run:.1f} s per run per worker -> n = {n}")

        X = latin_hypercube(n, args.seed)
        print(f"running {n} pigsim cases on {args.workers} workers -> {args.out}")
        t0 = time.perf_counter()
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(COLUMNS)
            for k, (i, x, r) in enumerate(pool.imap_unordered(_run, list(enumerate(X))), 1):
                w.writerow([i, *x, *(r[o] for o in OUTPUTS),
                            int(r["arrived"]), r["n_steps"], r["wall_s"]])
                f.flush()
                if k % 20 == 0 or k == n:
                    el = time.perf_counter() - t0
                    print(f"  {k}/{n} done, {el/60:.1f} min elapsed, "
                          f"~{el/k*(n-k)/60:.1f} min left", flush=True)

    # sort by id so the file is deterministic regardless of finishing order
    rows = list(csv.DictReader(open(args.out)))
    rows.sort(key=lambda r: int(r["id"]))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    n_fail = sum(r["arrived"] != "1" for r in rows)
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min; "
          f"{len(rows) - n_fail} arrived, {n_fail} did not")


if __name__ == "__main__":
    main()
