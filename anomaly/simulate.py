"""Anomaly-detection data: sensor time series from normal and faulty pigsim runs.

    python anomaly/simulate.py              # pilot, then ~35 min on 4 cores
    python anomaly/simulate.py --scale 0.1  # small set for a smoke test

Six signals per run, recorded at fixed stations and resampled to 1 s:

    p_inlet   pressure at s = 0.5 m            [Pa]
    p_25      pressure at s = 0.25 L
    p_50      pressure at s = 0.50 L
    p_75      pressure at s = 0.75 L
    p_outlet  pressure at s = L - 0.5 m
    m_outlet  mass flow at s = L - 0.5 m        [kg/s]

Inputs follow the stage-1 box (surrogate/pipeline.py) and are drawn by LHS.
Kinds of run (physical faults go through the pipeline hooks, the solver is
untouched):

    normal     nominal operation
    friction   dynamic (and static) friction x 1.5-2.5 once the pig passes
               s_a = 0.3-0.7 L (e.g. a dented or dirty section); onset = time
               the pig reaches s_a
    valve      outlet valve closes to 30-60 % opening over 5 s at t_a
    supply     inlet (supply) pressure falls to 50-80 % of its excess over
               1.5 bar over 5 s at t_a
    sensor     normal runs that later get a sensor drift or spikes added
               in post-processing (anomaly/detect.py), not in pigsim

t_a = 0.3-0.6 x the arrival time that the stage-1 surrogate predicts.
A run ends when the pig arrives or at t_end = 2 x predicted arrival + 60 s
(a stalled pig never arrives).  Output: runs.npz.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from surrogate import pipeline as P  # noqa: E402

SIGNALS = ["p_inlet", "p_25", "p_50", "p_75", "p_outlet", "m_outlet"]
DT = 1.0
COUNTS = {"normal": 400, "friction": 100, "valve": 100, "supply": 100, "sensor": 100}
RAMP = 5.0


def lhs(n, seed):
    unit = qmc.LatinHypercube(d=4, seed=seed).random(n)
    lo = np.array([P.RANGES[k][0] for k in P.INPUTS])
    hi = np.array([P.RANGES[k][1] for k in P.INPUTS])
    return qmc.scale(unit, lo, hi)


def _ramp(t, t0):
    return float(np.clip((t - t0) / RAMP, 0.0, 1.0))


def run_one(job):
    """job = (id, kind, x, params, t_pred) -> dict with the resampled signals."""
    i, kind, x, prm, t_pred = job
    p_in, mass, F, L = x
    hooks = {}
    if kind == "friction":
        s_a, k = prm["s_a"], prm["factor"]
        hooks["F_fric_of_s"] = lambda s: F * (1.0 + (k - 1.0) * float(np.clip((s - s_a) / 5.0, 0, 1)))
    elif kind == "valve":
        t_a, c = prm["t_a"], prm["chi"]
        hooks["chi_of_t"] = lambda t: 1.0 - (1.0 - c) * _ramp(t, t_a)
    elif kind == "supply":
        t_a, r = prm["t_a"], prm["ratio"]
        hooks["p_in_of_t"] = lambda t, p: p - (1.0 - r) * (p - P.P0) * _ramp(t, t_a)
    t0 = time.perf_counter()
    t_end = 2.0 * t_pred + 60.0
    try:
        sol = P.build(p_in, mass, F, L, **hooks)
        sol.set_stations([0.5, 0.25 * L, 0.5 * L, 0.75 * L, L - 0.5])
        sol.run(t_end, dt0=0.02, dt_max=1.0, cfl_pig=0.02, s_stop=0.999 * L, verbose=False)
    except Exception as exc:
        print(f"  run {i} ({kind}) failed: {exc!r}", flush=True)
        return None
    st = sol.station_arrays()
    h = sol.history()
    t = st["t"]
    raw = np.column_stack([st["p"], st["mdot"][:, -1]])
    grid = np.arange(0.0, t[-1] + 1e-9, DT)
    sig = np.column_stack([np.interp(grid, t, raw[:, j]) for j in range(raw.shape[1])])
    arrived = bool(h["s_pig"][-1] >= 0.999 * L)
    onset = prm.get("t_a", np.nan)
    if kind == "friction":   # the fault starts when the pig reaches s_a
        hit = np.nonzero(h["s_pig"] >= prm["s_a"])[0]
        onset = float(h["t"][hit[0]]) if len(hit) else np.nan
    return {"id": i, "kind": kind, "x": np.asarray(x), "signals": sig.astype(np.float32),
            "onset": onset, "arrived": arrived, "t_end": float(t[-1]), "t_pred": t_pred,
            "params": prm, "wall_s": time.perf_counter() - t0}


def make_jobs(counts, seed):
    """Inputs by LHS per kind; fault parameters from a seeded generator."""
    from surrogate.train_surrogate import load_ensemble
    ens = load_ensemble()
    rng = np.random.default_rng(seed)
    jobs, i = [], 0
    for k_idx, (kind, n) in enumerate(counts.items()):
        if n == 0:
            continue
        X = lhs(n, seed + 10 * k_idx)
        t_pred = ens.predict(X)[0][:, 0]
        for x, tp in zip(X, t_pred):
            prm = {}
            if kind == "friction":
                prm = {"s_a": float(rng.uniform(0.3, 0.7) * x[3]), "factor": float(rng.uniform(1.5, 2.5))}
            elif kind in ("valve", "supply"):
                prm = {"t_a": float(rng.uniform(0.3, 0.6) * tp)}
                prm["chi" if kind == "valve" else "ratio"] = float(
                    rng.uniform(0.3, 0.6) if kind == "valve" else rng.uniform(0.5, 0.8))
            jobs.append((i, kind, tuple(x), prm, float(tp)))
            i += 1
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=None,
                    help="multiply all counts by this; default = from a pilot and a 40 min budget")
    ap.add_argument("--budget-min", type=float, default=40.0)
    ap.add_argument("--safety", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--out", default=os.path.join(HERE, "runs.npz"))
    args = ap.parse_args()

    with Pool(args.workers) as pool:
        scale = args.scale
        if scale is None:
            pil = make_jobs({k: 2 for k in COUNTS}, args.seed + 999)
            t0 = time.perf_counter()
            list(pool.imap_unordered(run_one, pil))
            per_run = (time.perf_counter() - t0) / len(pil) * args.workers
            n_fit = args.safety * args.budget_min * 60 * args.workers / per_run
            scale = min(1.0, n_fit / sum(COUNTS.values()))
            print(f"pilot: {len(pil)} runs, {per_run:.1f} s per run per worker -> "
                  f"{n_fit:.0f} runs fit the budget -> scale {scale:.2f}", flush=True)
        counts = {k: int(round(v * scale)) for k, v in COUNTS.items()}
        jobs = make_jobs(counts, args.seed)
        print(f"running {len(jobs)} runs: {counts}", flush=True)
        t0 = time.perf_counter()
        res = []
        for k, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            res.append(r)
            if k % 50 == 0 or k == len(jobs):
                el = time.perf_counter() - t0
                print(f"  {k}/{len(jobs)} done, {el/60:.1f} min, ~{el/k*(len(jobs)-k)/60:.1f} min left",
                      flush=True)
    res = sorted([r for r in res if r is not None], key=lambda r: r["id"])
    lengths = np.array([len(r["signals"]) for r in res])
    np.savez_compressed(
        args.out,
        signals=np.concatenate([r["signals"] for r in res]), lengths=lengths,
        X=np.array([r["x"] for r in res]), kind=np.array([r["kind"] for r in res]),
        onset=np.array([r["onset"] for r in res]), arrived=np.array([r["arrived"] for r in res]),
        t_pred=np.array([r["t_pred"] for r in res]), id=np.array([r["id"] for r in res]),
        params=np.array([repr(r["params"]) for r in res]), signal_names=np.array(SIGNALS),
        wall_s=np.array([r["wall_s"] for r in res]))
    kinds, n = np.unique([r["kind"] for r in res], return_counts=True)
    stalled = {k: int(sum(1 for r in res if r["kind"] == k and not r["arrived"])) for k in kinds}
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min: {dict(zip(kinds, n))}; "
          f"not arrived by t_end: {stalled}; -> {args.out} "
          f"({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
