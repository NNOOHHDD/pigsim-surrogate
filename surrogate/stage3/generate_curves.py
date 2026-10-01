"""Stage 3 data: whole pig-speed curves V(xi) instead of two scalars.

    python surrogate/stage3/generate_curves.py            # pilot, then ~35 min on 4 cores
    python surrogate/stage3/generate_curves.py --n 100    # fixed sample count

Same four inputs and ranges as surrogate/pipeline.py (stage 1).  For every run
the pig history (t, s_pig, V_pig) is put on a fixed grid of 128 points

    xi_k = (s_k - s0) / (0.999 L - s0),   xi_k = (k / 127)^2,   k = 0 .. 127

i.e. xi = 0 where the pig starts (s0 = 2 m) and xi = 1 at the arrival point.
This is s/L shifted by s0 / L (at most 0.4 %), so that xi = 0 is the start for
every L.  The square spacing puts 40 of the 128 points in xi < 0.1, where the
launch transient happens.  If the pig ever moves backwards, the value at the
*first* passage of each grid position is used.

Stored per run (curves.npz):
    X        (n, 4)    p_in, mass, F_fric, L
    V        (n, 128)  pig speed on the xi grid [m/s]
    T        (n, 128)  time at which the pig first passes each grid point [s]
    t_arrive, v_max, s_vmax, t_break   pigsim scalars (t_break = first motion)
    n_back   number of time steps with V < 0 (pig moving backwards)
Runs where the pig does not arrive are dropped (none are expected in this box).
Every finished run is also written to _runs/<id>.npz (git-ignored), so an
interrupted campaign restarts where it stopped (--resume) instead of from zero.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import qmc

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from surrogate.pipeline import INPUTS, RANGES, S0, build  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
N_GRID = 128
XI = (np.arange(N_GRID) / (N_GRID - 1)) ** 2
T_END = 2000.0


def lhs(n, seed):
    unit = qmc.LatinHypercube(d=len(INPUTS), seed=seed).random(n)
    lo = np.array([RANGES[k][0] for k in INPUTS])
    hi = np.array([RANGES[k][1] for k in INPUTS])
    return qmc.scale(unit, lo, hi)


def to_grid(t, s, V, L):
    """First-passage V and t on the xi grid (history starts with t=0, s=s0, V=0)."""
    s_end = 0.999 * L
    s_k = S0 + XI * (s_end - S0)
    s_max = np.maximum.accumulate(s)
    new = np.concatenate([[True], np.diff(s_max) > 0])   # first time each new maximum is reached
    sm, tm, Vm = s_max[new], t[new], V[new]
    return np.interp(s_k, sm, Vm), np.interp(s_k, sm, tm)


RUN_DIR = os.path.join(HERE, "_runs")


def run_one(job):
    i, x = job
    res = _run_one(job)
    if res[2] is not None:
        np.savez(os.path.join(RUN_DIR, f"{i:05d}.npz"), x=x, **res[2])
    return res


def _run_one(job):
    i, x = job
    t0 = time.perf_counter()
    try:
        sol = build(*x)
        L = x[3]
        sol.run(T_END, dt0=0.02, dt_max=1.0, cfl_pig=0.02, s_stop=0.999 * L, verbose=False)
        h = sol.history()
        t = np.concatenate([[0.0], h["t"]])
        s = np.concatenate([[S0], h["s_pig"]])
        V = np.concatenate([[0.0], h["V_pig"]])
        arrived = bool(s[-1] >= 0.999 * L)
        if not arrived:
            return i, x, None
        t_arr = float(np.interp(0.999 * L, s[-2:], t[-2:]))
        s = np.minimum(s, 0.999 * L)
        t[-1] = t_arr
        Vg, Tg = to_grid(t, s, V, L)
        moving = np.nonzero(V > 0)[0]
        k = int(np.argmax(V))
        return i, x, {"V": Vg, "T": Tg, "t_arrive": t_arr, "v_max": float(V[k]),
                      "s_vmax": float(s[k]), "t_break": float(t[moving[0] - 1]) if len(moving) else np.nan,
                      "n_back": int(np.sum(V < 0)), "wall_s": time.perf_counter() - t0}
    except Exception as exc:  # recorded, not hidden
        print(f"  run {i} failed: {exc!r}", flush=True)
        return i, x, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--budget-min", type=float, default=40.0)
    ap.add_argument("--safety", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--out", default=os.path.join(HERE, "curves.npz"))
    ap.add_argument("--resume", action="store_true",
                    help="reuse finished runs in _runs/ (requires --n, the same n and seed)")
    args = ap.parse_args()
    os.makedirs(RUN_DIR, exist_ok=True)

    with Pool(args.workers) as pool:
        n = args.n
        if n is None:
            n_pilot = 2 * args.workers
            Xp = lhs(n_pilot, args.seed + 1000)
            t0 = time.perf_counter()
            list(pool.imap_unordered(run_one, list(enumerate(Xp))))
            t_run = (time.perf_counter() - t0) / n_pilot * args.workers
            n = int(args.safety * args.budget_min * 60 * args.workers / t_run // 10 * 10)
            print(f"pilot: {n_pilot} runs, {t_run:.1f} s per run per worker -> "
                  f"budget {args.budget_min:.0f} min x {args.safety:.0%} -> n = {n}", flush=True)
        X = lhs(n, args.seed)
        res, jobs = [], list(enumerate(X))
        if args.resume:
            done = {int(f[:5]) for f in os.listdir(RUN_DIR) if f.endswith(".npz")}
            for i in sorted(done & set(range(n))):
                z = np.load(os.path.join(RUN_DIR, f"{i:05d}.npz"))
                assert np.allclose(z["x"], X[i]), "run files do not match this n / seed"
                res.append((i, X[i], {k: z[k] for k in z.files if k != "x"}))
            jobs = [j for j in jobs if j[0] not in done]
            print(f"resume: {len(res)} runs already done, {len(jobs)} to go", flush=True)
        else:
            for f in os.listdir(RUN_DIR):
                os.remove(os.path.join(RUN_DIR, f))
        t0 = time.perf_counter()
        n_todo = len(jobs)
        for k, r in enumerate(pool.imap_unordered(run_one, jobs), 1):
            res.append(r)
            if k % 50 == 0 or k == n_todo:
                el = time.perf_counter() - t0
                print(f"  {k}/{n_todo} done, {el/60:.1f} min, ~{el/k*(n_todo-k)/60:.1f} min left",
                      flush=True)

    res.sort(key=lambda r: r[0])
    ok = [r for r in res if r[2] is not None]
    out = {"X": np.array([r[1] for r in ok]), "xi": XI, "id": np.array([r[0] for r in ok])}
    for key in ("V", "T"):
        out[key] = np.array([r[2][key] for r in ok], dtype=np.float32)
    for key in ("t_arrive", "v_max", "s_vmax", "t_break", "n_back", "wall_s"):
        out[key] = np.array([r[2][key] for r in ok])
    np.savez_compressed(args.out, **out)
    print(f"done in {(time.perf_counter()-t0)/60:.1f} min: {len(ok)}/{n} arrived "
          f"-> {args.out} ({os.path.getsize(args.out)/1e6:.2f} MB); "
          f"runs with backward motion: {int(np.sum(out['n_back'] > 0))}")


if __name__ == "__main__":
    main()
