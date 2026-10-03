"""Fit the lumped model to pigsim and measure its error on separate runs.

    python control/calibrate_lumped.py           # pigsim runs (~1 min, 4 cores) + fit (~1 min)
    python control/calibrate_lumped.py --cached  # reuse control/data/pigsim_openloop.npz

Three groups of open-loop pigsim runs (valve opening given as a function of time):
    authority   mean pig, fixed openings 0.1 / 0.33 / 1.0 -- does the valve
                change the launch peak?
    calib       4 runs the four lumped constants (n, c_g, f_mult, K_in) are fitted to
    valid       7 other runs (other pigs, other valve schedules, one with
                friction above the training range) for the error table

Writes control/lumped_params.json (constants + both tables) and
control/figures/lumped_vs_pigsim.png.
"""

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control import scenario as S  # noqa: E402
from control.controllers import Schedule  # noqa: E402
from control.lumped import DEFAULT_PARAMS, PARAMS_FILE, LumpedPig  # noqa: E402
from control.rollout import metrics  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "pigsim_openloop.npz")
FIG = os.path.join(HERE, "figures", "lumped_vs_pigsim.png")
T_MAX = 300.0


def schedule(name, t):
    if name.startswith("const"):
        return float(name[5:])
    if name == "steps_a":
        return 0.6 if t < 20 else 0.1 if t < 50 else 0.8 if t < 80 else 0.4
    if name == "steps_b":
        return 0.2 if t < 30 else 0.7 if t < 60 else 0.3
    if name == "sine":
        return 0.45 + 0.3 * math.sin(2.0 * math.pi * t / 40.0)
    if name == "ramp":
        return min(0.1 + 0.01 * t, 0.9)
    raise ValueError(name)


# (group, mass, F_fric, schedule)
RUNS = [
    ("authority", 110.0, 2500.0, "const0.1"),
    ("authority", 110.0, 2500.0, "const0.33"),
    ("authority", 110.0, 2500.0, "const1.0"),
    ("calib", 40.0, 1500.0, "const0.4"),
    ("calib", 180.0, 3500.0, "const0.4"),
    ("calib", 40.0, 1500.0, "steps_a"),
    ("calib", 180.0, 3500.0, "steps_a"),
    ("valid", 70.0, 1200.0, "steps_b"),
    ("valid", 150.0, 2000.0, "sine"),
    ("valid", 30.0, 3000.0, "ramp"),
    ("valid", 110.0, 2500.0, "const0.6"),
    ("valid", 190.0, 3900.0, "steps_b"),
    ("valid", 20.0, 3800.0, "sine"),
    ("valid", 100.0, 5000.0, "const0.5"),   # friction above the training range
]


def _run(args, plant=None, t_max=T_MAX):
    _, mass, F, name = args
    if plant is None:
        from control.pigsim_plant import PigsimPlant
        plant = PigsimPlant()
    ctrl = Schedule(lambda t: schedule(name, t))
    plant.reset(mass, F, u0=ctrl.reset())
    while not plant.arrived and plant.t < t_max - 1e-9:
        plant.step(ctrl(plant.obs()))
    tr = plant.trace()
    return {k: np.asarray(tr[k]) for k in ("t", "s", "V", "u")}


def pigsim_runs(cached):
    if cached and os.path.exists(DATA):
        d = np.load(DATA)
        return [{k: d[f"{i}_{k}"] for k in ("t", "s", "V", "u")} for i in range(len(RUNS))]
    t0 = time.perf_counter()
    with Pool(4) as pool:
        runs = pool.map(_run, RUNS)
    print(f"pigsim: {len(RUNS)} runs in {time.perf_counter() - t0:.0f} s")
    os.makedirs(os.path.dirname(DATA), exist_ok=True)
    np.savez_compressed(DATA, **{f"{i}_{k}": r[k] for i, r in enumerate(runs)
                                 for k in ("t", "s", "V", "u")})
    return runs


S_GRID = np.linspace(0.02, 0.995, 200) * S.L


def first_passage(tr, grid=S_GRID):
    s_max = np.maximum.accumulate(tr["s"])
    keep = np.concatenate([[True], np.diff(s_max) > 0])
    s, V, t = s_max[keep], tr["V"][keep], tr["t"][keep]
    ok = grid <= s[-1]
    Vg = np.full(grid.shape, np.nan)
    tg = np.full(grid.shape, np.nan)
    Vg[ok] = np.interp(grid[ok], s, V)
    tg[ok] = np.interp(grid[ok], s, t)
    return Vg, tg


def lumped_run(args, params):
    return _run(args, plant=LumpedPig(params=params))


def compare(ref, lum):
    """Error measures of one lumped run against the pigsim run."""
    mp, ml = metrics(ref), metrics(lum)
    Vp, _ = first_passage(ref)
    Vl, _ = first_passage(lum)
    w = S_GRID >= S.S_LIMIT
    both = np.isfinite(Vp) & np.isfinite(Vl)

    def pct(a, b):
        return 100.0 * (b - a) / a if np.isfinite(a) and np.isfinite(b) else float("nan")

    return {
        "t_arrive_pigsim": mp["t_arrive"], "t_arrive_lumped": ml["t_arrive"],
        "t_arrive_err_pct": pct(mp["t_arrive"], ml["t_arrive"]),
        "v_launch_pigsim": mp["v_max_launch"], "v_launch_lumped": ml["v_max_launch"],
        "v_launch_err_pct": pct(mp["v_max_launch"], ml["v_max_launch"]),
        "v_lim_pigsim": mp["v_max_limited"], "v_lim_lumped": ml["v_max_limited"],
        "v_lim_err_pct": pct(mp["v_max_limited"], ml["v_max_limited"]),
        "rms_V_second_half": float(np.sqrt(np.mean((Vp - Vl)[both & w] ** 2))),
        "rms_V_all": float(np.sqrt(np.mean((Vp - Vl)[both] ** 2))),
    }


KEYS = ["n", "c_g", "f_mult", "K_in"]
BOUNDS = {"n": (1.0, 1.4), "c_g": (0.0, 1.5), "f_mult": (0.5, 2.0), "K_in": (0.0, 20.0)}


def fit(ref_runs, idx):
    def unpack(z):
        p = dict(DEFAULT_PARAMS)
        for k, v in zip(KEYS, z):
            lo, hi = BOUNDS[k]
            p[k] = float(np.clip(v, lo, hi))
        return p

    def loss(z):
        p = unpack(z)
        tot = 0.0
        for i in idx:
            c = compare(ref_runs[i], lumped_run(RUNS[i], p))
            tot += c["rms_V_all"] / 5.0 + abs(c["t_arrive_err_pct"]) / 2.0
        return tot / len(idx)

    z0 = np.array([DEFAULT_PARAMS[k] for k in KEYS])
    res = minimize(loss, z0, method="Nelder-Mead",
                   options={"maxfev": 250, "xatol": 1e-3, "fatol": 1e-4})
    return unpack(res.x), float(loss(z0)), float(res.fun)


def figure(ref_runs, lum_runs, idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(idx)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(3.4 * ((n + 1) // 2), 6.2), sharex=True)
    for ax, i in zip(axes.ravel(), idx):
        g, m, F, name = RUNS[i]
        for tr, lab, st in ((ref_runs[i], "pigsim", "k-"), (lum_runs[i], "lumped", "C1--")):
            ax.plot(tr["s"] / S.L, tr["V"], st, lw=1.2, label=lab)
        ax.axhline(S.V_CAP, color="C3", lw=0.8, ls=":")
        ax.axvline(S.S_LIMIT / S.L, color="0.6", lw=0.8, ls=":")
        tag = " (F above training range)" if F > S.TRAIN["F_fric"][1] else ""
        ax.set_title(f"m={m:.0f} kg, F={F / 1e3:.1f} kN, u: {name}{tag}", fontsize=8)
        ax.set_ylim(0, 50)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("pig position s / L")
    for ax in axes[:, 0]:
        ax.set_ylabel("pig speed [m/s]")
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Lumped model vs pigsim, validation runs (open-loop valve schedules); "
                 "dotted: speed limit and where it starts", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    args = ap.parse_args()
    ref = pigsim_runs(args.cached)

    cal = [i for i, r in enumerate(RUNS) if r[0] == "calib"]
    val = [i for i, r in enumerate(RUNS) if r[0] == "valid"]
    aut = [i for i, r in enumerate(RUNS) if r[0] == "authority"]

    t0 = time.perf_counter()
    params, loss0, loss1 = fit(ref, cal)
    print(f"fit: loss {loss0:.3f} -> {loss1:.3f} in {time.perf_counter() - t0:.0f} s; {params}")

    lum = [lumped_run(r, params) for r in RUNS]
    table = []
    for i in cal + val:
        g, m, F, name = RUNS[i]
        table.append({"group": g, "mass": m, "F_fric": F, "valve": name, **compare(ref[i], lum[i])})
    authority = []
    for i in aut:
        mp = metrics(ref[i])
        Vp, _ = first_passage(ref[i])
        authority.append({"u": schedule(RUNS[i][3], 0.0), "v_launch": mp["v_max_launch"],
                          "V_at_half": float(Vp[np.argmin(abs(S_GRID - S.S_LIMIT))]),
                          "v_lim": mp["v_max_limited"], "t_arrive": mp["t_arrive"]})

    def summary(rows):
        out = {}
        for k in ("t_arrive_err_pct", "v_launch_err_pct", "v_lim_err_pct"):
            v = np.array([r[k] for r in rows])
            out[k.replace("_pct", "_mean_abs_pct")] = float(np.nanmean(np.abs(v)))
        out["rms_V_second_half_mean"] = float(np.mean([r["rms_V_second_half"] for r in rows]))
        return out

    out = {
        "params": params,
        "fit_loss": {"before": loss0, "after": loss1},
        "valve_authority_pigsim": authority,
        "table": table,
        "summary_calib": summary([r for r in table if r["group"] == "calib"]),
        "summary_valid": summary([r for r in table if r["group"] == "valid"]),
    }
    with open(PARAMS_FILE, "w") as fh:
        json.dump(out, fh, indent=1)
    figure(ref, lum, val)

    print("\nvalve authority (pigsim, mean pig):")
    for a in authority:
        print("  u={u:.2f}  launch peak {v_launch:5.1f}  V at s=L/2 {V_at_half:5.1f}  "
              "peak s>=L/2 {v_lim:5.1f}  t_arrive {t_arrive:6.1f}".format(**a))
    print("\nlumped vs pigsim:")
    for r in table:
        print("  {group:5s} m={mass:5.0f} F={F_fric:5.0f} {valve:9s} t_arr {t_arrive_pigsim:6.1f}/"
              "{t_arrive_lumped:6.1f} ({t_arrive_err_pct:+5.1f}%)  launch {v_launch_err_pct:+5.1f}%  "
              "peak s>=L/2 {v_lim_pigsim:5.1f}/{v_lim_lumped:5.1f} ({v_lim_err_pct:+5.1f}%)  "
              "rms V 2nd half {rms_V_second_half:4.2f} m/s".format(**r))
    print("calib:", out["summary_calib"])
    print("valid:", out["summary_valid"])


if __name__ == "__main__":
    main()
