"""Score the three controllers on the lumped model and on pigsim.

    python control/evaluate.py            # ~6 min on 4 cores (120 closed-loop pigsim runs)
    python control/evaluate.py --cached   # re-plot / re-tabulate from control/data/

    lumped, training range     200 pigs (seed 100)
    lumped, higher friction    200 pigs (seed 101), F_fric 4-6 kN (outside training)
    pigsim, training range      20 pigs (seed 200), also run on the lumped model
    pigsim, higher friction     10 pigs (seed 201), also run on the lumped model

Controllers: (a) fixed opening, (b) PI, (c) PPO, and (c') PPO trained with a
0.5 m/s margin on a jittered lumped model (an attempt to close the gap to pigsim).

Writes control/metrics.json and control/figures/{timeseries,metrics_bars,
pigsim_check,ppo_learning_curve}.png.
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control import scenario as S  # noqa: E402
from control.controllers import (PI, PPO_FILE, PPO_ROBUST_FILE, FixedOpening,  # noqa: E402
                                 PPOController)
from control.lumped import PARAMS_FILE, LumpedPig  # noqa: E402
from control.rollout import run  # noqa: E402
from control.tune_baselines import load as load_baselines  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figures")
DATA = os.path.join(HERE, "data")
NAMES = ["fixed", "pi", "ppo", "ppo_robust"]
LABEL = {"fixed": "(a) fixed opening", "pi": "(b) PI", "ppo": "(c) PPO",
         "ppo_robust": "(c') PPO, margin + jittered model"}
COLOR = {"fixed": "#7f7f7f", "pi": "#1f77b4", "ppo": "#d62728", "ppo_robust": "#ff7f0e"}

SETS = {
    "lumped_train": ("lumped", S.TRAIN, 200, 100),
    "lumped_highF": ("lumped", S.HIGH_FRICTION, 200, 101),
    "pigsim_train": ("pigsim", S.TRAIN, 20, 200),
    "pigsim_highF": ("pigsim", S.HIGH_FRICTION, 10, 201),
}

_cache = {}


def example_cases():
    """The three pigsim runs drawn in timeseries.png: lowest and highest friction
    of the training-range set, highest friction of the higher-friction set."""
    def friction(set_name):
        _, ranges, n, seed = SETS[set_name]
        return S.sample_conditions(n, seed, ranges)[:, 1]

    F, Fh = friction("pigsim_train"), friction("pigsim_highF")
    return [("pigsim_train", int(np.argmin(F))), ("pigsim_train", int(np.argmax(F))),
            ("pigsim_highF", int(np.argmax(Fh)))]


def make_controller(name):
    if name.startswith("ppo"):
        if name not in _cache:
            import torch
            torch.set_num_threads(1)
            _cache[name] = PPOController(PPO_FILE if name == "ppo" else PPO_ROBUST_FILE)
        return _cache[name]
    b = load_baselines()
    if name == "fixed":
        return FixedOpening(b["fixed"]["u"])
    return PI(**{k: b["pi"][k] for k in ("v_ref", "kp", "ki", "u_min")})


def make_plant(kind):
    if kind == "pigsim":
        from control.pigsim_plant import PigsimPlant
        return PigsimPlant()
    return LumpedPig(record=True)


def _job(args):
    kind, name, mass, F = args
    t0 = time.perf_counter()
    m, tr = run(make_plant(kind), make_controller(name), mass, F)
    m["wall_s"] = time.perf_counter() - t0
    keep = {k: np.asarray(tr[k], dtype=np.float32) for k in ("t", "s", "V", "u")}
    return m, keep


def summarise(rows):
    g = lambda k: np.array([r[k] for r in rows], dtype=float)  # noqa: E731
    t = g("t_arrive")
    return {
        "n": len(rows),
        "violation_rate": float(g("violation").mean()),
        "late_rate": float(g("late").mean()),
        "arrived_rate": float(g("arrived").mean()),
        "v_max_limited_mean": float(g("v_max_limited").mean()),
        "v_max_limited_p95": float(np.percentile(g("v_max_limited"), 95)),
        "v_max_limited_max": float(g("v_max_limited").max()),
        "t_arrive_mean": float(np.nanmean(t)) if np.isfinite(t).any() else float("nan"),
        "t_arrive_max": float(np.nanmax(t)) if np.isfinite(t).any() else float("nan"),
        "valve_travel_mean": float(g("valve_travel").mean()),
        "return_mean": float(g("ret").mean()),
        "v_max_launch_mean": float(g("v_max_launch").mean()),
    }


def run_all():
    jobs, index = [], []
    for set_name, (kind, ranges, n, seed) in SETS.items():
        conds = S.sample_conditions(n, seed, ranges)
        for name in NAMES:
            for i, (m, F) in enumerate(conds):
                jobs.append((kind, name, float(m), float(F)))
                index.append((set_name, name, i))
                if kind == "pigsim":     # the same pigs on the lumped model, for the gap
                    jobs.append(("lumped", name, float(m), float(F)))
                    index.append((set_name + "_lumped", name, i))
    # pigsim jobs first so the pool stays busy
    order = sorted(range(len(jobs)), key=lambda j: jobs[j][0] != "pigsim")
    t0 = time.perf_counter()
    with Pool(4) as pool:
        out = pool.map(_job, [jobs[j] for j in order], chunksize=1)
    print(f"{len(jobs)} runs in {time.perf_counter() - t0:.0f} s")
    res = {}
    traces = {}
    examples = example_cases()
    for j, (m, tr) in zip(order, out):
        set_name, name, i = index[j]
        res.setdefault(set_name, {}).setdefault(name, [None] * SETS.get(
            set_name.replace("_lumped", ""), (0, 0, 0, 0))[2])[i] = m
        if (set_name, i) in examples:
            for k, v in tr.items():
                traces[f"{set_name}|{name}|{i}|{k}"] = v
    os.makedirs(DATA, exist_ok=True)
    with open(os.path.join(DATA, "eval_runs.json"), "w") as fh:
        json.dump(res, fh)
    np.savez_compressed(os.path.join(DATA, "pigsim_eval_traces.npz"), **traces)
    return res


def load_cached():
    with open(os.path.join(DATA, "eval_runs.json")) as fh:
        return json.load(fh)


def gap_table(res, base):
    """pigsim minus lumped on the same pigs."""
    out = {}
    for name in NAMES:
        a = summarise(res[base + "_lumped"][name])
        b = summarise(res[base][name])
        out[name] = {
            "lumped": a, "pigsim": b,
            "delta": {k: b[k] - a[k] for k in ("violation_rate", "late_rate", "v_max_limited_mean",
                                                 "t_arrive_mean", "valve_travel_mean", "return_mean")},
        }
        # paired per-pig differences
        pl, pp = res[base + "_lumped"][name], res[base][name]
        dv = np.array([q["v_max_limited"] - p["v_max_limited"] for p, q in zip(pl, pp)])
        dt = np.array([q["t_arrive"] - p["t_arrive"] for p, q in zip(pl, pp)])
        out[name]["paired"] = {"dv_lim_mean": float(dv.mean()), "dv_lim_absmax": float(abs(dv).max()),
                               "dt_arrive_mean": float(np.nanmean(dt)),
                               "dt_arrive_absmax": float(np.nanmax(abs(dt)))}
    return out


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_timeseries(res):
    plt = _plt()
    tr = np.load(os.path.join(DATA, "pigsim_eval_traces.npz"))
    cases = example_cases()
    fig, axes = plt.subplots(2, len(cases), figsize=(4.6 * len(cases), 6.4), sharex=True)
    for c, (set_name, i) in enumerate(cases):
        r0 = res[set_name]["pi"][i]
        for name in NAMES:
            key = f"{set_name}|{name}|{i}"
            t, s, V, u = (tr[f"{key}|{k}"] for k in ("t", "s", "V", "u"))
            axes[0, c].plot(t, V, color=COLOR[name], lw=1.2, label=LABEL[name])
            axes[1, c].step(t, u, where="post", color=COLOR[name], lw=1.2)
            j = np.argmax(s >= S.S_LIMIT) if (s >= S.S_LIMIT).any() else None
            if j is not None:
                axes[0, c].plot(t[j], V[j], "|", color=COLOR[name], ms=12)
        axes[0, c].axhline(S.V_CAP, color="k", ls=":", lw=0.8)
        axes[0, c].axvline(S.T_TARGET, color="k", ls="--", lw=0.8)
        axes[1, c].axvline(S.T_TARGET, color="k", ls="--", lw=0.8)
        axes[0, c].set_ylim(0, 20)
        tag = "training range" if set_name == "pigsim_train" else "friction above training range"
        axes[0, c].set_title(f"pigsim: m = {r0['mass']:.0f} kg, F = {r0['F_fric'] / 1e3:.1f} kN\n({tag})",
                             fontsize=9)
        axes[1, c].set_xlabel("time [s]")
        axes[1, c].set_ylim(-0.03, 1.03)
    axes[0, 0].set_ylabel("pig speed [m/s]\n(launch surge above 20 m/s cut off)")
    axes[1, 0].set_ylabel("valve opening u")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Closed loop on pigsim. Dotted: speed limit (applies after the tick mark, "
                 "s = L/2). Dashed: arrival target.", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "timeseries.png"), dpi=130)


def fig_bars(summ):
    plt = _plt()
    sets = [("lumped_train", "lumped, 200 pigs\ntraining range"),
            ("lumped_highF", "lumped, 200 pigs\nhigher friction"),
            ("pigsim_train", "pigsim, 20 pigs\ntraining range"),
            ("pigsim_highF", "pigsim, 10 pigs\nhigher friction")]
    panels = [("violation_rate", "speed-limit violation rate", None),
              ("late_rate", "late-arrival rate", None),
              ("v_max_limited_mean", "mean peak speed, s >= L/2 [m/s]", S.V_CAP),
              ("t_arrive_mean", "mean arrival time [s]", S.T_TARGET),
              ("valve_travel_mean", "valve travel, sum |du|", None)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 3.8))
    x = np.arange(len(sets))
    w = 0.8 / len(NAMES)
    for ax, (k, title, ref) in zip(axes, panels):
        for j, name in enumerate(NAMES):
            vals = [summ[s][name][k] for s, _ in sets]
            ax.bar(x + (j - (len(NAMES) - 1) / 2) * w, vals, w, color=COLOR[name],
                   label=LABEL[name])
        if ref is not None:
            ax.axhline(ref, color="k", ls=":", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in sets], fontsize=7)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "metrics_bars.png"), dpi=130)


def fig_pigsim_check(res):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name in NAMES:
        for base, mk in (("pigsim_train", "o"), ("pigsim_highF", "^")):
            pl, pp = res[base + "_lumped"][name], res[base][name]
            axes[0].scatter([p["v_max_limited"] for p in pl], [q["v_max_limited"] for q in pp],
                            color=COLOR[name], marker=mk, s=22,
                            label=f"{LABEL[name]}" if base == "pigsim_train" else None)
            axes[1].scatter([p["t_arrive"] for p in pl], [q["t_arrive"] for q in pp],
                            color=COLOR[name], marker=mk, s=22)
    for ax, ref, lab in ((axes[0], S.V_CAP, "peak speed for s >= L/2 [m/s]"),
                         (axes[1], S.T_TARGET, "arrival time [s]")):
        lo, hi = ax.get_xlim()
        lo2, hi2 = ax.get_ylim()
        lo, hi = min(lo, lo2), max(hi, hi2)
        ax.plot([lo, hi], [lo, hi], "k-", lw=0.6)
        ax.axvline(ref, color="k", ls=":", lw=0.8)
        ax.axhline(ref, color="k", ls=":", lw=0.8)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("lumped model: " + lab)
        ax.set_ylabel("pigsim: " + lab)
    axes[0].legend(fontsize=8, title="circles: training range\ntriangles: higher friction",
                   title_fontsize=7)
    fig.suptitle("Same controllers, same pigs: lumped model (where they were tuned / trained) "
                 "vs pigsim", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "pigsim_check.png"), dpi=130)


def _curve(name):
    path = os.path.join(HERE, "models", name + "_curve.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fig_learning():
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for name, key in (("ppo_valve", "ppo"), ("ppo_valve_robust", "ppo_robust")):
        c = _curve(name)
        if c is None:
            continue
        ax.plot([p["steps"] for p in c["curve"]], [p["val_return"] for p in c["curve"]], "o-",
                color=COLOR[key], ms=2.5, lw=1, label=LABEL[key])
    b = load_baselines()
    ax.axhline(b["pi"]["mean_return_tuning_set"], color=COLOR["pi"], ls="--", lw=1,
               label="(b) PI, on its tuning pigs")
    ax.set_xlabel("PPO training steps (lumped model)")
    ax.set_ylabel("mean return, 32 validation pigs")
    ax.set_ylim(-8, 0)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "ppo_learning_curve.png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", action="store_true")
    args = ap.parse_args()
    res = load_cached() if args.cached else run_all()
    os.makedirs(FIGS, exist_ok=True)

    summ = {s: {n: summarise(res[s][n]) for n in NAMES} for s in SETS}
    gap = {base: gap_table(res, base) for base in ("pigsim_train", "pigsim_highF")}
    with open(PARAMS_FILE) as fh:
        lum = json.load(fh)
    b = load_baselines()
    ppo_info = {}
    for name, key in (("ppo_valve", "ppo"), ("ppo_valve_robust", "ppo_robust")):
        c = _curve(name)
        if c is not None:
            ppo_info[key] = {k: c[k] for k in ("steps", "seed", "wall_s", "best_val_return")}
    wall = {s: {n: float(np.mean([r["wall_s"] for r in res[s][n]])) for n in NAMES}
            for s in ("lumped_train", "pigsim_train")}

    out = {
        "scenario": {"L": S.L, "p_in": S.P_IN, "chi_max": S.CHI_MAX, "dt_ctrl": S.DT_CTRL,
                     "du_max": S.DU_MAX, "u0": S.U0, "s_limit": S.S_LIMIT, "v_cap": S.V_CAP,
                     "t_target": S.T_TARGET, "t_max": S.T_MAX, "train": S.TRAIN,
                     "high_friction": S.HIGH_FRICTION},
        "lumped_model": {"params": lum["params"], "summary_calib": lum["summary_calib"],
                         "summary_valid": lum["summary_valid"],
                         "valve_authority_pigsim": lum["valve_authority_pigsim"]},
        "controllers": {"fixed": b["fixed"], "pi": b["pi"], **ppo_info},
        "results": summ,
        "sim_to_sim_gap": gap,
        "mean_wall_s_per_run": wall,
    }
    with open(os.path.join(HERE, "metrics.json"), "w") as fh:
        json.dump(out, fh, indent=1)

    fig_timeseries(res)
    fig_bars(summ)
    fig_pigsim_check(res)
    fig_learning()

    for s in SETS:
        print(f"\n{s}")
        for n in NAMES:
            r = summ[s][n]
            print(f"  {n:6s} viol {r['violation_rate']:.3f}  late {r['late_rate']:.3f}  "
                  f"v_lim mean {r['v_max_limited_mean']:5.2f} max {r['v_max_limited_max']:5.2f}  "
                  f"t_arr mean {r['t_arrive_mean']:6.1f} max {r['t_arrive_max']:6.1f}  "
                  f"travel {r['valve_travel_mean']:.2f}  return {r['return_mean']:7.2f}  "
                  f"arrived {r['arrived_rate']:.2f}")
    for base in gap:
        print(f"\ngap {base} (pigsim - lumped, same pigs)")
        for n in NAMES:
            d, p = gap[base][n]["delta"], gap[base][n]["paired"]
            print(f"  {n:6s} d_viol {d['violation_rate']:+.2f} d_late {d['late_rate']:+.2f} "
                  f"d_vlim {d['v_max_limited_mean']:+.2f} d_tarr {d['t_arrive_mean']:+.1f} "
                  f"d_return {d['return_mean']:+.2f}  paired |dv| max {p['dv_lim_absmax']:.2f}")


if __name__ == "__main__":
    main()
