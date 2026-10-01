"""Stage 3: train and evaluate the curve models on curves.npz.

    python surrogate/stage3/run_stage3.py              # ~11 min on 4 cores
    python surrogate/stage3/run_stage3.py --quick      # smoke test, ~1 min

Experiments
    main      70 / 15 / 15 random split (numpy default_rng(0)), all 5 models
    extrap    train without the top 10 % of L (or of p_in), test on that 10 %
    lcurve    the main test set, training sets of 50 .. all runs
Outputs (next to this file): metrics.json, figures/*.png, models/deeponet.pt,
test_predictions.npz (for --plots-only)
"""

import argparse
import json
import os
import sys
import time

# one BLAS / OpenMP thread per process: the experiments run 4 processes side by side
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from surrogate import pipeline as P  # noqa: E402
from surrogate.stage3.models import (  # noqa: E402
    DeepONetEnsemble, MLPEnsemble, NearestNeighbour, PCAGP, PCAPoly,
)

FIG = os.path.join(HERE, "figures")
AREA = np.pi * P.D ** 2 / 4
START = 0.1                     # xi < 0.1 = launch segment, the rest = cruise
NAMES = ["DeepONet x5", "MLP x5", "PCA + poly3", "PCA + GP", "nearest neighbour"]
COL = {"DeepONet x5": "#2a78d6", "MLP x5": "#4a3aa7", "PCA + poly3": "#1baf7a",
       "PCA + GP": "#eb6834", "nearest neighbour": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def make_models(xi):
    return {"DeepONet x5": DeepONetEnsemble(xi), "MLP x5": MLPEnsemble(n_out=len(xi)),
            "PCA + poly3": PCAPoly(), "PCA + GP": PCAGP(),
            "nearest neighbour": NearestNeighbour()}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def t_break_analytic(X):
    """Time at which the inlet ramp overcomes static friction (pig at rest before)."""
    dp = X[:, 0] - P.P0
    return np.minimum(P.T_RAMP * P.STATIC_OVER_DYNAMIC * X[:, 2] / AREA / dp, P.T_RAMP)


def travel_time(V, X, xi):
    """sum 2 ds / (V_k + V_k+1): exact if the acceleration is constant per interval."""
    L = X[:, 3:4]
    s = P.S0 + xi[None, :] * (0.999 * L - P.S0)
    Vc = np.maximum(V, 1e-3)
    return np.sum(2 * np.diff(s, axis=1) / (Vc[:, 1:] + Vc[:, :-1]), axis=1)


def quad_weights(xi):
    w = np.zeros_like(xi)
    w[1:] += 0.5 * np.diff(xi)
    w[:-1] += 0.5 * np.diff(xi)
    return w


def rel_l2(V, Vh, w, mask):
    num = np.sum(w[mask] * (Vh[:, mask] - V[:, mask]) ** 2, axis=1)
    den = np.sum(w[mask] * V[:, mask] ** 2, axis=1)
    return np.sqrt(num / den)


def evaluate(V, Vh, sd, X, d, xi):
    """Every number is computed per test run, then summarised."""
    w = quad_weights(xi)
    out = {}
    for seg, mask in (("all", np.ones_like(xi, bool)), ("launch", xi < START),
                      ("cruise", xi >= START)):
        r = 100 * rel_l2(V, Vh, w, mask)
        out[f"relL2_{seg}_%"] = {"mean": float(r.mean()), "median": float(np.median(r)),
                                 "p95": float(np.percentile(r, 95)), "max": float(r.max())}
    span = 0.999 * X[:, 3] - P.S0
    k_t, k_p = V.argmax(1), Vh.argmax(1)
    vm_t, vm_p = V.max(1), Vh.max(1)
    out["v_max_rel_err_%"] = {"mean": float(np.mean(100 * np.abs(vm_p - vm_t) / vm_t)),
                              "max": float(np.max(100 * np.abs(vm_p - vm_t) / vm_t))}
    loc = np.abs(xi[k_p] - xi[k_t]) * span
    out["v_max_location_err_m"] = {"mean": float(loc.mean()), "median": float(np.median(loc)),
                                   "max": float(loc.max())}
    t_pred = t_break_analytic(X) + travel_time(Vh, X, xi)
    e = 100 * np.abs(t_pred - d["t_arrive"]) / d["t_arrive"]
    out["t_arrive_rel_err_%"] = {"mean": float(e.mean()), "p95": float(np.percentile(e, 95)),
                                 "max": float(e.max())}
    if sd is not None:
        err = np.abs(Vh - V)
        out["coverage"] = {f"{k}sigma": float(np.mean(err <= k * sd)) for k in (1, 2, 3)}
        out["coverage_launch_2sigma"] = float(np.mean(err[:, xi < START] <= 2 * sd[:, xi < START]))
        out["mean_std_m_s"] = float(sd.mean())
    return out


def floor_metrics(d, idx, xi):
    """What the 128-point grid itself loses, using the *true* curves."""
    X, V = d["X"][idx], d["V"][idx]
    tb_true = d["t_break"][idx]
    t_true_quad = tb_true + travel_time(V, X, xi)
    t_analytic_quad = t_break_analytic(X) + travel_time(V, X, xi)
    ta = d["t_arrive"][idx]
    return {
        "t_arrive_from_true_curve_rel_err_%": float(np.mean(100 * np.abs(t_true_quad - ta) / ta)),
        "t_arrive_true_curve_analytic_t_break_rel_err_%":
            float(np.mean(100 * np.abs(t_analytic_quad - ta) / ta)),
        "t_break_analytic_minus_pigsim_s": {
            "mean": float(np.mean(t_break_analytic(X) - tb_true)),
            "max_abs": float(np.max(np.abs(t_break_analytic(X) - tb_true)))},
        "grid_v_max_vs_pigsim_v_max_rel_%": float(np.mean(100 * np.abs(V.max(1) - d["v_max"][idx])
                                                         / d["v_max"][idx])),
    }


# ---------------------------------------------------------------------------
def fit_all(models, Xtr, Vtr, Xva, Vva, verbose=True):
    times = {}
    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(Xtr, Vtr, Xva, Vva)
        times[name] = time.perf_counter() - t0
        if verbose:
            extra = f", k = {m.k} PCA components" if hasattr(m, "k") else ""
            print(f"  {name:18s} fitted in {times[name]:6.1f} s{extra}", flush=True)
    return times


def split(n, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    a, b = int(0.70 * n), int(0.85 * n)
    return idx[:a], idx[a:b], idx[b:]


def _job(spec):
    """One training set -> fit all 5 models -> evaluate on one test set."""
    torch.set_num_threads(1)
    if spec.get("quick"):
        _make_quick()
    d, xi, tr, va, te = spec["d"], spec["d"]["xi"], spec["tr"], spec["va"], spec["te"]
    X, V = d["X"], d["V"]
    models = make_models(xi)
    times = {}
    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(X[tr], V[tr], X[va], V[va])
        times[name] = time.perf_counter() - t0
        print(f"    {spec['name']}: {name} fitted in {times[name]:.0f} s", flush=True)
    out = {"fit_time_s": times, "metrics": {}, "preds": {}}
    for name, m in models.items():
        t0 = time.perf_counter()
        mu, sd = m.predict(X[te])
        out.setdefault("predict_time_per_run_ms", {})[name] = 1e3 * (time.perf_counter() - t0) / len(te)
        out["metrics"][name] = evaluate(V[te], mu, sd, X[te], {"t_arrive": d["t_arrive"][te]}, xi)
        if spec["keep_preds"]:
            out["preds"][name] = (mu, sd)
    if spec["keep_preds"]:
        out["pca_components"] = {k: int(m.k) for k, m in models.items() if hasattr(m, "k")}
        out["net_info"] = {k: m.info for k, m in models.items() if hasattr(m, "info")}
        don = models["DeepONet x5"]
        os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
        torch.save({"state_dicts": [mm.state_dict() for mm in don.models], "p": don.p,
                    "width": don.width, "x_mean": don.xs.mean, "x_std": don.xs.std,
                    "v_scale": don.v_scale, "xi": xi, "inputs": P.INPUTS},
                   os.path.join(HERE, "models", "deeponet.pt"))
    print(f"  done: {spec['name']}", flush=True)
    return spec["name"], out


def _make_quick():
    import surrogate.stage3.models as M
    if getattr(M, "_quick", False):
        return
    M._quick = True
    M._NetEnsemble.n_members = 2
    _orig = M._train_net
    M._train_net = lambda *a, **k: _orig(*a, **dict(k, epochs=40, patience=10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--plots-only", action="store_true",
                    help="redraw figures from metrics.json and test_predictions.npz")
    args = ap.parse_args()
    d = dict(np.load(os.path.join(HERE, "curves.npz")))
    d["V"] = d["V"].astype(float)
    xi, X = d["xi"], d["X"]
    n = len(X)
    tr, va, te = split(n)
    if args.plots_only:
        res = json.load(open(os.path.join(HERE, "metrics.json")))
        z = np.load(os.path.join(HERE, "test_predictions.npz"))
        preds = {k: (z[f"{k}|mean"].astype(float),
                     z[f"{k}|std"].astype(float) if f"{k}|std" in z.files else None) for k in NAMES}
        os.makedirs(FIG, exist_ok=True)
        plots(d, z["te"], xi, preds, res)
        print("-> figures/")
        return
    print(f"{n} curves: {len(tr)} train / {len(va)} val / {len(te)} test; "
          f"backward motion in {int(np.sum(d['n_back'] > 0))} runs", flush=True)

    res = {"n_runs": int(n), "split": [len(tr), len(va), len(te)], "grid_points": int(len(xi)),
           "launch_segment": f"xi < {START}", "floor": floor_metrics(d, te, xi)}
    print("floor:", json.dumps(res["floor"]), flush=True)

    jobs = [{"name": "main", "tr": tr, "va": va, "te": te, "keep_preds": True}]
    extrap = {}
    for key, j in (("L", 3), ("p_in", 0)):
        cut = np.percentile(X[:, j], 90)
        hold = np.nonzero(X[:, j] > cut)[0]
        keep = np.random.default_rng(1).permutation(np.nonzero(X[:, j] <= cut)[0])
        ntr = int(0.85 * len(keep))
        extrap[key] = {"cut": float(cut), "n_train": int(ntr), "n_test": int(len(hold))}
        jobs.append({"name": f"extrap_{key}", "tr": keep[:ntr], "va": keep[ntr:], "te": hold,
                     "keep_preds": False})
    sizes = [50, 100, 200, 300]
    for nt in sizes:
        jobs.append({"name": f"lcurve_{nt}", "tr": tr[:nt], "va": va, "te": te, "keep_preds": False})
    for jb in jobs:
        jb["d"], jb["quick"] = d, args.quick

    from multiprocessing import Pool
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        out = dict(pool.map(_job, jobs, chunksize=1))
    print(f"all fits done in {(time.perf_counter()-t0)/60:.1f} min", flush=True)

    m = out["main"]
    res["fit_time_s"] = m["fit_time_s"]
    res["predict_time_per_run_ms"] = m["predict_time_per_run_ms"]
    res["pca_components"] = m["pca_components"]
    res["net_info"] = m["net_info"]
    res["main"] = m["metrics"]
    res["extrap"] = {k: dict(v, **out[f"extrap_{k}"]["metrics"]) for k, v in extrap.items()}
    res["lcurve"] = {"sizes": sizes + [len(tr)]}
    for name in NAMES:
        rows = []
        for key in [f"lcurve_{s}" for s in sizes] + ["main"]:
            r = out[key]["metrics"][name]
            rows.append({"relL2_all_%": r["relL2_all_%"]["mean"],
                         "relL2_launch_%": r["relL2_launch_%"]["mean"],
                         "t_arrive_rel_err_%": r["t_arrive_rel_err_%"]["mean"],
                         "coverage_2sigma": r.get("coverage", {}).get("2sigma")})
        res["lcurve"][name] = rows

    print("\nmain split")
    for name in NAMES:
        r = res["main"][name]
        print(f"  {name:18s} fit {res['fit_time_s'][name]:6.1f} s  relL2 all {r['relL2_all_%']['mean']:6.3f} %  "
              f"launch {r['relL2_launch_%']['mean']:6.3f} %  cruise {r['relL2_cruise_%']['mean']:6.3f} %  "
              f"v_max {r['v_max_rel_err_%']['mean']:5.2f} %  loc {r['v_max_location_err_m']['mean']:5.1f} m  "
              f"t_arr {r['t_arrive_rel_err_%']['mean']:5.2f} %"
              + (f"  2sigma {100*r['coverage']['2sigma']:4.0f} %" if "coverage" in r else ""))
    for key in ("L", "p_in"):
        for name in NAMES:
            r = res["extrap"][key][name]
            print(f"  extrap {key:5s} {name:18s} relL2 {r['relL2_all_%']['mean']:6.3f} %  "
                  f"t_arr {r['t_arrive_rel_err_%']['mean']:5.2f} %"
                  + (f"  2sigma {100*r['coverage']['2sigma']:4.0f} %" if "coverage" in r else ""))
    for name in NAMES:
        print(f"  lcurve {name:18s} " + "  ".join(f"{x['relL2_all_%']:.3f}" for x in res["lcurve"][name]))

    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    # test-set predictions, so the figures can be redrawn without retraining (--plots-only)
    np.savez_compressed(os.path.join(HERE, "test_predictions.npz"), te=te,
                        **{f"{k}|mean": v[0].astype(np.float32) for k, v in m["preds"].items()},
                        **{f"{k}|std": v[1].astype(np.float32) for k, v in m["preds"].items()
                           if v[1] is not None})
    os.makedirs(FIG, exist_ok=True)
    plots(d, te, xi, m["preds"], res)
    print("-> metrics.json, figures/, models/deeponet.pt")


# ---------------------------------------------------------------------------
def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plots(d, te, xi, preds, res):
    import matplotlib.pyplot as plt
    V, X = d["V"][te], d["X"][te]
    mu, sd = preds["DeepONet x5"]
    w = quad_weights(xi)
    err = rel_l2(V, mu, w, np.ones_like(xi, bool))
    picks = [int(np.argmin(err)), int(np.argsort(err)[len(err) // 2]), int(np.argmax(err))]

    # 1. example curves: best / median / worst DeepONet test run
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for c, (i, lab) in enumerate(zip(picks, ("best", "median", "worst"))):
        for r, (lo, hi) in enumerate(((0, 1), (0, START))):
            ax = axes[r, c]
            m = (xi >= lo) & (xi <= hi)
            ax.plot(xi[m], V[i, m], color=INK, lw=2.4, label="pigsim")
            for name in NAMES:
                ls = "-" if name == "DeepONet x5" else "--"
                ax.plot(xi[m], preds[name][0][i, m], color=COL[name], lw=1.3, ls=ls, label=name)
            ax.fill_between(xi[m], (mu - 2 * sd)[i, m], (mu + 2 * sd)[i, m], color=COL["DeepONet x5"],
                            alpha=0.18, lw=0, label="DeepONet ± 2 std")
            _style(ax)
            if r == 0:
                x = X[i]
                ax.set_title(f"{lab} DeepONet test run (rel. L2 {100*err[i]:.2f} %)\n"
                             f"p_in {x[0]/1e5:.2f} bar, mass {x[1]:.0f} kg, F {x[2]/1e3:.2f} kN, "
                             f"L {x[3]:.0f} m", fontsize=9.5, loc="left", color=INK)
            ax.set_xlabel("xi = (s − s0) / (0.999 L − s0)" + ("  [launch segment]" if r else ""),
                          color=INK, fontsize=9)
            if c == 0:
                ax.set_ylabel("pig speed V [m/s]", color=INK)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "curves_examples.png"), dpi=140)
    plt.close(fig)

    # 2. error per segment and model (test set)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    rng = np.random.default_rng(0)
    panels = [("all", "whole curve"), ("launch", f"launch (xi < {START})"), ("cruise", "cruise")]
    for ax, (seg, title) in zip(axes[:3], panels):
        mask = {"all": np.ones_like(xi, bool), "launch": xi < START, "cruise": xi >= START}[seg]
        for j, name in enumerate(NAMES):
            e = 100 * rel_l2(V, preds[name][0], w, mask)
            ax.scatter(j + rng.uniform(-0.18, 0.18, len(e)), e, s=8, color=COL[name], alpha=0.6, lw=0)
            ax.plot([j - 0.3, j + 0.3], [np.mean(e)] * 2, color=INK, lw=1.6)
        ax.set_yscale("log")
        ax.set_xticks(range(len(NAMES)))
        ax.set_xticklabels([n.replace(" ", "\n", 1) for n in NAMES], fontsize=8, color=INK)
        ax.set_title(f"relative L2 error, {title}", fontsize=10, loc="left", color=INK)
        _style(ax)
    axes[0].set_ylabel("per test run [%]  (bar = mean)", color=INK)
    ax = axes[3]
    t_true = d["t_arrive"][te]
    for name in NAMES:
        tp = t_break_analytic(X) + travel_time(preds[name][0], X, xi)
        ax.scatter(t_true, 100 * (tp - t_true) / t_true, s=8, color=COL[name], alpha=0.6, lw=0,
                   label=name)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([30, 100, 300])
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("pigsim t_arrive [s]", color=INK)
    ax.set_ylabel("(∫ds/V + t_break − pigsim) / pigsim [%]", color=INK)
    ax.set_title("arrival time from the predicted curve", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=7.5)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "errors_by_segment.png"), dpi=140)
    plt.close(fig)

    # 3. learning curves + 4. calibration
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    lc = res["lcurve"]
    for name in NAMES:
        axes[0].plot(lc["sizes"], [r["relL2_all_%"] for r in lc[name]], "o-", color=COL[name], label=name)
        axes[1].plot(lc["sizes"], [r["relL2_launch_%"] for r in lc[name]], "o-", color=COL[name])
    for ax, t in zip(axes[:2], ("whole curve", f"launch segment (xi < {START})")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(lc["sizes"])
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("training runs", color=INK)
        ax.set_ylabel("mean relative L2 error, test [%]", color=INK)
        ax.set_title(f"learning curve, {t}", fontsize=10, loc="left", color=INK)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    ax = axes[2]
    ks = np.array([1, 2, 3])
    from scipy.stats import norm
    ax.plot(ks, 100 * (2 * norm.cdf(ks) - 1), "k--", lw=1, label="calibrated Gaussian")
    for name in ("DeepONet x5", "MLP x5", "PCA + GP"):
        cov = res["main"][name]["coverage"]
        ax.plot(ks, [100 * cov[f"{k}sigma"] for k in ks], "o-", color=COL[name], label=name)
    ax.set_xticks(ks)
    ax.set_xticklabels(["1 std", "2 std", "3 std"])
    ax.set_ylabel("test grid points with |error| ≤ k std [%]", color=INK)
    ax.set_title("calibration of the predicted std (test set)", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "learning_and_calibration.png"), dpi=140)
    plt.close(fig)

    # 5. extrapolation
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, key in zip(axes, ("L", "p_in")):
        ex = res["extrap"][key]
        vals = [ex[n]["relL2_all_%"]["mean"] for n in NAMES]
        inside = [res["main"][n]["relL2_all_%"]["mean"] for n in NAMES]
        xs = np.arange(len(NAMES))
        ax.bar(xs - 0.2, inside, 0.38, color=GRID, edgecolor=MUTED, label="random test split")
        ax.bar(xs + 0.2, vals, 0.38, color=[COL[n] for n in NAMES], label=f"top 10 % of {key} held out")
        for x, v in zip(xs, vals):
            ax.text(x + 0.2, v * 1.08, f"{v:.2g}", ha="center", fontsize=8, color=INK)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([n.replace(" ", "\n", 1) for n in NAMES], fontsize=8, color=INK)
        unit = "m" if key == "L" else "bar"
        cut = ex["cut"] if key == "L" else ex["cut"] / 1e5
        ax.set_title(f"extrapolation: trained on {key} ≤ {cut:.0f} {unit}, tested above",
                     fontsize=10, loc="left", color=INK)
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    axes[0].set_ylabel("mean relative L2 error [%]", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "extrapolation.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
