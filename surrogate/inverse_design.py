"""Stage 2c: inverse design with the wide-box surrogate, checked against pigsim.

    python surrogate/inverse_design.py                       # defaults below, ~2 min
    python surrogate/inverse_design.py --L 2500 --t-max 600 --v-max 35

Question: for a pipeline of length L, which (p_in, mass, F_fric) get the pig
to the outlet within t_max with a peak speed of at most v_max?  Among those,
prefer the lowest inlet pressure p_in (the cheapest drive).

    minimise    p_in [bar]  +  lam * (std_t / t + std_v / v) [%]      <- uncertainty penalty
    subject to  t + k std_t <= t_max,  v + k std_v <= v_cap            <- the targets (k = 2)
                P_arrive(x) >= 0.95                                    <- arrival classifier
                x inside the training box                              <- hard bounds

t, v and their std come from the 5-member wide-box MLP ensemble, P_arrive from
the 5-member classifier (both written by regime.py).  The constraints are
enforced with a quadratic penalty whose weight grows from 1e2 to 1e5 during
the run (penalty continuation); the box is enforced exactly by writing
x = lo + (hi - lo) * sigmoid(u) and optimising the unconstrained u.

Everything is a PyTorch tensor, so autograd gives d(loss)/du through both
networks.  256 random starts are optimised together in one batch with Adam.
The same search is repeated without the classifier, the std penalty and the
k std margin ("naive") to show what those two terms are for.  Up to 5 distinct candidates
and up to 5 naive ones are re-run in pigsim.

Output: inverse.json, figures/inverse_design.png
"""

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate.generate_wide_data import T_END, WIDE_RANGES, simulate_ext  # noqa: E402
from surrogate.regime import load_classifier, p_start  # noqa: E402
from surrogate.train_surrogate import (  # noqa: E402
    BLUE, FIG, HERE, INK, MUTED, ORANGE, INPUTS, _style, load_ensemble,
)

DESIGN = ["p_in", "mass", "F_fric"]          # L is given, not designed
P_MIN_ARRIVE = 0.95


class TorchSurrogate:
    """Differentiable versions of Ensemble.predict and ArrivalClassifier.predict_proba."""

    def __init__(self):
        ens = load_ensemble(os.path.join(HERE, "models", "wide_ensemble.pt"))
        clf = load_classifier()
        self.regs, self.cls = ens.models, clf.models
        f = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
        self.xm, self.xsd = f(ens.xs.mean), f(ens.xs.std)
        self.ym, self.ysd = f(ens.ys.mean), f(ens.ys.std)
        self.cm, self.csd = f(clf.xs.mean), f(clf.xs.std)
        for m in self.regs + self.cls:
            for p in m.parameters():
                p.requires_grad_(False)          # we differentiate w.r.t. x, not weights

    def __call__(self, X):
        """X: (n, 4) tensor in physical units -> dict of (n,) tensors."""
        z = torch.stack([m((X - self.xm) / self.xsd) for m in self.regs])   # (5, n, 2)
        y = z * self.ysd + self.ym
        t = torch.exp(y[..., 0])                 # log t_arrive -> seconds
        v = y[..., 1]
        pa = torch.stack([torch.sigmoid(m((X - self.cm) / self.csd))[:, 0] for m in self.cls])
        return {"t": t.mean(0), "t_sd": t.std(0, unbiased=False),
                "v": v.mean(0), "v_sd": v.std(0, unbiased=False), "p_arrive": pa.mean(0)}


def optimise(sur, L, t_max, v_cap, use_classifier, lam, k_sigma, n_starts=256, iters=1500,
             seed=0):
    lo = torch.tensor([WIDE_RANGES[k][0] for k in DESIGN])
    hi = torch.tensor([WIDE_RANGES[k][1] for k in DESIGN])
    g = torch.Generator().manual_seed(seed)
    u = torch.randn(n_starts, len(DESIGN), generator=g).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=0.05)

    def evaluate(u, rho=0.0):
        x = lo + (hi - lo) * torch.sigmoid(u)                      # always inside the box
        X = torch.cat([x, torch.full((len(x), 1), float(L))], dim=1)
        o = sur(X)
        t_hi = o["t"] + k_sigma * o["t_sd"]                          # pessimistic side
        v_hi = o["v"] + k_sigma * o["v_sd"]
        pen = torch.relu(t_hi / t_max - 1) ** 2 + torch.relu(v_hi / v_cap - 1) ** 2
        if use_classifier:
            pen = pen + torch.relu(P_MIN_ARRIVE - o["p_arrive"]) ** 2
        unc = 100 * (o["t_sd"] / o["t"] + o["v_sd"] / o["v"])      # relative std in %
        J = x[:, 0] / 1e5 + lam * unc
        return x, o, J, J + rho * pen, pen

    for it in range(iters):
        # penalty continuation: start soft (rho = 1e2) so the starts can move
        # freely, end stiff (rho = 1e5) so the constraints are met almost exactly
        rho = 10 ** (2 + 3 * it / (iters - 1))
        opt.zero_grad()
        _, _, _, loss, _ = evaluate(u, rho)
        loss.sum().backward()       # starts are independent, so sum = optimise each
        opt.step()
    with torch.no_grad():
        x, o, J, _, pen = evaluate(u)
    res = {k: v.numpy() for k, v in o.items()}
    res.update(x=x.numpy(), J=J.numpy(), pen=pen.numpy())
    return res


def pick(res, feasible, k=5, min_dist=0.12):
    """Greedy: best objective first, then only points far enough from those chosen."""
    lo = np.array([WIDE_RANGES[d][0] for d in DESIGN])
    hi = np.array([WIDE_RANGES[d][1] for d in DESIGN])
    unit = (res["x"] - lo) / (hi - lo)
    chosen = []
    for i in np.argsort(np.where(feasible, res["J"], np.inf)):
        if not feasible[i] or len(chosen) == k:
            break
        if all(np.linalg.norm(unit[i] - unit[j]) > min_dist for j in chosen):
            chosen.append(i)
    return chosen


def _sim(x):
    return simulate_ext(*x)


def plot(results, L, t_max, v_cap, path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    ax = axes[0]
    for key, col, lab in (("naive", MUTED, "naive (no classifier, no std penalty)"),
                          ("full", BLUE, "with classifier + std penalty")):
        r = results[key]
        f = r["feasible"]
        ax.scatter(r["x"][f, 2] / 1e3, r["x"][f, 0] / 1e5, s=10, color=col, alpha=0.5, lw=0,
                   label=f"{lab}: {int(f.sum())} converged starts")
    F = np.linspace(*WIDE_RANGES["F_fric"], 100)
    ax.plot(F / 1e3, p_start(F) / 1e5, color=INK, lw=1, ls="--", label="analytic start threshold")
    for key, mk in (("naive", "X"), ("full", "o")):
        for c in results[key]["candidates"]:
            ax.scatter(c["x"][2] / 1e3, c["x"][0] / 1e5, s=80, marker=mk,
                       color=ORANGE if not c["pigsim"]["ok"] else BLUE,
                       edgecolor=INK, lw=0.8, zorder=5)
    ax.set_xlabel("F_fric [kN]", color=INK)
    ax.set_ylabel("p_in [bar]", color=INK)
    ax.set_title("Optimised designs (circle / cross = re-run in pigsim;\n"
                 "orange = pigsim misses a target)", color=INK, loc="left", fontsize=10)
    _style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")

    cands = [("full", i, c) for i, c in enumerate(results["full"]["candidates"])] + \
            [("naive", i, c) for i, c in enumerate(results["naive"]["candidates"])]
    labels = [f"{'A' if k == 'full' else 'N'}{i+1}" for k, i, _ in cands]
    for ax, key, sdk, cap, unit, name in ((axes[1], "t", "t_sd", t_max, "s", "t_arrive"),
                                          (axes[2], "v", "v_sd", v_cap, "m/s", "v_max")):
        for j, (k, _, c) in enumerate(cands):
            ax.errorbar(j - 0.12, c["pred"][key], yerr=2 * c["pred"][sdk], fmt="o",
                        color=BLUE if k == "full" else MUTED, ms=6, capsize=3,
                        label="surrogate ± 2 std" if j == 0 else None)
            ps = c["pigsim"]
            val = ps["t_arrive"] if key == "t" else ps["v_max"]
            if key == "t" and not ps["arrived"]:
                ax.scatter(j + 0.12, cap * 1.02, marker="^", s=60, color=ORANGE, zorder=5,
                           label="pigsim: did not arrive" if key == "t" else None)
            else:
                ax.scatter(j + 0.12, val, marker="D", s=40, color=INK, zorder=5,
                           label="pigsim" if j == 0 else None)
        ax.axhline(cap, color=ORANGE, lw=1, ls="--", label=f"target {cap:g} {unit}")
        ax.set_xticks(range(len(cands)))
        ax.set_xticklabels(labels, color=INK)
        ax.set_ylabel(f"{name} [{unit}]", color=INK)
        ax.set_title(f"{name}: surrogate vs pigsim (A = full, N = naive)",
                     color=INK, loc="left", fontsize=10)
        _style(ax)
        h, lab = ax.get_legend_handles_labels()
        uniq = dict(zip(lab, h))
        ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=7.5, loc="lower left",
                  scatterpoints=1, numpoints=1)
    fig.suptitle(f"Inverse design, L = {L:.0f} m: t_arrive ≤ {t_max:g} s, "
                 f"v_max ≤ {v_cap:g} m/s, minimum p_in", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=2500.0)
    ap.add_argument("--t-max", type=float, default=600.0)
    ap.add_argument("--v-max", type=float, default=35.0)
    ap.add_argument("--lam", type=float, default=0.05,
                    help="bar of p_in traded for 1 %% of relative ensemble std")
    ap.add_argument("--k-sigma", type=float, default=2.0,
                    help="targets must hold at mean + k * ensemble std")
    ap.add_argument("--n-candidates", type=int, default=5)
    ap.add_argument("--tag", default="", help="suffix for inverse<tag>.json / .png")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    sur = TorchSurrogate()

    results = {}
    for key, use_cls, lam, ks in (("full", True, args.lam, args.k_sigma),
                                  ("naive", False, 0.0, 0.0)):
        r = optimise(sur, args.L, args.t_max, args.v_max, use_cls, lam, ks)
        ok = ((r["t"] + ks * r["t_sd"] <= args.t_max * 1.001)
              & (r["v"] + ks * r["v_sd"] <= args.v_max * 1.001))
        if use_cls:
            ok &= r["p_arrive"] >= P_MIN_ARRIVE - 1e-3
        r["feasible"] = ok
        idx = pick(r, ok, k=args.n_candidates)
        # cheap physical check of every converged start: can the pig start at all?
        r["below_threshold"] = r["x"][:, 0] <= p_start(r["x"][:, 2])
        r["candidates"] = [{"x": r["x"][i].tolist(),
                            "pred": {k: float(r[k][i]) for k in ("t", "t_sd", "v", "v_sd", "p_arrive")},
                            "J": float(r["J"][i])} for i in idx]
        results[key] = r
        print(f"{key}: {ok.sum()}/{len(ok)} starts end feasible (surrogate), "
              f"{(ok & r['below_threshold']).sum()} of those below the analytic start "
              f"threshold, {len(idx)} candidates")

    jobs = [c["x"] + [args.L] for key in ("full", "naive") for c in results[key]["candidates"]]
    with Pool(4) as pool:
        sims = pool.map(_sim, jobs)
    k = 0
    for key in ("full", "naive"):
        for c in results[key]["candidates"]:
            s = sims[k]
            k += 1
            s["ok"] = bool(s["arrived"] and s["t_arrive"] <= args.t_max and s["v_max"] <= args.v_max)
            c["pigsim"] = {kk: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                           for kk, v in s.items()}
            p = c["pred"]
            x = c["x"]
            if s["arrived"]:
                cmp = (f"t {p['t']:6.1f}±{p['t_sd']:4.1f} vs {s['t_arrive']:6.1f} "
                       f"({100*(p['t']-s['t_arrive'])/s['t_arrive']:+5.1f} %)  "
                       f"v {p['v']:5.2f}±{p['v_sd']:4.2f} vs {s['v_max']:5.2f} "
                       f"({100*(p['v']-s['v_max'])/s['v_max']:+5.1f} %)")
            else:
                cmp = (f"t {p['t']:6.1f}±{p['t_sd']:4.1f} vs NOT ARRIVED "
                       f"(started={s['started']}, s_end={s['s_end']:.0f} m)")
            print(f"  {key:5s} p_in {x[0]/1e5:5.3f} bar  mass {x[1]:5.1f} kg  F {x[2]/1e3:5.2f} kN  "
                  f"P(arr) {p['p_arrive']:.3f}  margin {(x[0]-p_start(x[2]))/1e5:+.3f} bar  "
                  f"{cmp}  -> {'OK' if s['ok'] else 'MISSES TARGET'}")

    out = {"problem": {"L": args.L, "t_max": args.t_max, "v_max": args.v_max, "lam": args.lam, "k_sigma": args.k_sigma,
                       "p_min_arrive": P_MIN_ARRIVE, "design": DESIGN, "bounds": WIDE_RANGES},
           **{key: {"n_starts": int(len(results[key]["feasible"])),
                    "n_feasible": int(results[key]["feasible"].sum()),
                    "n_feasible_below_threshold": int((results[key]["feasible"]
                                                       & results[key]["below_threshold"]).sum()),
                    "candidates": results[key]["candidates"]} for key in ("full", "naive")}}
    with open(os.path.join(HERE, f"inverse{args.tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    plot(results, args.L, args.t_max, args.v_max, os.path.join(FIG, f"inverse_design{args.tag}.png"))
    print(f"-> inverse{args.tag}.json, figures/inverse_design{args.tag}.png")


if __name__ == "__main__":
    main()
