"""Stage 2b: regime transition -- pigs that never start or never arrive.

    python surrogate/regime.py          # ~3 min (training + 14 pigsim runs for the sweep)

Data: data.csv (stage 1, 650 runs, all arrived) + data_wide.csv (wider box,
includes non-arrivals).  Splits:
    data.csv       the stage-1 split (default_rng(0) -> 454 / 97 / 99), unchanged
    data_wide.csv  its own 70 / 15 / 15 split (default_rng(1))
Train / val / test of the two files are concatenated.

Models (all fitted on the combined training split):
    regressors on *arrived* runs only (t_arrive is undefined otherwise):
        poly3 (log x), GP, MLP x5 (same recipe as stage 1, retrained)
    classifier on *all* runs: "does the pig reach the outlet before 2000 s?"
        MLP x5 classifier (PyTorch, BCEWithLogitsLoss), probability = mean of members
    reference rule (no learning): the pig can only start if
        p_in > p_res + F_static / A = 1.5 bar + 1.2 F_fric / A

Questions answered on the wide test split (the only test rows that contain
non-arrivals):
    1. arrival yes/no -- regressors alone can only say "t_pred <= 2000 s";
       the GP may also flag doubt through its std; the classifier answers directly
    2. t_arrive / v_max accuracy for arrived runs, binned by the distance to the
       start threshold, dp_margin = p_in - (1.5 bar + 1.2 F_fric / A)

Outputs: regime.json, models/wide_ensemble.pt, models/arrival_classifier.pt,
figures/regime_sweep.png, figures/regime_map.png
"""

import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate import pipeline as P  # noqa: E402
from surrogate.baselines import GPModel, PolyModel, score  # noqa: E402
from surrogate.generate_wide_data import T_END, WIDE_RANGES, simulate_ext  # noqa: E402
from surrogate.train_surrogate import (  # noqa: E402
    BLUE, FIG, GRID, HERE, INK, MUTED, ORANGE, OUTPUTS, Ensemble, INPUTS, Scaler,
    _style, from_target, load_and_split, load_ensemble, make_mlp, to_target, train_member,
)

AQUA = "#1baf7a"
AREA = np.pi * P.D ** 2 / 4
N_MEMBERS = 5


def p_start(F_fric):
    """Lowest inlet pressure at which the pig can break free [Pa]."""
    return P.P_RES + P.STATIC_OVER_DYNAMIC * F_fric / AREA


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_all():
    """Combined table with a 'split' column and a 'src' column (stage1 / wide)."""
    d1, _, _, sp = load_and_split(os.path.join(HERE, "data.csv"), np.random.default_rng(0))
    d1 = d1.copy()
    d1["split"] = ""
    for k, idx in sp.items():
        d1.loc[idx, "split"] = k
    d1["src"] = "stage1"
    d1["set"] = "stage1"
    d1["started"] = 1

    d2 = pd.read_csv(os.path.join(HERE, "data_wide.csv"))
    d2 = d2[np.isfinite(d2["s_end"])].reset_index(drop=True)      # drop crashed runs
    idx = np.random.default_rng(1).permutation(len(d2))
    n_tr, n_va = int(0.70 * len(d2)), int(0.15 * len(d2))
    d2["split"] = ""
    d2.loc[idx[:n_tr], "split"] = "train"
    d2.loc[idx[n_tr:n_tr + n_va], "split"] = "val"
    d2.loc[idx[n_tr + n_va:], "split"] = "test"
    d2["src"] = "wide"

    cols = ["src", "set", "split"] + INPUTS + OUTPUTS + ["arrived", "started"]
    df = pd.concat([d1[cols], d2[cols]], ignore_index=True)
    df["dp_margin"] = df["p_in"] - p_start(df["F_fric"])
    return df


# ---------------------------------------------------------------------------
# arrival classifier (PyTorch)
# ---------------------------------------------------------------------------
def train_classifier_member(Xtr, ytr, Xva, yva, seed, epochs=2000, patience=200,
                            lr=3e-3, weight_decay=1e-5, batch=64):
    """Binary classifier: 4 inputs -> 1 logit.  Same loop as train_member.

    The network outputs a *logit* z (any real number); p = sigmoid(z).
    BCEWithLogitsLoss = binary cross entropy computed from the logit directly,
    which is numerically safer than sigmoid followed by BCELoss.
    """
    torch.manual_seed(seed)
    boot = np.random.default_rng(seed).integers(0, len(Xtr), len(Xtr))
    xt = torch.tensor(Xtr[boot], dtype=torch.float32)
    yt = torch.tensor(ytr[boot, None], dtype=torch.float32)       # shape (n, 1)
    xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(yva[:, None], dtype=torch.float32)

    model = make_mlp(xt.shape[1], 1, width=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, wait, n_ep = float("inf"), None, 0, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(xv), yv).item()
        n_ep = ep + 1
        if v < best - 1e-6:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait > patience:
                break
    model.load_state_dict(best_state)
    print(f"  classifier seed={seed}: {n_ep} epochs, best val BCE {best:.3f}")
    return model


class ArrivalClassifier:
    def __init__(self, models, xs):
        self.models, self.xs = models, xs

    def predict_proba(self, X):
        x = torch.tensor(self.xs.fwd(X), dtype=torch.float32)
        with torch.no_grad():
            p = np.stack([torch.sigmoid(m(x)).numpy()[:, 0] for m in self.models])
        return p.mean(axis=0), p.std(axis=0)


def load_classifier(path=os.path.join(HERE, "models", "arrival_classifier.pt")):
    ck = torch.load(path, weights_only=False)
    models = []
    for sd in ck["state_dicts"]:
        m = make_mlp(len(ck["inputs"]), 1, width=32)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    xs = Scaler.__new__(Scaler)
    xs.mean, xs.std = ck["x_mean"], ck["x_std"]
    return ArrivalClassifier(models, xs)


# ---------------------------------------------------------------------------
def class_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    return {
        "n": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)),
        # the dangerous error: model says "arrives" but the pig is stuck / late
        "false_arrive": int(np.sum(~y_true & y_pred)),
        "missed_arrive": int(np.sum(y_true & ~y_pred)),
        "n_not_arrived": int(np.sum(~y_true)),
    }


BINS = [(-np.inf, 0.3e5, "< 0.3 bar"), (0.3e5, 1.0e5, "0.3 - 1 bar"), (1.0e5, np.inf, "> 1 bar")]


def _sweep_sim(x):
    r = simulate_ext(*x)
    return r["t_arrive"], r["v_max"], r["arrived"], r["started"]


def plot_sweep(sweep, path):
    import matplotlib.pyplot as plt
    x = sweep["p_in"] / 1e5
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 2, 1.4]})
    ok = sweep["p_cls"] >= 0.5
    for r, name in enumerate(OUTPUTS):
        ax = axes[r]
        ax.plot(x, sweep["poly3"][:, r], color=AQUA, lw=1.6, label="poly3 (log x)")
        ax.fill_between(x, sweep["gp_mu"][:, r] - 2 * sweep["gp_sd"][:, r],
                        sweep["gp_mu"][:, r] + 2 * sweep["gp_sd"][:, r],
                        color=ORANGE, alpha=0.15, lw=0)
        ax.plot(x, sweep["gp_mu"][:, r], color=ORANGE, lw=1.6, label="GP ± 2 std")
        mu, sd = sweep["nn_mu"][:, r], sweep["nn_sd"][:, r]
        ax.fill_between(x, mu - 2 * sd, mu + 2 * sd, where=ok, color=BLUE, alpha=0.18, lw=0)
        ax.plot(np.where(ok, x, np.nan), mu, color=BLUE, lw=2.2,
                label="MLP x5 ± 2 std, where classifier says arrives")
        ax.plot(np.where(~ok, x, np.nan), mu, color=BLUE, lw=1, ls=":",
                label="MLP x5 where classifier says no")
        ax.plot(x, sweep["stage1"][:, r], color=MUTED, lw=1, ls="--",
                label="stage-1 MLP (trained on p_in ≥ 3 bar)")
        sa = sweep["sim_arrived"]
        ax.scatter(sweep["sim_p"][sa] / 1e5, sweep["sim"][sa, r], s=40, color=INK, zorder=5,
                   label="pigsim")
        if r == 1:      # v_max exists for non-arrivals too (0 if the pig never moved)
            ax.scatter(sweep["sim_p"][~sa] / 1e5, sweep["sim"][~sa, r], s=40, facecolor="white",
                       edgecolor=INK, zorder=5, label="pigsim, pig did not arrive")
            ax.legend(frameon=False, fontsize=8, loc="upper left", handles=[
                ax.collections[-1]])
        ax.axvline(sweep["p_start"] / 1e5, color=MUTED, lw=1)
        ax.set_ylabel(["t_arrive [s]", "v_max [m/s]"][r], color=INK)
        _style(ax)
    axes[0].set_yscale("log")
    axes[0].set_ylim(30, 20000)
    axes[0].axhline(T_END, color=MUTED, lw=0.8, ls="--")
    axes[0].text(x[-1], T_END * 1.08, "t_end = 2000 s", ha="right", fontsize=8, color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].set_ylim(-5, None)
    ax = axes[2]
    ax.plot(x, sweep["p_cls"], color=BLUE, lw=2, label="classifier P(arrive)")
    ax.fill_between(x, np.clip(sweep["p_cls"] - 2 * sweep["p_cls_sd"], 0, 1),
                    np.clip(sweep["p_cls"] + 2 * sweep["p_cls_sd"], 0, 1),
                    color=BLUE, alpha=0.15, lw=0)
    ax.scatter(sweep["sim_p"] / 1e5, sweep["sim_arrived"].astype(float), s=36, color=INK,
               zorder=5, label="pigsim (1 = arrived)")
    ax.axvline(sweep["p_start"] / 1e5, color=MUTED, lw=1, label="analytic start threshold")
    ax.set_ylabel("P(arrive)", color=INK)
    ax.set_xlabel("inlet pressure p_in [bar]", color=INK)
    ax.set_ylim(-0.08, 1.08)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    c = sweep["fixed"]
    fig.suptitle(f"p_in sweep across the start threshold "
                 f"(mass {c['mass']:.0f} kg, F_fric {c['F_fric']/1e3:.1f} kN, L {c['L']:.0f} m)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_map(df, clf, path):
    """All wide-box runs in the (F_fric, p_in) plane, coloured by outcome."""
    import matplotlib.pyplot as plt
    w = df[df["src"] == "wide"]
    fig, ax = plt.subplots(figsize=(8.5, 6))
    groups = [(w["arrived"] == 1, BLUE, "arrived"),
              ((w["arrived"] == 0) & (w["started"] == 1), ORANGE, "started, not arrived by 2000 s"),
              (w["started"] == 0, INK, "never started")]
    for m, col, lab in groups:
        ax.scatter(w.loc[m, "F_fric"] / 1e3, w.loc[m, "p_in"] / 1e5,
                   s=60 if col == ORANGE else 16, color=col,
                   edgecolor="white", lw=0.4, label=f"{lab} ({int(m.sum())})", zorder=3)
    F = np.linspace(*WIDE_RANGES["F_fric"], 200)
    ax.plot(F / 1e3, p_start(F) / 1e5, color=MUTED, lw=1.5, label="analytic start threshold")
    # classifier boundary P = 0.5 for three pipe lengths (mass at mid-range)
    pp = np.linspace(1.6e5, 4.0e5, 240)
    FF, PP = np.meshgrid(F, pp)
    for L, ls in ((500.0, ":"), (1750.0, "--"), (3000.0, "-")):
        X = np.column_stack([PP.ravel(), np.full(PP.size, 110.0), FF.ravel(), np.full(PP.size, L)])
        pr = clf.predict_proba(X)[0].reshape(PP.shape)
        cs = ax.contour(FF / 1e3, PP / 1e5, pr, levels=[0.5], colors=[BLUE], linestyles=[ls],
                        linewidths=1.2)
        ax.plot([], [], color=BLUE, ls=ls, lw=1.2, label=f"classifier P = 0.5, L = {L:.0f} m")
        del cs
    ax.set_ylim(1.55, 4.0)
    ax.set_xlabel("dynamic friction F_fric [kN]", color=INK)
    ax.set_ylabel("inlet pressure p_in [bar]", color=INK)
    ax.set_title("Wide-box runs below 4 bar: outcome vs friction and inlet pressure",
                 color=INK, loc="left", fontsize=11)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    torch.set_num_threads(1)
    df = load_all()
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        print(f"{s:5s}: {len(d)} runs ({int(d['arrived'].sum())} arrived), "
              f"wide-box part {int((d['src'] == 'wide').sum())}")
    X = df[INPUTS].to_numpy(float)
    Y = df[OUTPUTS].to_numpy(float)
    A = df["arrived"].to_numpy(int)
    tr = (df["split"] == "train").to_numpy()
    va = (df["split"] == "val").to_numpy()
    te = (df["split"] == "test").to_numpy()
    wide_te = te & (df["src"] == "wide").to_numpy()
    s1_te = te & (df["src"] == "stage1").to_numpy()

    # ---- regressors on arrived runs ----
    tra, vaa = tr & (A == 1), va & (A == 1)
    t0 = time.perf_counter()
    poly = PolyModel(3, log_x=True).fit(X[tra], Y[tra])
    gp = GPModel().fit(X[tra], Y[tra])
    print(f"poly3 + GP fitted in {time.perf_counter()-t0:.1f} s on {tra.sum()} arrived runs")

    xs = Scaler(X[tra])
    ys = Scaler(to_target(Y[tra]))
    t0 = time.perf_counter()
    regs = [train_member(xs.fwd(X[tra]), ys.fwd(to_target(Y[tra])),
                         xs.fwd(X[vaa]), ys.fwd(to_target(Y[vaa])), seed=i)[0]
            for i in range(N_MEMBERS)]
    ens = Ensemble(regs, xs, ys)
    print(f"MLP regression ensemble: {time.perf_counter()-t0:.0f} s")

    # ---- classifier on all runs ----
    xc = Scaler(X[tr])
    t0 = time.perf_counter()
    cls = [train_classifier_member(xc.fwd(X[tr]), A[tr].astype(float),
                                   xc.fwd(X[va]), A[va].astype(float), seed=i)
           for i in range(N_MEMBERS)]
    clf = ArrivalClassifier(cls, xc)
    print(f"classifier ensemble: {time.perf_counter()-t0:.0f} s")

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    torch.save({"state_dicts": [m.state_dict() for m in regs],
                "x_mean": xs.mean, "x_std": xs.std, "y_mean": ys.mean, "y_std": ys.std,
                "inputs": INPUTS, "outputs": OUTPUTS, "ranges": WIDE_RANGES},
               os.path.join(HERE, "models", "wide_ensemble.pt"))
    torch.save({"state_dicts": [m.state_dict() for m in cls],
                "x_mean": xc.mean, "x_std": xc.std, "inputs": INPUTS, "t_end": T_END,
                "ranges": WIDE_RANGES},
               os.path.join(HERE, "models", "arrival_classifier.pt"))

    stage1 = load_ensemble()

    # ---- predictions on the test rows ----
    pred = {}
    for name, fn in (("poly3 (log x)", lambda Z: (poly.predict(Z)[0], None)),
                     ("GP", gp.predict),
                     ("MLP x5", lambda Z: ens.predict(Z)[:2]),
                     ("stage-1 MLP", lambda Z: stage1.predict(Z)[:2])):
        mu, sd = fn(X)
        pred[name] = (mu, sd)
    p_cls, _ = clf.predict_proba(X)

    out = {"counts": {}, "classification": {}, "regression": {}, "not_arrived_predictions": {}}
    for s in ("train", "val", "test"):
        d = df[df["split"] == s]
        out["counts"][s] = {"total": int(len(d)), "arrived": int(d["arrived"].sum()),
                            "wide_box": int((d["src"] == "wide").sum()),
                            "wide_not_arrived": int(((d["src"] == "wide") & (d["arrived"] == 0)).sum())}
    w = df[df["src"] == "wide"]
    out["counts"]["wide_all"] = {
        "total": int(len(w)), "arrived": int(w["arrived"].sum()),
        "started_not_arrived": int(((w["started"] == 1) & (w["arrived"] == 0)).sum()),
        "never_started": int((w["started"] == 0).sum())}

    # 1. arrival yes/no on the wide test rows
    yt = A[wide_te] == 1
    rules = {
        "analytic start rule": X[wide_te, 0] > p_start(X[wide_te, 2]),
        "poly3: t_pred <= 2000 s": pred["poly3 (log x)"][0][wide_te, 0] <= T_END,
        "GP: t_pred <= 2000 s": pred["GP"][0][wide_te, 0] <= T_END,
        "GP: t_pred + 2 std <= 2000 s": (pred["GP"][0][wide_te, 0]
                                        + 2 * pred["GP"][1][wide_te, 0]) <= T_END,
        "MLP x5: t_pred <= 2000 s": pred["MLP x5"][0][wide_te, 0] <= T_END,
        "stage-1 MLP: t_pred <= 2000 s": pred["stage-1 MLP"][0][wide_te, 0] <= T_END,
        "classifier P >= 0.5": p_cls[wide_te] >= 0.5,
    }
    print("\n1. arrival yes/no on the wide-box test rows")
    for k, yp in rules.items():
        m = class_metrics(yt, yp)
        out["classification"][k] = m
        print(f"  {k:32s} acc {100*m['accuracy']:5.1f} %  false 'arrives' "
              f"{m['false_arrive']:2d}/{m['n_not_arrived']}  missed {m['missed_arrive']}")
    # log-loss / Brier of the classifier on the test rows
    pc = np.clip(p_cls[wide_te], 1e-6, 1 - 1e-6)
    out["classification"]["classifier P >= 0.5"]["brier"] = float(np.mean((pc - yt) ** 2))

    # what the regressors say for the pigs that did NOT arrive (all wide rows not
    # used for training their regressors, i.e. every non-arrived wide row)
    na = (df["src"] == "wide").to_numpy() & (A == 0)
    print(f"\n   predictions for all {na.sum()} wide-box runs that did not arrive")
    for name, (mu, sd) in pred.items():
        t = mu[na, 0]
        d = {"median_t_pred": float(np.median(t)), "frac_t_pred_le_t_end": float(np.mean(t <= T_END))}
        if sd is not None:
            d["median_rel_std_t"] = float(np.median(sd[na, 0] / t))
            d["frac_upper2sd_gt_t_end"] = float(np.mean(t + 2 * sd[na, 0] > T_END))
        out["not_arrived_predictions"][name] = d
        print(f"  {name:14s} {d}")
    d = {"median_p": float(np.median(p_cls[na])), "frac_p_ge_0.5": float(np.mean(p_cls[na] >= 0.5))}
    out["not_arrived_predictions"]["classifier"] = d
    print(f"  classifier     {d}")

    # 2. regression accuracy on arrived test rows, by margin
    print("\n2. arrived test rows: MAPE by distance to the start threshold")
    groups = {"stage-1 box test": s1_te & (A == 1)}
    marg = df["dp_margin"].to_numpy()
    for lo, hi, lab in BINS:
        groups[f"wide test, margin {lab}"] = wide_te & (A == 1) & (marg >= lo) & (marg < hi)
    for g, m in groups.items():
        out["regression"][g] = {"n": int(m.sum())}
        for name, (mu, sd) in pred.items():
            out["regression"][g][name] = {
                o: score(Y[m, j], mu[m, j], None if sd is None else sd[m, j])
                for j, o in enumerate(OUTPUTS)} if m.sum() > 2 else None
        line = "  ".join(
            f"{name} {out['regression'][g][name]['t_arrive']['MAPE_%']:6.2f}/"
            f"{out['regression'][g][name]['v_max']['MAPE_%']:5.2f}"
            for name in pred if out["regression"][g][name])
        print(f"  {g:32s} n={m.sum():3d}  {line}")

    # ---- sweep with pigsim truth ----
    fixed = {"mass": 110.0, "F_fric": 4500.0, "L": 2500.0}
    ps = p_start(fixed["F_fric"])
    grid = np.linspace(1.6e5, 4.0e5, 300)
    Xs = np.column_stack([grid, np.full_like(grid, fixed["mass"]),
                          np.full_like(grid, fixed["F_fric"]), np.full_like(grid, fixed["L"])])
    sim_p = np.array([1.8, 2.1, 2.2, 2.25, 2.3, 2.35, 2.4, 2.5, 2.6, 2.8, 3.0, 3.25, 3.5, 4.0]) * 1e5
    with Pool(4) as pool:
        sims = pool.map(_sweep_sim, [(p, fixed["mass"], fixed["F_fric"], fixed["L"]) for p in sim_p])
    sim = np.array([[s[0], s[1]] for s in sims])
    gmu, gsd = gp.predict(Xs)
    nmu, nsd, _ = ens.predict(Xs)
    pcl, pcl_sd = clf.predict_proba(Xs)
    sweep = {"p_in": grid, "poly3": poly.predict(Xs)[0], "gp_mu": gmu, "gp_sd": gsd,
             "nn_mu": nmu, "nn_sd": nsd, "stage1": stage1.predict(Xs)[0],
             "p_cls": pcl, "p_cls_sd": pcl_sd, "sim_p": sim_p, "sim": sim,
             "sim_arrived": np.array([s[2] for s in sims]), "p_start": ps, "fixed": fixed}
    out["sweep"] = {"fixed": fixed, "p_start_bar": ps / 1e5,
                    "pigsim": [{"p_in_bar": p / 1e5, "t_arrive": s[0], "v_max": s[1],
                                "arrived": bool(s[2]), "started": bool(s[3])}
                               for p, s in zip(sim_p, sims)]}
    print("\nsweep pigsim:", [(round(p / 1e5, 2), round(s[0], 1), round(s[1], 2), s[2])
                            for p, s in zip(sim_p, sims)])
    Xsim = np.column_stack([sim_p, np.full_like(sim_p, fixed["mass"]),
                            np.full_like(sim_p, fixed["F_fric"]), np.full_like(sim_p, fixed["L"])])
    out["sweep"]["models_at_pigsim_points"] = {
        "poly3": poly.predict(Xsim)[0].tolist(),
        "GP_mu": gp.predict(Xsim)[0].tolist(), "GP_sd": gp.predict(Xsim)[1].tolist(),
        "MLP_mu": ens.predict(Xsim)[0].tolist(), "MLP_sd": ens.predict(Xsim)[1].tolist(),
        "classifier_p": clf.predict_proba(Xsim)[0].tolist(),
        "stage1_MLP": stage1.predict(Xsim)[0].tolist()}

    with open(os.path.join(HERE, "regime.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    os.makedirs(FIG, exist_ok=True)
    plot_sweep(sweep, os.path.join(FIG, "regime_sweep.png"))
    plot_map(df, clf, os.path.join(FIG, "regime_map.png"))
    print("-> regime.json, models/wide_ensemble.pt, models/arrival_classifier.pt, "
          "figures/regime_sweep.png, figures/regime_map.png")


if __name__ == "__main__":
    main()
