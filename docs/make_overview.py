"""Draw docs/overview.png, the figure at the top of README.md (~10 s, no pigsim runs).

    python docs/make_overview.py

Left:  three held-out pig-speed curves (stage 3) -- pigsim vs. three curve models.
Right: the start threshold (stage 2) -- pigsim, the wide-range MLP ensemble and
       the arrival classifier along p_in, other inputs fixed.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from surrogate import registry  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"MLP x5": "#4a3aa7", "PCA + GP": "#eb6834", "DeepONet x5": "#2a78d6"}


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def main():
    s3 = os.path.join(ROOT, "surrogate", "stage3")
    d = np.load(os.path.join(s3, "curves.npz"))
    z = np.load(os.path.join(s3, "test_predictions.npz"))
    te, xi = z["te"], d["xi"]
    X = d["X"][te]
    picks = [int(np.argmin(np.abs(X[:, 0] - p))) for p in (3.3e5, 5.5e5, 7.8e5)]

    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.25, 1], height_ratios=[2.2, 1], hspace=0.12, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    for k, i in enumerate(picks):
        ax.plot(xi, d["V"][te][i], color=INK, lw=2.6, label="pigsim (held-out run)" if k == 0 else None)
        for name in COL:
            ax.plot(xi, z[f"{name}|mean"][i], color=COL[name], lw=1.2, ls="--",
                    label=name if k == 0 else None)
        x = X[i]
        ax.annotate(f"{x[0]/1e5:.1f} bar, {x[3]/1e3:.1f} km", (xi[np.argmax(d["V"][te][i])],
                    d["V"][te][i].max()), xytext=(14, 6), textcoords="offset points", fontsize=8.5, color=MUTED)
    ax.set_xlabel("position along the pipe  ξ = (s − s₀)/(0.999 L − s₀)", color=INK)
    ax.set_ylabel("pig speed V [m/s]", color=INK)
    ax.set_title("Stage 3: whole speed curve from 4 inputs (test error ≈ 0.8 % rel. L2)",
                 loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    style(ax)

    r = json.load(open(os.path.join(ROOT, "surrogate", "regime.json")))["sweep"]
    fx = r["fixed"]
    p = np.linspace(1.6e5, 4.0e5, 300)
    Xs = np.column_stack([p, np.full_like(p, fx["mass"]), np.full_like(p, fx["F_fric"]),
                          np.full_like(p, fx["L"])])
    pr = registry.predict(Xs)
    ok = pr["p_arrive"] >= 0.5
    sim = r["pigsim"]
    sp = np.array([q["p_in_bar"] for q in sim])
    arr = np.array([q["arrived"] for q in sim])
    st = np.array([np.nan if q["t_arrive"] is None or not q["arrived"] else q["t_arrive"] for q in sim],
                  dtype=float)

    a1 = fig.add_subplot(gs[0, 1])
    a1.fill_between(p / 1e5, pr["t_arrive"] - 2 * pr["t_arrive_std"], pr["t_arrive"] + 2 * pr["t_arrive_std"],
                    where=ok, color=COL["MLP x5"], alpha=0.18, lw=0)
    a1.plot(np.where(ok, p, np.nan) / 1e5, pr["t_arrive"], color=COL["MLP x5"], lw=2,
            label="MLP ensemble ± 2 std\n(drawn where P(arrive) ≥ 0.5)")
    a1.scatter(sp[arr], st[arr], color=INK, s=28, zorder=5, label="pigsim")
    a1.axvline(r["p_start_bar"], color=MUTED, lw=1)
    a1.set_yscale("log")
    a1.set_ylabel("arrival time [s]", color=INK)
    a1.set_title("Stage 2: below the start threshold the pig never moves", loc="left",
                 fontsize=10.5, color=INK)
    a1.legend(frameon=False, fontsize=8, loc="upper right")
    a1.tick_params(labelbottom=False)
    style(a1)

    a2 = fig.add_subplot(gs[1, 1], sharex=a1)
    a2.plot(p / 1e5, pr["p_arrive"], color=COL["DeepONet x5"], lw=2, label="classifier P(arrive)")
    a2.scatter(sp, arr.astype(float), color=INK, s=22, zorder=5, label="pigsim (1 = arrived)")
    a2.axvline(r["p_start_bar"], color=MUTED, lw=1, label="analytic start threshold")
    a2.set_ylim(-0.1, 1.1)
    a2.set_xlabel(f"inlet pressure p_in [bar]   (mass {fx['mass']:.0f} kg, friction "
                  f"{fx['F_fric']/1e3:.1f} kN, L {fx['L']:.0f} m)", color=INK, fontsize=9)
    a2.legend(frameon=False, fontsize=7.5, loc="center right")
    style(a2)
    fig.savefig(os.path.join(ROOT, "docs", "overview.png"), dpi=150, bbox_inches="tight")
    print("-> docs/overview.png")


if __name__ == "__main__":
    main()
