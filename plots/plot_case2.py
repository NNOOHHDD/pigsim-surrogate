"""Reproduce Figures 10-14 of Nieckele et al. (2000) for the area-change case."""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")


def split(s, y, n_up):
    """Insert a break at the pig so the discontinuity is not drawn as a line."""
    return (np.concatenate([s[:n_up], [np.nan], s[n_up:]]),
            np.concatenate([y[:n_up], [np.nan], y[n_up:]]))


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(os.path.join(RES, "case2.npz"))
    iso_path = os.path.join(RES, "case2_iso.npz")
    di = np.load(iso_path) if os.path.exists(iso_path) else None

    km = 1e-3
    n_up = int(d["n_up"])
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130,
                         "axes.grid": True, "grid.alpha": 0.3})

    # ---- Fig 10: pig velocity vs pig position -------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(d["h_s_pig"] * km, d["h_V_pig"], lw=1.2, label="non-isothermal")
    if di is not None:
        ax.plot(di["h_s_pig"] * km, di["h_V_pig"], "--", lw=1.0, label="isothermal")
    for x in (10, 20):
        ax.axvline(x, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"$x_{pig}$ (km)")
    ax.set_ylabel("pig velocity (m/s)")
    ax.set_title("Fig. 10 - pig velocity vs. pig position")
    ax.set_xlim(0, 30)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case2_fig10_velocity.png"))

    # ---- Fig 11: pig position vs time ---------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(d["h_t"], d["h_s_pig"] * km, lw=1.2, label="non-isothermal")
    if di is not None:
        ax.plot(di["h_t"], di["h_s_pig"] * km, "--", lw=1.0, label="isothermal")
    for y in (10, 20):
        ax.axhline(y, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$x_{pig}$ (km)")
    ax.set_title("Fig. 11 - pig position with time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case2_fig11_position.png"))

    # ---- Fig 12: axial pressure ---------------------------------------
    ts = d["snap_t"]
    want = [300, 500, 700, 1000, 1400]
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for tw in want:
        j = int(np.argmin(np.abs(ts - tw)))
        s, p = split(d["snap_s"][j], d["snap_p"][j], n_up)
        ax.plot(s * km, p / 1e5, lw=1.1, label=f"t = {ts[j]:.0f} s")
    for x in (10, 20):
        ax.axvline(x, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title("Fig. 12 - axial pressure distribution")
    ax.set_xlim(0, 30)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case2_fig12_pressure.png"))

    # ---- Fig 13: axial temperature ------------------------------------
    groups = [[0, 100, 200, 300], [400, 500, 600, 700], [800, 900, 1000, 1400]]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    for ax, grp in zip(axs, groups):
        for tw in grp:
            j = int(np.argmin(np.abs(ts - tw)))
            s, T = split(d["snap_s"][j], d["snap_T"][j] - 273.15, n_up)
            ax.plot(s * km, T, lw=1.1, label=f"t = {ts[j]:.0f} s")
        for x in (10, 20):
            ax.axvline(x, color="0.6", lw=0.8, ls=":")
        ax.set_xlabel("x (km)")
        ax.set_xlim(0, 30)
        ax.legend(fontsize=8)
    axs[0].set_ylabel("T (C)")
    fig.suptitle("Fig. 13 - temperature distribution along the pipeline", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case2_fig13_temperature.png"),
                bbox_inches="tight")

    # ---- Fig 14: temperature history at stations ----------------------
    st_t = d["st_t"]
    st_T = d["st_T"] - 273.15
    labels = ["5 km", "10 km", "15 km", "20 km", "25 km", "30 km"]
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.4), sharey=True)
    for k, ax in enumerate(axs):
        for j in (2 * k, 2 * k + 1):
            ax.plot(st_t, st_T[:, j], lw=1.0, label=f"x = {labels[j]}")
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
    axs[0].set_ylabel("T (C)")
    fig.suptitle("Fig. 14 - temperature variation with time at fixed stations",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case2_fig14_T_history.png"),
                bbox_inches="tight")

    print("figures written to", OUT)


if __name__ == "__main__":
    main()
