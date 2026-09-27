"""Reproduce Figures 3-8 of Nieckele et al. (2000) for the riser-dewatering case."""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")

S_POINTS = dict(inlet=0.0, p1=50.0, p2=350.0, p3=650.0, p4=750.0,
                p5=850.0, p6=1150.0, p7=1450.0, outlet=1500.0)
STATION_LABELS = ["inlet", "point 2", "point 3", "point 5", "point 7", "outlet"]


def split(s, y, n_up):
    return (np.concatenate([s[:n_up], [np.nan], s[n_up:]]),
            np.concatenate([y[:n_up], [np.nan], y[n_up:]]))


def mark(ax):
    for k in ("p1", "p2", "p3", "p4", "p5", "p6", "p7"):
        ax.axvline(S_POINTS[k], color="0.75", lw=0.7, ls=":")


def main():
    os.makedirs(OUT, exist_ok=True)
    d = np.load(os.path.join(RES, "case1.npz"))
    ip = os.path.join(RES, "case1_iso.npz")
    di = np.load(ip) if os.path.exists(ip) else None
    n_up = int(d["n_up"])
    ts = d["snap_t"]
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130,
                         "axes.grid": True, "grid.alpha": 0.3})

    # ---- Fig 2: riser geometry ----------------------------------------
    xy = np.array([(0, 0), (50, 0), (50, -300), (50, -600), (150, -600),
                   (250, -600), (250, -300), (250, 0), (300, 0)], dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.plot(xy[:, 0], xy[:, 1], "-o", ms=3, lw=1.4)
    for i, (x, y) in enumerate(xy):
        lbl = ["Inlet", "1", "2", "3", "4", "5", "6", "7", "Outlet"][i]
        ax.annotate(lbl, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=7)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Fig. 2 - riser geometry")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case1_fig2_geometry.png"))

    # ---- Fig 3: axial pressure ----------------------------------------
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for tw in [0, 200, 400, 600, 800, 1000, 1100, 1150, 1200]:
        j = int(np.argmin(np.abs(ts - tw)))
        s, p = split(d["snap_s"][j], d["snap_p"][j], n_up)
        ax.plot(s, p / 1e5, lw=1.1, label=f"t = {ts[j]:.0f} s")
    mark(ax)
    ax.set_xlabel("s (m)")
    ax.set_ylabel("Pressure (bar)")
    ax.set_title("Fig. 3 - pressure distribution")
    ax.set_xlim(0, 1500)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case1_fig3_pressure.png"))

    # ---- Figs 4 & 5: gas / liquid mass flow at fixed stations ---------
    st_t = d["st_t"]
    mdot = d["st_mdot"]
    is_up = d["st_is_up"]
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for j, lbl in enumerate(STATION_LABELS):
        g = np.where(is_up[:, j], mdot[:, j], np.nan)     # gas side of the pig
        axs[0].plot(st_t, g, lw=1.0, color=colors[j % len(colors)], label=lbl)
    # the liquid curves coincide (water is nearly incompressible) and differ
    # only in when they drop out, so draw the longest-lived one first
    for j in reversed(range(len(STATION_LABELS))):
        w = np.where(is_up[:, j], np.nan, mdot[:, j])     # liquid side
        axs[1].plot(st_t, w, lw=1.0 + 0.5 * (len(STATION_LABELS) - 1 - j),
                    color=colors[j % len(colors)], label=STATION_LABELS[j])
    axs[1].legend(*[list(reversed(x)) for x in axs[1].get_legend_handles_labels()])
    axs[0].set_title("Fig. 4 - gas mass flow rate")
    axs[1].set_title("Fig. 5 - liquid mass flow rate")
    for ax in axs:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("mass flow rate (kg/s)")
    axs[0].legend(fontsize=7)
    axs[1].set_ylim(-20, 400)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case1_fig4_5_massflow.png"))

    # ---- Fig 6: axial temperature -------------------------------------
    groups = [[0, 50, 200, 400, 600], [800, 1000, 1100, 1150]]
    fig, axs = plt.subplots(1, 2, figsize=(10.0, 3.6), sharey=True)
    for ax, grp in zip(axs, groups):
        for tw in grp:
            j = int(np.argmin(np.abs(ts - tw)))
            s, T = split(d["snap_s"][j], d["snap_T"][j] - 273.15, n_up)
            ax.plot(s, T, lw=1.1, label=f"t = {ts[j]:.0f} s")
        mark(ax)
        ax.set_xlabel("s (m)")
        ax.set_xlim(0, 1500)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("T (C)")
    fig.suptitle("Fig. 6 - axial temperature distribution", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case1_fig6_temperature.png"),
                bbox_inches="tight")

    # ---- Fig 7: pig velocity vs position ------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(d["h_s_pig"], d["h_V_pig"], lw=1.2, label="non-isothermal")
    if di is not None:
        ax.plot(di["h_s_pig"], di["h_V_pig"], "--", lw=1.0, label="isothermal")
    mark(ax)
    ax.set_xlabel(r"$x_{pig}$ (m)")
    ax.set_ylabel("pig velocity (m/s)")
    ax.set_title("Fig. 7 - pig velocity vs. pig position")
    ax.set_xlim(0, 1500)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case1_fig7_velocity.png"))

    # log scale companion, the low-speed part is otherwise invisible
    ax.set_yscale("log")
    ax.set_title("Fig. 7 - pig velocity vs. pig position (log scale)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case1_fig7_velocity_log.png"))

    # ---- Fig 8: pig position vs time ----------------------------------
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(d["h_t"], d["h_s_pig"], lw=1.2, label="non-isothermal")
    if di is not None:
        ax.plot(di["h_t"], di["h_s_pig"], "--", lw=1.0, label="isothermal")
    for k in ("p3", "p5"):
        ax.axhline(S_POINTS[k], color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$x_{pig}$ (m)")
    ax.set_title("Fig. 8 - pig position with time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "case1_fig8_position.png"))

    print("figures written to", OUT)


if __name__ == "__main__":
    main()

