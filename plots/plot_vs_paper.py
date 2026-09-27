"""Overlay re-read paper-figure values on the computed curves."""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    from plots.paper_reference import P_F3, P_F7_NON, P_F10, P_T1, P_T2
except ModuleNotFoundError:  # direct execution: python plots/plot_vs_paper.py
    from paper_reference import P_F3, P_F7_NON, P_F10, P_T1, P_T2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "figures")

ATM = 101325.0

# Fig. 7 isothermal comparison points were not part of the re-reading pass.
P_F7_ISO = [(1000, 4.0), (1100, 6.0), (1200, 9.0), (1300, 15.0)]

STYLE = dict(marker="o", ms=5, mfc="none", mew=1.4, ls="none")


def main():
    d1 = np.load(os.path.join(RES, "case1.npz"))
    d1i = np.load(os.path.join(RES, "case1_iso.npz"))
    d2 = np.load(os.path.join(RES, "case2.npz"))
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                         "font.family": "Malgun Gothic", "axes.unicode_minus": False,
                         "mathtext.fontset": "dejavusans"})
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.2))

    # (a) Case 1 - Fig. 7
    ax = axs[0, 0]
    ax.plot(d1["h_s_pig"], d1["h_V_pig"], lw=1.3, color="#1b5e7e",
            label="본 구현 (비등온)")
    ax.plot(d1i["h_s_pig"], d1i["h_V_pig"], lw=1.0, ls="--", color="#7aa9bd",
            label="본 구현 (등온)")
    x, y = zip(*P_F7_NON)
    ax.plot(x, y, color="#9a3a2c", label="논문 Fig. 7 판독 (비등온)", **STYLE)
    x, y = zip(*P_F7_ISO)
    ax.plot(x, y, color="#c98a7d", marker="s", ms=4.5, mfc="none", mew=1.2,
            ls="none", label="논문 Fig. 7 판독 (등온)")
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 16)
    ax.set_xlabel(r"$x_{pig}$ (m)")
    ax.set_ylabel(r"$V_{pig}$ (m/s)")
    ax.set_title("(a) Case 1 · pig 속도 vs 위치  [논문 Fig. 7]", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper left")

    # (b) Case 1 - Fig. 3 inlet pressure
    ax = axs[0, 1]
    t_in = d1["st_t"]
    p_in = d1["st_p"][:, 0] / ATM
    ax.plot(t_in, p_in, lw=1.3, color="#1b5e7e", label="본 구현")
    x, y = zip(*P_F3)
    ax.plot(x, y, color="#9a3a2c", label="논문 Fig. 3 판독", **STYLE)
    ax.set_xlim(0, 1250)
    ax.set_ylim(0, 80)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("입구 압력 (atm)")
    ax.set_title("(b) Case 1 · 입구 압력 이력  [논문 Fig. 3]", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")

    # (c) Case 2 - Fig. 10
    ax = axs[1, 0]
    ax.plot(d2["h_s_pig"] / 1000, d2["h_V_pig"], lw=1.3, color="#1b5e7e",
            label="본 구현 (비등온)")
    x, y = zip(*P_F10)
    ax.plot(x, y, color="#9a3a2c", label="논문 Fig. 10 판독", **STYLE)
    for xv in (10, 20):
        ax.axvline(xv, color="0.6", lw=.8, ls=":")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 70)
    ax.set_xlabel(r"$x_{pig}$ (km)")
    ax.set_ylabel(r"$V_{pig}$ (m/s)")
    ax.set_title("(c) Case 2 · pig 속도 vs 위치  [논문 Fig. 10]", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")

    # (d) both cases - position vs time
    ax = axs[1, 1]
    ax.plot(d1["h_t"], d1["h_s_pig"] / 1400 * 100, lw=1.3, color="#1b5e7e",
            label="본 구현 · Case 1")
    ax.plot(d2["h_t"], d2["h_s_pig"] / 30000 * 100, lw=1.3, color="#2c6a4d",
            label="본 구현 · Case 2")
    x, y = zip(*P_T1)
    ax.plot(x, np.array(y) / 1400 * 100, color="#9a3a2c",
            label="논문 본문 · Case 1", **STYLE)
    x, y = zip(*P_T2)
    ax.plot(x, np.array(y) / 30.0 * 100, color="#87590a", marker="s", ms=5,
            mfc="none", mew=1.4, ls="none", label="논문 본문 · Case 2")
    ax.set_xlim(0, 1650)
    ax.set_ylim(0, 105)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("주행 거리 (전장 대비 %)")
    ax.set_title("(d) 두 케이스 · pig 궤적  [논문 Fig. 8 / Fig. 11]", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")

    fig.tight_layout()
    path = os.path.join(OUT, "vs_paper_overlay.png")
    fig.savefig(path, dpi=140)
    print(path)


if __name__ == "__main__":
    main()
