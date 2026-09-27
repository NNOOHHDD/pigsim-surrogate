"""Print the quantitative comparison between this code and the paper's text."""

import os

import numpy as np

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")


def crossing(t, s, s_target):
    """First time the pig passes a station."""
    i = np.argmax(s >= s_target)
    if s[i] < s_target:
        return np.nan
    if i == 0:
        return float(t[0])
    return float(np.interp(s_target, s[i - 1:i + 1], t[i - 1:i + 1]))


def row(label, paper, mine, unit="s"):
    if np.isnan(mine):
        print(f"  {label:<44s} {paper:>14s} {'--':>12s}")
        return
    err = "" if not isinstance(paper, (int, float)) else \
        f"  ({100 * (mine - paper) / paper:+.1f} %)"
    ps = paper if isinstance(paper, str) else f"{paper:.0f} {unit}"
    print(f"  {label:<44s} {ps:>14s} {mine:>9.1f} {unit}{err}")


def main():
    print("\n" + "=" * 84)
    print("CASE 1 - dewatering of a riser        "
          "                paper (text)     this code")
    print("=" * 84)
    d = np.load(os.path.join(RES, "case1.npz"))
    di = np.load(os.path.join(RES, "case1_iso.npz"))
    t, s = d["h_t"], d["h_s_pig"]
    row("pig passes point 2 (s = 350 m)", 270.0, crossing(t, s, 350.0))
    row("pig passes point 3 (s = 650 m)", 820.0, crossing(t, s, 650.0))
    row("pig passes point 5 (s = 850 m)", 1080.0, crossing(t, s, 850.0))
    print(f"  {'peak pig velocity in the ascending leg':<44s} "
          f"{'tens of m/s':>14s} {np.max(d['h_V_pig']):>9.1f} m/s")
    print(f"  {'inlet pressure while pig is on the bottom':<44s} "
          f"{'~constant':>14s} "
          f"{d['snap_p'][:, 0][np.argmin(np.abs(d['snap_t'] - 800))] / 1e5:>9.1f} bar"
          f" -> "
          f"{d['snap_p'][:, 0][np.argmin(np.abs(d['snap_t'] - 1100))] / 1e5:.1f} bar")
    print(f"  {'exit time, non-isothermal vs isothermal':<44s} "
          f"{'iso arrives 1st':>14s} {t[-1]:>9.1f} s  vs {di['h_t'][-1]:.1f} s")

    print("\n" + "=" * 84)
    print("CASE 2 - horizontal line, severe area change"
          "          paper (text)     this code")
    print("=" * 84)
    d = np.load(os.path.join(RES, "case2.npz"))
    di = np.load(os.path.join(RES, "case2_iso.npz"))
    t, s, V = d["h_t"], d["h_s_pig"], d["h_V_pig"]
    row("pig reaches the large section (10 km)", 300.0, crossing(t, s, 10e3))
    row("pig at the centre of that section (15 km)", 500.0, crossing(t, s, 15e3))
    row("pig reaches the last section (20 km)", 600.0, crossing(t, s, 20e3))
    stopped = d["h_stopped"].astype(bool) & (s > 19.9e3) & (s < 20.1e3)
    if stopped.any():
        print(f"  {'pig held by static friction at 20 km':<44s} "
              f"{'yes, ~600-760 s':>14s} "
              f"{t[stopped][0]:>9.1f} -> {t[stopped][-1]:.1f} s")
    row("pig leaves the pipeline (30 km)", 1500.0, crossing(t, s, 29.9e3))
    print(f"  {'peak pig velocity':<44s} {'~100 m/s (Fig 10)':>14s} "
          f"{np.max(V):>9.1f} m/s")
    print(f"  {'exit time, non-isothermal vs isothermal':<44s} "
          f"{'nearly equal':>14s} {t[-1]:>9.1f} s  vs {di['h_t'][-1]:.1f} s")
    print("=" * 84 + "\n")


if __name__ == "__main__":
    main()
