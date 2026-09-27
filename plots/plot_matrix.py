"""Show the actual sparsity of the assembled linear system.

The unknowns are interleaved as [p_0, h_0, V_0, p_1, h_1, V_1, ...], which is
what makes the system narrowly banded -- the paper's "hepta-diagonal" matrix.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pigsim.domain as dm  # noqa: E402
from pigsim import PipeRoute, Domain, nitrogen  # noqa: E402

CAPTURED = {}


def main():
    N = 12                      # small enough to see the structure
    route = PipeRoute(xy=[(0, 0), (1000, 0)], D=0.85, wall=2.54e-3, rough=1.8e-6,
                      E=2e11, nu=0.3,
                      T_amb=lambda s: np.full(np.shape(s), 291.15),
                      U_gas=lambda s: np.full(np.shape(s), 10.0))
    fl = nitrogen()
    dom = Domain(fl, N, route, "up", cluster=1.0)
    dom.set_mesh(1000.0, 1.0)
    dom.init_uniform(lambda s: np.full(np.shape(s), 5e5),
                     lambda s: np.full(np.shape(s), 300.0))
    dom.V[:] = 2.0

    real_solve = dm.solve_banded

    def spy(lu, ab, b, **kw):
        CAPTURED["ab"] = np.array(ab, copy=True)
        return real_solve(lu, ab, b, **kw)

    dm.solve_banded = spy
    dom.solve_step(0.1, dom.p.copy(), dom.h.copy(), dom.V.copy(),
                   {"type": "p", "p": 7e5, "T": 320.0},
                   {"type": "V", "V": 1.0}, n_iter=1)
    dm.solve_banded = real_solve

    ab = CAPTURED["ab"]
    n = ab.shape[1]
    dense = np.zeros((n, n))
    for d in range(ab.shape[0]):
        off = dm.KU - d                     # row - col
        for j in range(n):
            i = j + off
            if 0 <= i < n:
                dense[i, j] = ab[d, j]

    plt.rcParams.update({"font.size": 8.5, "font.family": "Malgun Gothic",
                         "axes.unicode_minus": False})
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6),
                            gridspec_kw={"width_ratios": [1, 1.05]})

    ax = axs[0]
    nz = np.abs(dense) > 0
    ax.imshow(nz, cmap="Blues", interpolation="nearest", vmin=0, vmax=1.4)
    for k in range(0, n, 3):
        ax.axhline(k - .5, color="0.85", lw=.4)
        ax.axvline(k - .5, color="0.85", lw=.4)
    ax.set_title(f"조립된 계수행렬 ({n}×{n}, N = {12}셀)\n"
                 f"대역폭 {dm.KL} — 논문의 hepta-diagonal 구조", fontsize=9)
    ax.set_xlabel("미지수  [p0 h0 V0 | p1 h1 V1 | ...]")
    ax.set_ylabel("방정식  [연속 . 에너지 . 운동량] x 셀")

    ax = axs[1]
    band = np.abs(ab) > 0
    ax.imshow(band, aspect="auto", cmap="Blues", interpolation="nearest",
              vmin=0, vmax=1.4)
    ax.set_yticks(range(ab.shape[0]))
    ax.set_yticklabels([("+" if dm.KU - d >= 0 else "-") + str(abs(dm.KU - d))
                        for d in range(ab.shape[0])])
    ax.set_title("실제 저장 형태 — LAPACK 밴드\n"
                 f"({dm.KL + dm.KU + 1} x n 만 저장, n x n 아님)", fontsize=9)
    ax.set_xlabel("열 (미지수)")
    ax.set_ylabel("대각선 오프셋 (행-열)")

    fig.tight_layout()
    path = os.path.join(ROOT, "figures", "matrix_structure.png")
    fig.savefig(path, dpi=150)
    print(path)


if __name__ == "__main__":
    main()
