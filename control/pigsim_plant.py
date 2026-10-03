"""Closed-loop pigsim: the controller sets the outlet valve between solver segments.

The only connection to the solver is the existing chi_of_t hook of
surrogate/pipeline.build(): it returns whatever opening the controller last
commanded.  The run is advanced one control period at a time with the normal
PigFlowSolver.run(), so the solver and its numerics are untouched; the valve
opening is a zero-order hold between control instants, exactly as in the
lumped model.

Measured signals: inlet pressure (first upstream cell), outlet pressure (last
downstream cell, just before the valve), pig position and speed, time.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate import pipeline as P  # noqa: E402

from . import scenario as S  # noqa: E402


class PigsimPlant:
    """Same interface as control.lumped.LumpedPig."""

    def __init__(self, L=S.L, p_in=S.P_IN, chi_max=S.CHI_MAX, dt_ctrl=S.DT_CTRL,
                 N=40):
        self.L, self.p_in, self.chi_max, self.dt_ctrl, self.N = L, p_in, chi_max, dt_ctrl, N
        self.s_arrive = 0.999 * L

    def reset(self, mass, F_fric, u0=S.U0):
        self.u = float(u0)
        self.sol = P.build(self.p_in, mass, F_fric, self.L, N=self.N,
                           chi_of_t=lambda t: self.chi_max * self.u)
        self._dt = 0.02
        self._u_hist = [(0.0, self.u)]
        self.step_stats = {"vmax_lim": 0.0, "excess": 0.0}
        return self.obs()

    @property
    def t(self):
        return self.sol.t

    @property
    def s(self):
        return self.sol.pig.s

    @property
    def V(self):
        return self.sol.pig.V

    @property
    def arrived(self):
        return self.sol.pig.s >= self.s_arrive

    def obs(self):
        sol = self.sol
        return {"t": sol.t, "p_in": float(sol.up.p[0]), "p_out": float(sol.dn.p[-1]),
                "s": sol.pig.s, "V": sol.pig.V, "u": self.u}

    def step(self, u):
        self.u = min(max(float(u), 0.0), 1.0)
        self._u_hist.append((self.sol.t, self.u))
        n0 = len(self.sol.hist["t"])
        self.sol.run(self.sol.t + self.dt_ctrl, dt0=self._dt, dt_max=1.0, cfl_pig=0.02,
                     s_stop=self.s_arrive, verbose=False)
        h = self.sol.hist
        dts = h["dt"][n0:]
        if dts:
            self._dt = float(np.clip(max(dts[-3:]), 0.02, 1.0))
        s = np.asarray(h["s_pig"][n0:])
        V = np.asarray(h["V_pig"][n0:])
        dt = np.asarray(dts)
        w = s >= S.S_LIMIT
        self.step_stats = {
            "vmax_lim": float(V[w].max()) if w.any() else 0.0,
            "excess": float(np.sum(np.maximum(V[w] - S.V_CAP, 0.0) * dt[w])),
        }
        return self.obs()

    def trace(self):
        h = self.sol.history()
        t = np.concatenate([[0.0], h["t"]])
        tu, uu = zip(*self._u_hist)
        idx = np.searchsorted(np.asarray(tu), t, side="left") - 1
        return {"t": list(t), "s": [P.S0] + list(h["s_pig"]), "V": [0.0] + list(h["V_pig"]),
                "u": list(np.asarray(uu)[np.clip(idx, 0, None)])}
