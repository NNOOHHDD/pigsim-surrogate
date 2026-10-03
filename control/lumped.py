"""A lumped-parameter pig model, ~400x faster than pigsim, for training controllers.

Two gas volumes (upstream / downstream of the pig) with a polytropic pressure
law, a pig force balance with stick/slip friction, the gas columns' wall
friction and inertia, and the same valve law as pigsim:

    upstream   p1 = P0 (rho1/rho0)^n,  rho1 = m1 / (A s)
               inflow u_in:  rho1 l_in du_in/dt = (p_s - p1) - (f s/2D + K_in) rho1 u_in|u_in|/2
               dm1/dt = rho1 A u_in
    downstream p2 = P0 (rho2/rho0)^n,  rho2 = m2 / (A (L - s))
               outflow (quasi-steady):  p2 - p_res = rho2 u_v^2 [f (L-s)/4D + (A / chi CdA)^2]
               dm2/dt = -rho2 A u_v
    pig        (m + m_gas) dV/dt = (p1 - p2) A - fr1 - fr2 - F_c
               fr_i = f l_i/2D rho_i V|V|/2 A        (half of each column, moving at V)
               m_gas = c_g A (rho1 s + rho2 (L - s)) / 2

The valve law is pigsim's (paper eq. 8): mdot = rho chi CdA sqrt(dp / rho).
Calibrated constants (n, c_g, friction multiplier, K_in) are fitted to a few
pigsim runs by control/calibrate_lumped.py and stored in lumped_params.json.
What this model leaves out: pressure waves (the gas columns are lumped),
temperature as a separate state, wall elasticity.

Integration: semi-implicit Euler (velocities first, friction implicit) with a
sub-step chosen from the fastest local time scale.
"""

import json
import math
import os

from . import scenario as S

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_FILE = os.path.join(HERE, "lumped_params.json")

# fixed line data, the same numbers as surrogate/pipeline.py
D = 0.30
A = 0.25 * math.pi * D ** 2
ROUGH = 4.6e-5
R_N2 = 8.314462618 / 0.0280134
T_GAS = 293.15
MU = 1.76e-5
P0 = 1.5e5
P_RES = 1.5e5
T_RAMP = 10.0
CDA_VALVE = 0.002
STATIC_OVER_DYNAMIC = 1.2
S0 = 2.0
RHO0 = P0 / (R_N2 * T_GAS)

DEFAULT_PARAMS = {"n": 1.2, "c_g": 0.5, "f_mult": 1.0, "K_in": 1.0, "l_in": 2.0}


def load_params():
    if os.path.exists(PARAMS_FILE):
        with open(PARAMS_FILE) as fh:
            return {**DEFAULT_PARAMS, **json.load(fh)["params"]}
    return dict(DEFAULT_PARAMS)


def _fric_factor(u, rho):
    """Haaland's explicit form of Colebrook (turbulent), floored at low Re."""
    re = max(rho * abs(u) * D / MU, 4000.0)
    return (-1.8 * math.log10((ROUGH / D / 3.7) ** 1.11 + 6.9 / re)) ** -2


class LumpedPig:
    """Same interface as control.pigsim_plant.PigsimPlant.

    reset(mass, F_fric) -> obs;  step(u) -> obs  (valve opening u held for
    DT_CTRL);  obs is a dict of the measurable signals.  trace() returns the
    fine-grained history (t, s, V, u) used for the metrics.
    """

    def __init__(self, params=None, L=S.L, p_in=S.P_IN, chi_max=S.CHI_MAX,
                 dt_ctrl=S.DT_CTRL, s_arrive=None, record=True):
        self.p = dict(params or load_params())
        self.record = record
        self.v_cap = S.V_CAP        # only the PPO "margin" training variant changes this
        self.L, self.p_in, self.chi_max, self.dt_ctrl = L, p_in, chi_max, dt_ctrl
        self.s_arrive = s_arrive if s_arrive is not None else 0.999 * L

    # ------------------------------------------------------------------
    def reset(self, mass, F_fric, u0=S.U0):
        self.mass, self.F = float(mass), float(F_fric)
        self.t, self.s, self.V, self.u_in = 0.0, S0, 0.0, 0.0
        self.m1 = RHO0 * A * self.s
        self.m2 = RHO0 * A * (self.L - self.s)
        self.stuck = True
        self.u = float(u0)
        self._tr = {"t": [0.0], "s": [self.s], "V": [0.0], "u": [self.u]}
        self._p_out = P0
        self.step_stats = {"vmax_lim": 0.0, "excess": 0.0}
        return self.obs()

    def p_supply(self, t):
        return P0 + (self.p_in - P0) * min(t / T_RAMP, 1.0)

    def _pressure(self, m, vol):
        return P0 * (m / vol / RHO0) ** self.p["n"]

    @property
    def arrived(self):
        return self.s >= self.s_arrive

    def obs(self):
        return {"t": self.t, "p_in": self.p_supply(self.t), "p_out": self._p_out,
                "s": self.s, "V": self.V, "u": self.u}

    def trace(self):
        return {k: list(v) for k, v in self._tr.items()}

    # ------------------------------------------------------------------
    def step(self, u):
        """Advance one control period with the valve opening held at u."""
        self.u = min(max(float(u), 0.0), 1.0)
        t_end = self.t + self.dt_ctrl
        self.step_stats = {"vmax_lim": 0.0, "excess": 0.0}
        while self.t < t_end - 1e-12 and not self.arrived:
            self._substep(t_end)
        return self.obs()

    def _substep(self, t_end):
        P = self.p
        L, s, V = self.L, self.s, self.V
        l1, l2 = s, L - s
        vol1, vol2 = A * l1, A * l2
        rho1, rho2 = self.m1 / vol1, self.m2 / vol2
        p1, p2 = self._pressure(self.m1, vol1), self._pressure(self.m2, vol2)
        n = P["n"]
        m_eff = self.mass + P["c_g"] * A * (rho1 * l1 + rho2 * l2) / 2.0

        # time step from the fastest local time scale
        c1sq = n * p1 / rho1
        l_in = P["l_in"] + 0.5 * l1
        w_in = math.sqrt(c1sq / (l1 * l_in))
        w_pig = math.sqrt(n * A * (p1 / l1 + p2 / l2) / m_eff)
        dt = min(0.05, 0.15 / max(w_in, w_pig), t_end - self.t)

        f1 = P["f_mult"] * _fric_factor(V, rho1)
        f2 = P["f_mult"] * _fric_factor(V, rho2)

        # inflow velocity (implicit friction)
        p_s = self.p_supply(self.t + dt)
        k_in = (P["f_mult"] * _fric_factor(self.u_in, rho1) * 0.5 * l1 / D + P["K_in"]) * 0.5
        self.u_in = (self.u_in + dt * (p_s - p1) / (rho1 * l_in)) / \
            (1.0 + dt * k_in * abs(self.u_in) / l_in)

        # outflow (quasi-steady, valve + half the downstream column)
        chi_cda = self.chi_max * self.u * CDA_VALVE
        k_out = f2 * l2 / (4.0 * D)
        if chi_cda > 0.0 and p2 > P_RES:
            u_v = math.sqrt((p2 - P_RES) / (rho2 * (k_out + (A / chi_cda) ** 2)))
        else:
            u_v = 0.0
        self._p_out = p2 - rho2 * u_v * u_v * k_out

        # pig (stick / slip, implicit column friction)
        drive = (p1 - p2) * A
        F_stat = STATIC_OVER_DYNAMIC * self.F
        if self.stuck:
            if drive > F_stat:
                self.stuck = False
        if not self.stuck:
            kf = (f1 * l1 * rho1 + f2 * l2 * rho2) / (4.0 * D) * A
            V_new = (V + dt * (drive - self.F) / m_eff) / (1.0 + dt * kf * abs(V) / m_eff)
            if V_new <= 0.0:
                V_new = 0.0
                if drive <= F_stat:
                    self.stuck = True
        else:
            V_new = 0.0

        s_new = min(s + 0.5 * (V + V_new) * dt, self.s_arrive)
        self.m1 += rho1 * A * self.u_in * dt
        self.m2 -= rho2 * A * u_v * dt
        self.m1 = max(self.m1, 1e-6)
        self.m2 = max(self.m2, 1e-6)
        self.V, self.s = V_new, s_new
        self.t += dt
        if self.s >= S.S_LIMIT:
            st = self.step_stats
            st["vmax_lim"] = max(st["vmax_lim"], V_new)
            st["excess"] += max(V_new - self.v_cap, 0.0) * dt
        if not self.record:
            return
        self._tr["t"].append(self.t)
        self._tr["s"].append(self.s)
        self._tr["V"].append(self.V)
        self._tr["u"].append(self.u)
