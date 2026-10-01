"""A short, generic gas pipeline used to generate surrogate-model training data.

This is a made-up example, not a model of any real installation.  It reuses the
pigsim solver unchanged; only the geometry, fluid and operating point are new.

    inlet (pressure ramp) ──[ pig ]──────────────── outlet valve ── reservoir
    |<──────────────────────── L ─────────────────────────────>|

Fixed:
    horizontal steel pipe, D = 0.30 m, wall 6.35 mm, roughness 0.046 mm
    nitrogen at rest, 1.5 bar abs and 20 C
    inlet pressure ramped linearly from 1.5 bar to p_in over the first 10 s
    outlet valve (Cd A)_0 = 0.002 m2 (~3 % of the bore), fully open, into a
    1.5 bar reservoir -- it throttles the cruise speed to a realistic 5-15 m/s
    sealing pig (no by-pass), static friction = 1.2 x dynamic friction

Varied (the four surrogate inputs):
    p_in   inlet pressure            [Pa]
    mass   pig mass                  [kg]
    F_fric dynamic friction force    [N]
    L      pipeline length           [m]

Outputs:
    t_arrive  time for the pig to reach the outlet (s = 0.999 L)   [s]
    v_max     peak pig speed                                        [m/s]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pigsim import PipeRoute, Pig, PigFlowSolver, nitrogen  # noqa: E402

D = 0.30
WALL = 6.35e-3
ROUGH = 4.6e-5
T_AMB = 20.0 + 273.15
P0 = 1.5e5
P_RES = 1.5e5
T_RAMP = 10.0
CDA_VALVE = 0.002
STATIC_OVER_DYNAMIC = 1.2
S0 = 2.0                  # initial pig position (m); s = 0 would give an empty upstream domain

# Input ranges for the design of experiments: name -> (low, high, unit)
RANGES = {
    "p_in":   (3.0e5, 8.0e5, "Pa"),
    "mass":   (20.0, 200.0, "kg"),
    "F_fric": (1.0e3, 6.0e3, "N"),
    "L":      (500.0, 3000.0, "m"),
}
INPUTS = list(RANGES)
OUTPUTS = ["t_arrive", "v_max"]


def build(p_in, mass, F_fric, L, N=40, isothermal=False,
          p_in_of_t=None, chi_of_t=None, F_fric_of_s=None):
    """Set up one pigsim run.

    The three optional hooks change operating conditions only -- the solver
    and its numerics are untouched.  With all three left at None the run is
    identical to stage 1 (tests/test_pipeline_hooks.py checks this).

        p_in_of_t(t, p_nominal) -> inlet pressure [Pa]   e.g. a supply pressure drop
        chi_of_t(t)             -> outlet valve opening   e.g. a partly closed valve
        F_fric_of_s(s)          -> dynamic friction [N]   e.g. more friction after s_a
                                   (static friction stays 1.2 x dynamic)
    """
    const = lambda v: (lambda s: np.full(np.shape(s), v))  # noqa: E731
    route = PipeRoute(xy=[(0.0, 0.0), (L, 0.0)], D=D, wall=WALL, rough=ROUGH,
                      E=2.0e11, nu=0.3, T_amb=const(T_AMB), U_gas=const(10.0))
    if F_fric_of_s is None:
        pig = Pig(mass=mass, F_stat=STATIC_OVER_DYNAMIC * F_fric, F_dyn=F_fric)
    else:
        pig = Pig(mass=mass, F_stat=lambda s: STATIC_OVER_DYNAMIC * F_fric_of_s(s),
                  F_dyn=F_fric_of_s)

    def inlet(t):
        p = P0 + (p_in - P0) * min(t / T_RAMP, 1.0)
        if p_in_of_t is not None:
            p = p_in_of_t(t, p)
        return {"type": "p", "p": p, "T": T_AMB}

    def outlet(t):
        chi = 1.0 if chi_of_t is None else chi_of_t(t)
        return {"type": "valve", "CdA": CDA_VALVE, "chi": chi, "p_res": P_RES}

    sol = PigFlowSolver(route, pig, nitrogen(), N_up=N, N_down=N, cluster=1.5,
                        bc_inlet=inlet, bc_outlet=outlet,
                        T_iso=T_AMB if isothermal else None)
    sol.initialise(s_pig=S0, p_of_s=const(P0), T_up_of_s=const(T_AMB))
    return sol


def simulate(p_in, mass, F_fric, L, N=40, isothermal=False, t_end=2000.0,
             dt_max=1.0):
    """Run one pig transit and return the two outputs plus bookkeeping.

    N = 40 cells per side: N = 20 / 40 / 80 give the same t_arrive and v_max to
    within 0.05 %.  The full energy equation is kept (isothermal=False): the
    isothermal shortcut shifts v_max by ~5 %, which is not negligible here.
    """
    t0 = time.perf_counter()
    sol = build(p_in, mass, F_fric, L, N=N, isothermal=isothermal)
    s_stop = 0.999 * L
    sol.run(t_end, dt0=0.02, dt_max=dt_max, cfl_pig=0.02, s_stop=s_stop,
            verbose=False)
    h = sol.history()
    arrived = bool(h["s_pig"][-1] >= s_stop)
    if arrived:
        # interpolate the crossing time inside the last step
        t_arr = float(np.interp(s_stop, h["s_pig"][-2:], h["t"][-2:]))
    else:
        t_arr = float("nan")
    return {
        "t_arrive": t_arr,
        "v_max": float(np.max(h["V_pig"])),
        "arrived": arrived,
        "n_steps": len(h["t"]),
        "wall_s": time.perf_counter() - t0,
    }


if __name__ == "__main__":
    mid = {k: 0.5 * (lo + hi) for k, (lo, hi, _) in RANGES.items()}
    print("mid-range case:", mid)
    print(simulate(**mid))
