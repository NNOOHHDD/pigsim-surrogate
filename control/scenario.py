"""The control task: one fixed line, a controllable outlet valve, an unknown pig.

Everything not listed here is the stage-1 line of surrogate/pipeline.py
(D = 0.30 m, nitrogen, 1.5 bar at rest, inlet pressure ramped from 1.5 bar to
p_in over the first 10 s, outlet valve into a 1.5 bar reservoir).

    inlet (3 bar after 10 s) ──[ pig ]────────────── control valve ── 1.5 bar
    |<───────────────────────── L = 1500 m ───────────────────────────>|
                                        |<── speed limit applies ──>|
                                       s = 0.5 L

Valve.  Opening u in [0, 1] maps to the pipeline.build() hook as
chi = CHI_MAX u, i.e. (Cd A) = u x 0.006 m2 (three times the stage-1 valve).
The command may change by at most DU_MAX per control step (DT_CTRL).

Why the speed limit starts at half way.  The line is at 1.5 bar when the pig
is launched, so the pig first surges into the low-pressure gas (40-50 m/s
here) and only slows down once the gas ahead of it has been compressed.
During that surge the gas at the valve is still near reservoir pressure and
the valve passes almost nothing: in pigsim the launch peak is the same for
every opening (control/RESULTS_control.md, table 1).  An outlet valve
cannot limit it, so the limit is put on the approach to the receiver, where
the valve does have authority.

Unknown to the controller and drawn per episode:
    mass    pig mass                 20 - 200 kg
    F_fric  dynamic friction force   1.0 - 4.0 kN    (static = 1.2 x dynamic)
Out-of-range test ("higher friction"): F_fric 4.0 - 6.0 kN.
"""

import numpy as np

L = 1500.0                 # line length [m]
P_IN = 3.0e5               # final inlet pressure [Pa] (lower end of stage 1)
CHI_MAX = 3.0              # chi = CHI_MAX * u  ->  (Cd A)_max = 0.006 m2
DT_CTRL = 1.0              # control period [s]
DU_MAX = 0.10              # max change of opening per control period
U0 = 0.5                   # valve opening at t = 0 (PI and RL start here)

S_LIMIT = 0.5 * L          # the speed limit applies for s >= S_LIMIT
V_CAP = 8.0                # speed limit [m/s]
T_TARGET = 150.0           # arrival-time target [s]
T_MAX = 400.0              # episode ends here if the pig has not arrived [s]
S_ARRIVE = 0.999 * L       # arrival point (same rule as stage 1)

TRAIN = {"mass": (20.0, 200.0), "F_fric": (1.0e3, 4.0e3)}
HIGH_FRICTION = {"mass": (20.0, 200.0), "F_fric": (4.0e3, 6.0e3)}
MEAN_CONDITION = {"mass": 110.0, "F_fric": 2.5e3}


def sample_conditions(n, seed, ranges=TRAIN):
    """n (mass, F_fric) pairs, uniform in the given box (Latin hypercube)."""
    from scipy.stats import qmc
    unit = qmc.LatinHypercube(d=2, seed=seed).random(n)
    lo = np.array([ranges["mass"][0], ranges["F_fric"][0]])
    hi = np.array([ranges["mass"][1], ranges["F_fric"][1]])
    return qmc.scale(unit, lo, hi)
