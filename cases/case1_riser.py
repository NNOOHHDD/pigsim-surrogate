"""Paper case 1: dewatering of a sub-sea riser.

Nieckele, Braga & Azevedo (2000), IPC2000-175, section "Dewatering of a riser"
(pp. 4-6) -- Figures 2 to 8.

The riser is initially full of water; nitrogen is injected at the inlet and a
*sealing* pig separates the two fluids, so the upstream domain is gas and the
downstream domain is liquid.

Geometry (Table 1), total length 1500 m, D = 10 in, e = 6 mm,
eps/D = 1.8e-4, E = 2.1e5 MPa, nu = 0.3.
Ambient  25 C at sea level falling to 4 C at the 600 m deep bottom.
         U_G = 1 / 10 W/(m2 K) above water and 10 / 100 W/(m2 K) below,
         for gas / liquid respectively.
Inlet    gas mass flow ramped 0 -> 3 kg/s in 10 s, injected at 25 C.
Outlet   constant pressure, 10 atm.
Pig      3 kg, sealing, Fstat = Fdyn = 4982 N in both directions.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pigsim import PipeRoute, Pig, PigFlowSolver, nitrogen, water  # noqa: E402
from pigsim.pig import G  # noqa: E402

# Table 1 -- (x, y) of the way-points; s follows automatically
WAYPOINTS = [(0, 0), (50, 0), (50, -300), (50, -600),
             (150, -600), (250, -600), (250, -300), (250, 0), (300, 0)]
S_POINTS = dict(inlet=0.0, p1=50.0, p2=350.0, p3=650.0, p4=750.0,
                p5=850.0, p6=1150.0, p7=1450.0, outlet=1500.0)

D = 0.254                 # 10 in
WALL = 6.0e-3
EPS = 1.8e-4 * D
T_SURF = 25.0 + 273.15
T_BOT = 4.0 + 273.15
Y_BOT = -600.0
P_OUT = 10.0 * 101325.0
MDOT = 3.0
RAMP = 10.0
RHO_W = 1.0e3


def build(isothermal=False, N=200, scheme_h="upwind", s0=10.0):
    route = PipeRoute(
        xy=WAYPOINTS, D=D, wall=WALL, rough=EPS, E=2.1e11, nu=0.3,
        T_amb=lambda s: np.full(np.shape(s), T_SURF),   # replaced below
        U_gas=lambda s: np.where((s < S_POINTS["p1"]) | (s > S_POINTS["p7"]),
                                 1.0, 10.0),
        U_liq=lambda s: np.where((s < S_POINTS["p1"]) | (s > S_POINTS["p7"]),
                                 10.0, 100.0),
    )
    # ambient temperature drops linearly with depth (25 C -> 4 C at -600 m)
    route.T_amb = lambda s: T_SURF + (T_BOT - T_SURF) * np.clip(
        -route.elevation(s) / (-Y_BOT), 0.0, 1.0)

    pig = Pig(mass=3.0, F_stat=4982.0, F_dyn=4982.0, n_holes=0)

    def inlet(t):
        return {"type": "mdot", "mdot": MDOT * min(t / RAMP, 1.0), "T": T_SURF}

    def outlet(t):
        return {"type": "p", "p": P_OUT, "T": T_SURF}

    sol = PigFlowSolver(route, pig, fluid_up=nitrogen(), fluid_down=water(),
                        N_up=N, N_down=N, cluster=1.5,
                        bc_inlet=inlet, bc_outlet=outlet, scheme_h=scheme_h,
                        T_iso=T_SURF if isothermal else None)

    # initial state: fluid at rest, hydrostatic pressure, T = T_ambient
    def p0(s):
        return P_OUT + RHO_W * G * (0.0 - route.elevation(s))

    sol.initialise(s_pig=s0, p_of_s=p0,
                   T_up_of_s=lambda s: np.full(np.shape(s), T_SURF),
                   T_dn_of_s=route.T_amb)
    sol.set_stations([S_POINTS[k] for k in
                      ("inlet", "p2", "p3", "p5", "p7")] + [1499.0])
    return sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-end", type=float, default=1400.0)
    ap.add_argument("--dt-max", type=float, default=2.0)
    ap.add_argument("--N", type=int, default=200,
                    help="cells in each flow domain")
    ap.add_argument("--s0", type=float, default=10.0,
                    help="initial pig position in m")
    ap.add_argument("--isothermal", action="store_true")
    ap.add_argument("--scheme-h", default="upwind",
                    choices=["upwind", "central"])
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    args = ap.parse_args()

    sol = build(isothermal=args.isothermal, N=args.N,
                scheme_h=args.scheme_h, s0=args.s0)
    snaps = [50, 200, 400, 600, 800, 1000, 1100, 1150, 1200]
    sol.run(args.t_end, dt0=1e-3, dt_max=args.dt_max, cfl_pig=0.01,
            dV_max=1.0, snapshot_times=[t for t in snaps if t <= args.t_end],
            s_stop=1480.0)

    os.makedirs(args.out, exist_ok=True)
    tag = "case1_iso" if args.isothermal else "case1"
    if args.scheme_h != "upwind":
        tag += "_h" + args.scheme_h
    h = sol.history()
    np.savez(os.path.join(args.out, tag + ".npz"),
             **{f"h_{k}": v for k, v in h.items()},
             **{f"st_{k}": v for k, v in sol.station_arrays().items()},
             stations=sol.stations, n_up=sol.up.N,
             snap_t=np.array([s["t"] for s in sol.snapshots], dtype=float),
             snap_s=np.array([s["s"] for s in sol.snapshots]),
             snap_p=np.array([s["p"] for s in sol.snapshots]),
             snap_T=np.array([s["T"] for s in sol.snapshots]),
             snap_V=np.array([s["V"] for s in sol.snapshots]),
             snap_mdot=np.array([s["mdot"] for s in sol.snapshots]),
             snap_spig=np.array([s["s_pig"] for s in sol.snapshots]))
    print(f"\nfinished: t = {sol.t:.1f} s, pig at {sol.pig.s:.1f} m, "
          f"Vp = {sol.pig.V:.3f} m/s")
    return sol


if __name__ == "__main__":
    main()
