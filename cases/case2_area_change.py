"""Paper case 2: horizontal gas pipeline with severe area changes.

Nieckele, Braga & Azevedo (2000), IPC2000-175, section "Horizontal pipeline
with severe area changes" (p. 6) -- Figures 9 to 14.

Geometry   3 x 10 km, D = 85 / 86.5 / 85 cm, e = 2.54 / 2.5 / 2.54 mm,
           roughness 1.8e-3 mm, E = 2e5 MPa, nu = 0.3.
Fluid      nitrogen, initially 1.5 bar and 18 C, at rest.
Inlet      pressure ramped 1.5 -> 7 bar in 20 s; hot N2 injected at 50 C.
Outlet     valve to a 1.5 bar reservoir, (Cd A)_0 = 0.15 m2, chi = 30 %.
Pig        100 kg, 8 x 17.5 mm by-pass holes, K = 1.5.
           Fdyn / Fstat = 1.471e5 / 1.724e5 N in the end sections,
                          3.971e4 / 4.766e4 N in the central section.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pigsim import PipeRoute, Pig, PigFlowSolver, nitrogen  # noqa: E402

L_SEC = 10_000.0
L = 3 * L_SEC
D1, D2 = 0.85, 0.865
E1, E2 = 2.54e-3, 2.50e-3
TRANS = 20.0              # length over which the area step is smoothed (m)

T_AMB = 18.0 + 273.15
T_IN = 50.0 + 273.15
P0 = 1.5e5
P_IN = 7.0e5
P_RES = 1.5e5


def _blend(s, transition=TRANS):
    """0 in the end sections, 1 in the central section (smoothed step)."""
    s = np.asarray(s, dtype=float)
    return 0.5 * (np.tanh((s - L_SEC) / transition)
                 - np.tanh((s - 2 * L_SEC) / transition))


def build(isothermal=False, scheme_h="upwind", N=70,
          transition=TRANS, s0=5.0):
    route = PipeRoute(
        xy=[(0.0, 0.0), (L, 0.0)],
        D=lambda s: D1 + (D2 - D1) * _blend(s, transition),
        wall=lambda s: E1 + (E2 - E1) * _blend(s, transition),
        rough=1.8e-6,
        E=2.0e11, nu=0.3,
        T_amb=lambda s: np.full(np.shape(s), T_AMB),
        U_gas=lambda s: np.full(np.shape(s), 10.0),
    )

    in_end = lambda s: (s < L_SEC) or (s > 2 * L_SEC)   # noqa: E731
    pig = Pig(
        mass=100.0,
        F_stat=lambda s: 1.724e5 if in_end(s) else 4.766e4,
        F_dyn=lambda s: 1.471e5 if in_end(s) else 3.971e4,
        n_holes=8, d_hole=1.75e-2, K=1.5,
    )

    def inlet(t):
        p = P0 + (P_IN - P0) * min(t / 20.0, 1.0)
        return {"type": "p", "p": p, "T": T_IN}

    def outlet(t):
        return {"type": "valve", "CdA": 0.15, "chi": 0.30, "p_res": P_RES}

    sol = PigFlowSolver(route, pig, nitrogen(), N_up=N, N_down=N, cluster=1.5,
                        bc_inlet=inlet, bc_outlet=outlet, scheme_h=scheme_h,
                        T_iso=T_AMB if isothermal else None)
    sol.initialise(s_pig=s0,
                   p_of_s=lambda s: np.full(np.shape(s), P0),
                   T_up_of_s=lambda s: np.full(np.shape(s), T_AMB))
    sol.set_stations([5e3, 10e3, 15e3, 20e3, 25e3, 30e3 - 1.0])
    return sol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-end", type=float, default=1700.0)
    ap.add_argument("--dt-max", type=float, default=2.0)
    ap.add_argument("--N", type=int, default=70,
                    help="cells in each flow domain")
    ap.add_argument("--transition", type=float, default=TRANS,
                    help="tanh area-transition length in m")
    ap.add_argument("--s0", type=float, default=5.0,
                    help="initial pig position in m")
    ap.add_argument("--isothermal", action="store_true",
                    help="replace the energy equation by T = 18 C (paper's "
                         "isothermal comparison)")
    ap.add_argument("--scheme-h", default="upwind", choices=["upwind", "central"],
                    help="enthalpy convection scheme; 'central' is what the "
                         "paper states it used")
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"))
    args = ap.parse_args()

    sol = build(isothermal=args.isothermal, scheme_h=args.scheme_h,
                N=args.N, transition=args.transition, s0=args.s0)
    snaps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500]
    sol.run(args.t_end, dt0=0.02, dt_max=args.dt_max, cfl_pig=0.02,
            snapshot_times=[t for t in snaps if t <= args.t_end])

    os.makedirs(args.out, exist_ok=True)
    tag = "case2_iso" if args.isothermal else "case2"
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
    print(f"\nfinished: t = {sol.t:.1f} s, pig at {sol.pig.s/1000:.3f} km, "
          f"Vp = {sol.pig.V:.2f} m/s")
    return sol


if __name__ == "__main__":
    main()
