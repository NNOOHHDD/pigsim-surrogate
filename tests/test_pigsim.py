"""Verification of the flow solver against closed-form solutions.

Run with:  python tests/test_pigsim.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pigsim import PipeRoute, Domain, Pig, friction_factor, water, nitrogen  # noqa: E402
from pigsim.pig import G  # noqa: E402

FAILURES = []


def check(name, got, want, rtol):
    ok = abs(got - want) <= rtol * abs(want)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<52s} "
          f"got {got: .6g}  want {want: .6g}  ({100*abs(got-want)/abs(want):.2f} %)")
    if not ok:
        FAILURES.append(name)


def straight_route(L, D, sin_a=0.0, T=298.15, U=0.0, wall=6e-3, rough=4.6e-5):
    dy = sin_a * L
    dx = np.sqrt(max(L ** 2 - dy ** 2, 0.0))
    return PipeRoute(xy=[(0.0, 0.0), (dx, dy)], D=D, wall=wall, rough=rough,
                     E=2.1e11, nu=0.3,
                     T_amb=lambda s: np.full(np.shape(s), T),
                     U_gas=lambda s: np.full(np.shape(s), U),
                     U_liq=lambda s: np.full(np.shape(s), U))


def march(dom, bcL, bcR, dt, n):
    for _ in range(n):
        p_old, h_old, V_old = dom.p.copy(), dom.h.copy(), dom.V.copy()
        dom.solve_step(dt, p_old, h_old, V_old, bcL, bcR, n_iter=8, tol=1e-10)


# ---------------------------------------------------------------------------
def test_hydrostatic():
    """No flow in a vertical liquid column: dp/ds = -rho g."""
    print("\n1. hydrostatic column (liquid, no flow)")
    L, D = 100.0, 0.2
    route = straight_route(L, D, sin_a=1.0)
    fl = water()
    dom = Domain(fl, 60, route, "up", cluster=1.0)
    dom.set_mesh(L, 0.0)
    dom.init_uniform(lambda s: np.full(np.shape(s), 1e5),
                     lambda s: np.full(np.shape(s), 298.15))
    bcL = {"type": "V", "V": 0.0, "h": fl.h_of_T(298.15)}
    bcR = {"type": "p", "p": 1e5, "T": 298.15, "h": fl.h_of_T(298.15)}
    march(dom, bcL, bcR, 0.5, 400)
    rho = float(fl.props(1e5, fl.h_of_T(298.15)).rho)
    check("p(bottom) - p(top)", float(dom.p[0] - dom.p[-1]),
          rho * G * (dom.sc[-1] - dom.sc[0]), 2e-3)


def test_darcy_weisbach():
    """Steady liquid flow in a horizontal pipe: dp = f (L/D) rho V^2 / 2."""
    print("\n2. steady friction pressure drop (liquid, horizontal)")
    L, D, V = 500.0, 0.2, 2.0
    route = straight_route(L, D)
    fl = water()
    dom = Domain(fl, 80, route, "up", cluster=1.0)
    dom.set_mesh(L, 0.0)
    dom.init_uniform(lambda s: np.full(np.shape(s), 1e6),
                     lambda s: np.full(np.shape(s), 298.15))
    h = float(fl.h_of_T(298.15))
    bcL = {"type": "V", "V": V, "h": h}
    bcR = {"type": "p", "p": 1e6, "T": 298.15, "h": h}
    march(dom, bcL, bcR, 0.2, 1500)
    st = fl.props(dom.p, dom.h)
    rho = float(np.mean(st.rho))
    Re = rho * V * D / float(np.mean(st.mu))
    f = float(friction_factor(Re, route.roughness(0.0) / D))
    want = f * (dom.sc[-1] - dom.sc[0]) / D * rho * V ** 2 / 2.0
    check(f"dp over the pipe (Re = {Re:.3g}, f = {f:.4f})",
          float(dom.p[0] - dom.p[-1]), want, 5e-3)


def test_isothermal_gas_line():
    """Steady isothermal compressible pipe flow: p1^2 - p2^2 = f (L/D) G^2 R T."""
    print("\n3. steady isothermal compressible gas line")
    L, D = 2000.0, 0.3
    T = 298.15
    route = straight_route(L, D, T=T, wall=5e-3, rough=1.5e-5)
    fl = nitrogen()
    dom = Domain(fl, 80, route, "up", cluster=1.0, T_iso=T)
    dom.set_mesh(L, 0.0)
    dom.init_uniform(lambda s: np.full(np.shape(s), 2e6),
                     lambda s: np.full(np.shape(s), T))
    Gmass = 40.0                      # mass flux, kg/(m2 s)
    p_out = 2.0e6
    rho_in = p_out / (fl.R * T)
    h = float(fl.h_of_T(T))
    bcL = {"type": "V", "V": Gmass / rho_in, "h": h}
    bcR = {"type": "p", "p": p_out, "T": T, "h": h}
    march(dom, bcL, bcR, 0.05, 4000)
    st = fl.props(dom.p, dom.h)
    Re = Gmass * D / float(np.mean(st.mu))
    f = float(friction_factor(Re, route.roughness(0.0) / D))
    Leff = dom.sc[-1] - dom.sc[0]
    p2 = float(dom.p[-1])
    # p1^2 - p2^2 = f L/D G^2 R T   (the ln(p1/p2) acceleration term is
    # negligible here: f L/D = 30 >> 2 ln(p1/p2))
    want = np.sqrt(p2 ** 2 + f * Leff / D * Gmass ** 2 * fl.R * T)
    check(f"inlet pressure (Re = {Re:.3g}, f = {f:.4f})",
          float(dom.p[0]), want, 5e-3)


def test_bypass_roundtrip():
    """Pig.dp_bypass and Pig.bypass_flow must be exact inverses (eq. 5)."""
    print("\n4. by-pass pressure drop <-> flow round trip (eq. 5)")
    pig = Pig(mass=100.0, F_stat=1e5, F_dyn=1e5, n_holes=8, d_hole=1.75e-2, K=1.5)
    rho = 7.3
    for Qh in (0.1, 0.42, -0.3):
        dp = pig.dp_bypass(rho, 0.5675, Qh)
        check(f"Qh = {Qh:+.2f} m3/s", pig.bypass_flow(rho, dp), Qh, 1e-9)


def test_stick_slip():
    """Eq. (6): the contact force must saturate at the static limits."""
    print("\n5. stick / slip contact force (eq. 6)")
    pig = Pig(mass=3.0, F_stat=(1000.0, 800.0), F_dyn=(600.0, 500.0))
    check("stopped, driving 400 N  -> balanced", pig.contact_force(0, 0.0, 400.0),
          400.0, 1e-12)
    check("stopped, driving 5000 N -> static limit",
          pig.contact_force(0, 0.0, 5000.0), 1000.0, 1e-12)
    check("stopped, driving -5000 N-> static limit",
          pig.contact_force(0, 0.0, -5000.0), -800.0, 1e-12)
    check("moving forward          -> +Fdyn", pig.contact_force(0, 1.0, 5000.0),
          600.0, 1e-12)
    check("moving backward         -> -Fdyn", pig.contact_force(0, -1.0, -5000.0),
          -500.0, 1e-12)
    print(f"  [{'PASS' if pig.can_break_free(0, 1200.0) else 'FAIL'}] "
          "break-free above the static limit")
    print(f"  [{'PASS' if not pig.can_break_free(0, 900.0) else 'FAIL'}] "
          "no break-free below the static limit")


def main():
    print("=" * 84)
    print("pigsim verification against closed-form solutions")
    print("=" * 84)
    test_hydrostatic()
    test_darcy_weisbach()
    test_isothermal_gas_line()
    test_bypass_roundtrip()
    test_stick_slip()
    print("\n" + "=" * 84)
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
