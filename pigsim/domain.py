"""One side of the pig: adaptive moving mesh + implicit finite-difference solve.

The pipe is split by the pig into an upstream and a downstream domain, each
handled by a :class:`Domain`.  Inside a domain the governing equations are
written in the stretching coordinate (eta, t) of the paper's NUMERICAL METHOD
section, so that

    d/dt|_x  ->  d/dt|_eta - vg d/ds ,      vg = ds/dt|_eta

and the convective velocity becomes the *relative* velocity V - vg while the
mass-divergence term keeps the absolute velocity V.

Discretisation
--------------
Staggered mesh (Patankar 1980): p and h live at the N cell centres, V at the
N+1 faces.  The unknown vector is

    [p_0, h_0, V_0, p_1, h_1, V_1, ..., p_{N-1}, h_{N-1}, V_{N-1}, V_N]

so the coefficient matrix is narrowly banded (the paper solves the equivalent
hepta-diagonal system with a direct algorithm; here a dense banded solve is
used, which is cheap at these mesh sizes and easier to verify).

Time integration is fully implicit with the coefficients locally linearised
(Versteeg & Malalasekera 1995) -- i.e. Picard iteration on the advecting
velocity, the friction factor and the fluid properties.
"""

import numpy as np
from scipy.linalg import solve_banded

from .geometry import friction_factor
from .pig import G

# With the interleaved ordering [p_c, h_c, V_c, ...] every equation couples
# unknowns at most 5 places away, so the system is banded exactly as the
# paper's hepta-diagonal matrix.  A margin of one is kept.
KL = KU = 6


class Domain:
    """Flow domain on one side of the pig.

    Parameters
    ----------
    fluid : :class:`~pigsim.fluids.Fluid`
    N : int
        Number of cells (fixed; the mesh stretches instead of being re-numbered).
    route : :class:`~pigsim.geometry.PipeRoute`
    side : {'up', 'down'}
        'up'   -> s in [0, s_pig],  moving right boundary
        'down' -> s in [s_pig, L],  moving left boundary
    cluster : float
        Mesh-stretching exponent; >1 concentrates cells near the pig, where the
        gradients are steepest.
    scheme_p, scheme_h, scheme_V : {'central', 'upwind'}
        Spatial discretisation of the convective terms.  The paper uses central
        differences throughout; 'upwind' is available for the enthalpy, whose
        transport is convection-dominated (the diffusion k/cp is ~1e-5) and
        which otherwise develops wiggles at the sharp thermal fronts.
    """

    def __init__(self, fluid, N, route, side, cluster=1.5,
                 scheme_p="central", scheme_h="upwind", scheme_V="central",
                 T_lim=(50.0, 5000.0), p_lim=(1e2, 1e9), T_iso=None):
        self.T_iso = T_iso      # if set, the energy equation is replaced by T = T_iso
        self.fluid = fluid
        self.N = int(N)
        self.route = route
        self.side = side
        self.cluster = float(cluster)
        self.scheme_p = scheme_p
        self.scheme_h = scheme_h
        self.scheme_V = scheme_V
        self.h_lim = (fluid.h_of_T(T_lim[0]), fluid.h_of_T(T_lim[1]))
        self.p_lim = p_lim
        # below this domain length the stretching is relaxed towards a uniform
        # mesh, so that a pig near either end of the line does not generate
        # millimetre-sized cells
        self.span_ref = 0.02 * route.L

        u = np.linspace(0.0, 1.0, self.N + 1)
        self._u = u
        if side == "up":
            self._eta_cl = 1.0 - (1.0 - u) ** self.cluster   # clustered at eta = 1
        else:
            self._eta_cl = u ** self.cluster                  # clustered at eta = 0
        self.eta = self._eta_cl

        self.p = np.zeros(self.N)
        self.h = np.zeros(self.N)
        self.V = np.zeros(self.N + 1)
        self.sf = np.zeros(self.N + 1)
        self.set_mesh(0.0, 0.0)

    # ------------------------------------------------------------------
    # mesh
    # ------------------------------------------------------------------
    def set_mesh(self, s_pig, Vp):
        """Place the mesh for the current pig position and grid velocity."""
        L = self.route.L
        lo, hi = (0.0, s_pig) if self.side == "up" else (s_pig, L)
        span = max(hi - lo, 1e-12)
        w = min(1.0, span / self.span_ref)          # relax to uniform when short
        self.eta = (1.0 - w) * self._u + w * self._eta_cl
        self.sf = lo + span * self.eta
        if self.side == "up":
            self.vg_f = Vp * (self.sf - lo) / span
        else:
            self.vg_f = Vp * (hi - self.sf) / span

        self.sc = 0.5 * (self.sf[:-1] + self.sf[1:])
        self.dsc = np.diff(self.sf)                       # cell widths
        self.vg_c = 0.5 * (self.vg_f[:-1] + self.vg_f[1:])
        # distance between cell centres, with half cells at the two ends
        self.dsf = np.empty(self.N + 1)
        self.dsf[1:-1] = np.diff(self.sc)
        self.dsf[0] = self.sc[0] - self.sf[0]
        self.dsf[-1] = self.sf[-1] - self.sc[-1]

        self.A_c = self.route.area(self.sc)
        self.A_f = self.route.area(self.sf)
        self.D_c = self.route.D(self.sc)
        self.D_f = self.route.D(self.sf)
        self.eps_c = self.route.roughness(self.sc)
        self.eps_f = self.route.roughness(self.sf)
        self.sin_a_f = self.route.sin_alpha(self.sf)
        self.sin_a_c = self.route.sin_alpha(self.sc)
        self.T_amb_c = self.route.T_amb(self.sc)
        U = self.route.U_gas if self.fluid.is_gas else self.route.U_liq
        self.U_c = U(self.sc)

    # ------------------------------------------------------------------
    # pig-face states
    # ------------------------------------------------------------------
    @property
    def i_pig(self):
        """Index of the cell adjacent to the pig (0 for 'down', N-1 for 'up')."""
        return self.N - 1 if self.side == "up" else 0

    def p_pig_face(self):
        """Pressure linearly extrapolated to the pig face."""
        if self.side == "up":
            g = (self.p[-1] - self.p[-2]) / (self.sc[-1] - self.sc[-2])
            return self.p[-1] + g * (self.sf[-1] - self.sc[-1])
        g = (self.p[1] - self.p[0]) / (self.sc[1] - self.sc[0])
        return self.p[0] + g * (self.sf[0] - self.sc[0])

    def state_pig_face(self):
        """(p, rho, A) at the pig face."""
        i = self.i_pig
        st = self.fluid.props(self.p[i], self.h[i])
        A = self.A_f[-1] if self.side == "up" else self.A_f[0]
        return float(self.p_pig_face()), float(st.rho), float(A)

    # ------------------------------------------------------------------
    # initialisation helpers
    # ------------------------------------------------------------------
    def init_uniform(self, p_of_s, T_of_s):
        self.p = np.asarray(p_of_s(self.sc), dtype=float) * np.ones(self.N)
        self.h = self.fluid.h_of_T(np.asarray(T_of_s(self.sc), dtype=float)) * np.ones(self.N)
        self.V = np.zeros(self.N + 1)

    # ------------------------------------------------------------------
    # implicit solve
    # ------------------------------------------------------------------
    def solve_step(self, dt, p_old, h_old, V_old, bc_left, bc_right,
                   n_iter=12, tol=1e-6, relax=0.8):
        """Advance one time step.  Returns the number of Picard iterations."""
        for it in range(n_iter):
            p_new, h_new, V_new = self._linear_solve(
                dt, p_old, h_old, V_old, bc_left, bc_right)
            if not (np.all(np.isfinite(p_new)) and np.all(np.isfinite(h_new))
                    and np.all(np.isfinite(V_new))):
                raise FloatingPointError("non-finite solution in Domain.solve_step")
            p_new = np.clip(p_new, *self.p_lim)
            h_new = np.clip(h_new, *self.h_lim)
            dp = np.max(np.abs(p_new - self.p)) / max(np.max(np.abs(self.p)), 1.0)
            dV = np.max(np.abs(V_new - self.V)) / max(np.max(np.abs(V_new)), 1.0)
            dh = np.max(np.abs(h_new - self.h)) / max(np.max(np.abs(self.h)), 1.0)
            w = relax
            self.p = (1 - w) * self.p + w * p_new
            self.h = (1 - w) * self.h + w * h_new
            self.V = (1 - w) * self.V + w * V_new
            if max(dp, dV, dh) < tol:
                return it + 1
        return n_iter

    # -- assembly --------------------------------------------------------
    def _linear_solve(self, dt, p_old, h_old, V_old, bc_left, bc_right):
        N = self.N
        n = 3 * N + 1
        M = np.zeros((KL + KU + 1, n))   # LAPACK banded storage
        b = np.zeros(n)

        ip = 3 * np.arange(N)          # p_c
        ih = ip + 1                    # h_c
        iV = np.empty(N + 1, dtype=int)
        iV[:N] = 3 * np.arange(N) + 2  # V_f, f < N
        iV[N] = 3 * N                  # V_N

        st = self.fluid.props(self.p, self.h)
        rho, a, beta = st.rho, st.a, st.beta
        lam = self.route.lam(self.sc, rho, a)
        rho_a2 = rho * a ** 2 / lam
        chi_h = st.chi_h
        cp, kcond = self.fluid.cp, self.fluid.k

        # face-interpolated properties
        rho_f = np.empty(N + 1)
        rho_f[1:-1] = 0.5 * (rho[:-1] + rho[1:])
        rho_f[0], rho_f[-1] = rho[0], rho[-1]
        mu_f = np.empty(N + 1)
        mu_f[1:-1] = 0.5 * (st.mu[:-1] + st.mu[1:])
        mu_f[0], mu_f[-1] = st.mu[0], st.mu[-1]

        V_it = self.V
        Vc_it = 0.5 * (V_it[:-1] + V_it[1:])
        Uc = Vc_it - self.vg_c                       # relative velocity, cells
        Uf = V_it - self.vg_f                        # relative velocity, faces

        Re_f = rho_f * np.abs(V_it) * self.D_f / mu_f
        fD_f = friction_factor(Re_f, self.eps_f / self.D_f)
        Re_c = rho * np.abs(Vc_it) * self.D_c / st.mu
        fD_c = friction_factor(Re_c, self.eps_c / self.D_c)

        # boundary values available to the convective stencils
        p_bcL = bc_left.get("p") if bc_left["type"] == "p" else None
        p_bcR = bc_right.get("p") if bc_right["type"] == "p" else None
        h_bcL = bc_left.get("h")
        h_bcR = bc_right.get("h")

        gp = self._grad(self.sc, self.sf, Uc, p_bcL, p_bcR, self.scheme_p)
        gh = self._grad(self.sc, self.sf, Uc, h_bcL, h_bcR, self.scheme_h)

        c = np.arange(N)
        cm = np.clip(c - 1, 0, N - 1)
        cpl = np.clip(c + 1, 0, N - 1)

        # ---------------- continuity, row 3c ---------------------------
        r = ip
        _add(M, r, ip, 1.0 / dt)
        b[r] += p_old / dt
        _add(M, r, ip[cm], Uc * gp.wm)
        _add(M, r, ip, Uc * gp.w0)
        _add(M, r, ip[cpl], Uc * gp.wp)
        b[r] -= Uc * (gp.bl * _z(p_bcL) + gp.br * _z(p_bcR))

        div = rho_a2 / (self.A_c * self.dsc)
        _add(M, r, iV[1:], div * self.A_f[1:])
        _add(M, r, iV[:-1], -div * self.A_f[:-1])

        ch = rho_a2 * chi_h
        _add(M, r, ih, ch / dt)
        b[r] += ch * h_old / dt
        _add(M, r, ih[cm], ch * Uc * gh.wm)
        _add(M, r, ih, ch * Uc * gh.w0)
        _add(M, r, ih[cpl], ch * Uc * gh.wp)
        b[r] -= ch * Uc * (gh.bl * _z(h_bcL) + gh.br * _z(h_bcR))

        # ---------------- energy, row 3c+1 -----------------------------
        r = ih
        if self.T_iso is not None:
            # isothermal run: the energy equation is replaced by T = T_iso, so
            # the flow field is still fully compressible but at fixed temperature
            _add(M, r, ih, np.ones(N))
            b[r] += float(self.fluid.h_of_T(self.T_iso))
        else:
            self._energy_rows(M, b, r, ip, ih, c, cm, cpl, dt, p_old, h_old,
                              rho, Uc, gp, gh, p_bcL, p_bcR, h_bcL, h_bcR,
                              fD_c, Vc_it, cp, kcond)

        self._momentum_rows(M, b, iV, ip, dt, V_old, rho_f, fD_f, V_it, Uf,
                            bc_left, bc_right)

        x = solve_banded((KL, KU), M, b, check_finite=False)
        return x[ip], x[ih], x[iV]

    def _energy_rows(self, M, b, r, ip, ih, c, cm, cpl, dt, p_old, h_old,
                     rho, Uc, gp, gh, p_bcL, p_bcR, h_bcL, h_bcR,
                     fD_c, Vc_it, cp, kcond):
        N = self.N
        _add(M, r, ih, 1.0 / dt)
        b[r] += h_old / dt
        _add(M, r, ih[cm], Uc * gh.wm)
        _add(M, r, ih, Uc * gh.w0)
        _add(M, r, ih[cpl], Uc * gh.wp)
        b[r] -= Uc * (gh.bl * _z(h_bcL) + gh.br * _z(h_bcR))

        # - (1/rho) Dp/Dt
        _add(M, r, ip, -1.0 / (rho * dt))
        b[r] -= p_old / (rho * dt)
        _add(M, r, ip[cm], -Uc * gp.wm / rho)
        _add(M, r, ip, -Uc * gp.w0 / rho)
        _add(M, r, ip[cpl], -Uc * gp.wp / rho)
        b[r] += Uc * (gp.bl * _z(p_bcL) + gp.br * _z(p_bcR)) / rho

        # axial conduction, (1/(rho A)) d/ds (A (k/cp) dh/ds)
        Gam = kcond / cp
        cE = Gam * self.A_f[1:] / self.dsf[1:] / (rho * self.A_c * self.dsc)
        cW = Gam * self.A_f[:-1] / self.dsf[:-1] / (rho * self.A_c * self.dsc)
        _add(M, r, ih, cE + cW)
        _add(M, r[1:], ih[:-1], -cW[1:])
        _add(M, r[:-1], ih[1:], -cE[:-1])
        # ends: zero conductive flux unless a boundary enthalpy is known
        if h_bcL is None:
            _add(M, r[:1], ih[:1], -cW[:1])
        else:
            b[r[0]] += cW[0] * h_bcL
        if h_bcR is None:
            _add(M, r[-1:], ih[-1:], -cE[-1:])
        else:
            b[r[-1]] += cE[-1] * h_bcR

        # viscous dissipation (explicit) and wall heat transfer (implicit)
        b[r] += fD_c * np.abs(Vc_it) ** 3 / (2.0 * self.D_c)
        hcoef = 4.0 * self.U_c / (rho * cp * self.D_c)
        _add(M, r, ih, hcoef)
        b[r] += hcoef * self.fluid.h_of_T(self.T_amb_c)

    def _momentum_rows(self, M, b, iV, ip, dt, V_old, rho_f, fD_f, V_it, Uf,
                       bc_left, bc_right):
        N = self.N
        # interior faces f = 1 .. N-1
        f = np.arange(1, N)
        rf = iV[f]
        _add(M, rf, iV[f], 1.0 / dt)
        b[rf] += V_old[f] / dt
        if self.scheme_V == "central":
            dV = self.sf[f + 1] - self.sf[f - 1]
            _add(M, rf, iV[f + 1], Uf[f] / dV)
            _add(M, rf, iV[f - 1], -Uf[f] / dV)
        else:
            up = Uf[f] > 0
            dm = self.sf[f] - self.sf[f - 1]
            dp_ = self.sf[f + 1] - self.sf[f]
            _add(M, rf, iV[f], np.where(up, Uf[f] / dm, -Uf[f] / dp_))
            _add(M, rf, iV[f - 1], np.where(up, -Uf[f] / dm, 0.0))
            _add(M, rf, iV[f + 1], np.where(up, 0.0, Uf[f] / dp_))
        _add(M, rf, ip[f], 1.0 / (rho_f[f] * self.dsf[f]))
        _add(M, rf, ip[f - 1], -1.0 / (rho_f[f] * self.dsf[f]))
        _add(M, rf, iV[f], fD_f[f] * np.abs(V_it[f]) / (2.0 * self.D_f[f]))
        b[rf] -= G * self.sin_a_f[f]

        # left boundary face f = 0
        self._momentum_bc(M, b, iV, ip, 0, bc_left, dt, V_old, rho_f, fD_f, V_it, Uf)
        # right boundary face f = N
        self._momentum_bc(M, b, iV, ip, N, bc_right, dt, V_old, rho_f, fD_f, V_it, Uf)

    def _momentum_bc(self, M, b, iV, ip, f, bc, dt, V_old, rho_f, fD_f, V_it, Uf):
        N = self.N
        r = iV[f]
        kind = bc["type"]
        if kind == "V":
            _add(M, r, iV[f], 1.0)
            b[r] += bc["V"]
            return
        if kind == "valve":
            # Paper eq. (8), exactly as printed -- note there is NO factor of 2,
            # so this is not the textbook orifice equation:
            #     mdot = rho (Cd A)_0 chi sqrt((p - p_res)/rho)
            #   ->    V = C sqrt(dp/rho),   C = (Cd A)_0 chi / A
            ic = N - 1 if f == N else 0
            C = bc["CdA"] * bc["chi"] / self.A_f[f]
            st = self.fluid.props(self.p[ic], self.h[ic])
            dp = max(float(self.p[ic]) - bc["p_res"], 0.0)
            v = C * np.sqrt(dp / float(st.rho))
            dvdp = 0.5 * C / np.sqrt(float(st.rho) * max(dp, 1.0))
            sgn = 1.0 if f == N else -1.0
            _add(M, r, iV[f], 1.0)
            _add(M, r, ip[ic], -sgn * dvdp)
            b[r] += sgn * (v - dvdp * float(self.p[ic]))
            return
        # kind == 'p': half-cell momentum against the prescribed boundary pressure
        p_b = bc["p"]
        _add(M, r, iV[f], 1.0 / dt)
        b[r] += V_old[f] / dt
        if f == 0:
            dconv = self.sf[1] - self.sf[0]
            _add(M, r, iV[1], Uf[0] / dconv)
            _add(M, r, iV[0], -Uf[0] / dconv)
            _add(M, r, ip[0], 1.0 / (rho_f[0] * self.dsf[0]))
            b[r] += p_b / (rho_f[0] * self.dsf[0])
        else:
            dconv = self.sf[N] - self.sf[N - 1]
            _add(M, r, iV[N], Uf[N] / dconv)
            _add(M, r, iV[N - 1], -Uf[N] / dconv)
            _add(M, r, ip[N - 1], -1.0 / (rho_f[N] * self.dsf[N]))
            b[r] += -p_b / (rho_f[N] * self.dsf[N])
        _add(M, r, iV[f], fD_f[f] * abs(V_it[f]) / (2.0 * self.D_f[f]))
        b[r] -= G * self.sin_a_f[f]

    # -- gradient stencils ------------------------------------------------
    @staticmethod
    def _grad(sc, sf, U, phi_left, phi_right, scheme):
        """Three-point weights for d(phi)/ds at the cell centres.

        grad[c] = wm[c] phi[c-1] + w0[c] phi[c] + wp[c] phi[c+1]
                  + bl[c] phi_left + br[c] phi_right
        """
        N = len(sc)
        wm = np.zeros(N)
        w0 = np.zeros(N)
        wp = np.zeros(N)
        bl = np.zeros(N)
        br = np.zeros(N)

        c = np.arange(1, N - 1)
        if scheme == "central":
            d = sc[c + 1] - sc[c - 1]
            wm[c] = -1.0 / d
            wp[c] = 1.0 / d
        else:
            up = U[c] > 0
            dm = sc[c] - sc[c - 1]
            dpp = sc[c + 1] - sc[c]
            w0[c] = np.where(up, 1.0 / dm, -1.0 / dpp)
            wm[c] = np.where(up, -1.0 / dm, 0.0)
            wp[c] = np.where(up, 0.0, 1.0 / dpp)

        # first cell
        if phi_left is not None and (scheme == "central" or U[0] > 0):
            if scheme == "central" and N > 1:
                d = sc[1] - sf[0]
                wp[0] += 1.0 / d
                bl[0] += -1.0 / d
            else:
                d = sc[0] - sf[0]
                w0[0] += 1.0 / d
                bl[0] += -1.0 / d
        else:
            d = sc[1] - sc[0]
            w0[0] += -1.0 / d
            wp[0] += 1.0 / d

        # last cell
        if phi_right is not None and (scheme == "central" or U[-1] < 0):
            if scheme == "central" and N > 1:
                d = sf[-1] - sc[-2]
                wm[-1] += -1.0 / d
                br[-1] += 1.0 / d
            else:
                d = sf[-1] - sc[-1]
                w0[-1] += -1.0 / d
                br[-1] += 1.0 / d
        else:
            d = sc[-1] - sc[-2]
            wm[-1] += -1.0 / d
            w0[-1] += 1.0 / d

        return _Grad(wm, w0, wp, bl, br)


class _Grad:
    __slots__ = ("wm", "w0", "wp", "bl", "br")

    def __init__(self, wm, w0, wp, bl, br):
        self.wm, self.w0, self.wp, self.bl, self.br = wm, w0, wp, bl, br


def _add(M, rows, cols, vals):
    """Accumulate into LAPACK banded storage: M[KU + i - j, j] += A[i, j]."""
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    np.add.at(M, (KU + rows - cols, cols), vals)


def _z(v):
    return 0.0 if v is None else float(v)
