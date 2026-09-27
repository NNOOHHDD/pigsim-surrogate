"""Pipeline route geometry and ambient conditions.

The route is described by (x, y) way-points; the solver works in arc length
``s`` measured from the inlet.  Diameter, wall thickness, roughness, ambient
temperature and the global heat-transfer coefficient may all vary along ``s``.
"""

import numpy as np


class PipeRoute:
    """Piecewise-linear pipeline route.

    Parameters
    ----------
    xy : (n, 2) array
        Way-points, ``x`` horizontal and ``y`` vertical (m).  ``y`` increasing
        upwards, so a subsea riser has negative ``y``.
    D, wall, rough : float or (s, value) table
        Internal diameter, wall thickness and *absolute* roughness (m).  A
        table is given as a list of ``(s_start, s_end, value)`` triples and is
        evaluated as a piecewise-constant function of arc length.
    E, nu : float
        Young's modulus (Pa) and Poisson's ratio of the pipe material.
    T_amb : callable s -> K
    U_gas, U_liq : callable s -> W/(m2 K)
        Global heat-transfer coefficient, selected by whether the fluid in the
        cell is a gas or a liquid (the paper quotes different values for the
        two, e.g. 10 and 100 W/(m2 K) for the submerged part of the riser).
    """

    def __init__(self, xy, D, wall, rough, E, nu, T_amb, U_gas, U_liq=None):
        xy = np.asarray(xy, dtype=float)
        seg = np.diff(xy, axis=0)
        ds = np.hypot(seg[:, 0], seg[:, 1])
        self.xy = xy
        self.s_nodes = np.concatenate([[0.0], np.cumsum(ds)])
        self.L = float(self.s_nodes[-1])
        # sin(alpha) is constant on each straight segment
        self._sin_a = np.where(ds > 0, seg[:, 1] / np.where(ds > 0, ds, 1.0), 0.0)

        self._D = _as_table(D)
        self._wall = _as_table(wall)
        self._rough = _as_table(rough)
        self.E = E
        self.nu = nu
        self.T_amb = T_amb
        self.U_gas = U_gas
        self.U_liq = U_gas if U_liq is None else U_liq

    # -- geometry --------------------------------------------------------
    def _seg_index(self, s):
        i = np.searchsorted(self.s_nodes, np.asarray(s, dtype=float), side="right") - 1
        return np.clip(i, 0, len(self.s_nodes) - 2)

    def sin_alpha(self, s):
        return self._sin_a[self._seg_index(s)]

    def elevation(self, s):
        s = np.asarray(s, dtype=float)
        return np.interp(s, self.s_nodes, self.xy[:, 1])

    def D(self, s):
        return self._D(s)

    def area(self, s):
        d = self.D(s)
        return 0.25 * np.pi * d ** 2

    def dArea_ds(self, s, ds=1e-3):
        """Numerical d(A)/ds.  Zero except at the (smoothed) area steps."""
        s = np.asarray(s, dtype=float)
        return (self.area(s + ds) - self.area(s - ds)) / (2.0 * ds)

    def wall(self, s):
        return self._wall(s)

    def roughness(self, s):
        return self._rough(s)

    def lam(self, s, rho, a):
        """Pipe-deformation factor lambda = 1 + rho a^2 (1-nu^2) D/(e E).

        From the paper: lambda = 1 + rho a^2 2 C_D (D/D_ref) with
        C_D = (1 - nu^2) D_ref / (2 e E).  The effective (water-hammer) wave
        speed is a/sqrt(lambda).
        """
        return 1.0 + rho * a ** 2 * (1.0 - self.nu ** 2) * self.D(s) / (self.wall(s) * self.E)


def _as_table(value):
    """Return a callable f(s) from a scalar or a list of (s0, s1, v) triples."""
    if callable(value):
        return value
    if np.isscalar(value):
        v = float(value)
        return lambda s: np.full(np.shape(s), v, dtype=float)

    tbl = [(float(a), float(b), float(c)) for a, b, c in value]

    def f(s):
        s = np.asarray(s, dtype=float)
        out = np.full(s.shape, tbl[-1][2], dtype=float)
        for a, b, v in tbl:
            out = np.where((s >= a) & (s < b), v, out)
        out = np.where(s < tbl[0][0], tbl[0][2], out)
        return out

    return f


def friction_factor(Re, rel_rough):
    """Darcy friction factor.

    Laminar (Re < 2300): f = 64/Re.
    Turbulent: Miller's explicit correlation (Fox & McDonald),
        f = 0.25 / [log10(rel_rough/3.7 + 5.74/Re^0.9)]^2
    """
    Re = np.maximum(np.abs(Re), 1e-6)
    f_lam = 64.0 / Re
    arg = rel_rough / 3.7 + 5.74 / Re ** 0.9
    f_turb = 0.25 / np.log10(np.maximum(arg, 1e-12)) ** 2
    return np.where(Re < 2300.0, f_lam, f_turb)
