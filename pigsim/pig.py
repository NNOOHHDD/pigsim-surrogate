"""Pig model: force balance, by-pass pressure drop and stick/slip friction.

Paper eqs. (4)-(6).

    m dVp/dt = (p1 - p2) A - m g sin(alpha) - Fc(Vp)                    (4)
    p1 - p2  = (rho K / 2) (A/Ah)^2 (Q/A - Vp)^2                        (5)
    Fc       = -Fdyn_neg           if Vp < 0
               F   with -Fstat_neg <= F <= Fstat_pos   if Vp = 0        (6)
               +Fdyn_pos           if Vp > 0
"""

import numpy as np

G = 9.80665


class Pig:
    """A pig travelling along a :class:`~pigsim.geometry.PipeRoute`.

    Parameters
    ----------
    mass : float
        Pig mass (kg).
    F_stat, F_dyn : float or callable s -> float
        Static and dynamic contact force magnitudes (N).  Use callables to
        express the variation with axial position caused by area changes or by
        a varying pig/pipe friction coefficient (paper: ``Fc`` depends on xp).
        Separate values for forward/backward motion may be given as
        ``(pos, neg)`` pairs.
    n_holes, d_hole : int, float
        By-pass ports; ``n_holes = 0`` gives a sealing pig.
    K : float
        Localised pressure-drop coefficient of the by-pass ports.
    """

    def __init__(self, mass, F_stat, F_dyn, n_holes=0, d_hole=0.0, K=1.5,
                 s0=0.0, V0=0.0):
        self.mass = float(mass)
        self._F_stat = _force_pair(F_stat)
        self._F_dyn = _force_pair(F_dyn)
        self.n_holes = int(n_holes)
        self.d_hole = float(d_hole)
        self.K = float(K)
        self.A_hole = n_holes * 0.25 * np.pi * d_hole ** 2
        self.s = float(s0)
        self.V = float(V0)
        self.stopped = (V0 == 0.0)

    # -- contact forces ---------------------------------------------------
    def F_stat(self, s):
        pos, neg = self._F_stat
        return float(pos(s)), float(neg(s))

    def F_dyn(self, s):
        pos, neg = self._F_dyn
        return float(pos(s)), float(neg(s))

    def contact_force(self, s, Vp, driving):
        """Eq. (6).  ``driving`` is the net non-contact force (N).

        Returns the contact force with the sign convention of eq. (4), i.e. it
        is subtracted from the driving force.
        """
        fs_p, fs_n = self.F_stat(s)
        fd_p, fd_n = self.F_dyn(s)
        if Vp > 0.0:
            return fd_p
        if Vp < 0.0:
            return -fd_n
        # stopped: the contact force balances the driving force up to the
        # static limits
        return float(np.clip(driving, -fs_n, fs_p))

    def can_break_free(self, s, driving):
        fs_p, fs_n = self.F_stat(s)
        return driving > fs_p or driving < -fs_n

    # -- by-pass ----------------------------------------------------------
    @property
    def sealing(self):
        return self.A_hole <= 0.0

    def dp_bypass(self, rho, A, Q_hole):
        """Eq. (5) written with the volumetric by-pass flow rate Qh = Q - Vp A.

        p1 - p2 = (rho K / 2) (Qh / Ah)^2, signed with the flow direction.
        """
        if self.sealing:
            return 0.0
        vh = Q_hole / self.A_hole
        return 0.5 * rho * self.K * vh * abs(vh)

    def bypass_flow(self, rho, dp):
        """Inverse of :meth:`dp_bypass`: Qh from a given pressure drop."""
        if self.sealing:
            return 0.0
        vh = np.sign(dp) * np.sqrt(2.0 * abs(dp) / (rho * self.K))
        return vh * self.A_hole


def _force_pair(value):
    """Normalise a force specification to a (positive, negative) callable pair."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        pos, neg = value
    else:
        pos = neg = value
    return _as_callable(pos), _as_callable(neg)


def _as_callable(v):
    if callable(v):
        return v
    fv = float(v)
    return lambda s: fv
