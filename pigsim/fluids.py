"""Fluid property models.

Follows Nieckele, Braga & Azevedo (2000), ASME IPC2000-175, eqs. (7)-(9).

Enthalpy convention used throughout the code: h = cp * T with T in Kelvin and
cp constant.  This is exact for the ideal gas of eq. (7) and is the same
approximation the paper makes for liquids (cp constant, eq. 8/9).

Every model exposes ``props(p, h)`` returning a :class:`State` of arrays with
the quantities the flow solver needs:

    T      temperature                      [K]
    rho    density                          [kg/m3]
    a      isothermal speed of sound        [m/s]
    beta   thermal expansion coefficient    [1/K]
    mu     absolute viscosity               [Pa s]
    chi_h  (1/rho) * d(rho)/dh at const p   [kg/J]   == -beta/cp
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class State:
    T: np.ndarray
    rho: np.ndarray
    a: np.ndarray
    beta: np.ndarray
    mu: np.ndarray
    chi_h: np.ndarray


class Fluid:
    """Base class.  ``cp`` and ``k`` are constants (paper, eq. 9 remark)."""

    name = "fluid"
    is_gas = False

    def __init__(self, cp, k, mu_ref, c_mu_T=0.0, c_mu_p=0.0,
                 p_ref=101325.0, T_ref=298.15):
        self.cp = cp
        self.k = k
        self.mu_ref = mu_ref
        self.c_mu_T = c_mu_T
        self.c_mu_p = c_mu_p
        self.p_ref = p_ref
        self.T_ref = T_ref

    # -- helpers ---------------------------------------------------------
    def h_of_T(self, T):
        return self.cp * np.asarray(T, dtype=float)

    def T_of_h(self, h):
        return np.asarray(h, dtype=float) / self.cp

    def _mu(self, p, T):
        """Eq. (9): mu = mu_ref + c_mu_T (T - T_ref) + c_mu_p (p - p_ref)."""
        return np.maximum(
            self.mu_ref + self.c_mu_T * (T - self.T_ref)
            + self.c_mu_p * (p - self.p_ref),
            1e-8,
        )

    def props(self, p, h):  # pragma: no cover - abstract
        raise NotImplementedError


class IdealGas(Fluid):
    """Eq. (7):  rho = p/(Rgas T),  a^2 = Rgas T,  beta = 1/T."""

    is_gas = True

    def __init__(self, R, cp, k, mu_ref, c_mu_T=0.0, c_mu_p=0.0,
                 p_ref=101325.0, T_ref=298.15, name="gas"):
        super().__init__(cp, k, mu_ref, c_mu_T, c_mu_p, p_ref, T_ref)
        self.R = R
        self.name = name

    def props(self, p, h):
        p = np.asarray(p, dtype=float)
        T = self.T_of_h(h)
        a = np.sqrt(self.R * T)
        rho = p / (self.R * T)
        beta = 1.0 / T
        return State(T=T, rho=rho, a=a, beta=beta, mu=self._mu(p, T),
                     chi_h=-beta / self.cp)


class Liquid(Fluid):
    """Eq. (8):  rho = rho_ref [1 - beta (T - T_ref)] + (p - p_ref)/a^2.

    Speed of sound and thermal expansion coefficient are constants.
    """

    is_gas = False

    def __init__(self, rho_ref, a, beta, cp, k, mu_ref, c_mu_T=0.0, c_mu_p=0.0,
                 p_ref=101325.0, T_ref=298.15, name="liquid"):
        super().__init__(cp, k, mu_ref, c_mu_T, c_mu_p, p_ref, T_ref)
        self.rho_ref = rho_ref
        self.a0 = a
        self.beta0 = beta
        self.name = name

    def props(self, p, h):
        p = np.asarray(p, dtype=float)
        T = self.T_of_h(h)
        rho = (self.rho_ref * (1.0 - self.beta0 * (T - self.T_ref))
               + (p - self.p_ref) / self.a0 ** 2)
        a = np.full_like(rho, self.a0)
        beta = np.full_like(rho, self.beta0)
        return State(T=T, rho=rho, a=a, beta=beta, mu=self._mu(p, T),
                     chi_h=-beta / self.cp)


# ---------------------------------------------------------------------------
# Fluids used in the paper's test cases (ANALYSIS section, p. 4)
# ---------------------------------------------------------------------------

def nitrogen():
    """Rgas = 296.9 J/(kg K), mu_ref = 1.88e-5, c_mu_T = 2.255e-8*,
    cp = 1042 J/(kg K), k = 2.2e-2 W/(m K).

    * The scanned text reads ``c_muT = 2.255 10* N s/(m2 K)``; 2.255e-8 is the
      physically correct order of magnitude for N2 (dmu/dT ~ 4e-8 Pa s/K) and
      is what is used here.  The viscosity dependence has a negligible effect
      on the results.
    """
    return IdealGas(R=296.9, cp=1042.0, k=2.2e-2,
                    mu_ref=1.88e-5, c_mu_T=2.255e-8, c_mu_p=0.0,
                    p_ref=101325.0, T_ref=298.15, name="N2")


def water():
    """rho_ref = 1e3, a = 1200 m/s, beta = 0, mu_ref = 2.3e-3,
    c_mu_T = -7.5e-5, cp = 4180 J/(kg K), k = 0.6 W/(m K)."""
    return Liquid(rho_ref=1.0e3, a=1200.0, beta=0.0, cp=4180.0, k=0.6,
                  mu_ref=2.3e-3, c_mu_T=-7.5e-5, c_mu_p=0.0,
                  p_ref=101325.0, T_ref=298.15, name="H2O")
