"""The three controllers (and an open-loop schedule used for calibration).

Every controller sees only the measured signals of the plant's obs dict
(inlet / outlet pressure, pig position and speed, time, its own last valve
command) -- never the pig's mass or friction.
"""

import os

import numpy as np

from . import scenario as S

HERE = os.path.dirname(os.path.abspath(__file__))
PPO_FILE = os.path.join(HERE, "models", "ppo_valve.zip")
PPO_ROBUST_FILE = os.path.join(HERE, "models", "ppo_valve_robust.zip")


class FixedOpening:
    """(a) One opening for the whole run, set before launch."""
    rate_limited = False

    def __init__(self, u):
        self.u = float(u)

    def reset(self):
        return self.u

    def __call__(self, obs):
        return self.u


class Schedule:
    """Open-loop opening u(t), used only to compare the two plants."""
    rate_limited = False

    def __init__(self, fn):
        self.fn = fn

    def reset(self):
        return float(self.fn(0.0))

    def __call__(self, obs):
        return float(self.fn(obs["t"]))


class PI:
    """(b) PI speed controller in velocity form, with the same rate limit as RL.

        e_k  = V_ref - V_k
        u_k  = u_{k-1} + Kp (e_k - e_{k-1}) + Ki e_k dt,   clipped to [u_min, 1]

    The velocity form needs no separate anti-windup: the opening is clipped to
    [u_min, 1] and to +/- DU_MAX per step before it becomes the next u_{k-1}.
    During the launch surge (V >> V_ref) it closes the valve down to u_min;
    u_min (tuned like the gains) keeps it from stalling the pig there.
    """

    def __init__(self, v_ref, kp, ki, u_min=0.0, u0=S.U0):
        self.v_ref, self.kp, self.ki = float(v_ref), float(kp), float(ki)
        self.u_min, self.u0 = float(u_min), float(u0)

    def reset(self):
        self.u = self.u0
        self.e_prev = None
        return self.u

    def __call__(self, obs):
        e = self.v_ref - obs["V"]
        de = 0.0 if self.e_prev is None else e - self.e_prev
        u_cmd = self.u + self.kp * de + self.ki * e * S.DT_CTRL
        self.u = float(np.clip(np.clip(u_cmd, self.u - S.DU_MAX, self.u + S.DU_MAX),
                               self.u_min, 1.0))
        self.e_prev = e
        return self.u


def obs_vector(obs):
    """Normalised observation used by the RL policy (measurable signals only)."""
    return np.array([
        (obs["p_in"] - 1.5e5) / 1.5e5,
        (obs["p_out"] - 1.5e5) / 1.5e5,
        obs["s"] / S.L,
        obs["V"] / S.V_CAP,
        obs["t"] / S.T_TARGET,
        obs["u"],
    ], dtype=np.float32)


class PPOController:
    """(c) A trained stable-baselines3 PPO policy; action = change of opening."""

    def __init__(self, path=PPO_FILE, deterministic=True):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")
        self.deterministic = deterministic

    def reset(self):
        self.u = S.U0
        return self.u

    def __call__(self, obs):
        a, _ = self.model.predict(obs_vector(obs), deterministic=self.deterministic)
        self.u = float(np.clip(self.u + S.DU_MAX * float(np.clip(a[0], -1.0, 1.0)), 0.0, 1.0))
        return self.u
