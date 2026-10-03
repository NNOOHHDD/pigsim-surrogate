"""Gymnasium environment: steer the outlet valve while the pig travels.

    observation (6,)  inlet pressure, outlet pressure, pig position, pig speed,
                      elapsed time (all normalised, see controllers.obs_vector)
                      and the current valve opening
    action (1,)       change of opening in units of DU_MAX, clipped to [-1, 1]
    reward            - speed above the limit (s >= S_LIMIT), integrated and per step
                      - seconds past the arrival-time target
                      - valve movement
                      + progress along the line (= the penalty for not arriving
                        by T_MAX, paid densely; same objective up to a constant)
                      (weights in control/rollout.py)

Each reset draws a new pig (mass, friction) from the training box; the agent
is never told which.  The plant is the lumped model by default; pass
plant=PigsimPlant() to run the same environment on pigsim (slow).
"""

import gymnasium as gym
import numpy as np

from . import scenario as S
from .controllers import obs_vector
from .lumped import LumpedPig, load_params

ROBUST_MARGIN = 0.5     # m/s
ROBUST_JITTER = {"n": (1.0, 1.15), "c_g": (0.6, 1.6), "f_mult": (0.8, 1.25), "K_in": (0.5, 2.0)}
from .rollout import progress_reward, step_reward


class PigValveEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, plant=None, ranges=S.TRAIN, conditions=None, robust=False):
        super().__init__()
        # robust=True (training variant "PPO, robust"): the speed limit is
        # tightened by ROBUST_MARGIN and the four calibrated lumped-model constants
        # are redrawn every episode within ROBUST_JITTER, so the policy cannot rely
        # on one exact model.  Default False = the environment described above.
        self.robust = robust
        self.plant = plant if plant is not None else LumpedPig(record=False)
        self.ranges = ranges
        self.conditions = conditions      # optional fixed list of (mass, F_fric)
        self._k = 0
        self.observation_space = gym.spaces.Box(-5.0, 5.0, shape=(6,), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and "mass" in options:
            mass, F = options["mass"], options["F_fric"]
        elif self.conditions is not None:
            mass, F = self.conditions[self._k % len(self.conditions)]
            self._k += 1
        else:
            mass = self.np_random.uniform(*self.ranges["mass"])
            F = self.np_random.uniform(*self.ranges["F_fric"])
        self.condition = (float(mass), float(F))
        if self.robust:
            base = load_params()
            p = dict(base)
            for k, (lo, hi) in ROBUST_JITTER.items():
                p[k] = base[k] * self.np_random.uniform(lo, hi)
            p["n"] = min(max(p["n"], 1.0), 1.4)
            self.plant.p = p
            self.plant.v_cap = S.V_CAP - ROBUST_MARGIN
        self.obs = self.plant.reset(mass, F, u0=S.U0)
        return obs_vector(self.obs), {"mass": float(mass), "F_fric": float(F)}

    def step(self, action):
        a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
        u_old = self.plant.u
        u_new = float(np.clip(u_old + S.DU_MAX * a, 0.0, 1.0))
        t0, s0 = self.plant.t, self.plant.s
        self.obs = self.plant.step(u_new)
        r = step_reward(self.plant.step_stats, t0, self.plant.t, u_new - u_old,
                        v_cap=self.plant.v_cap if self.robust else S.V_CAP)
        # The "not arrived by T_MAX" penalty of rollout.timeout_penalty, paid out
        # as progress instead:  sum_k W (ds_k / L) = W (S_ARRIVE - S0)/L - W (left / L),
        # i.e. the same objective up to a constant, but without a reward that only
        # shows up hundreds of (discounted) steps after the valve was shut.
        r += progress_reward(s0, self.plant.s)
        terminated = bool(self.plant.arrived) or self.plant.t >= S.T_MAX - 1e-9
        return obs_vector(self.obs), float(r), terminated, False, {}
