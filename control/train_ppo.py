"""Train the PPO valve policy on the lumped model with randomised pigs.

    python control/train_ppo.py                    # 1.0 M steps, ~7 min on one core
    python control/train_ppo.py --steps 100000     # quick look
    python control/train_ppo.py --robust           # variant: 0.5 m/s margin, jittered model

Every 25 k steps the deterministic policy is scored on 32 validation pigs
(seed 2, not used for tuning or testing) and the best one is kept as
control/models/ppo_valve.zip (ppo_valve_robust.zip with --robust).  The
learning curve goes next to it (*_curve.json).  Validation always uses the
nominal lumped model and the real 8 m/s limit.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize  # noqa: E402

from control import scenario as S  # noqa: E402
from control.controllers import PPO_FILE, PPO_ROBUST_FILE, obs_vector  # noqa: E402
from control.env import PigValveEnv  # noqa: E402
from control.lumped import LumpedPig  # noqa: E402
from control.rollout import run  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VAL_SEED, N_VAL = 2, 32


class _Greedy:
    """The policy being trained, as a controller for rollout.run()."""

    def __init__(self, model):
        self.model = model

    def reset(self):
        self.u = S.U0
        return self.u

    def __call__(self, obs):
        a, _ = self.model.predict(obs_vector(obs), deterministic=True)
        self.u = float(np.clip(self.u + S.DU_MAX * float(np.clip(a[0], -1, 1)), 0.0, 1.0))
        return self.u


def evaluate(model, conds):
    """Mean return as reported in metrics.json (rollout.run, terminal timeout penalty)."""
    plant = LumpedPig(record=False)
    return float(np.mean([run(plant, _Greedy(model), m, F)[0]["ret"] for m, F in conds]))


class ValCallback(BaseCallback):
    def __init__(self, every, conds, path):
        super().__init__()
        self.every, self.conds, self.path = every, conds, path
        self.best, self.curve, self.next = -np.inf, [], every

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self.next += self.every
            r = evaluate(self.model, self.conds)
            self.curve.append({"steps": int(self.num_timesteps), "val_return": r})
            flag = ""
            if r > self.best:
                self.best = r
                self.model.save(self.path)
                flag = "  (saved)"
            print(f"  {self.num_timesteps:8d} steps  validation return {r:7.3f}{flag}", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=PPO_FILE)
    ap.add_argument("--eval-every", type=int, default=25_000)
    ap.add_argument("--robust", action="store_true",
                    help="train the 'robust' variant (0.5 m/s margin, jittered lumped model)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    if args.robust and args.out == PPO_FILE:
        args.out = PPO_ROBUST_FILE
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    venv = DummyVecEnv([lambda: PigValveEnv(robust=args.robust) for _ in range(8)])
    venv.seed(args.seed)
    venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.995)
    model = PPO("MlpPolicy", venv, n_steps=512, batch_size=256, n_epochs=10,
                gamma=0.995, gae_lambda=0.95, learning_rate=3e-4, clip_range=0.2,
                ent_coef=0.0, policy_kwargs=dict(net_arch=[64, 64], log_std_init=-0.5),
                seed=args.seed, device="cpu", verbose=0)
    cb = ValCallback(args.eval_every, S.sample_conditions(N_VAL, VAL_SEED), args.out)
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=cb)
    wall = time.perf_counter() - t0
    print(f"trained {args.steps} steps in {wall / 60:.1f} min; best validation return {cb.best:.3f}")
    curve = os.path.splitext(args.out)[0] + "_curve.json"
    with open(curve, "w") as fh:
        json.dump({"steps": args.steps, "seed": args.seed, "wall_s": wall,
                   "best_val_return": cb.best, "curve": cb.curve}, fh, indent=1)


if __name__ == "__main__":
    main()
