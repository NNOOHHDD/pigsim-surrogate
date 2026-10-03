"""Valve-control demo (control/): environment, plants, saved controllers.

    python -m pytest tests/test_control.py -m "not slow"     # ~10 s
    python -m pytest tests/test_control.py                   # + closed-loop pigsim and a short PPO run

The fast tests use the lumped model only; the slow ones run pigsim and train
PPO for a few thousand steps.
"""

import json
import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from control import scenario as S  # noqa: E402
from control.controllers import PI, FixedOpening, obs_vector  # noqa: E402
from control.env import PigValveEnv  # noqa: E402
from control.lumped import LumpedPig  # noqa: E402
from control.rollout import run  # noqa: E402

CTRL = os.path.join(ROOT, "control")


def test_lumped_model_arrives_with_measurable_signals():
    plant = LumpedPig()
    obs = plant.reset(110.0, 2500.0, u0=0.4)
    assert set(obs) == {"t", "p_in", "p_out", "s", "V", "u"}
    while not plant.arrived and plant.t < S.T_MAX:
        obs = plant.step(0.4)
        assert all(math.isfinite(v) for v in obs.values())
    assert plant.arrived and 60.0 < plant.t < S.T_MAX
    tr = plant.trace()
    assert max(tr["V"]) > 20.0          # the launch surge is there
    assert min(tr["V"]) >= 0.0


def test_closed_valve_holds_the_pig_back():
    a, b = LumpedPig(), LumpedPig()
    a.reset(110.0, 2500.0)
    b.reset(110.0, 2500.0)
    for _ in range(80):
        a.step(0.0)
        b.step(1.0)
    assert a.s < 0.7 * S.L < b.s


def test_env_api_and_episode_ends():
    from gymnasium.utils.env_checker import check_env
    env = PigValveEnv()
    check_env(env, skip_render_check=True)
    o, info = env.reset(seed=3)
    assert o.shape == (6,) and env.observation_space.contains(o)
    assert S.TRAIN["mass"][0] <= info["mass"] <= S.TRAIN["mass"][1]
    rng = np.random.default_rng(0)
    for k in range(int(S.T_MAX / S.DT_CTRL) + 1):
        o, r, term, trunc, _ = env.step(rng.uniform(-1, 1, size=1).astype(np.float32))
        assert np.isfinite(r) and np.all(np.isfinite(o))
        if term or trunc:
            break
    assert term


def test_controller_cannot_see_the_pig():
    """At t = 0 every pig gives the same observation; mass and friction are hidden."""
    env = PigValveEnv()
    o1, _ = env.reset(options={"mass": 20.0, "F_fric": 1000.0})
    o2, _ = env.reset(options={"mass": 200.0, "F_fric": 4000.0})
    np.testing.assert_array_equal(o1, o2)


def test_action_is_a_rate_limited_change_of_opening():
    env = PigValveEnv()
    env.reset(options={"mass": 110.0, "F_fric": 2500.0})
    env.step(np.array([1.0], dtype=np.float32))
    assert env.plant.u == pytest.approx(S.U0 + S.DU_MAX)
    env.step(np.array([-5.0], dtype=np.float32))       # clipped to -1
    assert env.plant.u == pytest.approx(S.U0)


def test_baselines_run_from_saved_settings():
    with open(os.path.join(CTRL, "baselines.json")) as fh:
        b = json.load(fh)
    plant = LumpedPig()
    m, _ = run(plant, FixedOpening(b["fixed"]["u"]), 110.0, 2500.0)
    assert m["arrived"] and m["valve_travel"] == 0.0
    pi = PI(**{k: b["pi"][k] for k in ("v_ref", "kp", "ki", "u_min")})
    m, tr = run(plant, pi, 110.0, 2500.0)
    assert m["arrived"]
    du = np.abs(np.diff(tr["u"]))
    assert du.max() <= S.DU_MAX + 1e-9


def test_policy_loads_and_reproduces_saved_result():
    pytest.importorskip("stable_baselines3")
    from control.controllers import PPO_FILE, PPOController
    assert os.path.exists(PPO_FILE)
    ctrl = PPOController()
    a, _ = ctrl.model.predict(obs_vector(LumpedPig().reset(110.0, 2500.0)), deterministic=True)
    assert a.shape == (1,)
    with open(os.path.join(CTRL, "data", "eval_runs.json")) as fh:
        saved = json.load(fh)["lumped_train"]["ppo"][0]
    m, _ = run(LumpedPig(), ctrl, saved["mass"], saved["F_fric"])
    assert m["arrived"] == saved["arrived"]
    assert m["t_arrive"] == pytest.approx(saved["t_arrive"], rel=1e-3)
    assert m["v_max_limited"] == pytest.approx(saved["v_max_limited"], rel=1e-3)


def test_metrics_file_is_complete():
    with open(os.path.join(CTRL, "metrics.json")) as fh:
        m = json.load(fh)
    for s in ("lumped_train", "lumped_highF", "pigsim_train", "pigsim_highF"):
        for c in ("fixed", "pi", "ppo", "ppo_robust"):
            r = m["results"][s][c]
            assert 0.0 <= r["violation_rate"] <= 1.0
            assert r["n"] == {"lumped_train": 200, "lumped_highF": 200,
                              "pigsim_train": 20, "pigsim_highF": 10}[s]


@pytest.mark.slow
def test_pigsim_plant_matches_a_single_run():
    """Running pigsim in 1-s segments through the chi_of_t hook changes nothing but
    the time-step sequence: same results as one uninterrupted run, to within
    pigsim's own time-step sensitivity (control/RESULTS_control.md, section 1)."""
    from control.pigsim_plant import PigsimPlant
    from surrogate import pipeline as P
    u, m, F = 0.355, 110.0, 2500.0
    plant = PigsimPlant()
    plant.reset(m, F, u0=u)
    while not plant.arrived and plant.t < S.T_MAX:
        plant.step(u)
    sol = P.build(S.P_IN, m, F, S.L, chi_of_t=lambda t: S.CHI_MAX * u)
    sol.run(S.T_MAX, dt0=0.02, dt_max=1.0, cfl_pig=0.02, s_stop=S.S_ARRIVE, verbose=False)
    h, hp = sol.history(), plant.sol.history()

    def v_lim(h):
        return h["V_pig"][h["s_pig"] >= S.S_LIMIT].max()

    assert hp["t"][-1] == pytest.approx(h["t"][-1], rel=5e-3)
    assert v_lim(hp) == pytest.approx(v_lim(h), abs=0.2)
    assert hp["V_pig"].max() == pytest.approx(h["V_pig"].max(), rel=2e-2)


@pytest.mark.slow
def test_ppo_training_runs(tmp_path):
    pytest.importorskip("stable_baselines3")
    import subprocess
    out = tmp_path / "p.zip"
    r = subprocess.run([sys.executable, os.path.join(CTRL, "train_ppo.py"), "--steps", "4096",
                        "--eval-every", "4096", "--out", str(out)],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr
    assert out.exists()
