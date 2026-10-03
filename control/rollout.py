"""Run one controller on one plant (lumped model or pigsim) and score the run.

The same loop, reward and metrics are used for both plants and all
controllers, so the numbers in metrics.json are directly comparable.
"""

import numpy as np

from . import scenario as S

# reward weights (also used to tune the baselines, so all three controllers
# optimise the same objective)
W_EXCESS = 5.0      # per (m/s x s) above the limit, divided by V_CAP
W_VIOL = 0.5        # per control step in which the limit is exceeded at all
W_LATE = 0.1        # per second past T_TARGET
W_VALVE = 0.1       # per full DU_MAX of valve movement
W_UNARRIVED = 50.0  # times the unfinished fraction of the line, at T_MAX


def step_reward(stats, t_before, t_after, du, v_cap=S.V_CAP):
    r = -W_EXCESS * stats["excess"] / S.V_CAP
    if stats["vmax_lim"] > v_cap:
        r -= W_VIOL
    late = max(t_after - max(t_before, S.T_TARGET), 0.0)
    r -= W_LATE * late
    r -= W_VALVE * abs(du) / S.DU_MAX
    return r


def timeout_penalty(s):
    return -W_UNARRIVED * max(S.S_ARRIVE - s, 0.0) / S.L


def progress_reward(s_before, s_after):
    """Dense form of timeout_penalty used by the gymnasium environment."""
    return W_UNARRIVED * (min(s_after, S.S_ARRIVE) - s_before) / S.L


def rate_limit(u_cmd, u_prev):
    return float(np.clip(np.clip(u_cmd, u_prev - S.DU_MAX, u_prev + S.DU_MAX), 0.0, 1.0))


def run(plant, controller, mass, F_fric, t_max=S.T_MAX):
    """Closed-loop run.  controller.reset() -> initial opening, controller(obs) -> opening.

    Controllers flagged ``rate_limited = False`` (the fixed opening, open-loop
    schedules) are applied as given; all others are rate-limited to DU_MAX.
    """
    u0 = controller.reset()
    obs = plant.reset(mass, F_fric, u0=u0)
    u = u0
    ret, travel = 0.0, 0.0
    while not plant.arrived and plant.t < t_max - 1e-9:
        u_cmd = controller(obs)
        u_new = rate_limit(u_cmd, u) if getattr(controller, "rate_limited", True) \
            else float(np.clip(u_cmd, 0.0, 1.0))
        du = u_new - u
        t0 = plant.t
        obs = plant.step(u_new)
        ret += step_reward(plant.step_stats, t0, plant.t, du)
        travel += abs(du)
        u = u_new
    if not plant.arrived:
        ret += timeout_penalty(plant.s)
    tr = plant.trace()
    m = metrics(tr)
    m.update(ret=ret, valve_travel=travel, mass=float(mass), F_fric=float(F_fric))
    return m, tr


def metrics(tr):
    t, s, V = (np.asarray(tr[k]) for k in ("t", "s", "V"))
    arrived = bool(s[-1] >= S.S_ARRIVE - 1e-6)
    if arrived:
        i = int(np.argmax(s >= S.S_ARRIVE - 1e-6))
        t_arr = float(np.interp(S.S_ARRIVE, s[max(i - 1, 0):i + 1], t[max(i - 1, 0):i + 1])) \
            if i > 0 else float(t[i])
    else:
        t_arr = float("nan")
    w = s >= S.S_LIMIT
    v_lim = float(V[w].max()) if w.any() else 0.0
    return {
        "arrived": arrived,
        "t_arrive": t_arr,
        "late": (not arrived) or t_arr > S.T_TARGET,
        "v_max_limited": v_lim,           # peak speed where the limit applies
        "violation": v_lim > S.V_CAP,
        "v_max_launch": float(V.max()),   # whole run, launch surge included
    }
