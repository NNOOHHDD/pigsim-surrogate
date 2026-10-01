"""Plain functions behind the four MCP tools (importable and testable without MCP).

Units at this interface are the ones people use: bar (absolute), kg, kN, m, s.
Models come from surrogate/registry.py (stage 2, wide box).
"""

import multiprocessing as mp
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from surrogate import registry  # noqa: E402
from surrogate.generate_wide_data import T_END, WIDE_RANGES  # noqa: E402

NEAR_THRESHOLD_BAR = 0.3

LIMITS = [
    "Trained on a made-up short horizontal gas line (D = 0.30 m, nitrogen, outlet valve "
    "into 1.5 bar); valid only inside the input ranges given here.",
    "The surrogate imitates pigsim; it carries none of pigsim's own model error "
    "(about +-6 % in arrival time against the reproduced paper).",
    "'Arrival' means reaching 0.999 L before 2000 s. Below the analytic start threshold "
    "p_in = 1.5 bar + 1.2 F_fric / A the pig never moves.",
    "Within 0.3 bar of the start threshold the regressors err by 1-13 % and the "
    "ensemble std underestimates the error by up to ~9x (RESULTS.md 9.4, 10.3).",
    "The std is the spread of 5 networks (model uncertainty only); it is not a calibrated "
    "probability, and v_max labels carry ~1 % numerical noise.",
]


def _x_si(p_in_bar, mass_kg, F_fric_kN, L_m):
    return np.array([p_in_bar * 1e5, mass_kg, F_fric_kN * 1e3, L_m], dtype=float)


def _range_warnings(x):
    out = []
    for k, v in zip(registry.INPUTS, x):
        lo, hi = WIDE_RANGES[k]
        if not lo <= v <= hi:
            out.append(f"{k} = {v:g} {registry.UNITS[k]} is outside the training range "
                       f"[{lo:g}, {hi:g}]: extrapolation, do not trust the numbers.")
    return out


def describe_model():
    rng = {
        "p_in_bar": [WIDE_RANGES["p_in"][0] / 1e5, WIDE_RANGES["p_in"][1] / 1e5],
        "mass_kg": list(WIDE_RANGES["mass"]),
        "F_fric_kN": [WIDE_RANGES["F_fric"][0] / 1e3, WIDE_RANGES["F_fric"][1] / 1e3],
        "L_m": list(WIDE_RANGES["L"]),
    }
    return {
        "what": "Surrogate of pigsim (transient pig motion, Nieckele et al. IPC2000-175) for a "
                "generic gas pipeline: 5-member MLP ensemble + arrival classifier.",
        "inputs": {
            "p_in_bar": "inlet (drive) pressure, absolute, reached by a 10 s ramp from 1.5 bar",
            "mass_kg": "pig mass",
            "F_fric_kN": "dynamic friction force (static = 1.2 x dynamic)",
            "L_m": "pipeline length",
        },
        "input_ranges": rng,
        "outputs": {"t_arrive_s": "time to reach 0.999 L", "v_max_m_s": "peak pig speed",
                    "p_arrive": "classifier probability that the pig arrives before 2000 s"},
        "known_limits": LIMITS,
        "accuracy": "random test split: t_arrive ~0.25 %, v_max ~0.2 % mean error; "
                    "near the start threshold 1-13 % (see surrogate/RESULTS.md)",
    }


def predict(p_in_bar, mass_kg, F_fric_kN, L_m):
    x = _x_si(p_in_bar, mass_kg, F_fric_kN, L_m)
    r = {k: float(v[0]) for k, v in registry.predict(x[None, :]).items()}
    warn = _range_warnings(x)
    margin = r["start_margin"] / 1e5
    if margin <= 0:
        warn.append(f"p_in is {-margin:.3f} bar BELOW the analytic start threshold: the pig "
                    "will not move; t_arrive / v_max below are meaningless extrapolations.")
    elif margin < NEAR_THRESHOLD_BAR:
        warn.append(f"Only {margin:.3f} bar above the start threshold: expect 1-13 % error "
                    "and a std that is too small.")
    if r["p_arrive"] < 0.5:
        warn.append(f"Classifier: P(arrive) = {r['p_arrive']:.2f} -> probably does not arrive.")
    return {"inputs": {"p_in_bar": p_in_bar, "mass_kg": mass_kg, "F_fric_kN": F_fric_kN, "L_m": L_m},
            "t_arrive_s": {"mean": r["t_arrive"], "std": r["t_arrive_std"]},
            "v_max_m_s": {"mean": r["v_max"], "std": r["v_max_std"]},
            "p_arrive": r["p_arrive"], "start_margin_bar": margin, "warnings": warn}


def _sim_worker(x, q):
    from surrogate.generate_wide_data import simulate_ext
    q.put(simulate_ext(*x))


def run_pigsim(p_in_bar, mass_kg, F_fric_kN, L_m, timeout_s=120.0):
    """Run the real solver in a child process; give up after timeout_s seconds.

    "spawn" starts a fresh interpreter (~2 s overhead).  A forked child would
    inherit the thread pools torch has just used in this process and can hang.
    """
    x = _x_si(p_in_bar, mass_kg, F_fric_kN, L_m)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    proc = ctx.Process(target=_sim_worker, args=(tuple(x), q))
    t0 = time.perf_counter()
    proc.start()
    r = None
    try:
        r = q.get(timeout=timeout_s)        # read before join: a full pipe would block the child
    except Exception:
        pass
    proc.join(5)
    if proc.is_alive():
        proc.terminate()
        proc.join()
    if r is None:
        status = "timeout" if time.perf_counter() - t0 >= timeout_s else "error"
        return {"status": status, "timeout_s": timeout_s,
                "warnings": [f"pigsim did not finish within {timeout_s:g} s (a run takes ~10 s)."
                             if status == "timeout" else "pigsim failed (no result)."]}
    return {"status": "ok", "arrived": bool(r["arrived"]), "started": bool(r["started"]),
            "t_arrive_s": None if not r["arrived"] else float(r["t_arrive"]),
            "v_max_m_s": float(r["v_max"]), "s_end_m": float(r["s_end"]),
            "wall_s": round(time.perf_counter() - t0, 2), "t_end_s": T_END,
            "warnings": _range_warnings(x)}


def inverse_design(L_m, t_max_s, v_max_cap_m_s=None, std_penalty=0.05, k_sigma=2.0,
                   n_starts=128, iters=1000, verify_with_pigsim=True, timeout_s=120.0):
    """Lowest p_in (with mass and F_fric free in their ranges) that meets the targets.

    minimise p_in [bar] + std_penalty * (relative ensemble std in %)
    s.t. t + k std_t <= t_max, v + k std_v <= v_cap, P(arrive) >= 0.95, inputs in range.
    """
    from surrogate.inverse_design import DESIGN, P_MIN_ARRIVE, optimise, pick
    lo, hi = WIDE_RANGES["L"]
    if not lo <= L_m <= hi:
        return {"status": "error", "warnings": [f"L_m must be in [{lo:g}, {hi:g}] m."]}
    v_cap = 1e6 if v_max_cap_m_s is None else float(v_max_cap_m_s)
    t0 = time.perf_counter()
    r = optimise(registry.torch_surrogate(), L_m, t_max_s, v_cap, True, std_penalty, k_sigma,
                 n_starts=n_starts, iters=iters)
    ok = ((r["t"] + k_sigma * r["t_sd"] <= t_max_s * 1.001)
          & (r["v"] + k_sigma * r["v_sd"] <= v_cap * 1.001)
          & (r["p_arrive"] >= P_MIN_ARRIVE - 1e-3))
    out = {"problem": {"L_m": L_m, "t_max_s": t_max_s, "v_max_cap_m_s": v_max_cap_m_s,
                       "std_penalty": std_penalty, "k_sigma": k_sigma, "design_variables": DESIGN},
           "n_starts": int(n_starts), "n_feasible": int(ok.sum()),
           "optimise_s": round(time.perf_counter() - t0, 1), "warnings": []}
    if not ok.any():
        out["status"] = "infeasible"
        out["warnings"].append("No start met all constraints: the target may be impossible "
                               "inside the training range.")
        return out
    cands = []
    for i in pick(r, ok, k=3):
        x = r["x"][i]
        cands.append({"p_in_bar": float(x[0] / 1e5), "mass_kg": float(x[1]), "F_fric_kN": float(x[2] / 1e3),
                      "predicted": {"t_arrive_s": float(r["t"][i]), "t_arrive_std": float(r["t_sd"][i]),
                                    "v_max_m_s": float(r["v"][i]), "v_max_std": float(r["v_sd"][i]),
                                    "p_arrive": float(r["p_arrive"][i])},
                      "start_margin_bar": float((x[0] - registry.p_start(x[2])) / 1e5)})
    out["status"] = "ok"
    out["best"], out["alternatives"] = cands[0], cands[1:]
    if cands[0]["start_margin_bar"] < NEAR_THRESHOLD_BAR:
        out["warnings"].append("Best design is within 0.3 bar of the start threshold, where the "
                               "surrogate is least accurate: rely on the pigsim check.")
    if verify_with_pigsim:
        b = cands[0]
        sim = run_pigsim(b["p_in_bar"], b["mass_kg"], b["F_fric_kN"], L_m, timeout_s)
        if sim["status"] == "ok":
            sim["meets_targets"] = bool(sim["arrived"] and sim["t_arrive_s"] <= t_max_s
                                        and (v_max_cap_m_s is None or sim["v_max_m_s"] <= v_max_cap_m_s))
            if sim["arrived"]:
                sim["t_arrive_rel_err_%"] = 100 * (b["predicted"]["t_arrive_s"] - sim["t_arrive_s"]) / sim["t_arrive_s"]
                sim["v_max_rel_err_%"] = 100 * (b["predicted"]["v_max_m_s"] - sim["v_max_m_s"]) / sim["v_max_m_s"]
        out["pigsim_check"] = sim
    return out
