"""Run reproducible numerical-sensitivity checks for the two paper cases."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cases.case1_riser import build as build_case1  # noqa: E402
from cases.case2_area_change import build as build_case2  # noqa: E402
from plots.paper_reference import P_F3, P_F7_NON, P_F10, P_T1, P_T2  # noqa: E402


def crossing(t, s, target):
    i = int(np.argmax(s >= target))
    if s[i] < target:
        return np.nan
    if i == 0:
        return float(t[0])
    return float(np.interp(target, s[i - 1:i + 1], t[i - 1:i + 1]))


def stats(model, reference, conditions):
    model = np.asarray(model, float)
    reference = np.asarray(reference, float)
    error = model - reference
    abs_error = np.abs(error)
    rel = abs_error / np.maximum(np.abs(reference), 1e-12)
    ia = int(np.nanargmax(abs_error))
    ir = int(np.nanargmax(rel))
    return {
        "n": int(len(reference)),
        "relative_rms_pct": float(100 * np.sqrt(np.nanmean(rel ** 2))),
        "max_abs_error": float(abs_error[ia]),
        "max_abs_condition": str(conditions[ia]),
        "max_abs_signed": float(error[ia]),
        "max_rel_error_pct": float(100 * rel[ir]),
        "max_rel_condition": str(conditions[ir]),
    }


def run_case1(N, dt_max, s0):
    sol = build_case1(N=N, s0=s0)
    sol.run(1400.0, dt0=1e-3, dt_max=dt_max, cfl_pig=0.01,
            dV_max=1.0, s_stop=1480.0, verbose=False)
    h = sol.history()
    st = sol.station_arrays()

    x = np.array([v[0] for v in P_F7_NON], float)
    ref = np.array([v[1] for v in P_F7_NON], float)
    vel = np.interp(x, h["s_pig"], h["V_pig"])

    t_ref = np.array([v[0] for v in P_F3], float)
    p_ref = np.array([v[1] for v in P_F3], float)
    p_model = np.interp(t_ref, st["t"], st["p"][:, 0] / 101325.0)

    arrival_ref = np.array([v[0] for v in P_T1], float)
    stations = np.array([v[1] for v in P_T1], float)
    arrival = [crossing(h["t"], h["s_pig"], value) for value in stations]
    panels = {
        "fig7_velocity": stats(vel, ref, [f"{v:g} m" for v in x]),
        "fig3_inlet_pressure": stats(p_model, p_ref, [f"{v:g} s" for v in t_ref]),
        "fig8_arrival_time": stats(arrival, arrival_ref,
                                   [f"{v:g} m" for v in stations]),
    }
    return {
        "case": "case1", "N": N, "dt_max": dt_max, "s0": s0,
        "finish_time_s": float(h["t"][-1]), "panels": panels,
        "combined_relative_rms_pct": float(np.sqrt(np.mean([
            value["relative_rms_pct"] ** 2 for value in panels.values()
        ]))),
    }


def run_case2(N, dt_max, transition, s0):
    sol = build_case2(N=N, transition=transition, s0=s0)
    sol.run(1700.0, dt0=0.02, dt_max=dt_max, cfl_pig=0.02,
            verbose=False)
    h = sol.history()

    x = np.array([v[0] for v in P_F10], float)
    ref = np.array([v[1] for v in P_F10], float)
    vel = np.interp(x * 1000.0, h["s_pig"], h["V_pig"])

    arrival_ref = np.array([v[0] for v in P_T2], float)
    stations = np.array([v[1] for v in P_T2], float) * 1000.0
    arrival = [crossing(h["t"], h["s_pig"], value) for value in stations]
    panels = {
        "fig10_velocity": stats(vel, ref, [f"{v:g} km" for v in x]),
        "fig11_arrival_time": stats(arrival, arrival_ref,
                                    [f"{v/1000:g} km" for v in stations]),
    }
    return {
        "case": "case2", "N": N, "dt_max": dt_max,
        "transition": transition, "s0": s0,
        "finish_time_s": float(h["t"][-1]), "panels": panels,
        "combined_relative_rms_pct": float(np.sqrt(np.mean([
            value["relative_rms_pct"] ** 2 for value in panels.values()
        ]))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["case1", "case2"], required=True)
    ap.add_argument("--N", type=int)
    ap.add_argument("--dt-max", type=float, default=2.0)
    ap.add_argument("--transition", type=float, default=20.0)
    ap.add_argument("--s0", type=float)
    ap.add_argument("--output")
    args = ap.parse_args()
    if args.case == "case1":
        result = run_case1(args.N or 200, args.dt_max,
                           10.0 if args.s0 is None else args.s0)
    else:
        result = run_case2(args.N or 70, args.dt_max, args.transition,
                           5.0 if args.s0 is None else args.s0)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
