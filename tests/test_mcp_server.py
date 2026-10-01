"""Tests for the MCP tools (mcp_server/).

    python -m pytest tests/test_mcp_server.py -m "not slow"     # ~10 s
    python -m pytest tests/test_mcp_server.py                   # + pigsim runs, ~1 min
"""

import asyncio
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp_server import tools  # noqa: E402


def test_describe_model_lists_ranges_and_limits():
    d = tools.describe_model()
    assert set(d["input_ranges"]) == {"p_in_bar", "mass_kg", "F_fric_kN", "L_m"}
    assert d["input_ranges"]["p_in_bar"] == [1.6, 8.0]
    assert len(d["known_limits"]) >= 3


def test_predict_inside_range_is_quiet_and_sane():
    r = tools.predict(5.5, 110, 3.5, 1750)
    assert r["warnings"] == []
    assert r["t_arrive_s"]["mean"] == pytest.approx(99.4, rel=0.01)   # pigsim: 99.4 s
    assert r["p_arrive"] > 0.99


def test_predict_warns_out_of_range_and_below_threshold():
    assert any("outside the training range" in w for w in tools.predict(9.0, 110, 3.5, 1750)["warnings"])
    r = tools.predict(1.8, 110, 3.5, 1750)              # threshold ~2.09 bar
    assert r["start_margin_bar"] < 0
    assert any("BELOW the analytic start threshold" in w for w in r["warnings"])


def test_inverse_design_without_pigsim():
    r = tools.inverse_design(2000, 300, n_starts=32, iters=300, verify_with_pigsim=False)
    assert r["status"] == "ok"
    b = r["best"]
    assert b["predicted"]["t_arrive_s"] + 2 * b["predicted"]["t_arrive_std"] <= 300 * 1.001
    assert b["predicted"]["p_arrive"] >= 0.949


def test_stdio_roundtrip():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def go():
        params = StdioServerParameters(command=sys.executable,
                                       args=[os.path.join(ROOT, "mcp_server", "server.py")])
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                res = await s.call_tool("predict", {"p_in_bar": 5.5, "mass_kg": 110,
                                                    "F_fric_kN": 3.5, "L_m": 1750})
                return names, json.loads(res.content[0].text)

    names, out = asyncio.run(go())
    assert names == {"describe_model", "predict", "run_pigsim", "inverse_design"}
    assert out["t_arrive_s"]["mean"] > 0


@pytest.mark.slow
def test_run_pigsim_matches_stage1_point():
    r = tools.run_pigsim(5.5, 110, 3.5, 1750, timeout_s=120)
    assert r["status"] == "ok" and r["arrived"]
    assert r["t_arrive_s"] == pytest.approx(99.4, rel=0.01)    # RESULTS.md 1: 99.4 s


@pytest.mark.slow
def test_run_pigsim_timeout():
    r = tools.run_pigsim(2.5, 110, 1.0, 3000, timeout_s=0.5)
    assert r["status"] == "timeout"
