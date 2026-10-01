"""MCP server (stdio) for the pigsim surrogate.

    python mcp_server/server.py            # speaks MCP over stdin/stdout

Tools: describe_model, predict, run_pigsim, inverse_design.  Connection to
Claude Desktop / Claude Code: README_mcp.md.  Nothing may be printed to stdout
here -- stdout carries the protocol.
"""

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_server import tools  # noqa: E402

mcp = FastMCP("pigsim-surrogate")


@mcp.tool()
def describe_model() -> dict:
    """Input ranges and units, outputs, accuracy and known limits of the pigsim surrogate."""
    return tools.describe_model()


@mcp.tool()
def predict(p_in_bar: float, mass_kg: float, F_fric_kN: float, L_m: float) -> dict:
    """Surrogate prediction of pig arrival time and peak speed (mean, std) plus the
    probability that the pig arrives. Warns when inputs are out of range or near the
    start threshold. Takes ~1 ms; use run_pigsim to confirm important answers."""
    return tools.predict(p_in_bar, mass_kg, F_fric_kN, L_m)


@mcp.tool()
def run_pigsim(p_in_bar: float, mass_kg: float, F_fric_kN: float, L_m: float,
               timeout_s: float = 120.0) -> dict:
    """Run the real pigsim solver once (~10 s) for the same inputs; stops after timeout_s."""
    return tools.run_pigsim(p_in_bar, mass_kg, F_fric_kN, L_m, timeout_s)


@mcp.tool()
def inverse_design(L_m: float, t_max_s: float, v_max_cap_m_s: Optional[float] = None,
                   std_penalty: float = 0.05, k_sigma: float = 2.0, n_starts: int = 128,
                   verify_with_pigsim: bool = True, timeout_s: float = 120.0) -> dict:
    """Find the lowest inlet pressure (with pig mass and friction free in their ranges)
    that gets the pig through a pipe of length L_m within t_max_s (and under an optional
    speed cap). Uses the surrogate with an arrival-classifier constraint, a std penalty
    and a k_sigma safety margin, then re-runs the best design in pigsim."""
    return tools.inverse_design(L_m, t_max_s, v_max_cap_m_s, std_penalty, k_sigma, n_starts,
                                verify_with_pigsim=verify_with_pigsim, timeout_s=timeout_s)


if __name__ == "__main__":
    mcp.run()          # stdio transport
