# pigsim surrogate as an MCP server

`mcp_server/server.py` exposes the stage-2 surrogate (MLP ensemble + arrival
classifier, `surrogate/models/`) and the pigsim solver itself as four
[Model Context Protocol](https://modelcontextprotocol.io) tools, so an LLM
client such as Claude can query, check and invert the model. It is a FastMCP
server (MCP Python SDK 1.x) that talks over **stdio**.

| Tool | What it does | Cost |
|---|---|---|
| `describe_model` | input ranges and units, outputs, accuracy, known limits | instant |
| `predict(p_in_bar, mass_kg, F_fric_kN, L_m)` | t_arrive and v_max (mean, std), P(arrive), distance to the start threshold, warnings for out-of-range / near-threshold inputs | ~1 ms |
| `run_pigsim(p_in_bar, mass_kg, F_fric_kN, L_m, timeout_s=120)` | runs the real solver once in a child process, stops at `timeout_s` | ~10 s |
| `inverse_design(L_m, t_max_s, v_max_cap_m_s=None, std_penalty=0.05, k_sigma=2, n_starts=128, verify_with_pigsim=True)` | lowest inlet pressure (pig mass and friction free in range) that meets the targets; classifier and range constraints, std penalty, `k_sigma` margin; re-runs the best design in pigsim | ~30–60 s |

Units at the interface: bar (absolute), kg, kN, m, s. All model loading goes
through `surrogate/registry.py`.

## Install

```bash
pip install -r requirements.txt        # includes mcp[cli]<2 (FastMCP API)
python -m pytest tests/test_mcp_server.py -m "not slow"     # ~10 s
```

The MCP SDK 2.x renamed `FastMCP` to `MCPServer`; this server pins `mcp<2`.

## Connect to Claude Code

```bash
claude mcp add pigsim -- python /absolute/path/to/pigsim-surrogate/mcp_server/server.py
claude mcp list                        # should show "pigsim" as connected
```

- Everything after `--` is the command that starts the server. Use the Python
  that has the requirements installed (e.g. `/path/to/venv/bin/python`).
- `--scope local` (default) is for you in this project only, `--scope user` for all your
  projects, `--scope project` writes a shared `.mcp.json` into the project:

```json
{
  "mcpServers": {
    "pigsim": {
      "type": "stdio",
      "command": "python",
      "args": ["/absolute/path/to/pigsim-surrogate/mcp_server/server.py"]
    }
  }
}
```

Inside Claude Code, `/mcp` shows the server and its tools.

## Connect to Claude Desktop

Edit the config file and fully quit and reopen Claude Desktop:

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "pigsim": {
      "command": "/absolute/path/to/python",
      "args": ["/absolute/path/to/pigsim-surrogate/mcp_server/server.py"]
    }
  }
}
```

Use absolute paths; Claude Desktop does not start the server from the
repository folder.

## Demo

```bash
python mcp_server/demo_client.py               # L = 2000 m, t_arrive <= 300 s, ~1-2 min
```

The client starts the server over stdio (as Claude would), calls
`describe_model`, then `inverse_design` for "lowest inlet pressure that gets the
pig through 2000 m within 300 s" (again with a wider margin if pigsim says the
first answer misses), and finally checks whether 0.2 bar less would also do,
with `predict` and `run_pigsim`. The full transcript is in
[`mcp_server/demo_log.md`](mcp_server/demo_log.md) (`demo_log.json` for machines).

Result of the committed run (L = 2000 m, t_arrive <= 300 s):

| step | design | surrogate t_arrive | pigsim t_arrive | target met |
|---|---|---|---|---|
| `inverse_design` (k_sigma = 2) | p_in 2.393 bar, 134 kg, 1.03 kN | 293.8 +- 3.1 s | 300.02 s | no, by 0.02 s |
| `inverse_design` (k_sigma = 3), asked again | p_in 2.408 bar, 126 kg, 1.04 kN | 290.5 s | 296.9 s | yes |
| `predict` / `run_pigsim`, 0.2 bar lower | p_in 2.208 bar | 354.4 s | 357.7 s | no |

The first answer sits exactly on the constraint boundary (minimising p_in pushes it
there) and the surrogate was 2 % optimistic, so pigsim missed by 0.02 s. The
client therefore re-asks with a wider margin, which pigsim then confirms. The
0.2 bar lower check shows the answer is close to the true minimum. This is why
`inverse_design` re-runs pigsim by default.

Example questions for Claude once connected:

- "What inputs does the pigsim surrogate take, and where is it unreliable?"
- "For a 2.5 km line, what is the lowest inlet pressure that gets the pig out
  within 10 minutes with a peak speed under 35 m/s? Check it with pigsim."
- "Predict the arrival time at 2.2 bar, 110 kg, 3.5 kN, 1750 m. Should I trust it?"

## Limits

The server returns the same caveats in `describe_model` and in per-call
`warnings`: the model only covers the generic test pipeline of
`surrogate/pipeline.py`; near the start threshold (p_in within 0.3 bar of
1.5 bar + 1.2 F_fric / A) the surrogate errs by 1–13 % and its std is too small;
the std is model uncertainty only. Treat `run_pigsim` as the reference.
