"""The optional hooks added to surrogate/pipeline.build() must not change results.

    python -m pytest tests/test_pipeline_hooks.py          # ~40 s

1. Default arguments reproduce two stage-1 runs stored in surrogate/data.csv
   (t_arrive and v_max to 1e-9 relative), so the stage-1 data is still valid.
2. Hooks that return the nominal values give the exact same history as no hooks.
3. A hook that changes the conditions does change the result (it is wired up).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from surrogate import pipeline as P  # noqa: E402

pytestmark = pytest.mark.slow


def _run(x, **hooks):
    sol = P.build(*x, **hooks)
    sol.run(2000.0, dt0=0.02, dt_max=1.0, cfl_pig=0.02, s_stop=0.999 * x[3], verbose=False)
    return sol.history()


@pytest.mark.parametrize("row", [0, 1])
def test_default_reproduces_stage1_data(row):
    d = pd.read_csv(os.path.join(ROOT, "surrogate", "data.csv")).sort_values("L").iloc[row]
    r = P.simulate(d["p_in"], d["mass"], d["F_fric"], d["L"])
    assert r["t_arrive"] == pytest.approx(d["t_arrive"], rel=1e-9)
    assert r["v_max"] == pytest.approx(d["v_max"], rel=1e-9)


def test_identity_hooks_give_identical_history():
    x = (5.0e5, 80.0, 3000.0, 600.0)
    h0 = _run(x)
    h1 = _run(x, p_in_of_t=lambda t, p: p, chi_of_t=lambda t: 1.0,
              F_fric_of_s=lambda s: x[2])
    for k in ("t", "s_pig", "V_pig", "p1", "p2"):
        np.testing.assert_array_equal(h0[k], h1[k])


def test_hook_changes_result():
    x = (5.0e5, 80.0, 3000.0, 600.0)
    h0 = _run(x)
    h1 = _run(x, chi_of_t=lambda t: 0.3 if t > 5.0 else 1.0)
    assert h1["t"][-1] > 1.05 * h0["t"][-1]      # a throttled outlet slows the pig
