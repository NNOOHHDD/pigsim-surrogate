"""Stage 2a: classical baselines vs the MLP ensemble on the stage-1 split.

    python surrogate/baselines.py          # ~1 min

Same data (data.csv), same split (numpy default_rng(0) -> 454 / 97 / 99) and
the same output transform (log t_arrive) as train_surrogate.py, so the numbers
are directly comparable with metrics.json.

    linear        ordinary least squares on the 4 z-scored inputs
    power law     least squares on log inputs (= degree-1 polynomial in log x)
    poly2, poly3  full degree-2 / degree-3 polynomial in the z-scored log inputs
    GP            Gaussian process, ARD Matern-5/2 + white noise, one per output
    MLP x5        the trained stage-1 ensemble (models/ensemble.pt)

Polynomials and the GP are fitted on the 454 training points only; the
validation set is not needed by them (no early stopping), so it is unused.
Output: baselines.json, figures/baselines_error.png, figures/gp_uncertainty.png
"""

import json
import os
import sys
import time
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate.train_surrogate import (  # noqa: E402
    BLUE, FIG, GRID, HERE, INK, MUTED, ORANGE, OUTPUTS, _style, from_target,
    load_and_split, load_ensemble, to_target,
)

AQUA, YELLOW = "#1baf7a", "#eda100"


# ---------------------------------------------------------------------------
# models.  Every model maps physical inputs -> physical outputs, with the
# same log transform on t_arrive as the MLP, so only the regressor differs.
# ---------------------------------------------------------------------------
class PolyModel:
    """Least squares on polynomial features.  degree=1, log_x=False is plain linear."""

    def __init__(self, degree, log_x):
        self.degree, self.log_x = degree, log_x

    def fit(self, X, Y):
        steps = [FunctionTransformer(np.log)] if self.log_x else []
        steps += [StandardScaler(), PolynomialFeatures(self.degree, include_bias=False),
                  LinearRegression()]
        self.pipe = make_pipeline(*steps).fit(X, to_target(Y))
        return self

    def n_params(self):
        lr = self.pipe[-1]
        return int(lr.coef_.size + lr.intercept_.size)

    def predict(self, X):
        return from_target(self.pipe.predict(X)), None


class GPModel:
    """One GP per output on z-scored log inputs.

    Kernel: C * Matern(nu=2.5, one length scale per input) + White.
    The white-noise term lets the GP attribute part of the scatter to label
    noise instead of forcing the mean through every point.  predict() returns
    the posterior std *including* that noise term, i.e. the spread expected
    for a fresh pigsim label, which is what the coverage check compares with.

    t_arrive is modelled in log space; its std is mapped back with the delta
    method, std[t] ~= t * std[log t] (fine here because std[log t] << 1).
    """

    def __init__(self, restarts=3, seed=0):
        self.restarts, self.seed = restarts, seed

    def fit(self, X, Y):
        self.xs = make_pipeline(FunctionTransformer(np.log), StandardScaler()).fit(X)
        Z, T = self.xs.transform(X), to_target(Y)
        self.gps = []
        for j in range(T.shape[1]):
            k = (ConstantKernel(1.0, (1e-3, 1e3))
                 * Matern(length_scale=np.ones(X.shape[1]), length_scale_bounds=(1e-2, 1e3), nu=2.5)
                 + WhiteKernel(1e-4, (1e-9, 1e-1)))
            gp = GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=self.restarts,
                                          random_state=self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(Z, T[:, j])
            self.gps.append(gp)
        return self

    def predict(self, X):
        Z = self.xs.transform(X)
        out = [gp.predict(Z, return_std=True) for gp in self.gps]
        M = np.stack([m for m, _ in out], axis=1)
        S = np.stack([s for _, s in out], axis=1)
        mu = from_target(M)
        sd = S.copy()
        for j, name in enumerate(OUTPUTS):
            if name == "t_arrive":
                sd[:, j] = mu[:, j] * S[:, j]
        return mu, sd

    def noise_fraction(self):
        """Fitted white-noise std as a fraction of the output std (normalised space)."""
        return [float(np.sqrt(gp.kernel_.k2.noise_level)) for gp in self.gps]


class MLPModel:
    def __init__(self):
        self.ens = load_ensemble()

    def predict(self, X):
        mu, sd, _ = self.ens.predict(X)
        return mu, sd


# ---------------------------------------------------------------------------
def score(y, mu, sd=None):
    err = mu - y
    rel = np.abs(err) / np.abs(y)
    out = {
        "R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(100 * rel.mean()),
        "p95_rel_%": float(100 * np.percentile(rel, 95)),
        "max_rel_%": float(100 * rel.max()),
        "max_abs_err": float(np.abs(err).max()),
    }
    if sd is not None:
        out.update({
            "mean_std": float(sd.mean()),
            "frac_within_1std": float(np.mean(np.abs(err) <= sd)),
            "frac_within_2std": float(np.mean(np.abs(err) <= 2 * sd)),
            "corr_std_abs_err": float(np.corrcoef(sd, np.abs(err))[0, 1]),
        })
    return out


def plot_errors(results, Yte, path):
    """Strip plot of |relative error| per model and output, log scale."""
    import matplotlib.pyplot as plt
    names = list(results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    rng = np.random.default_rng(0)
    for j, (ax, out) in enumerate(zip(axes, OUTPUTS)):
        for i, n in enumerate(names):
            rel = 100 * np.abs(results[n]["pred"][:, j] - Yte[:, j]) / Yte[:, j]
            rel = np.maximum(rel, 1e-4)
            x = i + rng.uniform(-0.18, 0.18, len(rel))
            col = BLUE if n.startswith("MLP") else (ORANGE if n == "GP" else MUTED)
            ax.scatter(x, rel, s=9, color=col, alpha=0.55, lw=0)
            med = np.median(rel)
            ax.plot([i - 0.3, i + 0.3], [med, med], color=INK, lw=1.6)
            ax.text(i, 1.35 * rel.max(), f"{np.mean(rel):.2g} %", ha="center",
                    fontsize=8, color=INK)
        ax.set_yscale("log")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9, color=INK)
        ax.set_title(out, color=INK, loc="left", fontsize=11)
        _style(ax)
    axes[0].set_ylabel("|prediction − pigsim| / pigsim  [%]", color=INK)
    fig.suptitle("Test set (99 runs): relative error per model. "
                 "Bar = median, number = mean (MAPE)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_uncertainty(Yte, results, path):
    """|error| vs predicted std for GP and MLP ensemble, same axes per output."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for j, (ax, out) in enumerate(zip(axes, OUTPUTS)):
        lo, hi = np.inf, 0.0
        for n, col in (("MLP x5", BLUE), ("GP", ORANGE)):
            r = results[n]
            err = np.abs(r["pred"][:, j] - Yte[:, j])
            s = r["std"][:, j]
            m = r["metrics"][out]
            ax.scatter(s, np.maximum(err, 1e-5), s=20, color=col, edgecolor="white", lw=0.5,
                       zorder=3, label=f"{n}: {100*m['frac_within_2std']:.0f} % inside 2 std, "
                                       f"corr {m['corr_std_abs_err']:.2f}")
            lo, hi = min(lo, s.min()), max(hi, s.max(), err.max())
        xs = np.array([lo / 2, hi * 2])
        ax.fill_between(xs, 1e-5, 2 * xs, color=GRID, alpha=0.6, lw=0, label="|error| ≤ 2 std")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xs)
        ax.set_ylim(max(1e-4 * hi, 1e-5), 4 * hi)
        unit = "s" if out == "t_arrive" else "m/s"
        ax.set_xlabel(f"predicted std [{unit}]", color=INK)
        ax.set_ylabel(f"|prediction − pigsim| [{unit}]", color=INK)
        ax.set_title(out, color=INK, loc="left", fontsize=11)
        _style(ax)
        ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle("Does the predicted std cover the actual error? (test set, 99 runs)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    df, X, Y, split = load_and_split(os.path.join(HERE, "data.csv"), np.random.default_rng(0))
    tr, te = split["train"], split["test"]
    print(f"split: {len(tr)} train / {len(split['val'])} val / {len(te)} test")

    models = {
        "linear": PolyModel(1, log_x=False),
        "power law": PolyModel(1, log_x=True),
        "poly2 (log x)": PolyModel(2, log_x=True),
        "poly3 (log x)": PolyModel(3, log_x=True),
        "GP": GPModel(),
    }
    results = {}
    for name, m in models.items():
        t0 = time.perf_counter()
        m.fit(X[tr], Y[tr])
        t_fit = time.perf_counter() - t0
        mu, sd = m.predict(X[te])
        t1 = time.perf_counter()
        m.predict(np.repeat(X[te], 101, axis=0)[:10000])
        t_pred = (time.perf_counter() - t1) / 10000
        n_par = m.n_params() if isinstance(m, PolyModel) else len(tr) * 2
        results[name] = {"pred": mu, "std": sd, "fit_s": t_fit, "pred_s": t_pred,
                         "n_params": n_par}
        print(f"{name:14s} fit {t_fit:6.2f} s")

    mlp = MLPModel()
    mu, sd = mlp.predict(X[te])
    t1 = time.perf_counter()
    mlp.predict(np.repeat(X[te], 101, axis=0)[:10000])
    with open(os.path.join(HERE, "metrics.json")) as f:
        t_train = json.load(f)["train_time_s"]
    results["MLP x5"] = {"pred": mu, "std": sd, "fit_s": t_train,
                         "pred_s": (time.perf_counter() - t1) / 10000, "n_params": 5 * 4610}

    for name, r in results.items():
        r["metrics"] = {o: score(Y[te, j], r["pred"][:, j],
                                 None if r["std"] is None else r["std"][:, j])
                        for j, o in enumerate(OUTPUTS)}

    print(f"\n{'model':14s} {'':9s} {'R2':>9s} {'MAPE%':>7s} {'p95%':>6s} {'max%':>6s} "
          f"{'in2sd':>6s} {'corr':>5s}")
    for name, r in results.items():
        for o in OUTPUTS:
            m = r["metrics"][o]
            extra = (f"{100*m['frac_within_2std']:5.0f}% {m['corr_std_abs_err']:5.2f}"
                     if "frac_within_2std" in m else "")
            print(f"{name:14s} {o:9s} {m['R2']:9.5f} {m['MAPE_%']:7.3f} "
                  f"{m['p95_rel_%']:6.2f} {m['max_rel_%']:6.2f} {extra}")

    gp = models["GP"]
    for o, gpj, nf in zip(OUTPUTS, gp.gps, gp.noise_fraction()):
        print(f"GP {o}: kernel {gpj.kernel_}  -> noise std / output std = {nf:.4f}")

    summary = {
        "split": {k: int(len(v)) for k, v in split.items()},
        "models": {n: {"metrics": r["metrics"], "fit_s": round(r["fit_s"], 2),
                       "predict_s_per_point_batched": r["pred_s"], "n_params": r["n_params"]}
                   for n, r in results.items()},
        "gp_kernels": {o: str(g.kernel_) for o, g in zip(OUTPUTS, gp.gps)},
        "gp_noise_std_over_output_std": dict(zip(OUTPUTS, gp.noise_fraction())),
    }
    with open(os.path.join(HERE, "baselines.json"), "w") as f:
        json.dump(summary, f, indent=2)

    os.makedirs(FIG, exist_ok=True)
    plot_errors(results, Y[te], os.path.join(FIG, "baselines_error.png"))
    plot_uncertainty(Y[te], results, os.path.join(FIG, "gp_uncertainty.png"))
    print("-> baselines.json, figures/baselines_error.png, figures/gp_uncertainty.png")


if __name__ == "__main__":
    main()
