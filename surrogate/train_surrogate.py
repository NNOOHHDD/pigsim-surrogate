"""Step 3: train a 5-member MLP ensemble on surrogate/data.csv and plot it.

    python surrogate/train_surrogate.py

Pipeline
    data.csv -> split train/val/test (70/15/15)
             -> scale inputs and outputs with *training-set* statistics
             -> train 5 small MLPs, each on its own bootstrap resample and
                with its own random initialisation
             -> ensemble mean = prediction, ensemble std = uncertainty
             -> metrics.json, figures/*.png, models/ensemble.pt
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from surrogate.pipeline import INPUTS, OUTPUTS, RANGES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
N_MEMBERS = 5
SEED = 0

# t_arrive spans 10 .. 500 s and scales roughly like L / speed, so it is
# learnt in log space (relative errors matter, and the target becomes smoother).
LOG_OUTPUT = {"t_arrive": True, "v_max": False}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class Scaler:
    """z-score: (x - mean) / std, fitted on the training set only.

    Inputs differ by 6 orders of magnitude (p_in ~ 1e5 Pa, mass ~ 1e2 kg);
    without scaling the first layer would be dominated by p_in and Adam's
    single learning rate could not suit every weight.
    """

    def __init__(self, a):
        self.mean = a.mean(axis=0)
        self.std = a.std(axis=0)

    def fwd(self, a):
        return (a - self.mean) / self.std

    def inv(self, z):
        return z * self.std + self.mean


def to_target(Y):
    Y = Y.copy()
    for j, name in enumerate(OUTPUTS):
        if LOG_OUTPUT[name]:
            Y[:, j] = np.log(Y[:, j])
    return Y


def from_target(Y):
    Y = Y.copy()
    for j, name in enumerate(OUTPUTS):
        if LOG_OUTPUT[name]:
            Y[..., j] = np.exp(Y[..., j])
    return Y


def load_and_split(path, rng):
    df = pd.read_csv(path)
    n_all = len(df)
    df = df[df["arrived"] == 1].dropna(subset=OUTPUTS).reset_index(drop=True)
    print(f"{n_all} runs in {os.path.basename(path)}, {len(df)} usable "
          f"(pig reached the outlet)")
    idx = rng.permutation(len(df))
    n_tr = int(0.70 * len(df))
    n_va = int(0.15 * len(df))
    split = {"train": idx[:n_tr], "val": idx[n_tr:n_tr + n_va],
             "test": idx[n_tr + n_va:]}
    X = df[INPUTS].to_numpy(float)
    Y = df[OUTPUTS].to_numpy(float)
    return df, X, Y, split


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def make_mlp(n_in, n_out, width=64, depth=2):
    """4 -> 64 -> 64 -> 2 with SiLU.

    nn.Sequential chains layers; nn.Linear is y = W x + b with W, b learnable.
    SiLU (x * sigmoid(x)) is smooth, so the surrogate is differentiable
    everywhere -- useful later for gradient-based optimisation or sensitivity.
    ~4.6k parameters for ~450 training points: small on purpose.
    """
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers)


def train_member(Xtr, Ytr, Xva, Yva, seed, epochs=3000, patience=300,
                 lr=3e-3, weight_decay=1e-5, batch=64):
    """Train one ensemble member; returns (model, loss history).

    Two sources of diversity between members:
      * torch.manual_seed(seed) -> different initial weights
      * bootstrap: train on len(Xtr) points drawn *with replacement*
    Early stopping keeps the weights with the lowest validation loss.
    """
    torch.manual_seed(seed)
    g = np.random.default_rng(seed)
    boot = g.integers(0, len(Xtr), len(Xtr))
    xt = torch.tensor(Xtr[boot], dtype=torch.float32)   # numpy -> torch tensor
    yt = torch.tensor(Ytr[boot], dtype=torch.float32)
    xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(Yva, dtype=torch.float32)

    model = make_mlp(xt.shape[1], yt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=100)
    loss_fn = nn.MSELoss()

    best, best_state, wait = float("inf"), None, 0
    hist = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()                                     # training mode
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), batch):                # mini-batches
            b = perm[i:i + batch]
            opt.zero_grad()                               # 1. clear old gradients
            loss = loss_fn(model(xt[b]), yt[b])           # 2. forward pass + loss
            loss.backward()                               # 3. backprop: d loss / d weights
            opt.step()                                    # 4. Adam weight update
            tot += loss.item() * len(b)

        model.eval()                                      # evaluation mode
        with torch.no_grad():                             # no gradient bookkeeping
            v = loss_fn(model(xv), yv).item()
        sched.step(v)
        hist["train"].append(tot / len(xt))
        hist["val"].append(v)

        if v < best - 1e-7:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait > patience:
                break
    model.load_state_dict(best_state)
    print(f"  member seed={seed}: {len(hist['val'])} epochs, best val MSE {best:.2e}")
    return model, hist


class Ensemble:
    def __init__(self, models, xs, ys):
        self.models, self.xs, self.ys = models, xs, ys

    def predict(self, X):
        """X in physical units -> (mean, std, all members), physical units.

        Each member's prediction is mapped back to physical units first and
        the statistics are taken there, so the std is in s and m/s.
        """
        x = torch.tensor(self.xs.fwd(X), dtype=torch.float32)
        with torch.no_grad():
            z = np.stack([m(x).numpy() for m in self.models])   # (members, n, 2)
        P = from_target(self.ys.inv(z))
        return P.mean(axis=0), P.std(axis=0), P


def load_ensemble(path=os.path.join(HERE, "models", "ensemble.pt")):
    """Rebuild the trained Ensemble from models/ensemble.pt (written by main)."""
    ck = torch.load(path, weights_only=False)
    models = []
    for sd in ck["state_dicts"]:
        m = make_mlp(len(ck["inputs"]), len(ck["outputs"]))
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    xs = Scaler.__new__(Scaler)
    xs.mean, xs.std = ck["x_mean"], ck["x_std"]
    ys = Scaler.__new__(Scaler)
    ys.mean, ys.std = ck["y_mean"], ck["y_std"]
    return Ensemble(models, xs, ys)


# ---------------------------------------------------------------------------
# metrics and plots
# ---------------------------------------------------------------------------
def metrics(y, mu, sd):
    err = mu - y
    return {
        "R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE_%": float(100 * np.mean(np.abs(err) / np.abs(y))),
        "max_abs_err": float(np.max(np.abs(err))),
        "mean_ens_std": float(np.mean(sd)),
        # if the std were a calibrated 1-sigma Gaussian error bar, ~95 %
        # of test points would lie inside +-2 std
        "frac_within_2std": float(np.mean(np.abs(err) <= 2 * sd)),
    }


BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LABEL = {"t_arrive": "arrival time  t_arrive [s]", "v_max": "peak pig speed  v_max [m/s]"}


def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_parity(Y, mu, sd, path, m):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for j, (ax, name) in enumerate(zip(axes, OUTPUTS)):
        lo, hi = min(Y[:, j].min(), mu[:, j].min()), max(Y[:, j].max(), mu[:, j].max())
        pad = 0.05 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=1,
                ls="--", label="perfect prediction")
        ax.errorbar(Y[:, j], mu[:, j], yerr=2 * sd[:, j], fmt="none",
                    ecolor=BLUE, alpha=0.35, lw=1, label="±2 × ensemble std")
        ax.scatter(Y[:, j], mu[:, j], s=22, color=BLUE, edgecolor="white", lw=0.6,
                   zorder=3, label="ensemble mean")
        if LOG_OUTPUT[name]:
            from matplotlib.ticker import FormatStrFormatter, LogLocator, NullFormatter
            ax.set_xscale("log")
            ax.set_yscale("log")
            for axis in (ax.xaxis, ax.yaxis):
                axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
                axis.set_major_formatter(FormatStrFormatter("%g"))
                axis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("pigsim (true)  " + LABEL[name], color=INK)
        ax.set_ylabel("surrogate (predicted)", color=INK)
        mm = m[name]
        ax.set_title(f"{name}:  R² = {mm['R2']:.4f},  MAPE = {mm['MAPE_%']:.2f} %",
                     color=INK, fontsize=11, loc="left")
        _style(ax)
        ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(f"Test set ({len(Y)} unseen pigsim runs): predicted vs actual",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_uncertainty(Y, mu, sd, path, m):
    """Does the ensemble std track the actual error?  (x: std, y: |error|)"""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for j, (ax, name) in enumerate(zip(axes, OUTPUTS)):
        err = np.abs(mu[:, j] - Y[:, j])
        s = sd[:, j]
        top = 1.05 * max(err.max(), 2 * s.max())
        xs = np.linspace(0, top / 2, 2)
        ax.fill_between(xs, 0, 2 * xs, color=BLUE, alpha=0.08, lw=0,
                        label="|error| ≤ 2 std")
        ax.plot(xs, 2 * xs, color=MUTED, lw=1, ls="--")
        ax.scatter(s, err, s=22, color=BLUE, edgecolor="white", lw=0.6, zorder=3,
                   label="test point")
        ax.set_xlim(0, top / 2)
        ax.set_ylim(0, top)
        unit = "s" if name == "t_arrive" else "m/s"
        ax.set_xlabel(f"ensemble std [{unit}]", color=INK)
        ax.set_ylabel(f"|prediction − pigsim| [{unit}]", color=INK)
        r = np.corrcoef(s, err)[0, 1]
        ax.set_title(f"{name}: {100*m[name]['frac_within_2std']:.0f} % inside 2 std, "
                     f"corr(std, |err|) = {r:.2f}", color=INK, fontsize=11, loc="left")
        _style(ax)
        ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Ensemble uncertainty vs actual error (test set)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sweep(ens, path):
    """Vary one input past the training range with the others at mid-range.

    Inside the range the 5 members agree (narrow band); outside it they have
    no data to agree on and diverge -- the ensemble *knows* it is guessing.
    """
    import matplotlib.pyplot as plt
    mid = np.array([0.5 * (RANGES[k][0] + RANGES[k][1]) for k in INPUTS])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
    for c, key in enumerate(["L", "p_in"]):
        k = INPUTS.index(key)
        lo, hi = RANGES[key][:2]
        span = hi - lo
        # p_in below ~2 bar cannot push the pig out of a 1.5 bar line at all,
        # so the sweep stops there instead of extrapolating into nonsense
        x_min = 2.0e5 if key == "p_in" else 0.3 * lo
        x = np.linspace(x_min, hi + 0.6 * span, 200)
        X = np.tile(mid, (len(x), 1))
        X[:, k] = x
        mu, sd, P = ens.predict(X)
        scale = 1e-5 if key == "p_in" else 1.0
        xl = x * scale
        for r, name in enumerate(OUTPUTS):
            ax = axes[r, c]
            ax.axvspan(lo * scale, hi * scale, color=GRID, alpha=0.5, lw=0,
                       label="training range")
            for i in range(P.shape[0]):
                ax.plot(xl, P[i, :, r], color=MUTED, lw=0.8, alpha=0.6,
                        label="individual members" if i == 0 else None)
            ax.fill_between(xl, mu[:, r] - 2 * sd[:, r], mu[:, r] + 2 * sd[:, r],
                            color=BLUE, alpha=0.18, lw=0, label="mean ± 2 std")
            ax.plot(xl, mu[:, r], color=BLUE, lw=2, label="ensemble mean")
            ax.set_ylabel(LABEL[name], color=INK)
            _style(ax)
            if r == 1:
                ax.set_xlabel(("inlet pressure p_in [bar]" if key == "p_in"
                               else "pipeline length L [m]"), color=INK)
            if r == 0 and c == 0:
                ax.legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("1-D sweeps beyond the training range (other inputs at mid-range)",
                 color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_loss(hists, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, h in enumerate(hists):
        ax.plot(h["train"], color=BLUE, lw=1, alpha=0.7,
                label="train (bootstrap)" if i == 0 else None)
        ax.plot(h["val"], color=ORANGE, lw=1, alpha=0.7,
                label="validation" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("epoch", color=INK)
    ax.set_ylabel("MSE (scaled units)", color=INK)
    ax.set_title("Learning curves, 5 ensemble members", color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
def main():
    torch.set_num_threads(1)     # tiny networks: threading overhead > gain
    rng = np.random.default_rng(SEED)
    df, X, Y, split = load_and_split(os.path.join(HERE, "data.csv"), rng)
    tr, va, te = split["train"], split["val"], split["test"]
    print(f"split: {len(tr)} train / {len(va)} val / {len(te)} test")

    xs = Scaler(X[tr])
    ys = Scaler(to_target(Y[tr]))
    Xtr, Xva = xs.fwd(X[tr]), xs.fwd(X[va])
    Ytr, Yva = ys.fwd(to_target(Y[tr])), ys.fwd(to_target(Y[va]))

    t0 = time.perf_counter()
    models, hists = [], []
    for i in range(N_MEMBERS):
        m, h = train_member(Xtr, Ytr, Xva, Yva, seed=SEED + i)
        models.append(m)
        hists.append(h)
    t_train = time.perf_counter() - t0
    ens = Ensemble(models, xs, ys)

    mu, sd, _ = ens.predict(X[te])
    m = {name: metrics(Y[te, j], mu[:, j], sd[:, j]) for j, name in enumerate(OUTPUTS)}

    # cost: surrogate vs solver.  Two numbers, because they differ a lot:
    #   single call  -- one design point per predict(), dominated by Python/torch
    #                   call overhead (what an optimiser calling it in a loop sees)
    #   batched      -- 10 000 points in one predict() (what a Monte-Carlo sees)
    t_sim = float(df["wall_s"].mean())
    t1 = time.perf_counter()
    for x in X[te]:
        ens.predict(x[None, :])
    t_single = (time.perf_counter() - t1) / len(te)
    Xbig = np.repeat(X[te], 1 + 10000 // len(te), axis=0)[:10000]
    t1 = time.perf_counter()
    ens.predict(Xbig)
    t_batch = (time.perf_counter() - t1) / len(Xbig)

    summary = {
        "n_usable": int(len(df)),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "train_time_s": round(t_train, 1),
        "pigsim_s_per_run": round(t_sim, 2),
        "surrogate_s_single_call": t_single,
        "surrogate_s_per_point_batched": t_batch,
        "speedup_single_call": round(t_sim / t_single),
        "speedup_batched": round(t_sim / t_batch),
        "test": m,
    }
    print(json.dumps(summary, indent=2))
    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    os.makedirs(FIG, exist_ok=True)
    plot_parity(Y[te], mu, sd, os.path.join(FIG, "parity_test.png"), m)
    plot_uncertainty(Y[te], mu, sd, os.path.join(FIG, "uncertainty_vs_error.png"), m)
    plot_sweep(ens, os.path.join(FIG, "sweep_extrapolation.png"))
    plot_loss(hists, os.path.join(FIG, "learning_curves.png"))

    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    torch.save({"state_dicts": [mm.state_dict() for mm in models],
                "x_mean": xs.mean, "x_std": xs.std, "y_mean": ys.mean, "y_std": ys.std,
                "inputs": INPUTS, "outputs": OUTPUTS, "log_output": LOG_OUTPUT},
               os.path.join(HERE, "models", "ensemble.pt"))
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
