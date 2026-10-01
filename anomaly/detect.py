"""Anomaly detection on the pigsim sensor series in runs.npz.

    python anomaly/detect.py            # ~5 min

Every detector is trained on *normal* runs only.

Data handling
    * Measurement noise is added to every run (pressure: 0.002 bar + 0.1 %
      of reading, flow: 0.002 kg/s + 0.5 %), seeded per run.
    * "sensor" runs are normal pigsim runs that get one sensor fault from
      t_a = 0.3-0.6 x run length on one random channel:
        drift   offset that grows linearly to 0.05-0.15 bar (pressure) or
                5-15 % (flow) over 60 s and keeps growing at that rate
        spikes  5 single-sample spikes of 0.2-0.5 bar or 20-50 %
    * Normal runs: 60 % train / 15 % validation / 25 % test.  All faulty runs
      are test runs.

Two input variants
    raw         signals in bar (pressure above 1.5 bar) and kg/s, z-scored per channel
    normalised  residual = signal - expected normal signal for this run's
                operating point (p_in, mass, F_fric, L): the inverse-distance
                mean of the 5 nearest normal training runs (log-input space)
                at the same time; then z-scored per channel.  Training runs
                use leave-one-out neighbours.

Three detectors, each giving a score per second (causal: window ends at t)
    z-score     max over channels of |z(t)|
    PCA         reconstruction error of the last 32 s (6 x 32 values),
                PCA with 99 % of the normal training variance
    CNN-AE      1D convolutional autoencoder on the same 32 s window

Evaluation
    threshold   95th percentile of the run-maximum score on normal
                validation runs (so ~5 % of normal runs raise an alarm)
    ROC-AUC     run-maximum score, normal test runs vs. faulty runs of one kind
    detection   alarm at or after the onset; delay = first such alarm - onset
    false alarm normal test runs with any alarm; alarms before the onset
                of faulty runs are counted separately
Outputs: metrics.json, figures/*.png
"""

import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from surrogate import pipeline as P  # noqa: E402

FIG = os.path.join(HERE, "figures")
W = 32
FAULTS = ["friction", "valve", "supply", "drift", "spikes"]
DETECTORS = ["z-score", "PCA", "CNN-AE"]
VARIANTS = ["raw", "normalised"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COL = {"z-score": "#1baf7a", "PCA": "#eb6834", "CNN-AE": "#2a78d6"}


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load(path=os.path.join(HERE, "runs.npz")):
    d = np.load(path)
    off = np.concatenate([[0], np.cumsum(d["lengths"])])
    runs = []
    for i in range(len(d["lengths"])):
        s = d["signals"][off[i]:off[i + 1]].astype(float)
        s[:, :5] = (s[:, :5] - P.P0) / 1e5            # pressure above 1.5 bar, in bar
        runs.append({"sig": s, "x": d["X"][i], "kind": str(d["kind"][i]),
                     "onset": float(d["onset"][i]), "arrived": bool(d["arrived"][i]),
                     "id": int(d["id"][i])})
    return runs


def add_noise(runs, seed=0):
    for r in runs:
        g = np.random.default_rng(seed + r["id"])
        s = r["sig"]
        sd = np.empty_like(s)
        sd[:, :5] = 0.002 + 0.001 * np.abs(s[:, :5] + 1.5)
        sd[:, 5] = 0.002 + 0.005 * np.abs(s[:, 5])
        r["sig"] = s + sd * g.standard_normal(s.shape)


def add_sensor_faults(runs, seed=1):
    """Turn the 'sensor' runs into drift (first half) and spike (second half) runs."""
    sens = [r for r in runs if r["kind"] == "sensor"]
    g = np.random.default_rng(seed)
    for k, r in enumerate(sens):
        s, n = r["sig"], len(r["sig"])
        t_a = int(g.uniform(0.3, 0.6) * n)
        j = int(g.integers(0, 6))
        is_flow = j == 5
        level = max(np.abs(s[:, j]).max(), 1e-3)
        if k < len(sens) // 2:
            mag = g.uniform(0.05, 0.15) if not is_flow else g.uniform(0.05, 0.15) * level
            t = np.arange(n)
            s[:, j] += mag * np.clip(t - t_a, 0, None) / 60.0
            r["kind"], r["params"] = "drift", {"channel": j, "per_60s": float(mag)}
        else:
            idx = np.sort(g.choice(np.arange(t_a, n), size=min(5, n - t_a), replace=False))
            amp = g.uniform(0.2, 0.5, len(idx)) * (level if is_flow else 1.0)
            s[idx, j] += amp * g.choice([-1, 1], len(idx))
            t_a = int(idx[0])
            r["kind"], r["params"] = "spikes", {"channel": j}
        r["onset"] = float(t_a)


def split_normal(runs, seed=0):
    normal = [r for r in runs if r["kind"] == "normal"]
    idx = np.random.default_rng(seed).permutation(len(normal))
    a, b = int(0.60 * len(normal)), int(0.75 * len(normal))
    return [normal[i] for i in idx[:a]], [normal[i] for i in idx[a:b]], [normal[i] for i in idx[b:]]


class Expected:
    """Expected normal signal for an operating point: IDW mean of 5 nearest normal runs."""

    def __init__(self, train, k=5):
        self.train, self.k = train, k
        Z = np.log(np.array([r["x"] for r in train]))
        self.mu, self.sd = Z.mean(0), Z.std(0)
        self.Z = (Z - self.mu) / self.sd

    def __call__(self, r, exclude=None):
        z = (np.log(r["x"]) - self.mu) / self.sd
        dist = np.linalg.norm(self.Z - z, axis=1)
        if exclude is not None:
            dist[exclude] = np.inf
        nb = np.argsort(dist)[:self.k]
        w = 1.0 / (dist[nb] + 1e-6)
        n = len(r["sig"])
        acc = np.zeros((n, 6))
        for wi, i in zip(w, nb):
            s = self.train[i]["sig"]
            pad = np.vstack([s, np.repeat(s[-1:], max(0, n - len(s)), axis=0)])[:n]
            acc += wi * pad
        return acc / w.sum()


def features(runs, train, variant, expected=None):
    """List of (n_t, 6) arrays: raw signals or residuals; train runs use leave-one-out."""
    if variant == "raw":
        return [r["sig"].copy() for r in runs]
    tr_ids = {id(r): i for i, r in enumerate(train)}
    return [r["sig"] - expected(r, exclude=tr_ids.get(id(r))) for r in runs]


def windows(F, stride=1):
    """Causal windows (n_t, 6, W) ending at every t; the start is padded with the first row."""
    pad = np.vstack([np.repeat(F[:1], W - 1, axis=0), F])
    idx = np.arange(0, len(F), stride)[:, None] + np.arange(W)[None, :]
    return np.transpose(pad[idx], (0, 2, 1))


# ---------------------------------------------------------------------------
# detectors
# ---------------------------------------------------------------------------
class ZScore:
    def fit(self, Ftr, Fva):
        A = np.vstack(Ftr)
        self.mu, self.sd = A.mean(0), A.std(0)
        return self

    def score(self, F):
        return np.max(np.abs((F - self.mu) / self.sd), axis=1)


class PCAWindow:
    def fit(self, Ftr, Fva):
        A = np.vstack(Ftr)
        self.mu, self.sd = A.mean(0), A.std(0)
        Wtr = np.vstack([windows((F - self.mu) / self.sd, stride=2) for F in Ftr]).reshape(-1, 6 * W)
        self.pca = PCA(0.99).fit(Wtr)
        return self

    def score(self, F):
        X = windows((F - self.mu) / self.sd).reshape(-1, 6 * W)
        R = self.pca.inverse_transform(self.pca.transform(X))
        return np.mean((X - R) ** 2, axis=1)


class ConvAE(nn.Module):
    """(6, 32) -> conv -> (16, 16) -> 16 latent -> back.  ~10k parameters."""

    def __init__(self, latent=16):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(6, 16, 5, padding=2), nn.SiLU(),
                                 nn.Conv1d(16, 16, 5, stride=2, padding=2), nn.SiLU(),
                                 nn.Flatten(), nn.Linear(16 * W // 2, latent))
        self.dec_lin = nn.Linear(latent, 16 * W // 2)
        self.dec = nn.Sequential(nn.SiLU(), nn.ConvTranspose1d(16, 16, 4, stride=2, padding=1),
                                 nn.SiLU(), nn.Conv1d(16, 6, 5, padding=2))

    def forward(self, x):
        z = self.dec_lin(self.enc(x)).view(-1, 16, W // 2)
        return self.dec(z)


class CNNAE:
    def fit(self, Ftr, Fva, epochs=60, patience=8, seed=0):
        torch.manual_seed(seed)
        A = np.vstack(Ftr)
        self.mu, self.sd = A.mean(0), A.std(0)
        mk = lambda Fs: torch.tensor(np.vstack([windows((F - self.mu) / self.sd, stride=2)  # noqa: E731
                                                for F in Fs]), dtype=torch.float32)
        xt, xv = mk(Ftr), mk(Fva)
        self.model = ConvAE()
        opt = torch.optim.Adam(self.model.parameters(), lr=2e-3)
        best, best_state, wait = np.inf, None, 0
        for ep in range(epochs):
            self.model.train()
            perm = torch.randperm(len(xt))
            for i in range(0, len(xt), 256):
                b = xt[perm[i:i + 256]]
                opt.zero_grad()
                loss = ((self.model(b) - b) ** 2).mean()
                loss.backward()
                opt.step()
            self.model.eval()
            with torch.no_grad():
                v = ((self.model(xv) - xv) ** 2).mean().item()
            if v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in self.model.state_dict().items()}
            else:
                wait += 1
                if wait > patience:
                    break
        self.model.load_state_dict(best_state)
        self.epochs, self.val_mse = ep + 1, best
        return self

    def score(self, F):
        x = torch.tensor(windows((F - self.mu) / self.sd), dtype=torch.float32)
        with torch.no_grad():
            return ((self.model(x) - x) ** 2).mean(dim=(1, 2)).numpy()


MAKE = {"z-score": ZScore, "PCA": PCAWindow, "CNN-AE": CNNAE}


# ---------------------------------------------------------------------------
def evaluate(scores_va, scores_te_norm, scores_fault, faults):
    thr = float(np.percentile([s.max() for s in scores_va], 95))
    out = {"threshold": thr,
           "false_alarm_rate_normal_test": float(np.mean([s.max() > thr for s in scores_te_norm])),
           "mean_alarm_seconds_per_normal_run": float(np.mean([(s > thr).sum() for s in scores_te_norm]))}
    neg = np.array([s.max() for s in scores_te_norm])
    for kind in FAULTS:
        sel = [(s, r) for s, r in zip(scores_fault, faults) if r["kind"] == kind]
        if not sel:
            continue
        pos = np.array([s.max() for s, _ in sel])
        y = np.r_[np.zeros(len(neg)), np.ones(len(pos))]
        delays, early = [], 0
        for s, r in sel:
            t0 = int(np.ceil(r["onset"]))
            early += int(np.any(s[:t0] > thr))
            hit = np.nonzero(s[t0:] > thr)[0]
            delays.append(float(hit[0]) if len(hit) else np.nan)
        dl = np.array(delays)
        out[kind] = {"n": len(sel), "roc_auc": float(roc_auc_score(y, np.r_[neg, pos])),
                     "detected": float(np.mean(np.isfinite(dl))),
                     "delay_s_median": float(np.nanmedian(dl)) if np.isfinite(dl).any() else None,
                     "delay_s_p90": float(np.nanpercentile(dl, 90)) if np.isfinite(dl).any() else None,
                     "alarm_before_onset": float(early / len(sel))}
    return out


def main():
    torch.set_num_threads(1)
    t_start = time.perf_counter()
    runs = load()
    add_noise(runs)
    add_sensor_faults(runs)
    train, val, test_n = split_normal(runs)
    faults = [r for r in runs if r["kind"] in FAULTS]
    kinds, counts = np.unique([r["kind"] for r in runs], return_counts=True)
    print("runs:", dict(zip(kinds, counts.tolist())), f"normal split {len(train)}/{len(val)}/{len(test_n)}")
    expected = Expected(train)

    res = {"counts": dict(zip(kinds.tolist(), counts.tolist())),
           "normal_split": [len(train), len(val), len(test_n)],
           "faults_not_arrived": {k: int(sum(1 for r in faults if r["kind"] == k and not r["arrived"]))
                                  for k in FAULTS},
           "results": {}}
    keep = {}
    for variant in VARIANTS:
        F = {name: features(rs, train, variant, expected)
             for name, rs in (("train", train), ("val", val), ("test", test_n), ("fault", faults))}
        for det in DETECTORS:
            t0 = time.perf_counter()
            m = MAKE[det]().fit(F["train"], F["val"])
            S = {k: [m.score(f) for f in F[k]] for k in ("val", "test", "fault")}
            ev = evaluate(S["val"], S["test"], S["fault"], faults)
            ev["fit_and_score_s"] = time.perf_counter() - t0
            res["results"][f"{variant} / {det}"] = ev
            keep[(variant, det)] = (S, F)
            print(f"  {variant:10s} {det:7s} FA {100*ev['false_alarm_rate_normal_test']:4.0f} %  " +
                  "  ".join(f"{k} AUC {ev[k]['roc_auc']:.2f} det {100*ev[k]['detected']:3.0f}% "
                            f"delay {ev[k]['delay_s_median'] if ev[k]['delay_s_median'] is None else round(ev[k]['delay_s_median'])}"
                            for k in FAULTS if k in ev), flush=True)

    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    os.makedirs(FIG, exist_ok=True)
    plot_auc(res)
    plot_examples(faults, keep, res, expected, train)
    print(f"-> metrics.json, figures/  ({(time.perf_counter()-t_start)/60:.1f} min)")


# ---------------------------------------------------------------------------
def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def plot_auc(res):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    xs = np.arange(len(FAULTS))
    for ax, (metric, lab) in zip(axes, (("roc_auc", "ROC-AUC (run level)"),
                                        ("detected", "detected after onset (5 % FA threshold)"),
                                        ("delay_s_median", "median detection delay [s]"))):
        for k, det in enumerate(DETECTORS):
            for v, variant in enumerate(VARIANTS):
                r = res["results"][f"{variant} / {det}"]
                vals = [np.nan if r.get(f, {}).get(metric) is None else r[f][metric] for f in FAULTS]
                off = (k - 1) * 0.27 + (v - 0.5) * 0.12
                ax.bar(xs + off, vals, 0.11, color=COL[det], alpha=0.45 if variant == "raw" else 1.0,
                       label=f"{det}, {variant}")
        ax.set_xticks(xs)
        ax.set_xticklabels(FAULTS, color=INK)
        ax.set_title(lab, fontsize=10, loc="left", color=INK)
        _style(ax)
    axes[0].set_ylim(0.4, 1.02)
    axes[0].axhline(0.5, color=MUTED, lw=0.8, ls="--")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, fontsize=8.5, ncol=6, loc="lower center")
    fig.suptitle("Detectors trained on normal runs only. Pale = raw signals, solid = operating-point "
                 "normalised residuals", color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(FIG, "detection_summary.png"), dpi=140)
    plt.close(fig)


def plot_examples(faults, keep, res, expected, train):
    """One faulty run per kind: signals, residuals, and the normalised CNN-AE / PCA scores."""
    import matplotlib.pyplot as plt
    names = ["p_inlet", "p_25", "p_50", "p_75", "p_outlet", "m_outlet"]
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7", INK]
    fig, axes = plt.subplots(3, len(FAULTS), figsize=(4.2 * len(FAULTS), 9), sharex="col")
    for c, kind in enumerate(FAULTS):
        i = next((k for k, r in enumerate(faults) if r["kind"] == kind), None)
        if i is None:
            continue
        r = faults[i]
        t = np.arange(len(r["sig"]))
        ax = axes[0, c]
        for j in range(6):
            y = r["sig"][:, j] if j < 5 else r["sig"][:, j] / max(r["sig"][:, 5].max(), 1e-6) * r["sig"][:, :5].max()
            ax.plot(t, y, color=cols[j], lw=1, label=names[j] + (" (scaled)" if j == 5 else ""))
        ax.set_title(f"{kind}", loc="left", fontsize=11, color=INK)
        if c == 0:
            ax.set_ylabel("signal [bar above 1.5]", color=INK)
            ax.legend(frameon=False, fontsize=7)
        ax = axes[1, c]
        Fn = keep[("normalised", "z-score")][1]["fault"][i]
        for j in range(6):
            ax.plot(t, Fn[:, j], color=cols[j], lw=1)
        if c == 0:
            ax.set_ylabel("residual vs 5 nearest\nnormal runs", color=INK)
        ax = axes[2, c]
        for det in ("PCA", "CNN-AE"):
            for variant, ls in (("raw", ":"), ("normalised", "-")):
                s = keep[(variant, det)][0]["fault"][i]
                thr = res["results"][f"{variant} / {det}"]["threshold"]
                ax.plot(t, s / thr, color=COL[det], ls=ls, lw=1.2, label=f"{det}, {variant}")
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.set_yscale("log")
        if c == 0:
            ax.set_ylabel("score / threshold", color=INK)
            ax.legend(frameon=False, fontsize=7)
        ax.set_xlabel("time [s]", color=INK)
        for a in axes[:, c]:
            a.axvline(r["onset"], color="#e34948", lw=1)
            _style(a)
    fig.suptitle("One faulty test run per kind (red line = onset)", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "examples.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
