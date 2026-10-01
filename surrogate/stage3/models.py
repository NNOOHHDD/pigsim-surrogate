"""Stage 3 models: map the 4 inputs to a whole speed curve V(xi) on 128 points.

All models share one interface:

    m = Model(...).fit(X_train, V_train, X_val, V_val)
    mean, std = m.predict(X)        # (n, 128) each, m/s; std is None if the model has none

Inputs are always log-transformed and z-scored with training statistics.

    DeepONetEnsemble   branch net (inputs) x trunk net (xi) -> V(xi), 5 members
    MLPEnsemble        plain MLP 4 -> 128 values at once, 5 members
    PCAPoly            PCA of the training curves + cubic polynomial for the scores
    PCAGP              PCA of the training curves + one GP per score
    NearestNeighbour   copy the curve of the closest training run
"""

import warnings

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from torch import nn


class LogScaler:
    """x -> (log x - mean) / std, statistics from the training inputs."""

    def __init__(self, X):
        Z = np.log(X)
        self.mean, self.std = Z.mean(0), Z.std(0)

    def __call__(self, X):
        return (np.log(X) - self.mean) / self.std


def _mlp(n_in, n_out, width, depth, last_act=False):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.SiLU()]
        d = width
    layers.append(nn.Linear(d, n_out))
    if last_act:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# neural operators / networks
# ---------------------------------------------------------------------------
class DeepONet(nn.Module):
    """V(x, xi) = sum_k branch_k(x) * trunk_k(xi) + b0.

    branch: the 4 (scaled) inputs -> p coefficients
    trunk:  one position xi       -> p basis functions evaluated at xi
    The trunk sees [xi, sqrt(xi)] (mapped to [-1, 1]) because the launch
    transient is squeezed into small xi.  Because the trunk only depends on
    xi, it is evaluated once on the 128 grid points and the whole curve of a
    batch of runs is one matrix product: (n, p) @ (p, 128).
    """

    def __init__(self, n_in=4, p=32, width=64, depth=2):
        super().__init__()
        self.branch = _mlp(n_in, p, width, depth)
        self.trunk = _mlp(2, p, width, depth, last_act=True)
        self.b0 = nn.Parameter(torch.zeros(1))

    def forward(self, x, xi_feat):
        return self.branch(x) @ self.trunk(xi_feat).T + self.b0


def xi_features(xi):
    return torch.tensor(np.column_stack([2 * xi - 1, 2 * np.sqrt(xi) - 1]), dtype=torch.float32)


def _train_net(make, forward, Xtr, Ytr, Xva, Yva, seed, epochs=3000, patience=300,
               lr=3e-3, weight_decay=1e-6, batch=64, bootstrap=True):
    """Same loop as stage 1: Adam, ReduceLROnPlateau, early stopping on val MSE."""
    torch.manual_seed(seed)
    idx = (np.random.default_rng(seed).integers(0, len(Xtr), len(Xtr)) if bootstrap
           else np.arange(len(Xtr)))
    xt = torch.tensor(Xtr[idx], dtype=torch.float32)
    yt = torch.tensor(Ytr[idx], dtype=torch.float32)
    xv = torch.tensor(Xva, dtype=torch.float32)
    yv = torch.tensor(Yva, dtype=torch.float32)
    model = make()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=100)
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt))
        for i in range(0, len(xt), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = ((forward(model, xt[b]) - yt[b]) ** 2).mean()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = ((forward(model, xv) - yv) ** 2).mean().item()
        sched.step(v)
        if v < best - 1e-8:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait > patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, ep + 1, best


class _NetEnsemble:
    n_members = 5

    def fit(self, Xtr, Vtr, Xva, Vva, seed=0, verbose=False):
        self.xs = LogScaler(Xtr)
        self.v_scale = float(Vtr.std())
        Ztr, Zva = self.xs(Xtr), self.xs(Xva)
        self.models, self.info = [], []
        for k in range(self.n_members):
            m, ep, best = _train_net(self._make, self._forward, Ztr, Vtr / self.v_scale,
                                     Zva, Vva / self.v_scale, seed=seed + k)
            self.models.append(m)
            self.info.append({"epochs": ep, "best_val_mse": best})
            if verbose:
                print(f"    {type(self).__name__} member {k}: {ep} epochs, val MSE {best:.2e}")
        return self

    def members(self, X):
        x = torch.tensor(self.xs(X), dtype=torch.float32)
        with torch.no_grad():
            return np.stack([self._forward(m, x).numpy() for m in self.models]) * self.v_scale

    def predict(self, X):
        P = self.members(X)
        return P.mean(0), P.std(0)


class DeepONetEnsemble(_NetEnsemble):
    def __init__(self, xi, p=32, width=64):
        self.xif = xi_features(xi)
        self.p, self.width = p, width

    def _make(self):
        return DeepONet(p=self.p, width=self.width)

    def _forward(self, model, x):
        return model(x, self.xif)


class MLPEnsemble(_NetEnsemble):
    def __init__(self, n_out=128, width=128, depth=3):
        self.n_out, self.width, self.depth = n_out, width, depth

    def _make(self):
        return _mlp(4, self.n_out, self.width, self.depth)

    def _forward(self, model, x):
        return model(x)


# ---------------------------------------------------------------------------
# classical baselines
# ---------------------------------------------------------------------------
class _PCABase:
    """Curves are compressed to the first k principal components (enough for
    99.99 % of the training variance); a regressor then maps inputs -> scores."""

    var_kept = 0.9999

    def _fit_pca(self, Vtr):
        full = PCA().fit(Vtr)
        self.k = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), self.var_kept) + 1)
        self.pca = PCA(self.k).fit(Vtr)
        return self.pca.transform(Vtr)


class PCAPoly(_PCABase):
    def fit(self, Xtr, Vtr, *_, **__):
        self.xs = LogScaler(Xtr)
        S = self._fit_pca(Vtr)
        self.reg = make_pipeline(PolynomialFeatures(3, include_bias=False),
                                 LinearRegression()).fit(self.xs(Xtr), S)
        return self

    def predict(self, X):
        return self.pca.inverse_transform(self.reg.predict(self.xs(X))), None


class PCAGP(_PCABase):
    """One GP per PCA score; the curve variance is sum_j phi_j(xi)^2 var_j.

    (The scores are uncorrelated over the training set, so the GPs are fitted
    independently.)  Kernel as in stage 2: C * Matern-5/2 (ARD) + white noise.
    """

    def __init__(self, restarts=2):
        self.restarts = restarts

    def fit(self, Xtr, Vtr, *_, **__):
        self.xs = LogScaler(Xtr)
        S = self._fit_pca(Vtr)
        Z = self.xs(Xtr)
        self.gps = []
        for j in range(self.k):
            kern = (ConstantKernel(1.0, (1e-3, 1e3))
                    * Matern(np.ones(Z.shape[1]), (1e-2, 1e3), nu=2.5)
                    + WhiteKernel(1e-4, (1e-9, 1e-1)))
            gp = GaussianProcessRegressor(kern, normalize_y=True,
                                          n_restarts_optimizer=self.restarts, random_state=j)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                gp.fit(Z, S[:, j])
            self.gps.append(gp)
        return self

    def predict(self, X):
        Z = self.xs(X)
        out = [gp.predict(Z, return_std=True) for gp in self.gps]
        S = np.stack([m for m, _ in out], 1)
        sd = np.stack([s for _, s in out], 1)
        mean = self.pca.inverse_transform(S)
        std = np.sqrt((sd ** 2) @ (self.pca.components_ ** 2))
        return mean, std


class NearestNeighbour:
    def fit(self, Xtr, Vtr, *_, **__):
        self.xs = LogScaler(Xtr)
        self.knn = KNeighborsRegressor(n_neighbors=1).fit(self.xs(Xtr), Vtr)
        return self

    def predict(self, X):
        return self.knn.predict(self.xs(X)), None
