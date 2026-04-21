"""
Input Specific Neural Networks (ISNN) - Complete Implementation
================================================================
Implements ISNN-1 and ISNN-2 architectures from the paper:
"Input Specific Neural Networks" (arXiv:2503.00268v1)

Two implementations:
1. PyTorch (with autograd)
2. Manual NumPy (with manual backpropagation)

Toy datasets:
- Additive:       f = exp(-0.5x) + log(1+exp(0.4y)) + tanh(t) + sin(z) - 0.4
- Multiplicative: g = exp(-0.3x) * (0.15y)^2 * tanh(0.3t) * (0.2*sin(0.5z+2)+0.5)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import qmc

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────
# Latin Hypercube Sampling
# ─────────────────────────────────────────────

def lhs_sample(n, low, high, dim=4):
    sampler = qmc.LatinHypercube(d=dim, seed=SEED)
    s = sampler.random(n)
    return qmc.scale(s, low, high)


# ─────────────────────────────────────────────
# Dataset Generation
# ─────────────────────────────────────────────

def f_additive(X):
    x, y, t, z = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return np.exp(-0.5*x) + np.log(1 + np.exp(0.4*y)) + np.tanh(t) + np.sin(z) - 0.4

def g_multiplicative(X):
    x, y, t, z = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    fx = np.exp(-0.3*x)
    fy = (0.15*y)**2
    ft = np.tanh(0.3*t)
    fz = 0.2*np.sin(0.5*z + 2) + 0.5
    return fx * fy * ft * fz

def generate_datasets(func, train_low=0.0, train_high=4.0,
                      test_high=6.0, n_train=500, n_test=5000):
    X_train = lhs_sample(n_train, train_low, train_high)
    X_test  = lhs_sample(n_test,  train_low, test_high)
    y_train = func(X_train).reshape(-1, 1)
    y_test  = func(X_test).reshape(-1, 1)
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
# Activation Functions
# ─────────────────────────────────────────────

def softplus(x):           return np.log1p(np.exp(x))
def softplus_prime(x):     return 1.0 / (1.0 + np.exp(-x))
def softplus_double(x):    e = np.exp(-x); return e / (1+e)**2
def sigmoid(x):            return 1.0 / (1.0 + np.exp(-x))
def sigmoid_prime(x):      s = sigmoid(x); return s*(1-s)


# ══════════════════════════════════════════════════════════════
#  PART 1 – PyTorch Implementations
# ══════════════════════════════════════════════════════════════

class ISNN1_Torch(nn.Module):
    """
    ISNN-1: separate sub-networks per input type;
    all input branches merge into the x-branch at layer 1 only.

    Constraints:
    - x0: convex           → W_xx (h≥1) non-negative, σ_mc convex & monotone
    - y0: convex+monotone  → W_yy, W_xy non-negative, σ_mc
    - t0: monotone only    → W_tt, W_xt non-negative, σ_m
    - z0: arbitrary        → no constraints
    """
    def __init__(self, hidden=10, Hx=2, Hy=2, Ht=2, Hz=2):
        super().__init__()
        self.Hx, self.Hy, self.Ht, self.Hz = Hx, Hy, Ht, Hz
        h = hidden

        # ── x-branch ──────────────────────────────────────────
        # First layer: receives x0 + skip connections from all other branches
        self.Wxx0 = nn.Parameter(torch.randn(h, 1) * 0.1)
        self.bx0  = nn.Parameter(torch.zeros(h))
        self.Wxy  = nn.Parameter(torch.randn(h, h) * 0.1)
        self.Wxz  = nn.Parameter(torch.randn(h, h) * 0.1)
        self.Wxt  = nn.Parameter(torch.randn(h, h) * 0.1)
        # Hidden x layers (h≥1): weights must be non-negative
        self.Wxx  = nn.ParameterList([nn.Parameter(torch.randn(h, h).abs() * 0.1) for _ in range(Hx-1)])
        self.bx   = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(Hx-1)])

        # ── y-branch (convex + monotone) ──────────────────────
        self.Wyy = nn.ParameterList([nn.Parameter(torch.randn(h, 1 if i==0 else h).abs()*0.1) for i in range(Hy)])
        self.by  = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(Hy)])

        # ── t-branch (monotone only) ──────────────────────────
        self.Wtt = nn.ParameterList([nn.Parameter(torch.randn(h, 1 if i==0 else h).abs()*0.1) for i in range(Ht)])
        self.bt  = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(Ht)])

        # ── z-branch (arbitrary) ──────────────────────────────
        self.Wzz = nn.ParameterList([nn.Parameter(torch.randn(h, 1 if i==0 else h)*0.1) for i in range(Hz)])
        self.bz  = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(Hz)])

        # ── output scalar ─────────────────────────────────────
        self.W_out = nn.Parameter(torch.randn(1, h).abs() * 0.1)
        self.b_out = nn.Parameter(torch.zeros(1))

    def _softplus(self, x): return torch.log1p(torch.exp(x))
    def _sigmoid(self, x):  return torch.sigmoid(x)

    def forward(self, X):
        x0 = X[:, 0:1]; y0 = X[:, 1:2]; t0 = X[:, 2:3]; z0 = X[:, 3:4]

        # ── y sub-network ─────────────────────────────────────
        yh = y0
        for i in range(self.Hy):
            W = self.Wyy[i].abs()        # enforce non-negativity
            yh = self._softplus(yh @ W.T + self.by[i])

        # ── t sub-network ─────────────────────────────────────
        th = t0
        for i in range(self.Ht):
            W = self.Wtt[i].abs()        # enforce non-negativity
            th = self._sigmoid(th @ W.T + self.bt[i])

        # ── z sub-network ─────────────────────────────────────
        zh = z0
        for i in range(self.Hz):
            zh = self._sigmoid(zh @ self.Wzz[i].T + self.bz[i])

        # ── x sub-network ─────────────────────────────────────
        # First layer merges all branches
        x1 = self._softplus(
            x0 @ self.Wxx0.T + self.bx0
            + yh @ self.Wxy.abs().T
            + zh @ self.Wxz.T
            + th @ self.Wxt.abs().T
        )
        xh = x1
        for i in range(self.Hx - 1):
            W = self.Wxx[i].abs()       # enforce non-negativity for convexity
            xh = self._softplus(xh @ W.T + self.bx[i])

        out = xh @ self.W_out.abs().T + self.b_out
        return out


class ISNN2_Torch(nn.Module):
    """
    ISNN-2: all branches have the SAME depth H;
    skip connections from all inputs into every x-layer.
    """
    def __init__(self, hidden=15, H=2):
        super().__init__()
        self.H = H
        h = hidden

        # ── y branch (convex + monotone) ──────────────────────
        self.Wyy = nn.ParameterList([nn.Parameter(torch.randn(h, 1 if i==0 else h).abs()*0.1) for i in range(H-1)])
        self.by  = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(H-1)])

        # ── t branch (monotone) ───────────────────────────────
        self.Wtt = nn.ParameterList([nn.Parameter(torch.randn(h, 1 if i==0 else h).abs()*0.1) for i in range(H-1)])
        self.bt  = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(H-1)])

        # ── z branch (arbitrary) ──────────────────────────────
        self.Wzz = nn.ParameterList([nn.Parameter(torch.randn(h, 1 if i==0 else h)*0.1) for i in range(H-1)])
        self.bz  = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(H-1)])

        # ── x branch: first layer (all inputs merge) ──────────
        self.Wxx0_0 = nn.Parameter(torch.randn(h, 1) * 0.1)
        self.Wxy0   = nn.Parameter(torch.randn(h, 1).abs() * 0.1)
        self.Wxz0   = nn.Parameter(torch.randn(h, 1) * 0.1)
        self.Wxt0   = nn.Parameter(torch.randn(h, 1).abs() * 0.1)
        self.bx0    = nn.Parameter(torch.zeros(h))

        # ── x branch: subsequent layers with skip from x0 ─────
        self.Wxx   = nn.ParameterList([nn.Parameter(torch.randn(h, h).abs()*0.1) for _ in range(H-1)])
        self.Wxx0  = nn.ParameterList([nn.Parameter(torch.randn(h, 1)*0.1) for _ in range(H-1)])
        self.Wxy   = nn.ParameterList([nn.Parameter(torch.randn(h, h).abs()*0.1) for _ in range(H-1)])
        self.Wxz   = nn.ParameterList([nn.Parameter(torch.randn(h, h)*0.1) for _ in range(H-1)])
        self.Wxt   = nn.ParameterList([nn.Parameter(torch.randn(h, h).abs()*0.1) for _ in range(H-1)])
        self.bx    = nn.ParameterList([nn.Parameter(torch.zeros(h)) for _ in range(H-1)])

        # ── output ────────────────────────────────────────────
        self.W_out = nn.Parameter(torch.randn(1, h).abs() * 0.1)
        self.b_out = nn.Parameter(torch.zeros(1))

    def _softplus(self, x): return torch.log1p(torch.exp(x))
    def _sigmoid(self, x):  return torch.sigmoid(x)

    def forward(self, X):
        x0 = X[:, 0:1]; y0 = X[:, 1:2]; t0 = X[:, 2:3]; z0 = X[:, 3:4]

        # ── y sub-network ─────────────────────────────────────
        yh = y0
        for i in range(self.H - 1):
            yh = self._softplus(yh @ self.Wyy[i].abs().T + self.by[i])

        # ── t sub-network ─────────────────────────────────────
        th = t0
        for i in range(self.H - 1):
            th = self._sigmoid(th @ self.Wtt[i].abs().T + self.bt[i])

        # ── z sub-network ─────────────────────────────────────
        zh = z0
        for i in range(self.H - 1):
            zh = self._sigmoid(zh @ self.Wzz[i].T + self.bz[i])

        # ── x sub-network: first layer ────────────────────────
        x1 = self._softplus(
            x0 @ self.Wxx0_0.T
            + y0 @ self.Wxy0.abs().T
            + z0 @ self.Wxz0.T
            + t0 @ self.Wxt0.abs().T
            + self.bx0
        )
        xh = x1
        for i in range(self.H - 1):
            xh = self._softplus(
                xh @ self.Wxx[i].abs().T
                + x0 @ self.Wxx0[i].T
                + yh @ self.Wxy[i].abs().T
                + zh @ self.Wxz[i].T
                + th @ self.Wxt[i].abs().T
                + self.bx[i]
            )

        out = xh @ self.W_out.abs().T + self.b_out
        return out


class FFNN_Torch(nn.Module):
    """Standard feed-forward neural network (unconstrained baseline)."""
    def __init__(self, input_dim=4, hidden=30, n_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


def train_torch(model, X_tr, y_tr, X_te, y_te, epochs=30000, lr=1e-3, log_every=500):
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te, dtype=torch.float32)

    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses, test_losses = [], []
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(X_tr_t)
        loss = loss_fn(pred, y_tr_t)
        loss.backward()
        opt.step()

        if ep % log_every == 0 or ep == epochs-1:
            model.eval()
            with torch.no_grad():
                tl = loss_fn(model(X_te_t), y_te_t).item()
            train_losses.append(loss.item())
            test_losses.append(tl)

    return train_losses, test_losses


# ══════════════════════════════════════════════════════════════
#  PART 2 – Manual NumPy Implementations (with manual backprop)
# ══════════════════════════════════════════════════════════════
#
#  Backpropagation implemented from scratch following the
#  chain-rule derivations in Appendix A of the paper and the
#  standard backprop algorithm (Slide 194 reference).
#
#  Each model stores parameters in dicts; forward pass caches
#  pre-activation (z) and post-activation (a) values required
#  by backprop. Gradients are computed layer-by-layer and
#  used with Adam update rules.
# ══════════════════════════════════════════════════════════════

class AdamOptimizer:
    """Per-parameter Adam state."""
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr; self.beta1 = beta1; self.beta2 = beta2; self.eps = eps
        self.m = {}; self.v = {}; self.t = 0

    def step(self, params, grads):
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])
            g = grads[key]
            self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*g
            self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*g**2
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def relu_proj(W):
    """Project weights to non-negative (for constraint enforcement)."""
    return np.maximum(W, 0)


class ISNN1_NumPy:
    """
    Manual ISNN-1 with hand-coded backpropagation.
    Architecture mirrors ISNN1_Torch exactly.
    """
    def __init__(self, hidden=10, Hx=2, Hy=2, Ht=2, Hz=2, seed=42):
        rng = np.random.default_rng(seed)
        h = hidden
        self.Hx, self.Hy, self.Ht, self.Hz = Hx, Hy, Ht, Hz
        self.h = h
        P = {}

        # x-branch (first layer)
        P['Wxx0'] = rng.normal(0, 0.1, (h, 1))
        P['bx0']  = np.zeros(h)
        P['Wxy']  = np.abs(rng.normal(0, 0.1, (h, h)))
        P['Wxz']  = rng.normal(0, 0.1, (h, h))
        P['Wxt']  = np.abs(rng.normal(0, 0.1, (h, h)))
        for i in range(Hx - 1):
            P[f'Wxx_{i}'] = np.abs(rng.normal(0, 0.1, (h, h)))
            P[f'bx_{i}']  = np.zeros(h)

        # y-branch
        P['Wyy_0'] = np.abs(rng.normal(0, 0.1, (h, 1)))
        P['by_0']  = np.zeros(h)
        for i in range(1, Hy):
            P[f'Wyy_{i}'] = np.abs(rng.normal(0, 0.1, (h, h)))
            P[f'by_{i}']  = np.zeros(h)

        # t-branch
        P['Wtt_0'] = np.abs(rng.normal(0, 0.1, (h, 1)))
        P['bt_0']  = np.zeros(h)
        for i in range(1, Ht):
            P[f'Wtt_{i}'] = np.abs(rng.normal(0, 0.1, (h, h)))
            P[f'bt_{i}']  = np.zeros(h)

        # z-branch
        P['Wzz_0'] = rng.normal(0, 0.1, (h, 1))
        P['bz_0']  = np.zeros(h)
        for i in range(1, Hz):
            P[f'Wzz_{i}'] = rng.normal(0, 0.1, (h, h))
            P[f'bz_{i}']  = np.zeros(h)

        # output
        P['Wout'] = np.abs(rng.normal(0, 0.1, (1, h)))
        P['bout'] = np.zeros(1)

        self.P = P
        self.opt = AdamOptimizer()

    def _enforce_constraints(self):
        """Project constrained weights to non-negative after each update."""
        P = self.P
        P['Wxy'] = relu_proj(P['Wxy'])
        P['Wxt'] = relu_proj(P['Wxt'])
        for i in range(self.Hx - 1):
            P[f'Wxx_{i}'] = relu_proj(P[f'Wxx_{i}'])
        for i in range(self.Hy):
            P[f'Wyy_{i}'] = relu_proj(P[f'Wyy_{i}'])
        for i in range(self.Ht):
            P[f'Wtt_{i}'] = relu_proj(P[f'Wtt_{i}'])

    def forward(self, X):
        P = self.P
        x0 = X[:, 0:1]; y0 = X[:, 1:2]; t0 = X[:, 2:3]; z0 = X[:, 3:4]
        N = X.shape[0]
        cache = {'x0': x0, 'y0': y0, 't0': t0, 'z0': z0}

        # ── y sub-network ─────────────────────────────────────
        ya = [y0]
        yz = []
        yh = y0
        for i in range(self.Hy):
            W = relu_proj(P[f'Wyy_{i}'])
            z = yh @ W.T + P[f'by_{i}']
            yh = softplus(z)
            yz.append(z); ya.append(yh)
        cache['ya'] = ya; cache['yz'] = yz

        # ── t sub-network ─────────────────────────────────────
        ta = [t0]
        tz = []
        th = t0
        for i in range(self.Ht):
            W = relu_proj(P[f'Wtt_{i}'])
            z = th @ W.T + P[f'bt_{i}']
            th = sigmoid(z)
            tz.append(z); ta.append(th)
        cache['ta'] = ta; cache['tz'] = tz

        # ── z sub-network ─────────────────────────────────────
        za = [z0]
        zz = []
        zh = z0
        for i in range(self.Hz):
            W = P[f'Wzz_{i}']
            z = zh @ W.T + P[f'bz_{i}']
            zh = sigmoid(z)
            zz.append(z); za.append(zh)
        cache['za'] = za; cache['zz'] = zz

        # ── x sub-network (layer 1) ───────────────────────────
        z1 = (x0 @ P['Wxx0'].T + P['bx0']
              + yh @ relu_proj(P['Wxy']).T
              + zh @ P['Wxz'].T
              + th @ relu_proj(P['Wxt']).T)
        a1 = softplus(z1)
        xa = [x0, a1]; xz = [z1]

        xh = a1
        for i in range(self.Hx - 1):
            W = relu_proj(P[f'Wxx_{i}'])
            z = xh @ W.T + P[f'bx_{i}']
            xh = softplus(z)
            xz.append(z); xa.append(xh)
        cache['xa'] = xa; cache['xz'] = xz

        # ── output ────────────────────────────────────────────
        out = xh @ relu_proj(P['Wout']).T + P['bout']
        cache['xh_final'] = xh
        self.cache = cache
        return out

    def backward(self, X, y_true):
        P = self.P
        cache = self.cache
        N = X.shape[0]
        grads = {}

        # ── MSE loss derivative ───────────────────────────────
        y_pred = self.forward(X)
        dL_dout = 2.0 * (y_pred - y_true) / N          # (N,1)

        # ── output layer ─────────────────────────────────────
        xh_final = cache['xh_final']
        grads['Wout'] = dL_dout.T @ xh_final            # (1,h)
        grads['bout'] = dL_dout.sum(axis=0)
        dL_dxh = dL_dout @ relu_proj(P['Wout'])         # (N,h)

        # ── x-branch backward (layers 1..Hx-1) ───────────────
        xa = cache['xa']; xz = cache['xz']
        for i in range(self.Hx - 2, -1, -1):
            # layer i+1 in xa (index i+2 is xh, index i+1 is prev)
            dL_dz = dL_dxh * softplus_prime(xz[i+1])   # element-wise
            W = relu_proj(P[f'Wxx_{i}'])
            grads[f'Wxx_{i}'] = dL_dz.T @ xa[i+1]
            grads[f'bx_{i}']  = dL_dz.sum(axis=0)
            dL_dxh = dL_dz @ W

        # ── x-branch layer 0 (merged layer) ───────────────────
        dL_dz0 = dL_dxh * softplus_prime(xz[0])
        grads['Wxx0'] = dL_dz0.T @ cache['x0']
        grads['bx0']  = dL_dz0.sum(axis=0)

        # gradient to y-branch output
        grads['Wxy'] = dL_dz0.T @ cache['ya'][-1]
        dL_dyh = dL_dz0 @ relu_proj(P['Wxy'])

        # gradient to z-branch output
        grads['Wxz'] = dL_dz0.T @ cache['za'][-1]
        dL_dzh = dL_dz0 @ P['Wxz']

        # gradient to t-branch output
        grads['Wxt'] = dL_dz0.T @ cache['ta'][-1]
        dL_dth = dL_dz0 @ relu_proj(P['Wxt'])

        # ── y-branch backward ─────────────────────────────────
        ya = cache['ya']; yz = cache['yz']
        for i in range(self.Hy - 1, -1, -1):
            dL_dz = dL_dyh * softplus_prime(yz[i])
            grads[f'Wyy_{i}'] = dL_dz.T @ ya[i]
            grads[f'by_{i}']  = dL_dz.sum(axis=0)
            if i > 0:
                dL_dyh = dL_dz @ relu_proj(P[f'Wyy_{i}'])

        # ── t-branch backward ─────────────────────────────────
        ta = cache['ta']; tz = cache['tz']
        for i in range(self.Ht - 1, -1, -1):
            dL_dz = dL_dth * sigmoid_prime(tz[i])
            grads[f'Wtt_{i}'] = dL_dz.T @ ta[i]
            grads[f'bt_{i}']  = dL_dz.sum(axis=0)
            if i > 0:
                dL_dth = dL_dz @ relu_proj(P[f'Wtt_{i}'])

        # ── z-branch backward ─────────────────────────────────
        za = cache['za']; zz = cache['zz']
        for i in range(self.Hz - 1, -1, -1):
            dL_dz = dL_dzh * sigmoid_prime(zz[i])
            grads[f'Wzz_{i}'] = dL_dz.T @ za[i]
            grads[f'bz_{i}']  = dL_dz.sum(axis=0)
            if i > 0:
                dL_dzh = dL_dz @ P[f'Wzz_{i}']

        return grads

    def mse(self, X, y):
        pred = self.forward(X)
        return np.mean((pred - y)**2)

    def train_step(self, X, y):
        grads = self.backward(X, y)
        self.opt.step(self.P, grads)
        self._enforce_constraints()


class ISNN2_NumPy:
    """
    Manual ISNN-2 with hand-coded backpropagation.
    All branches have depth H; x-branch receives skip connections
    from x0 and all other sub-branches at every layer.
    """
    def __init__(self, hidden=15, H=2, seed=42):
        rng = np.random.default_rng(seed)
        h = hidden; self.H = H; self.h = h
        P = {}

        # y-branch (H-1 layers, convex+monotone)
        P['Wyy_0'] = np.abs(rng.normal(0,0.1,(h,1)))
        P['by_0']  = np.zeros(h)
        for i in range(1, H-1):
            P[f'Wyy_{i}'] = np.abs(rng.normal(0,0.1,(h,h)))
            P[f'by_{i}']  = np.zeros(h)

        # t-branch (H-1 layers, monotone)
        P['Wtt_0'] = np.abs(rng.normal(0,0.1,(h,1)))
        P['bt_0']  = np.zeros(h)
        for i in range(1, H-1):
            P[f'Wtt_{i}'] = np.abs(rng.normal(0,0.1,(h,h)))
            P[f'bt_{i}']  = np.zeros(h)

        # z-branch (H-1 layers, arbitrary)
        P['Wzz_0'] = rng.normal(0,0.1,(h,1))
        P['bz_0']  = np.zeros(h)
        for i in range(1, H-1):
            P[f'Wzz_{i}'] = rng.normal(0,0.1,(h,h))
            P[f'bz_{i}']  = np.zeros(h)

        # x-branch layer 0 (merges all raw inputs)
        P['Wxx0_0'] = rng.normal(0,0.1,(h,1))
        P['Wxy0']   = np.abs(rng.normal(0,0.1,(h,1)))
        P['Wxz0']   = rng.normal(0,0.1,(h,1))
        P['Wxt0']   = np.abs(rng.normal(0,0.1,(h,1)))
        P['bx0']    = np.zeros(h)

        # x-branch layers 1..H-1 (with skip from x0 + sub-branches)
        for i in range(H-1):
            P[f'Wxx_{i}']  = np.abs(rng.normal(0,0.1,(h,h)))
            P[f'Wxx0_{i}'] = rng.normal(0,0.1,(h,1))
            P[f'Wxy_{i}']  = np.abs(rng.normal(0,0.1,(h,h)))
            P[f'Wxz_{i}']  = rng.normal(0,0.1,(h,h))
            P[f'Wxt_{i}']  = np.abs(rng.normal(0,0.1,(h,h)))
            P[f'bx_{i}']   = np.zeros(h)

        P['Wout'] = np.abs(rng.normal(0,0.1,(1,h)))
        P['bout'] = np.zeros(1)

        self.P = P
        self.opt = AdamOptimizer()

    def _enforce_constraints(self):
        P = self.P
        P['Wxy0'] = relu_proj(P['Wxy0'])
        P['Wxt0'] = relu_proj(P['Wxt0'])
        for i in range(self.H - 1):
            P[f'Wyy_{i}']  = relu_proj(P[f'Wyy_{i}'])
            P[f'Wtt_{i}']  = relu_proj(P[f'Wtt_{i}'])
            P[f'Wxx_{i}']  = relu_proj(P[f'Wxx_{i}'])
            P[f'Wxy_{i}']  = relu_proj(P[f'Wxy_{i}'])
            P[f'Wxt_{i}']  = relu_proj(P[f'Wxt_{i}'])

    def forward(self, X):
        P = self.P
        x0 = X[:,0:1]; y0 = X[:,1:2]; t0 = X[:,2:3]; z0 = X[:,3:4]
        cache = {'x0':x0,'y0':y0,'t0':t0,'z0':z0}

        # ── y sub-network ─────────────────────────────────────
        ya=[y0]; yz=[]
        yh=y0
        for i in range(self.H-1):
            W=relu_proj(P[f'Wyy_{i}'])
            z=yh@W.T+P[f'by_{i}']
            yh=softplus(z)
            yz.append(z); ya.append(yh)
        cache['ya']=ya; cache['yz']=yz

        # ── t sub-network ─────────────────────────────────────
        ta=[t0]; tz=[]
        th=t0
        for i in range(self.H-1):
            W=relu_proj(P[f'Wtt_{i}'])
            z=th@W.T+P[f'bt_{i}']
            th=sigmoid(z)
            tz.append(z); ta.append(th)
        cache['ta']=ta; cache['tz']=tz

        # ── z sub-network ─────────────────────────────────────
        za=[z0]; zz=[]
        zh=z0
        for i in range(self.H-1):
            W=P[f'Wzz_{i}']
            z=zh@W.T+P[f'bz_{i}']
            zh=sigmoid(z)
            zz.append(z); za.append(zh)
        cache['za']=za; cache['zz']=zz

        # ── x sub-network layer 0 ─────────────────────────────
        z1 = (x0@P['Wxx0_0'].T + y0@relu_proj(P['Wxy0']).T
              + z0@P['Wxz0'].T  + t0@relu_proj(P['Wxt0']).T + P['bx0'])
        a1 = softplus(z1)
        xa=[x0,a1]; xz=[z1]
        xh=a1

        # ── x sub-network layers 1..H-1 ───────────────────────
        for i in range(self.H-1):
            z = (xh @ relu_proj(P[f'Wxx_{i}']).T
                 + x0 @ P[f'Wxx0_{i}'].T
                 + ya[-1] @ relu_proj(P[f'Wxy_{i}']).T
                 + za[-1] @ P[f'Wxz_{i}'].T
                 + ta[-1] @ relu_proj(P[f'Wxt_{i}']).T
                 + P[f'bx_{i}'])
            xh = softplus(z)
            xz.append(z); xa.append(xh)
        cache['xa']=xa; cache['xz']=xz
        cache['xh_final']=xh

        out = xh @ relu_proj(P['Wout']).T + P['bout']
        self.cache = cache
        return out

    def backward(self, X, y_true):
        P = self.P; cache = self.cache
        N = X.shape[0]
        grads = {}

        y_pred = self.forward(X)
        dL_dout = 2.0*(y_pred - y_true)/N

        grads['Wout'] = dL_dout.T @ cache['xh_final']
        grads['bout'] = dL_dout.sum(axis=0)
        dL_dxh = dL_dout @ relu_proj(P['Wout'])

        xa=cache['xa']; xz=cache['xz']
        ya=cache['ya']; ta=cache['ta']; za=cache['za']
        yz=cache['yz']; tz=cache['tz']; zz=cache['zz']
        x0=cache['x0']

        # Accumulators for sub-branch gradients (summed over x-layers)
        dL_dyh = np.zeros_like(ya[-1])
        dL_dth = np.zeros_like(ta[-1])
        dL_dzh = np.zeros_like(za[-1])

        # ── x layers H-1 → 1 ──────────────────────────────────
        for i in range(self.H-2, -1, -1):
            dL_dz = dL_dxh * softplus_prime(xz[i+1])
            grads[f'Wxx_{i}']  = dL_dz.T @ xa[i+1]
            grads[f'Wxx0_{i}'] = dL_dz.T @ x0
            grads[f'Wxy_{i}']  = dL_dz.T @ ya[-1]
            grads[f'Wxz_{i}']  = dL_dz.T @ za[-1]
            grads[f'Wxt_{i}']  = dL_dz.T @ ta[-1]
            grads[f'bx_{i}']   = dL_dz.sum(axis=0)
            dL_dxh   = dL_dz @ relu_proj(P[f'Wxx_{i}'])
            dL_dyh  += dL_dz @ relu_proj(P[f'Wxy_{i}'])
            dL_dzh  += dL_dz @ P[f'Wxz_{i}']
            dL_dth  += dL_dz @ relu_proj(P[f'Wxt_{i}'])

        # ── x layer 0 ─────────────────────────────────────────
        dL_dz0 = dL_dxh * softplus_prime(xz[0])
        grads['Wxx0_0'] = dL_dz0.T @ x0
        grads['Wxy0']   = dL_dz0.T @ cache['y0']
        grads['Wxz0']   = dL_dz0.T @ cache['z0']
        grads['Wxt0']   = dL_dz0.T @ cache['t0']
        grads['bx0']    = dL_dz0.sum(axis=0)

        # ── y-branch backward ─────────────────────────────────
        for i in range(self.H-2, -1, -1):
            dL_dz = dL_dyh * softplus_prime(yz[i])
            grads[f'Wyy_{i}'] = dL_dz.T @ ya[i]
            grads[f'by_{i}']  = dL_dz.sum(axis=0)
            if i > 0:
                dL_dyh = dL_dz @ relu_proj(P[f'Wyy_{i}'])

        # ── t-branch backward ─────────────────────────────────
        for i in range(self.H-2, -1, -1):
            dL_dz = dL_dth * sigmoid_prime(tz[i])
            grads[f'Wtt_{i}'] = dL_dz.T @ ta[i]
            grads[f'bt_{i}']  = dL_dz.sum(axis=0)
            if i > 0:
                dL_dth = dL_dz @ relu_proj(P[f'Wtt_{i}'])

        # ── z-branch backward ─────────────────────────────────
        for i in range(self.H-2, -1, -1):
            dL_dz = dL_dzh * sigmoid_prime(zz[i])
            grads[f'Wzz_{i}'] = dL_dz.T @ za[i]
            grads[f'bz_{i}']  = dL_dz.sum(axis=0)
            if i > 0:
                dL_dzh = dL_dz @ P[f'Wzz_{i}']

        return grads

    def mse(self, X, y):
        return np.mean((self.forward(X) - y)**2)

    def train_step(self, X, y):
        grads = self.backward(X, y)
        self.opt.step(self.P, grads)
        self._enforce_constraints()


class FFNN_NumPy:
    """Standard FFNN baseline with manual backprop (tanh activations)."""
    def __init__(self, input_dim=4, hidden=30, n_layers=2, seed=42):
        rng = np.random.default_rng(seed)
        P = {}
        dims = [input_dim] + [hidden]*n_layers + [1]
        for i in range(len(dims)-1):
            P[f'W_{i}'] = rng.normal(0, 0.1, (dims[i+1], dims[i]))
            P[f'b_{i}'] = np.zeros(dims[i+1])
        self.P = P; self.dims = dims
        self.n_layers = n_layers + 1
        self.opt = AdamOptimizer()

    def forward(self, X):
        a = X; cache = [X]; z_cache = []
        for i in range(self.n_layers - 1):
            z = a @ self.P[f'W_{i}'].T + self.P[f'b_{i}']
            z_cache.append(z)
            a = np.tanh(z); cache.append(a)
        # last layer: linear
        z = cache[-1] @ self.P[f'W_{self.n_layers-1}'].T + self.P[f'b_{self.n_layers-1}']
        z_cache.append(z)
        cache.append(z)
        self.cache = cache; self.z_cache = z_cache
        return z

    def backward(self, X, y_true):
        N = X.shape[0]
        y_pred = self.forward(X)
        dL = 2*(y_pred - y_true)/N
        grads = {}
        for i in range(self.n_layers, 0, -1):
            if i == self.n_layers:
                dL_dz = dL
            else:
                dL_dz = dL * (1 - np.tanh(self.z_cache[i-1])**2)
            grads[f'W_{i-1}'] = dL_dz.T @ self.cache[i-1]
            grads[f'b_{i-1}'] = dL_dz.sum(axis=0)
            dL = dL_dz @ self.P[f'W_{i-1}']
        return grads

    def mse(self, X, y):
        return np.mean((self.forward(X) - y)**2)

    def train_step(self, X, y):
        grads = self.backward(X, y)
        self.opt.step(self.P, grads)


def train_numpy(model, X_tr, y_tr, X_te, y_te, epochs=30000, log_every=500):
    train_losses, test_losses = [], []
    for ep in range(epochs):
        model.train_step(X_tr, y_tr)
        if ep % log_every == 0 or ep == epochs-1:
            train_losses.append(model.mse(X_tr, y_tr))
            test_losses.append(model.mse(X_te, y_te))
    return train_losses, test_losses


# ══════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════

def plot_loss_curves(results, dataset_name, epochs=30000, log_every=500, save_path=None):
    ep_axis = np.linspace(0, epochs, len(next(iter(results.values()))['train']))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Training & Test Loss — {dataset_name} Dataset', fontsize=14, fontweight='bold')

    colors = {'FFNN': '#e74c3c', 'ISNN-1': '#2ecc71', 'ISNN-2': '#3498db'}
    for name, res in results.items():
        c = colors.get(name, 'gray')
        axes[0].semilogy(ep_axis, res['train'], label=name, color=c)
        axes[1].semilogy(ep_axis, res['test'],  label=name, color=c)

    for ax, title in zip(axes, ['(a) Training Loss', '(b) Test Loss']):
        ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss')
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_behavior(models_dict, func, dataset_name, train_high=4.0, eval_high=6.0, save_path=None):
    """Plot interpolated vs extrapolated response (like Figures 4 & 6)."""
    t_vals = np.linspace(0, eval_high, 200)
    X_eval = np.column_stack([t_vals, t_vals, t_vals, t_vals])
    y_true = func(X_eval).ravel()

    mask_interp = t_vals <= train_high

    model_names = list(models_dict.keys())
    n = len(model_names)
    ncols = 2 if n <= 2 else 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows))
    if nrows == 1: axes = axes.reshape(1, -1)
    fig.suptitle(f'Predictive Behavior — {dataset_name} Dataset\n(x=y=t=z ∈ [0, {eval_high}])',
                 fontsize=13, fontweight='bold')

    for idx, (name, model) in enumerate(models_dict.items()):
        ax = axes[idx // ncols][idx % ncols]
        X_t = torch.tensor(X_eval, dtype=torch.float32) if hasattr(model, 'parameters') else None

        if X_t is not None:
            model.eval()
            with torch.no_grad():
                y_pred = model(X_t).numpy().ravel()
        else:
            y_pred = model.forward(X_eval).ravel()

        ax.plot(t_vals, y_true, 'k--', linewidth=1.5, label='True response')
        ax.plot(t_vals[mask_interp],  y_pred[mask_interp],  color='steelblue',  lw=2, label='Interpolated')
        ax.plot(t_vals[~mask_interp], y_pred[~mask_interp], color='tomato',     lw=2, label='Extrapolated')
        ax.axvline(train_high, color='gray', linestyle=':', alpha=0.7)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Input value'); ax.set_ylabel('Output')
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_combined_comparison(torch_results, numpy_results, dataset_name, save_path=None):
    """Side-by-side comparison: PyTorch vs NumPy implementations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'PyTorch vs NumPy — {dataset_name} Dataset', fontsize=14, fontweight='bold')

    implementations = [('PyTorch', torch_results), ('NumPy (Manual Backprop)', numpy_results)]
    colors = {'FFNN': '#e74c3c', 'ISNN-1': '#2ecc71', 'ISNN-2': '#3498db'}

    for row_idx, (impl_name, results) in enumerate(implementations):
        for col_idx, loss_type in enumerate(['train', 'test']):
            ax = axes[row_idx][col_idx]
            for name, res in results.items():
                ax.semilogy(res[loss_type], label=name, color=colors.get(name, 'gray'))
            ax.set_title(f'{impl_name} — {"Training" if col_idx==0 else "Test"} Loss')
            ax.set_xlabel('Log steps'); ax.set_ylabel('MSE Loss')
            ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

def run_experiment(func, func_name, train_high=4.0, test_high=6.0,
                   epochs=30000, n_train=500, n_test=5000):
    print(f"\n{'='*60}")
    print(f"  Dataset: {func_name}")
    print(f"{'='*60}")

    X_tr, y_tr, X_te, y_te = generate_datasets(
        func, 0.0, train_high, test_high, n_train, n_test)

    np.save(f'/mnt/user-data/outputs/{func_name}_X_train.npy', X_tr)
    np.save(f'/mnt/user-data/outputs/{func_name}_y_train.npy', y_tr)
    np.save(f'/mnt/user-data/outputs/{func_name}_X_test.npy',  X_te)
    np.save(f'/mnt/user-data/outputs/{func_name}_y_test.npy',  y_te)
    print(f"  Data saved. Train: {X_tr.shape}, Test: {X_te.shape}")

    log_every = max(1, epochs // 60)

    # ── PyTorch models ────────────────────────────────────────
    print("\n  [PyTorch] Training FFNN ...")
    ffnn_t  = FFNN_Torch();   tr_f,  te_f  = train_torch(ffnn_t,  X_tr, y_tr, X_te, y_te, epochs, log_every=log_every)
    print(f"    Final train: {tr_f[-1]:.6f}  test: {te_f[-1]:.6f}")

    print("  [PyTorch] Training ISNN-1 ...")
    isnn1_t = ISNN1_Torch();  tr_i1, te_i1 = train_torch(isnn1_t, X_tr, y_tr, X_te, y_te, epochs, log_every=log_every)
    print(f"    Final train: {tr_i1[-1]:.6f}  test: {te_i1[-1]:.6f}")

    print("  [PyTorch] Training ISNN-2 ...")
    isnn2_t = ISNN2_Torch();  tr_i2, te_i2 = train_torch(isnn2_t, X_tr, y_tr, X_te, y_te, epochs, log_every=log_every)
    print(f"    Final train: {tr_i2[-1]:.6f}  test: {te_i2[-1]:.6f}")

    torch_results = {
        'FFNN':   {'train': tr_f,  'test': te_f},
        'ISNN-1': {'train': tr_i1, 'test': te_i1},
        'ISNN-2': {'train': tr_i2, 'test': te_i2},
    }

    # ── NumPy models ─────────────────────────────────────────
    print("\n  [NumPy]   Training FFNN ...")
    ffnn_n  = FFNN_NumPy();   tr_fn,  te_fn  = train_numpy(ffnn_n,  X_tr, y_tr, X_te, y_te, epochs, log_every)
    print(f"    Final train: {tr_fn[-1]:.6f}  test: {te_fn[-1]:.6f}")

    print("  [NumPy]   Training ISNN-1 ...")
    isnn1_n = ISNN1_NumPy();  tr_i1n, te_i1n = train_numpy(isnn1_n, X_tr, y_tr, X_te, y_te, epochs, log_every)
    print(f"    Final train: {tr_i1n[-1]:.6f}  test: {te_i1n[-1]:.6f}")

    print("  [NumPy]   Training ISNN-2 ...")
    isnn2_n = ISNN2_NumPy();  tr_i2n, te_i2n = train_numpy(isnn2_n, X_tr, y_tr, X_te, y_te, epochs, log_every)
    print(f"    Final train: {tr_i2n[-1]:.6f}  test: {te_i2n[-1]:.6f}")

    numpy_results = {
        'FFNN':   {'train': tr_fn,  'test': te_fn},
        'ISNN-1': {'train': tr_i1n, 'test': te_i1n},
        'ISNN-2': {'train': tr_i2n, 'test': te_i2n},
    }

    # ── Plots ─────────────────────────────────────────────────
    print("\n  Generating plots ...")

    # Loss curves (PyTorch — mirrors paper Figures 3 & 5)
    plot_loss_curves(torch_results, f'{func_name} [PyTorch]', epochs, log_every,
                     save_path=f'/mnt/user-data/outputs/{func_name}_torch_loss.png')

    # Loss curves (NumPy)
    plot_loss_curves(numpy_results, f'{func_name} [NumPy Manual Backprop]', epochs, log_every,
                     save_path=f'/mnt/user-data/outputs/{func_name}_numpy_loss.png')

    # Behavioral response (PyTorch — mirrors paper Figures 4 & 6)
    torch_models = {'FFNN': ffnn_t, 'ISNN-1': isnn1_t, 'ISNN-2': isnn2_t}
    plot_behavior(torch_models, func, f'{func_name} [PyTorch]',
                  train_high=train_high, eval_high=test_high,
                  save_path=f'/mnt/user-data/outputs/{func_name}_torch_behavior.png')

    # Behavioral response (NumPy)
    numpy_models = {'FFNN': ffnn_n, 'ISNN-1': isnn1_n, 'ISNN-2': isnn2_n}
    plot_behavior(numpy_models, func, f'{func_name} [NumPy]',
                  train_high=train_high, eval_high=test_high,
                  save_path=f'/mnt/user-data/outputs/{func_name}_numpy_behavior.png')

    # Combined side-by-side
    plot_combined_comparison(torch_results, numpy_results, func_name,
                             save_path=f'/mnt/user-data/outputs/{func_name}_comparison.png')

    print(f"  Plots saved for {func_name}.")
    return torch_results, numpy_results


if __name__ == '__main__':
    EPOCHS = 30000

    # ── Experiment 1: Additive (Eq. 12) ──────────────────────
    run_experiment(f_additive, 'Additive',
                   train_high=4.0, test_high=6.0, epochs=EPOCHS)

    # ── Experiment 2: Multiplicative (Eq. 13) ────────────────
    run_experiment(g_multiplicative, 'Multiplicative',
                   train_high=4.0, test_high=10.0, epochs=EPOCHS)

    print("\n✓ All experiments complete. Results saved to /mnt/user-data/outputs/")
