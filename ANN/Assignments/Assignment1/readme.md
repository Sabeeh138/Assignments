# Input Specific Neural Networks (ISNN)


---

## Overview

This project implements the **Input Specific Neural Network (ISNN)** architecture in two ways:

1. **PyTorch** — using autograd for backpropagation
2. **NumPy** — using fully manual backpropagation (matrix operations only)

Both ISNN-1 and ISNN-2 architectures are implemented, along with a standard Feed-Forward Neural Network (FFNN) as a baseline. All models are trained and evaluated on two toy datasets from Section 3.1 of the paper.

---

## File Structure

```
submission/
├── isnn_implementation.py       # Complete source code (PyTorch + NumPy)
├── README.md                    # This file
├── ISNN_Assignment_Report.docx  # Written report
│
├── Additive_Xtrain.npy          # Additive dataset — training inputs  (500 x 4)
├── Additive_ytrain.npy          # Additive dataset — training targets (500 x 1)
├── Additive_Xtest.npy           # Additive dataset — test inputs      (5000 x 4)
├── Additive_ytest.npy           # Additive dataset — test targets     (5000 x 1)
│
├── Multiplicative_Xtrain.npy    # Multiplicative dataset — training inputs  (500 x 4)
├── Multiplicative_ytrain.npy    # Multiplicative dataset — training targets (500 x 1)
├── Multiplicative_Xtest.npy     # Multiplicative dataset — test inputs      (5000 x 4)
├── Multiplicative_ytest.npy     # Multiplicative dataset — test targets     (5000 x 1)
│
├── Additive_torch_loss.png      # Loss curves — Additive, PyTorch    (Fig. 3 equivalent)
├── Additive_numpy_loss.png      # Loss curves — Additive, NumPy
├── Additive_behavior.png        # Behavior plot — Additive            (Fig. 4 equivalent)
│
├── Multiplicative_torch_loss.png  # Loss curves — Multiplicative, PyTorch  (Fig. 5 equivalent)
├── Multiplicative_numpy_loss.png  # Loss curves — Multiplicative, NumPy
└── Multiplicative_behavior.png    # Behavior plot — Multiplicative          (Fig. 6 equivalent)
```

---

## Requirements

```bash
pip install torch numpy matplotlib scipy
```

---

## How to Run

```bash
python isnn_implementation.py
```

This will:
- Generate both toy datasets using Latin Hypercube Sampling
- Train FFNN, ISNN-1, and ISNN-2 in both PyTorch and NumPy
- Save all datasets as `.npy` files
- Save all plots as `.png` files

---

## Datasets

Both datasets have 4 inputs: `x, y, t, z`, all sampled from `[0, 4]` for training.

| Dataset | Function | Test Range |
|---|---|---|
| Additive | `exp(-0.5x) + log(1+exp(0.4y)) + tanh(t) + sin(z) - 0.4` | `[0, 6]` |
| Multiplicative | `exp(-0.3x) · (0.15y)² · tanh(0.3t) · (0.2·sin(0.5z+2)+0.5)` | `[0, 10]` |

The test range deliberately extends beyond training to measure **extrapolation**.

---

## Architectures

### Input Constraints

| Input | Constraint | Activation | Weight Condition |
|---|---|---|---|
| `x₀` | Convex | Softplus | `W_xx ≥ 0` (layers ≥ 1) |
| `y₀` | Convex + Monotone | Softplus | `W_yy, W_xy ≥ 0` |
| `t₀` | Monotone only | Sigmoid | `W_tt, W_xt ≥ 0` |
| `z₀` | Arbitrary | Sigmoid | No restriction |

### ISNN-1
- Each input has its **own sub-network** with potentially different depth
- Sub-network outputs merge into the **first layer of the x-branch only**
- Hyperparameters: `hidden=10`, `Hx=Hy=Ht=Hz=2` → ~1,600 parameters

### ISNN-2
- All branches share the **same depth H**
- Skip connections from `x₀` and all sub-branches feed into **every x-branch layer**
- Hyperparameters: `hidden=15`, `H=2` → ~1,877 parameters

### FFNN (Baseline)
- Standard 2-hidden-layer network, Tanh activations, no constraints
- `hidden=30` → ~2,041 parameters

---

## Manual Backpropagation

The NumPy implementation computes all gradients by hand. At each layer:

```
dL/dz_h   =  dL/da_h  ⊙  σ'(z_h)          # element-wise (Hadamard product)
dL/dW_h   =  dL/dz_hᵀ  @  a_{h-1}         # weight gradient
dL/da_{h-1} =  dL/dz_h  @  W_h            # propagate backwards
```

For the branching ISNN structure, gradients at the merge point are **split and routed** back through each sub-network independently.

Activation derivatives:
- **Softplus**: `σ'(x) = sigmoid(x)`
- **Sigmoid**: `σ'(x) = σ(x)(1 − σ(x))`

The Adam optimizer is also implemented from scratch with per-parameter momentum.

---

## Results Summary

Test MSE after 3,000 epochs (Adam, lr=0.001):

| Model | Additive Test MSE | Multiplicative Test MSE |
|---|---|---|
| FFNN (PyTorch) | 0.2627 | 0.0101 |
| ISNN-1 (PyTorch) | 0.0694 | 0.0215 |
| ISNN-2 (PyTorch) | 0.0561 | 0.0208 |
| FFNN (NumPy) | 0.2159 | 0.0057 |
| ISNN-1 (NumPy) | 0.0829 | 0.0216 |
| ISNN-2 (NumPy) | 0.0200 | 0.0200 |

**Key finding:** ISNNs significantly outperform FFNN on out-of-distribution test data. The structural constraints act as implicit regularization, enabling better extrapolation.

---

## Reference

> Jadoon, A., Seidl, D. T., Jones, R. E., & Fuhg, J. (2025).
> *Input Specific Neural Networks.* arXiv:2503.00268v1.
