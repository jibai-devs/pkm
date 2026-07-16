# Optimizer Reference

**Status:** Reference doc (Jul 2026). Covers optimizers used in training and
their tradeoffs.

---

## What an optimizer does

The loss function computes a single number ("how bad was this prediction?").
`loss.backward()` computes the gradient — for each parameter, "which direction
would make the loss go up?" The optimizer then updates each parameter in the
**opposite** direction (downhill).

The question is: **how big of a step do you take?** Different optimizers answer
this differently.

---

## The optimizers

### SGD (Stochastic Gradient Descent)

The simplest. Just move each parameter by `lr * gradient`:

```
param -= lr * grad
```

- All parameters get the same learning rate
- Problem: parameters with sparse gradients (like embedding tables — most card
  IDs never appear in a single batch) get infrequent updates and learn slowly

### SGD + Momentum

Adds a "velocity" term — accumulates past gradients with decay:

```
velocity = beta * velocity + grad
param -= lr * velocity
```

- Helps the optimizer roll through flat regions and small local minima
- Like a ball rolling downhill — builds up speed in consistent directions
- `beta` typically 0.9

### Adagrad (Adaptive Gradient)

Tracks per-parameter gradient history and **shrinks the learning rate for
frequently-updated parameters**:

```
cache += grad^2
param -= lr * grad / (sqrt(cache) + eps)
```

- Parameters that get large/frequent gradients → large cache → smaller
  effective LR
- Parameters that get rare/small gradients → small cache → larger effective LR
- Great for sparse features (card embeddings!)
- **Problem:** cache only grows, so LR shrinks to zero over time and the
  optimizer stops learning

### Adadelta

Fixes Adagrad's "LR dies" problem by using an exponential moving average
instead of accumulating forever:

```
cache = rho * cache + (1-rho) * grad^2
param -= rms(delta) / rms(cache) * grad
```

- No manual LR needed — self-adjusting
- But in practice slower to converge than Adam

### RMSprop

Hinton's informal lecture notes fix for Adagrad. Same idea as Adadelta but
simpler:

```
cache = decay * cache + (1-decay) * grad^2
param -= lr * grad / (sqrt(cache) + eps)
```

- Just Adagrad with exponential decay on the cache
- Keeps adaptive per-parameter LR but doesn't let it die
- Popular for RNNs

### Adam (Adaptive Moments) — **what we use**

Combines **momentum** (first moment = mean of gradients) and **RMSprop**
(second moment = variance of gradients):

```
m = beta1 * m + (1-beta1) * grad          # momentum (mean)
v = beta2 * v + (1-beta2) * grad^2         # RMSprop (variance)
m_hat = m / (1 - beta1^t)                  # bias correction
v_hat = v / (1 - beta2^t)                  # bias correction
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

Default betas: `beta1=0.9`, `beta2=0.999`, `eps=1e-8`.

**Bias correction** is the key detail — `m` and `v` start at zero, so early
steps are biased toward zero. The correction divides by `(1 - beta^t)` to
compensate, making early updates larger.

### AdamW (Adam with Weight Decay)

Adam but weight decay is **decoupled** from the gradient update:

```
# Adam update (same as above)
param -= lr * m_hat / (sqrt(v_hat) + eps)
# Plus explicit weight decay
param -= lr * wd * param
```

In vanilla Adam, weight decay was implemented as L2 regularization (added to
gradient), which interacts badly with the adaptive LR. AdamW separates them.

---

## Comparison table

| Optimizer | Per-param LR | Momentum | Bias correction | Weight decay | Best for |
|---|:--:|:--:|:--:|:--:|---|
| SGD | No | No | No | L2 | Final convergence quality |
| SGD+Momentum | No | Yes | No | L2 | Computer vision |
| Adagrad | Yes | No | No | No | Sparse features |
| Adadelta | Yes | No | No | No | No LR tuning |
| RMSprop | Yes | No | No | No | RNNs |
| **Adam** | Yes | Yes | Yes | L2 | Default, fast convergence |
| AdamW | Yes | Yes | Yes | Decoupled | Modern default |

## Why Adam for RL

Adam is the "just works" choice for RL specifically because:

1. **Fast convergence** — each rollout is expensive (full games), so you want
   maximum learning per update
2. **Sparse gradients** — card embeddings have 1268 entries but most batches
   only touch ~50-100 card IDs; Adam's per-param LR handles this naturally
3. **Little tuning** — `lr=3e-4` is a solid default; only needs adjustment if
   training is unstable

## Potential upgrade: AdamW

Swap one line in `pkm/rl/train.py`:

```python
# current
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# upgrade
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

Weight decay regularizes the network, prevents overfitting to recent batches.
Most modern training uses AdamW. This is a free improvement with minimal risk.

## What we use

- **PPO training** (`pkm/rl/train.py`): `Adam(lr=3e-4)`
- **Expert iteration** (`pkm/rl/exit_train.py`): `Adam(lr=1e-4)` (lower LR
  because MCTS targets are higher quality — don't want to overshoot)
