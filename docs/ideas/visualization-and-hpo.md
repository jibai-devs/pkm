# Visualization & Hyperparameter Optimization Tools

**Status:** Reference doc (Jul 2026). Covers monitoring, visualization, and
automated hyperparameter search.

---

## Monitoring & Visualization

### What we have

| Tool | Status | What it does |
|---|---|---|
| **TensorBoard** | Built-in | Real-time training dashboards |
| **CSV metrics** | Built-in | Flat logs for offline analysis |
| **Plotly notebook** | Built-in | Interactive charts from CSV |
| **wandb** | Integrated | Cloud-hosted dashboards, run comparison |

### TensorBoard (default)

Already integrated. Every training run writes to `runs/`:

```bash
tensorboard --logdir=runs
# opens http://localhost:6006
```

Logs: loss curves, win rate, entropy, clip fraction, eval metrics.

### wandb (Weights & Biases)

Cloud-hosted experiment tracking. Enable with `--wandb-project`:

```bash
# PPO training with wandb
pkm train --agent 02_dragapult --iterations 100 --wandb-project pkm-ppo

# expert iteration with wandb
pkm exit-train --agent 02_dragapult --iterations 20 --wandb-project pkm-exit

# via justfile
just train-wandb 02_dragapult pkm-ppo
just exit-wandb 02_dragapult pkm-exit
```

First run will ask you to log in (`wandb login`). Free tier is generous.

Gives you:
- Web dashboard (hosted, shareable via URL)
- **Run comparison** — overlay different hyperparameters side by side
- System metrics (CPU, memory)
- Model versioning
- Team collaboration

### Architecture visualization

**Netron** — visualize the network as a graph:

```bash
pip install netron
python -c "
import torch, netron
from pkm.rl.model import PolicyValueNet
model = PolicyValueNet()
dummy = torch.randn(1, 641)
torch.onnx.export(model.state_encoder, dummy, 'model.onnx')
netron.start('model.onnx')
"
```

Shows every layer, input/output shapes, connections. Good for documentation.

**torchviz** — computation graph of a forward pass:

```bash
pip install torchviz
```

```python
from torchviz import make_dot
y = model.act(d)
make_dot(y.value, params=dict(model.named_params())).render("graph")
```

Shows the exact gradient flow — useful for debugging.

**HiddenLayer** — simpler architecture diagram:

```bash
pip install hiddenlayer
```

```python
import hiddenlayer as hl
hl.build_graph(model, torch.zeros(1, 641)).save("arch.png")
```

### Custom Plotly charts

The notebook `notebooks/training_monitor.ipynb` already has interactive charts.
To compare multiple runs:

```python
import plotly.graph_objects as go
import pandas as pd

runs = {
    "lr=3e-4": pd.read_csv("metrics/ppo_train.csv"),
    "lr=1e-3": pd.read_csv("runs/lr_1e-3/ppo_train.csv"),
}

fig = go.Figure()
for name, df in runs.items():
    fig.add_trace(go.Scatter(x=df["iter"], y=df["eval_win_rate"], name=name))
fig.update_layout(title="Win Rate vs Learning Rate", xaxis_title="Iteration")
fig.show()
```

---

## Hyperparameter Optimization

### Optuna (automated search)

Integrated via `pkm sweep`. Searches over hyperparameters to maximize win rate:

```bash
# PPO sweep (50 trials)
pkm sweep --trials 50 --iterations 20 --games 8

# expert iteration sweep
pkm sweep exit --trials 30 --iterations 10

# with agent profile
pkm sweep --agent 02_dragapult --trials 100

# via justfile
just sweep 02_dragapult 50
just sweep-exit 02_dragapult 30
```

**What it searches:**

For PPO:
| Parameter | Range | Scale |
|---|---|---|
| `lr` | 1e-5 to 1e-2 | log |
| `gamma` | 0.95 to 0.999 | linear |
| `lam` | 0.9 to 0.99 | linear |
| `shaping_coef` | 0.0 to 0.5 | linear |
| `pool_size` | 4 to 16 | integer |
| `pool_prob` | 0.2 to 0.8 | linear |

For expert iteration:
| Parameter | Range | Scale |
|---|---|---|
| `lr` | 1e-5 to 1e-2 | log |
| `n_simulations` | 16 to 64 | step 8 |
| `n_determinizations` | 1 to 4 | integer |

**Persistent storage** — resume interrupted sweeps:

```bash
pkm sweep --trials 50 --storage sqlite:///sweep.db
# re-running picks up where it left off
pkm sweep --trials 100 --storage sqlite:///sweep.db
```

**Visualization** — Optuna has built-in charts:

```python
import optuna

study = optuna.load_study(study_name="ppo_sweep", storage="sqlite:///sweep.db")

# parameter importances
optuna.visualization.plot_param_importances(study)

# contour plot (2D slice of the search space)
optuna.visualization.plot_contour(study, params=["lr", "gamma"])

# optimization history
optuna.visualization.plot_optimization_history(study)

# parallel coordinate
optuna.visualization.plot_parallel_coordinate(study)
```

### Manual sweeps

For quick A/B tests, just change one parameter:

```bash
just train 02_dragapult 100 16 1e-3   # lr=1e-3
just train 02_dragapult 100 16 3e-4   # lr=3e-4
just train 02_dragapult 100 16 1e-4   # lr=1e-4
```

Compare the CSV metrics or overlay on TensorBoard/wandb.

---

## Metric Logging Architecture

Logging uses a pluggable backend system (`pkm/rl/logging.py`):

```
MetricLog
    ├── TensorBoardBackend  (always on)
    ├── WandbBackend        (if --wandb-project)
    └── CsvBackend          (always on, via csv.DictWriter)
```

Each backend implements the `Backend` protocol:

```python
class Backend(Protocol):
    def scalar(self, tag: str, value: float, step: int) -> None: ...
    def close(self) -> None: ...
```

To add a new backend (e.g., MLflow, custom HTTP endpoint), just implement
the protocol and register it:

```python
log = MetricLog()
log.add_tensorboard("runs/ppo")
log.add(MyCustomBackend(...))
```

---

## Quick Reference

| Task | Command |
|---|---|
| TensorBoard | `tensorboard --logdir=runs` |
| Training + wandb | `just train-wandb` |
| PPO sweep | `just sweep` |
| ExIt sweep | `just sweep-exit` |
| Compare runs | wandb dashboard or Plotly notebook |
| Visualize architecture | Netron / torchviz / HiddenLayer |
