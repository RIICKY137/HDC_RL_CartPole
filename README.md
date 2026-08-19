# HDC_RL_CartPole

Reinforcement learning experiments on **CartPole-v1**, comparing a **Baseline DQN** against **HDC-encoded DQN (HDC-DQN)** — a hybrid approach that maps environment states into hyperdimensional vectors before Q-value estimation.

## Project Structure

```
HDC_RL_CartPole/
├── src/
│   ├── hdc/                    # HDC core: hypervector ops & CartPole state encoding
│   ├── hdc_dqn_cartpole/       # HDC-DQN training & evaluation
│   ├── dqn_cartpole_v2/        # Baseline DQN (v2)
│   ├── agents/                 # Legacy agent implementations
│   ├── training/               # Training scripts & ablation runner
│   └── utils/                  # Plotting & metric comparison tools
├── artifacts/                  # Training metrics & plots (generated locally)
├── experiments/results/        # Legacy CSV experiment outputs
└── requirements.txt
```

## Setup

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

Run all training commands from the `src/` directory:

```bash
cd src
```

### Baseline DQN

```bash
python -m dqn_cartpole_v2.train
python -m dqn_cartpole_v2.evaluate --checkpoint artifacts/checkpoints/cartpole_dqn.pt
```

### HDC-DQN

CartPole's 4D continuous observation is encoded via **Level Encoding + Binding + Bundling** into a fixed-size hypervector, then fed into the Q-network:

```
State (4D) → HDC Encoder → Hypervector (8192D) → Q Network → Q-values
```

```bash
python -m hdc_dqn_cartpole.train
python -m hdc_dqn_cartpole.evaluate --checkpoint artifacts/checkpoints/cartpole_hdc_dqn.pt
```

### Compare Training Curves

```bash
python utils/compare_metrics.py \
  --baseline-metrics ../artifacts/train_metrics.json \
  --hdc-metrics artifacts/hdc_train_metrics.json \
  --output ../artifacts/comparison_dqn_vs_hdc_dqn.png
```

### HDC Hyperparameter Ablation

```bash
python training/run_hdc_ablation.py --episodes 500
```

## Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episodes` | 1500 | Number of training episodes |
| `--hdc-dim` | 8192 (2^13) | Hypervector dimensionality |
| `--hdc-n-bins` | 10 | Discretization bins per state dimension |
| `--hidden-sizes` | 256 128 | Q-network hidden layers (HDC-DQN) |
| `--learning-rate` | 5e-4 | Adam learning rate |
| `--epsilon-decay` | 0.997 | Epsilon-greedy decay factor |

See full CLI options with `--help`:

```bash
python -m hdc_dqn_cartpole.train --help
```

## HDC Encoding

- **Binding**: element-wise multiplication (`a * b`) to combine positional and level hypervectors
- **Bundling**: sum hypervectors and L2-normalize to produce the final state representation
- **Replay buffer**: stores raw 4D states; encoding happens at train/act time to save memory

The default `hdc_dim=8192` (2^13) is the power of two closest to 10,000, which aligns well with bitwise operations and memory layout.

## Reference Results

On CartPole-v1 (1500 episodes, seed=7):

| Method | Mean Reward | Best Moving Avg | Best Validation Reward |
|--------|-------------|-----------------|------------------------|
| Baseline DQN | ~199 | 500 | 500 |
| HDC-DQN | ~59 | ~131 | ~164 |

Discretizing continuous states loses precision; CartPole is sensitive to pole angle and angular velocity, so the current HDC-DQN underperforms a direct MLP. Ablation suggests increasing `n_bins` (e.g. 20) improves HDC performance.

Possible next steps: hybrid input (raw state + HDC vector), continuous projection encoding, or pure HDC Q-learning.

## Output Files

| Path | Description |
|------|-------------|
| `artifacts/train_metrics.json` | Baseline DQN training metrics |
| `artifacts/hdc_train_metrics.json` | HDC-DQN training metrics |
| `artifacts/checkpoints/*.pt` | Model checkpoints |
| `artifacts/comparison_dqn_vs_hdc_dqn.png` | Side-by-side learning curves |
| `artifacts/ablation/` | Hyperparameter ablation results |

## Dependencies

- Python 3.9+
- [Gymnasium](https://gymnasium.farama.org/) — CartPole-v1 environment
- PyTorch — DQN network and training
- NumPy / Matplotlib — HDC encoding and visualization

See `requirements.txt` for pinned versions.

## License

MIT (modify as needed)
