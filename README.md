# HDC_RL_CartPole

CartPole-v1 强化学习实验项目，包含 **Baseline DQN** 与 **HDC 状态编码 + DQN（HDC-DQN）** 两种方案，便于对比传统神经网络与超维计算（Hyperdimensional Computing, HDC）在 RL 中的效果。

## 项目结构

```
HDC_RL_CartPole/
├── src/
│   ├── hdc/                    # HDC 核心：超向量运算与 CartPole 状态编码
│   ├── hdc_dqn_cartpole/       # HDC-DQN 训练、评估
│   ├── dqn_cartpole_v2/        # Baseline DQN（v2 版本）
│   ├── agents/                 # 早期 agent 实现
│   ├── training/               # 训练脚本与 ablation
│   └── utils/                  # 绘图与指标对比工具
├── artifacts/                  # 训练指标与图表（本地生成）
├── experiments/results/        # 早期 CSV 实验结果
└── requirements.txt
```

## 环境安装

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## 快速开始

所有训练命令需在 `src/` 目录下执行：

```bash
cd src
```

### Baseline DQN

```bash
python -m dqn_cartpole_v2.train
python -m dqn_cartpole_v2.evaluate --checkpoint artifacts/checkpoints/cartpole_dqn.pt
```

### HDC-DQN

CartPole 4 维连续状态经 **Level Encoding + Binding + Bundling** 映射为固定长度超向量，再输入 Q 网络：

```
状态 (4D) → HDC Encoder → 超向量 (8192D) → Q Network → Q 值
```

```bash
python -m hdc_dqn_cartpole.train
python -m hdc_dqn_cartpole.evaluate --checkpoint artifacts/checkpoints/cartpole_hdc_dqn.pt
```

### 对比训练曲线

```bash
python utils/compare_metrics.py \
  --baseline-metrics ../artifacts/train_metrics.json \
  --hdc-metrics artifacts/hdc_train_metrics.json \
  --output ../artifacts/comparison_dqn_vs_hdc_dqn.png
```

### HDC 超参 Ablation

```bash
python training/run_hdc_ablation.py --episodes 500
```

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--episodes` | 1500 | 训练 episode 数 |
| `--hdc-dim` | 8192 (2^13) | 超向量维度 |
| `--hdc-n-bins` | 10 | 每个状态维度的离散档位数 |
| `--hidden-sizes` | 256 128 | Q 网络隐藏层（HDC-DQN） |
| `--learning-rate` | 5e-4 | Adam 学习率 |
| `--epsilon-decay` | 0.997 | ε-greedy 衰减系数 |

完整参数可通过 `--help` 查看，例如：

```bash
python -m hdc_dqn_cartpole.train --help
```

## HDC 编码说明

- **Binding**：逐元素乘法（`a * b`），组合位置超向量与 level 超向量
- **Bundling**：多路超向量求和后 L2 归一化，合成最终状态表示
- **Replay Buffer**：仍存储原始 4D 状态，训练时再编码，节省内存

默认 `hdc_dim=8192`（2^13），为最接近 10000 的 2 的幂，便于按位运算与内存对齐。

## 实验结果（参考）

在 CartPole-v1 上（1500 episodes，seed=7）：

| 方法 | 平均 Reward | 最佳滑动平均 | 最佳验证 Reward |
|------|-------------|--------------|-----------------|
| Baseline DQN | ~199 | 500 | 500 |
| HDC-DQN | ~59 | ~131 | ~164 |

HDC 离散化编码会损失连续状态精度，CartPole 对角度/速度较敏感，因此当前 HDC-DQN 弱于直接 MLP。Ablation 显示增大 `n_bins`（如 20）可改善 HDC 表现。

后续可尝试：混合输入（原始状态 + HDC 向量）、连续投影编码、或纯 HDC Q-learning。

## 输出文件

| 路径 | 说明 |
|------|------|
| `artifacts/train_metrics.json` | Baseline DQN 训练指标 |
| `artifacts/hdc_train_metrics.json` | HDC-DQN 训练指标 |
| `artifacts/checkpoints/*.pt` | 模型 checkpoint |
| `artifacts/comparison_dqn_vs_hdc_dqn.png` | 对比曲线 |
| `artifacts/ablation/` | 超参 ablation 结果 |

## 依赖

- Python 3.9+
- [Gymnasium](https://gymnasium.farama.org/) — CartPole-v1 环境
- PyTorch — DQN 网络与训练
- NumPy / Matplotlib — HDC 编码与可视化

详见 `requirements.txt`。

## License

MIT（可按需修改）
