from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hdc_dqn_cartpole.config import HDCDQNConfig
from hdc_dqn_cartpole.train import train


def run_ablation(
    episodes: int,
    seed: int,
    output_dir: Path,
    hdc_dims: list[int],
    hdc_n_bins_list: list[int],
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for hdc_dim in hdc_dims:
        config = HDCDQNConfig(
            seed=seed,
            episodes=episodes,
            hdc_dim=hdc_dim,
            hdc_n_bins=10,
            metrics_path=output_dir / f"hdc_dim_{hdc_dim}_metrics.json",
            checkpoint_path=output_dir / f"hdc_dim_{hdc_dim}.pt",
        )
        print(f"\n=== Ablation: hdc_dim={hdc_dim}, n_bins=10 ===")
        metrics = train(config)
        results.append(
            {
                "experiment": "hdc_dim",
                "hdc_dim": hdc_dim,
                "hdc_n_bins": 10,
                "mean_reward": metrics["mean_reward"],
                "best_moving_average_reward": metrics["best_moving_average_reward"],
                "best_validation_reward": metrics["best_validation_reward"],
                "metrics_path": str(config.metrics_path),
            }
        )

    for n_bins in hdc_n_bins_list:
        if n_bins == 10:
            continue
        default_hdc_dim = HDCDQNConfig().hdc_dim
        config = HDCDQNConfig(
            seed=seed,
            episodes=episodes,
            hdc_dim=default_hdc_dim,
            hdc_n_bins=n_bins,
            metrics_path=output_dir / f"hdc_bins_{n_bins}_metrics.json",
            checkpoint_path=output_dir / f"hdc_bins_{n_bins}.pt",
        )
        print(f"\n=== Ablation: hdc_dim={default_hdc_dim}, n_bins={n_bins} ===")
        metrics = train(config)
        results.append(
            {
                "experiment": "hdc_n_bins",
                "hdc_dim": default_hdc_dim,
                "hdc_n_bins": n_bins,
                "mean_reward": metrics["mean_reward"],
                "best_moving_average_reward": metrics["best_moving_average_reward"],
                "best_validation_reward": metrics["best_validation_reward"],
                "metrics_path": str(config.metrics_path),
            }
        )

    summary_path = output_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved ablation summary to {summary_path}")
    return results


def plot_ablation(results: list[dict], output_path: Path) -> None:
    dim_results = [item for item in results if item["experiment"] == "hdc_dim"]
    bin_results = [item for item in results if item["experiment"] == "hdc_n_bins"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if dim_results:
        dims = [item["hdc_dim"] for item in dim_results]
        best_ma = [item["best_moving_average_reward"] for item in dim_results]
        axes[0].bar([str(dim) for dim in dims], best_ma, color="#4C78A8")
        axes[0].set_title("Effect of hdc_dim (n_bins=10)")
        axes[0].set_xlabel("hdc_dim")
        axes[0].set_ylabel("Best Moving Average Reward")
        axes[0].grid(True, axis="y", alpha=0.3)

    if bin_results:
        bins = [item["hdc_n_bins"] for item in bin_results]
        best_ma = [item["best_moving_average_reward"] for item in bin_results]
        axes[1].bar([str(value) for value in bins], best_ma, color="#F58518")
        axes[1].set_title(f"Effect of n_bins (hdc_dim={HDCDQNConfig().hdc_dim})")
        axes[1].set_xlabel("n_bins")
        axes[1].set_ylabel("Best Moving Average Reward")
        axes[1].grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ablation plot to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HDC hyperparameter ablation on CartPole.")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/ablation"))
    parser.add_argument(
        "--hdc-dims",
        type=int,
        nargs="+",
        default=[1024, 4096, 8192],
    )
    parser.add_argument(
        "--hdc-n-bins",
        type=int,
        nargs="+",
        default=[5, 10, 20],
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("artifacts/ablation/hdc_ablation.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = run_ablation(
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        hdc_dims=args.hdc_dims,
        hdc_n_bins_list=args.hdc_n_bins,
    )
    plot_ablation(results, args.plot_path)
    print("\nAblation summary:")
    for item in results:
        print(
            f"  dim={item['hdc_dim']}, bins={item['hdc_n_bins']}: "
            f"best_ma={item['best_moving_average_reward']:.1f}, "
            f"best_val={item['best_validation_reward']:.1f}"
        )
    print(f"Average best_ma across runs: {mean(item['best_moving_average_reward'] for item in results):.1f}")


if __name__ == "__main__":
    main()
