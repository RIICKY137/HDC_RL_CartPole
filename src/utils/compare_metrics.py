from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def plot_comparison(
    metrics_paths: list[Path],
    labels: list[str],
    output_path: Path,
) -> None:
    if len(metrics_paths) != len(labels):
        raise ValueError("metrics_paths and labels must have the same length")

    plt.figure(figsize=(12, 6))

    for metrics_path, label in zip(metrics_paths, labels):
        metrics = load_metrics(metrics_path)
        config = metrics.get("config", {})
        episode_rewards = metrics["episode_rewards"]
        moving_average_rewards = metrics["moving_average_rewards"]
        episodes = list(range(1, len(episode_rewards) + 1))
        window = config.get("moving_average_window", 20)

        plt.plot(episodes, episode_rewards, alpha=0.15)
        plt.plot(
            episodes,
            moving_average_rewards,
            linewidth=2,
            label=f"{label} (MA-{window})",
        )

    solved_threshold = load_metrics(metrics_paths[0]).get("config", {}).get("solved_threshold", 475.0)
    plt.axhline(y=solved_threshold, linestyle="--", color="gray", label=f"Solved ({solved_threshold})")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("CartPole-v1: Baseline DQN vs HDC-DQN")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot to: {output_path}")


def summarize_metrics(metrics_path: Path, label: str) -> None:
    metrics = load_metrics(metrics_path)
    print(
        f"{label}: "
        f"mean={metrics['mean_reward']:.1f}, "
        f"best_ma={metrics['best_moving_average_reward']:.1f}, "
        f"best_val={metrics.get('best_validation_reward', float('nan')):.1f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare DQN and HDC-DQN training metrics.")
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("artifacts/train_metrics.json"),
    )
    parser.add_argument(
        "--hdc-metrics",
        type=Path,
        default=Path("artifacts/hdc_train_metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/comparison_dqn_vs_hdc_dqn.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summarize_metrics(args.baseline_metrics, "Baseline DQN")
    summarize_metrics(args.hdc_metrics, "HDC-DQN")
    plot_comparison(
        metrics_paths=[args.baseline_metrics, args.hdc_metrics],
        labels=["Baseline DQN", "HDC-DQN"],
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
