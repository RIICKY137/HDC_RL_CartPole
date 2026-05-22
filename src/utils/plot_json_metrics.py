import json
import os
import sys

import matplotlib.pyplot as plt


def plot_dqn_metrics(
    json_path,
    output_path=None,
):
    with open(json_path, "r") as f:
        metrics = json.load(f)

    config = metrics.get("config", {})

    episode_rewards = metrics["episode_rewards"]
    moving_average_rewards = metrics["moving_average_rewards"]

    episodes = list(range(1, len(episode_rewards) + 1))

    solved_threshold = config.get("solved_threshold", 475.0)
    moving_average_window = config.get("moving_average_window", 20)

    if output_path is None:
        base, _ = os.path.splitext(json_path)
        output_path = base + ".png"

    plt.figure(figsize=(12, 6))

    plt.plot(
        episodes,
        episode_rewards,
        label="Episode Reward",
        alpha=0.35,
    )

    plt.plot(
        episodes,
        moving_average_rewards,
        label=f"Moving Average Reward ({moving_average_window} episodes)",
        linewidth=2,
    )

    plt.axhline(
        y=solved_threshold,
        linestyle="--",
        label=f"Solved Threshold ({solved_threshold})",
    )

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN-v2 Training Performance on CartPole-v1")
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/utils/plot_json_metrics.py <metrics_json_path>")
        sys.exit(1)

    json_path = sys.argv[1]
    plot_dqn_metrics(json_path)