import csv
import os
import sys

import matplotlib.pyplot as plt


def load_results(csv_path):
    episodes = []
    rewards = []
    mean_rewards = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)

        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["total_reward"]))
            mean_rewards.append(float(row["mean_reward_last_100"]))

    return episodes, rewards, mean_rewards


def plot_results(csv_path, output_path=None, title=None):
    episodes, rewards, mean_rewards = load_results(csv_path)

    if output_path is None:
        base, _ = os.path.splitext(csv_path)
        output_path = base + ".png"

    if title is None:
        filename = os.path.basename(csv_path)
        title = filename.replace("_", " ").replace(".csv", "")

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, label="Episode Reward", alpha=0.5)
    plt.plot(episodes, mean_rewards, label="Mean Reward Last 100 Episodes", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/utils/plot_results.py <csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    plot_results(csv_path)