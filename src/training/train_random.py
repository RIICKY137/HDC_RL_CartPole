import csv
import os
import sys
from collections import deque

import gymnasium as gym


# Allow Python to find src/agents when running this file directly
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(SRC_DIR)

from agents.random_agent import RandomAgent


def train_random_agent(
    env_name="CartPole-v1",
    episodes=100,
    seed=42,
    output_path="experiments/results/random_agent_cartpole.csv",
):
    env = gym.make(env_name)
    agent = RandomAgent(env.action_space)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    recent_rewards = deque(maxlen=100)
    results = []

    for episode in range(1, episodes + 1):
        observation, info = env.reset(seed=seed + episode)

        total_reward = 0
        episode_length = 0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(observation)

            observation, reward, terminated, truncated, info = env.step(action)

            total_reward += reward
            episode_length += 1

        recent_rewards.append(total_reward)
        mean_reward_last_100 = sum(recent_rewards) / len(recent_rewards)

        results.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "episode_length": episode_length,
                "mean_reward_last_100": mean_reward_last_100,
            }
        )

        print(
            f"Episode {episode:3d} | "
            f"Reward: {total_reward:6.1f} | "
            f"Length: {episode_length:4d} | "
            f"Mean100: {mean_reward_last_100:6.2f}"
        )

    env.close()

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "episode",
                "total_reward",
                "episode_length",
                "mean_reward_last_100",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    train_random_agent()