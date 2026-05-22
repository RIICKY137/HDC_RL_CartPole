import csv
import os
import random
import sys
from collections import deque

import gymnasium as gym
import numpy as np
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(SRC_DIR)

from agents.dqn_agent import DQNAgent
from utils.replay_buffer import ReplayBuffer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_dqn(
    env_name="CartPole-v1",
    episodes=500,
    seed=42,
    batch_size=64,
    replay_capacity=10000,
    min_replay_size=1000,
    output_path="experiments/results/dqn_cartpole.csv",
    model_path="models/dqn_cartpole.pt",
):
    set_seed(seed)

    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=500,
        target_update_freq=10,
    )

    replay_buffer = ReplayBuffer(capacity=replay_capacity)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    recent_rewards = deque(maxlen=100)
    results = []

    for episode in range(1, episodes + 1):
        state, info = env.reset(seed=seed + episode)

        total_reward = 0.0
        episode_length = 0
        losses = []

        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = agent.select_action(state)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            episode_length += 1

            if len(replay_buffer) >= min_replay_size:
                transitions = replay_buffer.sample(batch_size)
                loss = agent.update(transitions)
                losses.append(loss)

        if episode % agent.target_update_freq == 0:
            agent.update_target_network()

        recent_rewards.append(total_reward)
        mean_reward_last_100 = sum(recent_rewards) / len(recent_rewards)
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        epsilon = agent.get_epsilon()

        results.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "episode_length": episode_length,
                "mean_reward_last_100": mean_reward_last_100,
                "mean_loss": mean_loss,
                "epsilon": epsilon,
            }
        )

        print(
            f"Episode {episode:4d} | "
            f"Reward: {total_reward:6.1f} | "
            f"Length: {episode_length:4d} | "
            f"Mean100: {mean_reward_last_100:7.2f} | "
            f"Loss: {mean_loss:8.4f} | "
            f"Epsilon: {epsilon:6.3f}"
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
                "mean_loss",
                "epsilon",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    agent.save(model_path)

    print(f"\nSaved results to: {output_path}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    train_dqn()