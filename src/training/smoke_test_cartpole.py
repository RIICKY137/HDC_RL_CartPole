import gymnasium as gym


def main():
    env = gym.make("CartPole-v1")

    obs, info = env.reset(seed=42)

    print("Initial observation:", obs)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    total_reward = 0

    for step in range(1000):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode ended at step {step + 1}")
            print(f"Total reward: {total_reward}")
            obs, info = env.reset()
            total_reward = 0

    env.close()


if __name__ == "__main__":
    main()