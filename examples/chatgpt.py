import torch

from pixyzrl.environments import Env


def test_environment(env_name="CartPole-v1", num_steps=10):
    """Test a reinforcement learning environment."""
    env = Env(env_name, action_var="a")  # `action_var` を明示的に指定
    obs, info = env.reset()
    print(f"Initial Observation: {obs}")

    for step in range(num_steps):
        action_value = env.action_space.sample()  # Select a random action
        action_tensor = torch.tensor(action_value, dtype=torch.float32).unsqueeze(0)  # Tensor に変換
        action = {"a": action_tensor}  # 辞書として渡す

        next_obs, reward, done, truncated, info = env.step(action)
        print(f"Step {step}: Action={action}, Reward={reward}, Done={done}")

        if done:
            print("Episode finished. Resetting environment.")
            obs, info = env.reset()

    env.close()


if __name__ == "__main__":
    test_environment()
