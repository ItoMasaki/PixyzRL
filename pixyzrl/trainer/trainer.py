import torch
from pixyz.models import Model

from pixyzrl.environments.vectorized_env import VectorizedEnv
from pixyzrl.memory.vectorized_memory import VectorizedReplayBuffer


class Trainer:
    """Trainer class to manage training of reinforcement learning agents."""

    def __init__(self, env: VectorizedEnv, memory: VectorizedReplayBuffer, agent: Model, device: str) -> None:
        """Initialize the trainer with the environment, memory, and agent."""
        self.env = env
        self.memory = memory
        self.agent = agent
        self.device = device

    def collect_experiences(self) -> None:
        """Collect experiences from the environment and store them in memory."""
        obs = self.env.reset()
        done = torch.zeros(self.env.get_num_envs(), dtype=torch.bool, device=self.device)

        while not done.all():
            action_dict = self.agent.select_action(torch.tensor(obs, dtype=torch.float32, device=self.device))
            action = action_dict["a"]
            next_obs, reward, dones, _ = self.env.step(action.cpu().numpy())

            self.memory.add(obs, action.cpu().numpy(), reward, dones)
            obs = next_obs
            done = torch.tensor(dones, dtype=torch.bool, device=self.device)

    def train(self, num_iterations: int) -> None:
        """Train the agent for a given number of iterations."""
        for iteration in range(num_iterations):
            print(f"Iteration {iteration + 1}/{num_iterations}")
            self.collect_experiences()
            batch = self.memory.sample()
            self.agent.update(batch)
            self.memory.buffer.clear()

    def save_model(self, path: str) -> None:
        """Save the trained model."""
        torch.save(self.agent.state_dict(), path)

    def load_model(self, path: str) -> None:
        """Load a trained model."""
        self.agent.load_state_dict(torch.load(path, map_location=self.device))
