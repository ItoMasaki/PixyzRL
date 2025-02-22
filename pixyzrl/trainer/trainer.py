"""Trainer class to manage training of reinforcement learning agents."""

import torch
from torchrl.objectives.value import ValueEstimatorBase

from pixyzrl.logger import Logger


class Trainer:
    """Trainer class to manage training of reinforcement learning agents."""

    def __init__(self, env, memory, agent, device, value_function: ValueEstimatorBase, logger: Logger = None):
        """Initialize the trainer with the environment, memory, agent, and optional logger.

        :param env: Environment for training.
        :param memory: Memory buffer.
        :param agent: RL agent.
        :param device: Device to run training on.
        :param value_function: An instance of ValueEstimatorBase (e.g., TD0Estimator, TD1Estimator, TDLambdaEstimator, GAE).
        :param logger: Optional logger.
        """
        self.env = env
        self.memory = memory
        self.agent = agent
        self.device = device
        self.value_function = value_function
        self.logger = logger

        if self.logger:
            self.logger.log("Trainer initialized.")

    def compute_value(self, obs, next_obs, reward, done):
        """Compute value based on the selected value function method."""
        return self.value_function(obs, next_obs, reward, done)

    def collect_experiences(self):
        """Collect experiences from the environment and store them in memory."""
        obs = self.env.reset()
        done = torch.zeros(self.env.get_num_envs(), dtype=torch.bool, device=self.device)

        while not done.all():
            action_dict = self.agent.select_action(torch.tensor(obs, dtype=torch.float32, device=self.device))
            action = action_dict["a"]
            next_obs, reward, dones, _ = self.env.step(action.cpu().numpy())
            value = self.compute_value(obs, next_obs, reward, dones)
            self.memory.add(obs, action.cpu().numpy(), reward, dones, value.cpu().numpy())
            obs = next_obs
            done = torch.tensor(dones, dtype=torch.bool, device=self.device)

        if self.logger:
            self.logger.log("Collected experiences from environment.")

    def train(self, num_iterations: int):
        """Train the agent for a given number of iterations."""
        for iteration in range(num_iterations):
            self.collect_experiences()
            batch = self.memory.sample()
            self.agent.update(batch)
            self.memory.buffer.clear()

            if self.logger:
                self.logger.log(f"Iteration {iteration + 1}/{num_iterations} completed.")

    def save_model(self, path: str):
        """Save the trained model."""
        torch.save(self.agent.state_dict(), path)
        if self.logger:
            self.logger.log(f"Model saved at {path}.")

    def load_model(self, path: str):
        """Load a trained model."""
        self.agent.load_state_dict(torch.load(path, map_location=self.device))
        if self.logger:
            self.logger.log(f"Model loaded from {path}.")
