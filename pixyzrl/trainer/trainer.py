import torch

from pixyzrl.logger import Logger


class Trainer:
    """Trainer class to manage training of reinforcement learning agents."""

    def __init__(self, env, memory, agent, device, logger: Logger = None, off_policy: bool = False):
        """Initialize the trainer with the environment, memory, agent, and optional logger.

        :param env: Environment for training.
        :param memory: Memory buffer.
        :param agent: RL agent.
        :param device: Device to run training on.
        :param logger: Optional logger.
        :param off_policy: Whether the agent follows an off-policy method (DQN, DDPG, etc.).
        """
        self.env = env
        self.memory = memory
        self.agent = agent
        self.device = device
        self.logger = logger
        self.off_policy = off_policy

        if self.logger:
            self.logger.log("Trainer initialized.")

    def collect_experiences(self):
        """Collect experiences from the environment and store them in memory."""
        obs, info = self.env.reset()
        done = False

        while not done:
            action = self.agent.select_action(torch.tensor(obs, dtype=torch.float32, device=self.device))

            if isinstance(self.env.action_space, torch.distributions.Categorical):
                next_obs, reward, done, _, _ = self.env.step(torch.argmax(action["a"]))
            else:
                next_obs, reward, done, _, _ = self.env.step(action["a"].cpu().numpy())

            # Store experience in memory
            self.memory.add(obs=obs, action=action["a"].cpu().numpy(), reward=reward, done=done, value=action["v"].cpu().numpy())

            obs = next_obs

            if self.off_policy:
                self.train_step()  # Train during experience collection for off-policy methods

        if self.logger:
            self.logger.log("Collected experiences from environment.")

    def train_step(self):
        """Perform a single training step."""
        if len(self.memory) < 128:
            return  # Ensure sufficient samples

        batch = self.memory.sample(128)
        loss = self.agent.train(batch)

        if self.logger:
            self.logger.log(f"Training step completed. Loss: {loss}")

    def train(self, num_iterations: int):
        """Train the agent for a given number of iterations."""
        for iteration in range(num_iterations):
            self.collect_experiences()

            if not self.off_policy:  # Train after collecting trajectories for on-policy methods
                self.train_step()
                self.memory.clear()

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
