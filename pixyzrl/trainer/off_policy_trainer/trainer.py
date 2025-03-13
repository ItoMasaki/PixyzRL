import torch

from pixyzrl.environments import BaseEnv
from pixyzrl.logger import Logger
from pixyzrl.memory import BaseBuffer
from pixyzrl.models.base_model import RLModel
from pixyzrl.trainer.base_trainer import BaseTrainer


class OffPolicyTrainer(BaseTrainer):
    """Trainer class for off-policy reinforcement learning methods (e.g., DQN, DDPG)."""

    def __init__(self, env: BaseEnv, memory: BaseBuffer, agent: RLModel, device: torch.device | str = "cpu", logger: Logger | None = None) -> None:
        super().__init__(env, memory, agent, device, logger)

    def collect_experiences(self) -> None:
        obs, info = self.env.reset()
        done = False

        while not done:
            action = self.agent.select_action({"o": obs.to(self.device)})

            if self.env.is_discrete:
                next_obs, reward, done, _, _ = self.env.step(torch.argmax(action[self.agent.action_var].cpu()))
            else:
                next_obs, reward, done, _, _ = self.env.step(action[self.agent.action_var].cpu().numpy())

            self.memory.add(obs=obs, action=action[self.agent.action_var].cpu().numpy(), reward=reward, done=done, value=action[self.agent.critic.var[0]].cpu().detach())
            obs = next_obs

        if self.logger:
            self.logger.log("Collected off-policy experiences.")

    def train_model(self, batch_size: int = 128, num_epochs: int = 4) -> None:
        if len(self.memory) < self.memory.buffer_size:
            return

        total_loss = self.agent.train_step(self.memory, batch_size, num_epochs)

        if self.logger:
            self.logger.log(f"Off-policy training step completed. Loss: {total_loss / num_epochs}")

    def train(self, num_iterations: int, batch_size: int = 128, num_epochs: int = 4) -> None:
        for iteration in range(num_iterations):
            self.collect_experiences()
            self.train_model(batch_size, num_epochs)
            if self.logger:
                self.logger.log(f"Off-policy Iteration {iteration + 1}/{num_iterations} completed.")
