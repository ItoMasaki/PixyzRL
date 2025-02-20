"""Proximal Policy Optimization (PPO) agent using Pixyz."""

# from typing import Any
#

import genesis as gs

# import numpy as np
import torch
from genesis.engine.entities.rigid_entity.rigid_entity import RigidEntity

# from losses import ClipLoss, MSELoss, RatioLoss
# from model import Actor, Critic, FeatureExtractor
# from numpy.typing import NDArray
# from pixyz.losses import Entropy, MinLoss, Parameter
# from pixyz.losses import Expectation as E
# from pixyz.models import Model
# from torch.optim import Adam
#
# from utils.utils import get_env_properties
#
B = 1


#
# ################################## set device ##################################
# device = torch.device("mps")
#
# TWO_CH = 2
# THREE_CH = 3
#
#
# class RolloutBuffer:
#     """Buffer for storing rollout data."""
#
#     def __init__(self) -> None:
#         """Initialize the buffer."""
#         self.actions = []
#         self.states = []
#         self.logprobs = []
#         self.rewards = []
#         self.state_values = []
#         self.is_terminals = []
#
#     def clear(self) -> None:
#         """Clear the buffer."""
#         del self.actions[:]
#         del self.states[:]
#         del self.logprobs[:]
#         del self.rewards[:]
#         del self.state_values[:]
#         del self.is_terminals[:]
#
#
# class PPO(Model):
#     """PPO agent using Pixyz."""
#
#     def __init__(self, config: dict[str, Any]) -> None:
#         """Initialize the PPO agent."""
#         env_properties = get_env_properties(config["env"]["name"])
#         self.action_dim = env_properties["action_dim"]
#         self.gamma = config["agent"]["hyperparameters"]["gamma"]
#         self.eps_clip = config["agent"]["hyperparameters"]["eps_clip"]
#         self.K_epochs = config["agent"]["hyperparameters"]["K_epochs"]
#         self.lr_actor = config["agent"]["hyperparameters"]["lr_actor"]
#         self.lr_critic = config["agent"]["hyperparameters"]["lr_critic"]
#
#         # Shared CNN layers
#         self.shared_cnn = FeatureExtractor().to(device)
#
#         # Actor network
#         self.actor = Actor(self.action_dim, "new").to(device)
#         self.actor_old = Actor(self.action_dim, "old").to(device)
#
#         # Critic network
#         self.critic = Critic().to(device)
#
#         # Buffer for storing rollout data
#         self.buffer = RolloutBuffer()
#
#         advantage = Parameter("A")
#         ratio = RatioLoss(self.actor, self.actor_old)
#         clip = ClipLoss(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
#         ppo_loss = -MinLoss(clip * advantage, ratio * advantage)
#
#         mse_loss = MSELoss(self.critic, "r")
#
#         loss = E(self.shared_cnn, ppo_loss + 0.5 * mse_loss - 0.01 * Entropy(self.actor)).mean()
#
#         super().__init__(loss, distributions=[self.actor, self.critic, self.shared_cnn], optimizer=Adam, optimizer_params={})
#
#         # Optimizer
#         self.optimizer = torch.optim.Adam(
#             [
#                 {"params": self.actor.parameters(), "lr": self.lr_actor},
#                 {"params": self.critic.parameters(), "lr": self.lr_critic},
#             ],
#         )
#
#     def preprocess_state(self, state: tuple[torch.Tensor | NDArray[Any]] | torch.Tensor | NDArray[Any]) -> NDArray[Any]:
#         """Preprocess the state."""
#         if isinstance(state, tuple):
#             state = state[0]
#
#         # NumPy配列に変換
#         if isinstance(state, torch.Tensor):
#             state = state.cpu().numpy()
#
#         # グレースケールの場合は3チャネルに複製
#         if len(state.shape) == TWO_CH:
#             state = np.stack([state] * 3, axis=-1)
#
#         # 正規化 (0-255 -> 0-1)
#         if state.dtype == np.uint8:
#             state = state.astype(np.float32) / 255.0
#
#         # チャネルの順序を変更 (H, W, C) -> (C, H, W)
#         if len(state.shape) == THREE_CH:
#             state = np.transpose(state, (2, 0, 1))
#
#         return state
#
#     def store_transition(self, reward: NDArray[Any], done: int | bool) -> None:
#         """Store transition data in the buffer."""
#         self.buffer.rewards.append(reward)
#         self.buffer.is_terminals.append(done)
#
#     def select_action(self, state: tuple[torch.Tensor | NDArray[Any]] | torch.Tensor | NDArray[Any]) -> NDArray[Any]:
#         """Select an action."""
#         state = self.preprocess_state(state)
#
#         with torch.no_grad():
#             state = torch.FloatTensor(state).unsqueeze(0).to(device)
#             s = self.shared_cnn.sample({"o": state})["s"]
#             action = self.actor_old.sample({"s": s})["a"]
#             state_val = self.critic.sample({"s": s})["v"]
#
#         self.buffer.states.append(state)
#         self.buffer.actions.append(action)
#         self.buffer.state_values.append(state_val)
#
#         return action.detach().cpu().numpy().flatten()
#
#     def update(self) -> None:
#         """Update the agent."""
#         # Monte Carlo estimate of returns
#         rewards = []
#         discounted_reward = 0
#         for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals), strict=False):
#             if is_terminal:
#                 discounted_reward = 0
#             discounted_reward = reward + (self.gamma * discounted_reward)
#             rewards.insert(0, discounted_reward)
#
#         # Normalizing the rewards
#         rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
#         rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
#
#         # Convert list to tensor
#         old_states = torch.cat(self.buffer.states, dim=0).detach()
#         old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
#         old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()
#
#         # Calculate advantages
#         advantages = rewards.detach() - old_state_values.detach()
#
#         # Optimize policy for K epochs
#         for _ in range(self.K_epochs):
#             # Evaluating old actions and values
#             self.train({"o": old_states, "a": old_actions, "A": advantages, "r": rewards})
#
#         # Copy new weights into old policy
#         self.actor_old.load_state_dict(self.actor.state_dict())
#
#         # Clear buffer
#         self.buffer.clear()
#
#
def distance(pos1, pos2):
    return torch.sqrt(torch.sum((pos1 - pos2) ** 2, dim=-1))


def main():
    gs.init(backend=gs.metal)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2, 0, 2),
            camera_lookat=(0, 0, 0),
            camera_fov=30,
        ),
        show_viewer=True,
        rigid_options=gs.options.RigidOptions(
            dt=1 / 32,
            gravity=(0, 0, -9.8),
        ),
    )

    # Create a ground plane
    ground = scene.add_entity(gs.morphs.Plane())
    target = scene.add_entity(gs.morphs.Sphere(radius=0.05, pos=(1, 0, 0)))
    # robot = scene.add_entity(gs.morphs.MJCF(file="xml/universal_robots_ur5e/ur5e.xml"))
    # robot = scene.add_entity(gs.morphs.MJCF(file="./mybot/sample.urdf"))
    robot: RigidEntity = scene.add_entity(gs.morphs.URDF(file="fetch_description/robots/fetch.urdf", pos=(0, 0, 1)))

    scene.build(n_envs=B, env_spacing=(1.5, 1.5))

    gs.tools.run_in_another_thread(run_sim, (scene, robot, target))
    scene.viewer.start()


def run_sim(scene: gs.Scene, robot: RigidEntity, target: RigidEntity):
    dofs_idx = [robot.get_joint(joint.name).dof_idx_local for joint in robot.joints][1:-1]
    dof_len = len(dofs_idx)

    robot.set_pos(torch.Tensor([0, 0, 0.5]).repeat(B, 1))
    robot.set_quat(torch.Tensor([0, 0, 0, 1]).repeat(B, 1))

    for _ in range(10):
        for _ in range(1000):
            pos = torch.randn(B, dof_len) - 0.5
            pos[6:8] += 0.5
            robot.control_dofs_position(
                pos,
                dofs_idx_local=dofs_idx,
            )

            # rewards = mse_loss(robot.get_joint("wrist_3").get_pos(), target.get_pos(), reduction="none").sum(dim=-1)
            # rewards = (-distance(robot.get_joint("wrist_3").get_pos(), target.get_pos())).exp()
            # print(f"Reward: {rewards}")

            scene.step()

        robot.set_dofs_position(torch.zeros(B, dof_len), dofs_idx_local=dofs_idx)
        robot.set_pos(torch.Tensor([0, 0, 0.45]).repeat(B, 1))
        robot.set_quat(torch.Tensor([0, 0, 0, 1]).repeat(B, 1))
        scene.step()


if __name__ == "__main__":
    main()
