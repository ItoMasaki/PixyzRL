<<<<<<< HEAD:policy/ppo.py
<<<<<<<< HEAD:policy/ppo.py
=======
>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy/ppo.py
import numpy as np
import torch
from pixyz.losses import MinLoss, Parameter, ValueLoss
from pixyz.models import Model
<<<<<<< HEAD:policy/ppo.py
========
"""Proximal Policy Optimization (PPO) agent using Pixyz."""  # noqa: INP001

from copy import deepcopy
from typing import Any

import torch
from numpy.typing import NDArray
from pixyz import distributions as dists
from pixyz.losses import Entropy, MinLoss, Parameter
from pixyz.losses import Expectation as E  # noqa: N817
from pixyz.models import Model
from torch.optim import Adam
>>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy_gradient/ppo/ppo.py
=======
>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy/ppo.py

from pixyzrl.losses import ClipLoss, MSELoss, RatioLoss
from pixyzrl.memory import RolloutBuffer


class PPO(Model):
<<<<<<< HEAD:policy/ppo.py
<<<<<<<< HEAD:policy/ppo.py
=======
>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy/ppo.py
    def __init__(self, actor, actor_old, critic, gamma, eps_clip, K_epochs, device, use_amp, normalize=True):
        ##############################
        #      Hyper parameters      #
        ##############################
<<<<<<< HEAD:policy/ppo.py
========
    """PPO agent using Pixyz."""

    def __init__(self, actor: dists.Distribution, critic: dists.Distribution, shared_cnn: dists.Distribution, gamma: float, eps_clip: float, k_epochs: int, lr_actor: float, lr_critic: float, device: str) -> None:
        """Initialize the PPO agent."""
>>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy_gradient/ppo/ppo.py
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = k_epochs
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.device = device

        # Shared CNN layers
        self.shared_cnn = shared_cnn

<<<<<<<< HEAD:policy/ppo.py
=======
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        self.use_amp = use_amp
        self.normalize = normalize

        #################################
        #      Actor-Critic models      #
        #################################
        self.actor = actor.to(self.device)
        self.actor_old = actor_old.to(self.device)
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic = critic.to(self.device)

>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy/ppo.py
        ###########################
        #      Loss function      #
        ###########################
        advantage = Parameter("\\hat{A}")
        ratio = RatioLoss(self.actor, self.actor_old) * advantage
        clip = ClipLoss(ratio, 1 - eps_clip, 1 + eps_clip) * advantage

        ppo_loss = -MinLoss(ratio, clip)
        value_loss = ValueLoss(0.5) * MSELoss("v", "r")
        # entropy = -ValueLoss(0.01)*Entropy(self.actor, sum_features=False)

        # loss_func = (value_loss + ppo_loss + entropy).mean()
        loss_func = (value_loss + ppo_loss).mean()

        #########################
        #      Setup model      #
        #########################
        super().__init__(loss=loss_func, distributions=[self.actor, self.critic], optimizer=torch.optim.Adam, optimizer_params={"lr": 0.0002})

        self.optimizer = torch.optim.Adam([{"params": self.actor.parameters(), "lr": 0.0002}, {"params": self.critic.parameters(), "lr": 0.0002}])
<<<<<<< HEAD:policy/ppo.py
========
        # Actor network
        self.actor = actor
        self.actor_old = deepcopy(actor)
        self.actor_old.name = "old"

        # Critic network
        self.critic = critic
>>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy_gradient/ppo/ppo.py

        # Buffer for storing rollout data
        self.buffer = RolloutBuffer()

<<<<<<<< HEAD:policy/ppo.py
    def select_action(self, belief, state):
========
        advantage = Parameter("A")
        ratio = RatioLoss(self.actor, self.actor_old)
        clip = ClipLoss(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
        ppo_loss = -MinLoss(clip * advantage, ratio * advantage)

        mse_loss = MSELoss(self.critic, "r")

        loss = E(self.shared_cnn, ppo_loss + 0.5 * mse_loss - 0.01 * Entropy(self.actor)).mean()

        super().__init__(loss, distributions=[self.actor, self.critic, self.shared_cnn], optimizer=Adam, optimizer_params={})

        # Optimizer
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": self.lr_actor},
                {"params": self.critic.parameters(), "lr": self.lr_critic},
            ]
        )

    def store_transition(self, reward: NDArray[Any], done: int | bool) -> None:
        """Store transition data in the buffer."""
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    def select_action(self, state: tuple[torch.Tensor | NDArray[Any]] | torch.Tensor | NDArray[Any]) -> NDArray[Any]:
        """Select an action."""

>>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy_gradient/ppo/ppo.py
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            s = self.shared_cnn.sample({"o": state})["s"]
            action = self.actor_old.sample({"s": s})["a"]
            state_val = self.critic.sample({"s": s})["v"]

<<<<<<<< HEAD:policy/ppo.py
=======

        self.buffer = RolloutBuffer()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def select_action(self, belief, state):
        with torch.no_grad():
            state = state.to(self.device).detach()
            belief = belief.to(self.device).detach()

>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy/ppo.py
            action = self.actor_old.sample({"s_t": state, "z_t": belief})["a_t"].detach()
            state_val = self.critic.sample({"s_t": state, "z_t": belief})["v_t"].detach()

            self.buffer.states.append(state)
            self.buffer.beliefs.append(belief)

            self.buffer.actions.append(action)
            self.buffer.state_values.append(state_val)

            return action.detach().cpu().numpy().astype(np.float64)

    def get_discount_reward(self):
        rewards = []
        discounted_reward = 0.0

        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals), strict=False):
            if is_terminal:
                discounted_reward = 0.0

            discounted_reward = reward + self.gamma * discounted_reward
<<<<<<< HEAD:policy/ppo.py
========
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.state_values.append(state_val)

        return action.detach().cpu().numpy().flatten()

    def update(self) -> None:
        """Update the agent."""
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals), strict=False):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
>>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy_gradient/ppo/ppo.py
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # Convert list to tensor
        old_states = torch.cat(self.buffer.states, dim=0).detach()
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach()

<<<<<<<< HEAD:policy/ppo.py
=======
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(dtype=torch.float32, device=self.device)

        if self.normalize:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy/ppo.py
        return rewards.detach()

    def get_advantages(self):
        advantages = []
        advantage = 0
        next_value = 0

        for r, v in zip(reversed(self.buffer.rewards), reversed(self.buffer.state_values), strict=False):
            td_error = r + next_value * self.gamma - v
            advantage = td_error + advantage * self.gamma * 0.99
            next_value = v
            advantages.insert(0, advantage)

        advantages = torch.stack(advantages).to(dtype=torch.float32, device=self.device)

        if self.normalize:
            advantages = (advantages - advantages.mean()) / advantages.std()

        return advantages

    def update(self):
        # Calculate discount rewards
        advantages = self.get_advantages().squeeze().detach()

        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_beliefs = torch.squeeze(torch.stack(self.buffer.beliefs, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        rewards = advantages + old_state_values

        advantages = advantages.unsqueeze(-1)
        rewards = rewards.unsqueeze(-1)

        total_loss = 0.0
<<<<<<< HEAD:policy/ppo.py
========
        # Calculate advantages
        advantages = rewards.detach() - old_state_values.detach()
>>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy_gradient/ppo/ppo.py

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
<<<<<<<< HEAD:policy/ppo.py
=======

        for _ in range(self.K_epochs):
>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy/ppo.py
            # Evaluating old values
            state_values = self.critic(s_tn1=old_states.detach(), h_tn1=old_beliefs.detach())["v_tn1"]

            # match state_values tensor dimensions with rewards tensor
            # state_values = torch.squeeze(state_values)

            # print("state_values", state_values.shape)
            # print("rewards", rewards.shape)
            # print("advantages", advantages.shape)
            # print("old_actions", old_actions.shape)
            # print("old_states", old_states.shape)
            # print("old_beliefs", old_beliefs.shape)

            loss, x_dict = self.train({"s_tn1": old_states.detach(), "h_tn1": old_beliefs.detach(), "a_tn1": old_actions, "v": state_values, "r": rewards, "\\hat{A}": advantages})

            total_loss += loss.item()
<<<<<<< HEAD:policy/ppo.py
========
            # Evaluating old actions and values
            self.train({"o": old_states, "a": old_actions, "A": advantages, "r": rewards})
>>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy_gradient/ppo/ppo.py
=======
>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy/ppo.py

        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())

<<<<<<< HEAD:policy/ppo.py
        # Clear buffer
        self.buffer.clear()
<<<<<<<< HEAD:policy/ppo.py

        return total_loss / self.K_epochs
========
>>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy_gradient/ppo/ppo.py
=======
        # clear buffer
        self.buffer.clear()

        return total_loss / self.K_epochs
>>>>>>> 611891be9389d5eb3272e7321e5428afcd150ec1:pixyzrl/policy/ppo.py
