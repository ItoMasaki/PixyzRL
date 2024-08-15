import numpy as np
import torch
from pixyz.losses import MinLoss, Parameter, ValueLoss
from pixyz.models import Model

from pixyzrl.losses import ClipLoss, MSELoss, RatioLoss
from pixyzrl.memory import RolloutBuffer


class PPO(Model):
    def __init__(self, actor, actor_old, critic, gamma, eps_clip, K_epochs, device, use_amp, normalize=True):
        ##############################
        #      Hyper parameters      #
        ##############################
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

        self.buffer = RolloutBuffer()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def select_action(self, belief, state):
        with torch.no_grad():
            state = state.to(self.device).detach()
            belief = belief.to(self.device).detach()

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
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(dtype=torch.float32, device=self.device)

        if self.normalize:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

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

        for _ in range(self.K_epochs):
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

        # Copy new weights into old policy
        self.actor_old.load_state_dict(self.actor.state_dict())

        # clear buffer
        self.buffer.clear()

        return total_loss / self.K_epochs
