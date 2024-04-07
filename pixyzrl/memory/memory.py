import numpy as np
import torch
from env import postprocess_observation, preprocess_observation_


class ExperienceReplay():
    def __init__(self, batch_size, batch_length, chunk_size, observation_size, action_size, max_episode_size, bit_depth, device):
        self.device = device
        self.observation_size = observation_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.batch_length = batch_length
        self.chunk_size = chunk_size
        self.max_episode_size = max_episode_size

        self.observations = [[]]
        self.actions = [[]]
        self.rewards = [[]]
        self.nonterminals = [[]]

        self.idx = 0
        self.full = False
        self.steps, self.episodes = 0, 0
        self.bit_depth = bit_depth

    def reset(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.nonterminals = []

        self.idx = 0
        self.full = False
        self.steps, self.episodes = 0, 0

    def append(self, observation, action, reward, done):
        reward = torch.ones(1, 1) * reward
        _done = torch.ones(1, 1) * (not done)

        if done:

            self.observations[self.episodes].append(observation)
            self.actions[self.episodes].append(action)
            self.rewards[self.episodes].append(reward)
            self.nonterminals[self.episodes].append(_done)

            self.episodes += 1
            
            self.observations.append([])
            self.actions.append([])
            self.rewards.append([])
            self.nonterminals.append([])

        else:
            self.observations[self.episodes].append(observation)
            self.actions[self.episodes].append(action)
            self.rewards[self.episodes].append(reward)
            self.nonterminals[self.episodes].append(_done)

        self.reset_sample = True
        self.idx = 0


    def sample(self):
        self.idx += 1

        if self.reset_sample:
            self.reset_sample = False

            observations = [[torch.empty(0, 3, 64, 64)] for _ in range(self.batch_size)]
            actions = [[torch.empty(0, self.action_size)] for _ in range(self.batch_size)]
            rewards = [[torch.empty(0, 1)] for _ in range(self.batch_size)]
            nonterminals = [[torch.empty(0, 1)] for _ in range(self.batch_size)]

            for i in range(self.batch_size):
                while observations[i][-1].shape[0] < self.chunk_size*100:
                    idx = np.random.randint(0, self.episodes)

                    observations[i][-1] = torch.concat((observations[i][-1], torch.stack(self.observations[idx]).squeeze(1)), dim=0)
                    actions[i][-1] = torch.concat((actions[i][-1], torch.stack(self.actions[idx]).squeeze(1)), dim=0)
                    rewards[i][-1] = torch.concat((rewards[i][-1], torch.stack(self.rewards[idx]).squeeze(1)), dim=0)
                    nonterminals[i][-1] = torch.concat((nonterminals[i][-1], torch.stack(self.nonterminals[idx]).squeeze(1)), dim=0)

            min_length = min([obs[-1].shape[0] for obs in observations])

            for i in range(self.batch_size):
                observations[i] = observations[i][-1][:min_length]
                actions[i] = actions[i][-1][:min_length]
                rewards[i] = rewards[i][-1][:min_length]
                nonterminals[i] = nonterminals[i][-1][:min_length]

            self._observations = torch.stack(observations)
            self._actions = torch.stack(actions)
            self._rewards = torch.stack(rewards)
            self._nonterminals = torch.stack(nonterminals)

            remaining_length = self._observations.shape[1] % self.chunk_size

            if remaining_length:
                self._observations = self._observations[:, :-remaining_length]
                self._actions = self._actions[:, :-remaining_length]
                self._rewards = self._rewards[:, :-remaining_length]
                self._nonterminals = self._nonterminals[:, :-remaining_length]

            sampled_observations = self._observations[:, (self.idx-1)*self.chunk_size:self.idx*self.chunk_size].transpose(0, 1)
            sampled_actions = self._actions[:, (self.idx-1)*self.chunk_size:self.idx*self.chunk_size].transpose(0, 1)
            sampled_rewards = self._rewards[:, (self.idx-1)*self.chunk_size:self.idx*self.chunk_size].transpose(0, 1)
            sampled_nonterminals = self._nonterminals[:, (self.idx-1)*self.chunk_size:self.idx*self.chunk_size].transpose(0, 1)

            return sampled_observations.to(self.device), sampled_actions.to(self.device), sampled_rewards.to(self.device), sampled_nonterminals.to(self.device)

        else:
            sampled_observations = self._observations[:, (self.idx-1)*self.chunk_size:self.idx*self.chunk_size].transpose(0, 1)
            sampled_actions = self._actions[:, (self.idx-1)*self.chunk_size:self.idx*self.chunk_size].transpose(0, 1)
            sampled_rewards = self._rewards[:, (self.idx-1)*self.chunk_size:self.idx*self.chunk_size].transpose(0, 1)
            sampled_nonterminals = self._nonterminals[:, (self.idx-1)*self.chunk_size:self.idx*self.chunk_size].transpose(0, 1)

            if (self.idx+1)*self.chunk_size > self._observations.shape[1]:
                self.idx = 0

            return sampled_observations.to(self.device), sampled_actions.to(self.device), sampled_rewards.to(self.device), sampled_nonterminals.to(self.device)
            



# class ExperienceReplay():
#   def __init__(self, size, symbolic_env, observation_size, action_size, bit_depth, device):
#     self.device = device
#     self.symbolic_env = symbolic_env
#     self.observation_size = observation_size
#     self.action_size = action_size
#     self.size = size
#     self.observations = np.empty((size, observation_size) if symbolic_env else (
#         size, 3, 64, 64), dtype=np.float32 if symbolic_env else np.uint8)
#     self.actions = np.empty((size, action_size), dtype=np.float32)
#     self.rewards = np.empty((size, 31), dtype=np.float32)
#     self.nonterminals = np.empty((size, 1), dtype=np.float32)
#     self.idx = 0
#     self.full = False  # Tracks if memory has been filled/all slots are valid
#     # Tracks how much experience has been used in total
#     self.steps, self.episodes = 0, 0
#     self.bit_depth = bit_depth
# 
#   def reset(self):
#     self.observations = np.empty((self.size, self.observation_size) if self.symbolic_env else (
#         self.size, 3, 64, 64), dtype=np.float32 if self.symbolic_env else np.uint8)
#     self.actions = np.empty((self.size, self.action_size), dtype=np.float32)
#     self.rewards = np.empty((self.size, 31), dtype=np.float32)
#     self.nonterminals = np.empty((self.size, 1), dtype=np.float32)
#     self.idx = 0
#     self.full = False  # Tracks if memory has been filled/all slots are valid
#     # Tracks how much experience has been used in total
#     self.steps, self.episodes = 0, 0
# 
#   def append(self, observation, action, reward, done):
#     if self.symbolic_env:
#       self.observations[self.idx] = observation.numpy()
#     else:
#       self.observations[self.idx] = postprocess_observation(observation.numpy(), self.bit_depth)
#     self.actions[self.idx] = action.numpy()
#     self.rewards[self.idx] = reward
#     self.nonterminals[self.idx] = not done
#     self.idx = (self.idx + 1) % self.size
#     self.full = self.full or self.idx == 0
#     self.steps, self.episodes = self.steps + \
#         1, self.episodes + (1 if done else 0)
# 
#   # Returns an index for a valid single sequence chunk uniformly sampled from the memory
#   def _sample_idx(self, L):
#     valid_idx = False
#     while not valid_idx:
#       idx = np.random.randint(0, self.size if self.full else self.idx - L)
#       idxs = np.arange(idx, idx + L) % self.size
#       # Make sure data does not cross the memory index
#       valid_idx = not self.idx in idxs[1:]
#     return idxs
# 
#   def _retrieve_batch(self, idxs, n, L):
#     vec_idxs = idxs.transpose().reshape(-1)  # Unroll indices
#     observations = torch.as_tensor(
#         self.observations[vec_idxs].astype(np.float32))
#     if not self.symbolic_env:
#       # Undo discretisation for visual observations
#       preprocess_observation_(observations, self.bit_depth)
#     return observations.reshape(L, n, *observations.shape[1:]), self.actions[vec_idxs].reshape(L, n, -1), self.rewards[vec_idxs].reshape(L, n, 31), self.nonterminals[vec_idxs].reshape(L, n, 1)
# 
#   # Returns a batch of sequence chunks uniformly sampled from the memory
#   def sample(self, n, L):
#     batch = self._retrieve_batch(np.asarray(
#         [self._sample_idx(L) for _ in range(n)]), n, L)
#     # print(np.asarray([self._sample_idx(L) for _ in range(n)]))
#     # [1578 1579 1580 ... 1625 1626 1627]                                                                                                                                        | 0/100 [00:00<?, ?it/s]
#     # [1049 1050 1051 ... 1096 1097 1098]
#     # [1236 1237 1238 ... 1283 1284 1285]
#     # ...
#     # [2199 2200 2201 ... 2246 2247 2248]
#     # [ 686  687  688 ...  733  734  735]
#     # [1377 1378 1379 ... 1424 1425 1426]]
#     return [torch.as_tensor(item).to(device=self.device) for item in batch]




class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.beliefs = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.beliefs[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def compute_returns_and_advantage(self, last_values, dones):
        last_values = last_values.cpu().flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.to(torch.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values
