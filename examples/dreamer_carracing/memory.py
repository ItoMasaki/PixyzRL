import numpy as np
import torch


class ExperienceReplay():

  def __init__(
      self,
      size,
      num_envs,
      symbolic_env,
      observation_size,
      action_size,
      bit_depth,
      device
  ):

    self.device = device
    self.symbolic_env = symbolic_env

    self.size = int(size)
    self.num_envs = int(num_envs)

    self.bit_depth = bit_depth

    # -----------------------------
    # observation buffer
    # -----------------------------
    if symbolic_env:
      self.observations = np.empty(
          (self.size, observation_size),
          dtype=np.float32
      )
    else:
      self.observations = np.empty(
          (self.size, 3, 64, 64),
          dtype=np.uint8
      )

    self.actions = np.empty(
        (self.size, action_size),
        dtype=np.float32
    )

    self.rewards = np.empty(
        (self.size,),
        dtype=np.float32
    )

    self.nonterminals = np.empty(
        (self.size, 1),
        dtype=np.float32
    )

    self.discounts = np.empty(
        (self.size, 1),
        dtype=np.float32
    )

    # -----------------------------
    # sequence safety
    # -----------------------------
    self.env_ids = np.empty(
        (self.size,),
        dtype=np.int32
    )

    self.episode_ids = np.empty(
        (self.size,),
        dtype=np.int64
    )

    self._episode_counters = np.zeros(
        (self.num_envs,),
        dtype=np.int64
    )

    # -----------------------------
    # ring buffer pointer
    # -----------------------------
    self.idx = 0
    self.full = False

    self.steps = 0
    self.episodes = 0


  # ============================================================
  # utilities
  # ============================================================

  def __len__(self):
    return self.size if self.full else self.idx


  def _to_np(self, x):
    if torch.is_tensor(x):
      return x.detach().cpu().numpy()
    return np.asarray(x)


  # ------------------------------------------------------------
  # observation encode/decode
  # ------------------------------------------------------------

  def _encode_obs(self, obs):

    obs = np.clip(obs, 0.0, 1.0)

    return (obs * 255).astype(np.uint8)


  def _decode_obs(self, obs):

    return obs.astype(np.float32) / 255.0


  # ============================================================
  # append (完全vector化)
  # ============================================================

  def append(
      self,
      observation,
      action,
      reward,
      done,
      discount=0.99
  ):

    obs_np = self._to_np(observation)
    act_np = self._to_np(action)
    rew_np = self._to_np(reward).astype(np.float32).reshape(self.num_envs)
    done_np = self._to_np(done).astype(np.bool_).reshape(self.num_envs)
    discount_np = self._to_np(discount).astype(np.float32).reshape(self.num_envs, 1)

    # 画像観測はuint8に変換
    if not self.symbolic_env:
      obs_np = self._encode_obs(obs_np)

    # ----------------------------------------------------------
    # 書き込みindex
    # ----------------------------------------------------------

    idxs = (self.idx + np.arange(self.num_envs)) % self.size

    # ----------------------------------------------------------
    # 書き込み
    # ----------------------------------------------------------

    self.observations[idxs] = obs_np
    self.actions[idxs] = act_np.astype(np.float32)
    self.rewards[idxs] = rew_np
    self.discounts[idxs] = discount_np

    self.nonterminals[idxs, 0] = (~done_np).astype(np.float32)

    self.env_ids[idxs] = np.arange(self.num_envs)

    self.episode_ids[idxs] = self._episode_counters

    # ----------------------------------------------------------
    # episode更新
    # ----------------------------------------------------------

    done_envs = np.where(done_np)[0]

    if len(done_envs) > 0:

      self._episode_counters[done_envs] += 1
      self.episodes += len(done_envs)

    # ----------------------------------------------------------
    # ring buffer pointer
    # ----------------------------------------------------------

    self.idx = (self.idx + self.num_envs) % self.size

    if self.idx < self.num_envs:
      self.full = True

    self.steps += 1


  # ============================================================
  # sampling
  # ============================================================

  def _sample_idx(self, L):

    max_t = self.size if self.full else self.idx

    stride = self.num_envs

    needed = (L - 1) * stride + 1

    if max_t < needed:
      raise ValueError("Not enough data in replay buffer")

    while True:

      start = np.random.randint(0, max_t - needed + 1)

      idxs = start + np.arange(L) * stride

      # ring buffer境界回避
      if self.full and (self.idx in idxs[1:]):
        continue

      # episode跨ぎ防止
      ep0 = self.episode_ids[idxs[0]]

      if np.all(self.episode_ids[idxs] == ep0):

        # env混入防止
        e0 = self.env_ids[idxs[0]]

        if np.all(self.env_ids[idxs] == e0):

          return idxs


  # ============================================================
  # batch retrieval
  # ============================================================

  def _retrieve_batch(self, idxs, n, L):

    vec_idxs = idxs.transpose().reshape(-1)

    obs = self.observations[vec_idxs]

    if not self.symbolic_env:
      obs = self._decode_obs(obs)

    obs = torch.from_numpy(obs)

    act = torch.from_numpy(self.actions[vec_idxs])
    rew = torch.from_numpy(self.rewards[vec_idxs])
    nt = torch.from_numpy(self.nonterminals[vec_idxs])
    dis = torch.from_numpy(self.discounts[vec_idxs])

    obs = obs.reshape(L, n, *obs.shape[1:])
    act = act.reshape(L, n, -1)
    rew = rew.reshape(L, n)
    nt = nt.reshape(L, n, 1)
    dis = dis.reshape(L, n, 1)

    return obs, act, rew, nt, dis


  # ============================================================
  # public sample
  # ============================================================

  def sample(self, n, L):

    idxs = np.asarray(
        [self._sample_idx(L) for _ in range(n)]
    )

    batch = self._retrieve_batch(idxs, n, L)

    return [t.to(self.device) for t in batch]
  

# import numpy as np
# import torch


# class ExperienceReplay():
#   def __init__(self, size, num_envs, symbolic_env, observation_size, action_size, bit_depth, device):
#     self.device = device
#     self.symbolic_env = symbolic_env
#     self.size = int(size)
#     self.num_envs = int(num_envs)
#     self.bit_depth = bit_depth

#     # observation buffer
#     if symbolic_env:
#       self.observations = np.empty((self.size, observation_size), dtype=np.float32)
#     else:
#       # 画像はuint8で保存
#       self.observations = np.empty((self.size, 3, 64, 64), dtype=np.uint8)

#     self.actions = np.empty((self.size, action_size), dtype=np.float32)
#     self.rewards = np.empty((self.size,), dtype=np.float32)
#     self.nonterminals = np.empty((self.size, 1), dtype=np.float32)
#     self.discounts = np.empty((self.size, 1), dtype=np.float32)

#     self.env_ids = np.empty((self.size,), dtype=np.int32)
#     self.episode_ids = np.empty((self.size,), dtype=np.int64)

#     self._episode_counters = np.zeros((self.num_envs,), dtype=np.int64)

#     self.idx = 0
#     self.full = False

#     self.steps = 0
#     self.episodes = 0


#   def __len__(self):
#     return self.size if self.full else self.idx


#   def _to_np(self, x):
#     return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


#   def _encode_obs(self, obs):
#     """float → uint8"""
#     obs = np.clip(obs, 0, 1)
#     obs = (obs * 255).astype(np.uint8)
#     return obs


#   def _decode_obs(self, obs):
#     """uint8 → float"""
#     return obs.astype(np.float32) / 255.0


#   def append(self, observation, action, reward, done, discount=0.99):

#     obs_np = self._to_np(observation)
#     act_np = self._to_np(action)
#     rew_np = self._to_np(reward).astype(np.float32).reshape(self.num_envs)
#     done_np = self._to_np(done).astype(np.bool_).reshape(self.num_envs)
#     discount_np = self._to_np(discount).astype(np.float32).reshape(self.num_envs, 1)

#     for e in range(self.num_envs):

#       obs = obs_np[e]

#       if not self.symbolic_env:
#         obs = self._encode_obs(obs)

#       self.observations[self.idx] = obs
#       self.actions[self.idx] = act_np[e].astype(np.float32)
#       self.rewards[self.idx] = rew_np[e]
#       self.discounts[self.idx] = discount_np[e]

#       self.nonterminals[self.idx, 0] = 0.0 if done_np[e] else 1.0

#       self.env_ids[self.idx] = e
#       self.episode_ids[self.idx] = self._episode_counters[e]

#       if done_np[e]:
#         self._episode_counters[e] += 1
#         self.episodes += 1

#       self.idx = (self.idx + 1) % self.size
#       self.full = self.full or (self.idx == 0)

#     self.steps += 1


#   def _sample_idx(self, L):

#     max_t = self.size if self.full else self.idx
#     stride = self.num_envs

#     needed = (L - 1) * stride + 1

#     if max_t < needed:
#       raise ValueError("Not enough data")

#     while True:

#       start = np.random.randint(0, max_t - needed + 1)

#       idxs = start + np.arange(L) * stride

#       if self.full and (self.idx in idxs[1:]):
#         continue

#       ep0 = self.episode_ids[idxs[0]]

#       if np.all(self.episode_ids[idxs] == ep0):

#         e0 = self.env_ids[idxs[0]]

#         if np.all(self.env_ids[idxs] == e0):

#           return idxs


#   def _retrieve_batch(self, idxs, n, L):

#     vec_idxs = idxs.transpose().reshape(-1)

#     obs = self.observations[vec_idxs]

#     if not self.symbolic_env:
#       obs = self._decode_obs(obs)

#     obs = torch.from_numpy(obs)

#     act = torch.from_numpy(self.actions[vec_idxs])
#     rew = torch.from_numpy(self.rewards[vec_idxs])
#     nt = torch.from_numpy(self.nonterminals[vec_idxs])
#     dis = torch.from_numpy(self.discounts[vec_idxs])

#     obs = obs.reshape(L, n, *obs.shape[1:])
#     act = act.reshape(L, n, -1)
#     rew = rew.reshape(L, n)
#     nt = nt.reshape(L, n, 1)
#     dis = dis.reshape(L, n, 1)

#     return obs, act, rew, nt, dis


#   def sample(self, n, L):

#     idxs = np.asarray([self._sample_idx(L) for _ in range(n)])

#     batch = self._retrieve_batch(idxs, n, L)

#     return [t.to(self.device) for t in batch]
  

# import numpy as np
# import torch


# class ExperienceReplay():
#   def __init__(self, size, num_envs, symbolic_env, observation_size, action_size, bit_depth, device):
#     self.device = device
#     self.symbolic_env = symbolic_env
#     self.size = int(size)
#     self.num_envs = int(num_envs)
#     self.bit_depth = bit_depth

#     self.observations = np.empty(
#       (self.size, observation_size) if symbolic_env else (self.size, 3, 64, 64), dtype=np.float32
#     )
#     self.actions = np.empty((self.size, action_size), dtype=np.float32)
#     self.rewards = np.empty((self.size,), dtype=np.float32)
#     self.nonterminals = np.empty((self.size, 1), dtype=np.float32)
#     self.discounts = np.empty((self.size, 1), dtype=np.float32)

#     # 追加: どのenv由来か/どのepisodeか を記録（シーケンスの混入防止）
#     self.env_ids = np.empty((self.size,), dtype=np.int32)
#     self.episode_ids = np.empty((self.size,), dtype=np.int64)
#     self._episode_counters = np.zeros((self.num_envs,), dtype=np.int64)

#     self.idx = 0
#     self.full = False
#     self.steps, self.episodes = 0, 0

#   def __len__(self):
#     return self.size if self.full else self.idx

#   def append(self, observation, action, reward, done, discount=0.99):
#     """
#     VecEnv想定:
#       observation: (N, ...) torch/np
#       action:      (N, act_dim)
#       reward:      (N,)
#       done:        (N,)  (Gymnasiumなら done = terminated|truncated を渡す)
#     """

#     def to_np(x):
#       return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)

#     obs_np = to_np(observation)
#     act_np = to_np(action)
#     rew_np = to_np(reward).astype(np.float32).reshape(self.num_envs)
#     done_np = to_np(done).astype(np.bool_).reshape(self.num_envs)
#     discount_np = to_np(discount).astype(np.float32).reshape(self.num_envs, 1)
#     # envごとにフラットに1件ずつ格納
#     for e in range(self.num_envs):
#       self.observations[self.idx] = obs_np[e].astype(np.float32)
#       self.actions[self.idx] = act_np[e].astype(np.float32)
#       self.rewards[self.idx] = rew_np[e]
#       self.discounts[self.idx] = discount_np[e]
#       self.nonterminals[self.idx, 0] = 0.0 if done_np[e] else 1.0

#       self.env_ids[self.idx] = e
#       self.episode_ids[self.idx] = self._episode_counters[e]

#       # done なら次ステップから episode_id を進める
#       if done_np[e]:
#         self._episode_counters[e] += 1
#         self.episodes += 1

#       self.idx = (self.idx + 1) % self.size
#       self.full = self.full or (self.idx == 0)

#     self.steps += 1  # vec-step count

#   def _sample_idx(self, L):
#     """
#     フラット(interleaved)格納で、同一envの連続Lステップを取る。
#     連続スロットではなく stride=num_envs で飛び飛びに拾う。
#     """
#     max_t = self.size if self.full else self.idx
#     stride = self.num_envs

#     # 同一envの L ステップには最低 (L-1)*stride + 1 スロット分が必要
#     needed = (L - 1) * stride + 1
#     if max_t < needed:
#       raise ValueError(f"Not enough data: have {max_t}, need {needed} (L={L}, num_envs={stride})")

#     while True:
#       # start は「どのenvの列から始めるか」を含む
#       start = np.random.randint(0, max_t - needed + 1)
#       idxs = start + np.arange(L) * stride  # ★ここが肝

#       # 書き込みポインタ self.idx を跨がない（リングの境界を避ける）
#       if self.full and (self.idx in idxs[1:]):
#         continue

#       # 同一episodeかどうか（done跨ぎ防止）
#       ep0 = self.episode_ids[idxs[0]]
#       if np.all(self.episode_ids[idxs] == ep0):
#         # （append順が固定なら env_ids は自動で揃うのでチェック省略可）
#         # 念のため入れるなら↓
#         e0 = self.env_ids[idxs[0]]
#         if np.all(self.env_ids[idxs] == e0):
#           return idxs

#   def _retrieve_batch(self, idxs, n, L):
#     vec_idxs = idxs.transpose().reshape(-1)  # (L*n,)

#     obs = torch.as_tensor(self.observations[vec_idxs].astype(np.float32))
#     act = torch.as_tensor(self.actions[vec_idxs])
#     rew = torch.as_tensor(self.rewards[vec_idxs])
#     nt  = torch.as_tensor(self.nonterminals[vec_idxs])
#     dis = torch.as_tensor(self.discounts[vec_idxs])

#     obs = obs.reshape(L, n, *obs.shape[1:])
#     act = act.reshape(L, n, -1)
#     rew = rew.reshape(L, n)
#     nt  = nt.reshape(L, n, 1)
#     dis = dis.reshape(L, n, 1)
#     return obs, act, rew, nt, dis

#   def sample(self, n, L):
#     idxs = np.asarray([self._sample_idx(L) for _ in range(n)])
#     batch = self._retrieve_batch(idxs, n, L)
#     return [t.to(device=self.device) for t in batch]