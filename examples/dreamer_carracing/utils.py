"""Utility functions for training and evaluation."""

from collections.abc import Iterable
from typing import Any

import cv2
import matplotlib.animation as animation
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from torch.nn import Module


def imagine_ahead(prev_state: torch.Tensor, prev_belief: torch.Tensor, policy: Any, transition_model: Any, stochastic_model: Any, planning_horizon: int = 12) -> list[torch.Tensor]:
    """Implements the imagination step of the Dreamer algorithm.

    Args:
        prev_state (torch.Tensor): Previous state tensor.
        prev_belief (torch.Tensor): Previous belief tensor.
        policy (Any): Policy model to sample actions from.
        transition_model (Any): Transition model to compute next beliefs.
        stochastic_model (Any): Stochastic model to compute next states.
        planning_horizon (int, optional): Number of steps to imagine ahead. Defaults to 12.
    Returns:
        list[torch.Tensor]: List containing the imagined beliefs and prior states.
    """

    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = planning_horizon
    beliefs = [None] * T
    prior_states = [None] * T
    actions = [None] * (T - 1)
    beliefs[0], prior_states[0] = prev_belief, prev_state

    # Loop over time sequence
    for t in range(T - 1):
        t_t = torch.ones(prior_states[t].size(0), 1).to(prior_states[t].device)

        action = policy.sample_action(beliefs[t], prior_states[t])
        actions[t] = action

        # Compute belief (deterministic hidden state)
        beliefs[t + 1] = transition_model.sample({"h_t": beliefs[t], "s_t": prior_states[t], "a_t": action, "e_t": t_t})["h_tp1"]
        # Compute state prior by applying transition dynamics
        prior_states[t + 1] = stochastic_model.sample({"h_t": beliefs[t + 1], "a_t": action}, reparam=True)["s_t"]

    return [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(actions, dim=0)]


def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
    next_values = torch.cat([value_pred[1:], bootstrap.unsqueeze(0)], dim=0)

    disc = discount.detach() if isinstance(discount, torch.Tensor) else torch.full_like(imged_reward, discount)

    inputs = imged_reward + disc * next_values * (1.0 - lambda_)
    last = bootstrap.detach()
    outs = []
    for t in reversed(range(inputs.shape[0])):
        last = inputs[t] + disc[t] * lambda_ * last
        outs.append(last)

    return torch.stack(list(reversed(outs)), dim=0)


def td_lambda(imged_reward, value_pred, discount=0.99, lambda_=0.95):
    """Calculate TD(lambda) returns given rewards and value predictions."""
    advantages = []
    advantage = 0
    next_value = 0
    device = imged_reward.device

    for r, v in zip(reversed(imged_reward), reversed(value_pred), strict=False):
        td_error = r + next_value * discount - v
        advantage = td_error + advantage * discount * lambda_
        next_value = v
        advantages.insert(0, advantage)

    advantages = torch.stack(advantages).to(dtype=torch.float32, device=device)
    returns = advantages + value_pred

    return advantages, returns


def gae_and_return(imged_reward, value_pred, discount=0.99, lambda_=0.95, normalize=True):
    """Calculate Generalized Advantage Estimation (GAE) given rewards and value predictions."""
    advantages = []
    advantage = 0
    next_value = value_pred[-1]  # Use the last value prediction as the bootstrap value
    device = imged_reward.device
    discount = discount if isinstance(discount, torch.Tensor) else torch.full_like(imged_reward, discount)

    # if normalize:
    #     imged_reward = (imged_reward - imged_reward.mean()) / (imged_reward.std() + 1e-8)

    for r, v, d in zip(reversed(imged_reward), reversed(value_pred), reversed(discount), strict=False):
        td_error = r + next_value * d - v
        advantage = td_error + advantage * d * lambda_
        next_value = v
        advantages.insert(0, advantage)

    advantages = torch.stack(advantages).to(dtype=torch.float32, device=device)
    returns = advantages + value_pred

    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages, returns


def advantage_and_return(imged_reward, value_pred, discount=0.99, lambda_=0.95, normalize=True):
    """Calculate advantages given rewards and value predictions."""
    advantages = []
    advantage = 0
    next_value = 0
    device = imged_reward.device

    for r, v in zip(reversed(imged_reward), reversed(value_pred), strict=False):
        td_error = r + next_value * discount - v
        advantage = td_error + advantage * discount * lambda_
        next_value = v
        advantages.insert(0, advantage)

    advantages = torch.stack(advantages).to(dtype=torch.float32, device=device)
    returns = advantages + value_pred

    if normalize:
        advantages = (advantages - advantages.mean()) / advantages.std()

    return advantages, returns


class ActivateParameters:
    def __init__(self, modules: Iterable[Module]):
        """Context manager to locally Activate the gradients.

        Example:
        -------
        ```
        with ActivateParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.

        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            # print(param.requires_grad)
            param.requires_grad = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


# "get_parameters" and "FreezeParameters" are from the following repo
# https://github.com/juliusfrost/dreamer-pytorch
def get_parameters(modules: Iterable[Module]):
    """Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.

        Example:
        -------
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.

        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]


# Preprocesses an observation inplace (from float32 Tensor [0, 255] to [-0.5, 0.5])
def preprocess_observation_(observation, bit_depth):
    # Quantise to given bit depth and centre
    observation.div_(2 ** (8 - bit_depth)).floor_().div_(2**bit_depth).sub_(0.5)
    # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)
    observation.add_(torch.rand_like(observation).div_(2**bit_depth))


# Postprocess an observation for storage (from float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
def postprocess_observation(observation, bit_depth):
    if isinstance(observation, torch.Tensor):
        observation = observation.cpu().numpy()
    return np.clip(np.floor((observation + 0.5) * 2**bit_depth) * 2 ** (8 - bit_depth), 0, 2**8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth, image_size):
    images = torch.tensor(cv2.resize(images, image_size, interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
    # Quantise, centre and dequantise inplace
    preprocess_observation_(images, bit_depth)
    return images.unsqueeze(dim=0)  # Add batch dimension


def make_animation_(imgs, recs, h_ts, s_ts, rssm_losses, actor_losses, value_losses, rewards, save_path="animation.mp4"):
    # 書き出し設定
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, 30.0, (width, height))

    color_range = np.arange(len(imgs))

    for i in range(len(imgs)):
        # plt.clf()  # グラフをクリア

        # 高速描画用の backend
        fig = plt.figure(figsize=(16, 4))  # 横に4つ並べる → 幅縮小
        canvas = FigureCanvas(fig)

        # 軸レイアウト
        gs = fig.add_gridspec(1, 4)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2], projection="3d")
        ax4 = fig.add_subplot(gs[0, 3], projection="3d")

        # 画像描画
        ax1.imshow(imgs[i])
        ax1.set_title("Original Images")
        ax1.axis("off")

        ax2.imshow(recs[i])
        ax2.set_title("Reconstructed Images")
        ax2.axis("off")

        # PCA描画
        ax3.scatter(h_ts[: i + 1, 0], h_ts[: i + 1, 1], h_ts[: i + 1, 2], c=color_range[: i + 1], cmap="viridis")
        ax3.set_title("RSSM Beliefs (PCA)")
        ax3.axis("off")

        ax4.scatter(s_ts[: i + 1, 0], s_ts[: i + 1, 1], s_ts[: i + 1, 2], c=color_range[: i + 1], cmap="plasma")
        ax4.set_title("RSSM States (PCA)")
        ax4.axis("off")

        # キャンバス描画 & OpenCV変換
        canvas.draw()
        img = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8)
        # img = img.reshape(canvas.get_width_height()[::-1] + (4,))  # ARGB形式で取得
        # ARGB -> RGBA に変換
        img_w, img_h = canvas.get_width_height()
        img = img.reshape(img_h, img_w, 4)
        # ARGB -> RGBA に変換
        img = img[:, :, [1, 2, 3, 0]]

        # アルファチャンネルを取り除きBGRに変換
        img = img[:, :, :3]

        # RGBA -> BGR に変換
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        out.write(img)

        plt.close(fig)
        print(f"Frame {i + 1}/{len(imgs)} processed.")

    out.release()
    print(f"Animation saved to {save_path}")


def make_animation(imgs, recs, h_ts, s_ts, rssm_losses, actor_losses, value_losses, rewards, save_path="animation.mp4"):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(32, 16))
    ax1.set_title("Original Images")
    ax2.set_title("Reconstructed Images")
    ax3.axis("off")
    ax4.axis("off")
    ax3 = fig.add_subplot(143, projection="3d")
    ax4 = fig.add_subplot(144, projection="3d")
    ax1.axis("off")
    ax2.axis("off")
    color_range = np.arange(len(imgs))

    def update(frame):
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()

        ax1.imshow(imgs[frame])
        ax2.imshow(recs[frame])
        ax3.scatter(h_ts[: frame + 1, 0], h_ts[: frame + 1, 1], h_ts[: frame + 1, 2], c=color_range[: frame + 1], cmap="viridis", label="Beliefs (PCA)")
        ax4.scatter(s_ts[: frame + 1, 0], s_ts[: frame + 1, 1], s_ts[: frame + 1, 2], c=color_range[: frame + 1], cmap="plasma", label="States (PCA)")
        ax1.set_title("Original Images")
        ax2.set_title("Reconstructed Images")
        ax3.set_title("RSSM Beliefs (PCA)")
        ax4.set_title("RSSM States (PCA)")
        ax1.axis("off")
        ax2.axis("off")

    ani = animation.FuncAnimation(fig, update, frames=len(imgs), repeat=False)
    ani.save(save_path, writer="ffmpeg", fps=64)

    plt.close(fig)  # Close the figure to free memory
