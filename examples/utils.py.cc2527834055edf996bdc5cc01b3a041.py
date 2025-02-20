"""Utility functions for training and evaluation."""

from collections.abc import Iterable

import cv2
import numpy as np
import torch
from torch.nn import Module


def imagine_ahead(prev_state, prev_belief, policy, transition_model, stochastic_model, planning_horizon=12):
    # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
    T = planning_horizon
    beliefs, prior_states, actions = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
    beliefs[0], prior_states[0] = prev_belief, prev_state

    # Loop over time sequence
    for t in range(T - 1):
        t_t = torch.ones(prior_states[t].size(0), 1).to(prior_states[t].device)

        _actions = policy.sample({"z_t": beliefs[t].detach(), "s_t": prior_states[t].detach()})["a_t"]
        _actions[0] = torch.clamp(_actions[0], -1, 1)
        _actions[1] = torch.clamp(_actions[1], 0, 1)
        _actions[2] = torch.clamp(_actions[2], 0, 1)

        actions[t] = _actions
        # Compute belief (deterministic hidden state)
        beliefs[t + 1] = transition_model.sample({"z_t": beliefs[t], "s_t": prior_states[t], "a_t": actions[t], "t_t": t_t})["z_tp1"]
        # Compute state prior by applying transition dynamics
        prior_states[t + 1] = stochastic_model.sample({"z_t": beliefs[t + 1]}, reparam=True)["s_t"]

    imagined_traj = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0), torch.stack(actions[:-1], dim=0)]
    return imagined_traj


def lambda_return(imged_reward, value_pred, bootstrap, discount=0.99, lambda_=0.95):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    next_values = torch.cat([value_pred[1:], bootstrap[None]], 0)
    discount_tensor = discount * torch.ones_like(imged_reward)  # pcont
    inputs = imged_reward + discount_tensor * next_values * (1 - lambda_)
    last = bootstrap
    indices = reversed(range(len(inputs)))
    outputs = []
    for index in indices:
        inp, disc = inputs[index], discount_tensor[index]
        last = inp + disc * lambda_ * last
        outputs.append(last)
    outputs.reverse()
    outputs = torch.stack(outputs, 0)
    returns = outputs
    return returns


def advantage_and_return(imged_reward, value_pred, discount=0.99, lambda_=0.95, normalize=False):
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
    return np.clip(np.floor((observation + 0.5) * 2**bit_depth) * 2 ** (8 - bit_depth), 0, 2**8 - 1).astype(np.uint8)


def _images_to_observation(images, bit_depth):
    images = torch.tensor(cv2.resize(images, (64, 64), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1), dtype=torch.float32)  # Resize and put channel first
    # Quantise, centre and dequantise inplace
    preprocess_observation_(images, bit_depth)
    return images.unsqueeze(dim=0)  # Add batch dimension

