# -*- coding: utf-8 -*-
"""
Implementation of a Dueling Deep Q-Network (DQN) agent with an
Intrinsic Curiosity Module (ICM) designed to play Super Mario Bros.

This agent utilizes several reinforcement learning techniques:
- Dueling DQN architecture (inspired by Rainbow DQN).
- Noisy Networks for parameter-space exploration.
- Prioritized Experience Replay (PER) for efficient learning from transitions.
- N-step bootstrapping to propagate rewards faster.
- Intrinsic Curiosity Module (ICM) to generate intrinsic rewards based on
  prediction errors, encouraging exploration of novel states.
- Environment wrappers for preprocessing:
  - Frame skipping
  - Grayscale conversion and resizing (to 84x90)
  - Frame stacking
- Custom reward shaping to augment the extrinsic reward signal.
- Centralized configuration via a Hyperparameters dataclass.
- Logs extrinsic (shaped/unshaped) and intrinsic rewards during training.
"""

import datetime
import math
import os
import random
import time
from collections import deque, namedtuple
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import gym
import gym_super_mario_bros
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.spaces import Box
from gym.wrappers import TimeLimit

# from gym.wrappers.frame_stack import FrameStack # Using custom FrameStack below
# from gym.wrappers.gray_scale_observation import GrayScaleObservation # Replaced by GrayScaleResize
# from gym.wrappers.resize_observation import ResizeObservation # Replaced by GrayScaleResize
from torchvision import transforms as T  # For GrayScaleResize
from PIL import Image  # For GrayScaleResize

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Optional: For TensorBoard logging
# from torch.utils.tensorboard import SummaryWriter


# === Configuration ===
@dataclass
class Hyperparameters:
    """Centralized hyperparameters for training."""

    # --- Environment ---
    env_id: str = "SuperMarioBros-v0"  # Changed back to default for broader use
    life_mode: int = 3  # Life mode (1 means terminate episode on first life lost)
    starting_lives: int = 3  # Default starting lives in Super Mario Bros
    skip_frames: int = 4
    stack_frames: int = 4
    resize_dim: Tuple[int, int] = (
        84,
        90,
    )  # Resize dimension (Height, Width) - Matches RainbowDQN expectation
    max_episode_steps: int = 4500  # Max steps per episode (via TimeLimit wrapper)

    # --- Reward Shaping ---
    death_penalty: float = -100
    move_reward: float = 0.1  # Small reward for moving right
    stuck_penalty: float = -0.1  # Penalty for staying in the same x-pos
    step_penalty: float = -0.01  # Small penalty per step to encourage progress

    # --- Training ---
    total_train_steps: int = 5_000_000
    batch_size: int = 32
    learning_rate: float = 0.0001  # Adam learning rate for DQN
    adam_eps: float = 1.5e-4  # Adam epsilon for DQN

    gamma: float = 0.9  # Discount factor for Bellman equation (extrinsic)
    target_update_freq: int = 10000  # Steps between target network updates
    gradient_clip_norm: float = 10.0  # Clip gradients to this norm

    # --- Replay Buffer (PER) ---
    buffer_size: int = 10000
    per_alpha: float = 0.6  # Priority exponent
    per_beta_start: float = 0.4  # Initial importance sampling exponent
    per_beta_frames: int = 2_000_000  # Steps to anneal beta to 1.0
    per_epsilon: float = (
        1e-5  # Small value added to priorities (using value from first script)
    )

    # --- N-Step Returns ---
    n_step: int = 5  # Number of steps for N-step returns

    # --- Noisy Nets ---
    noisy_sigma_init: float = 2.5  # Initial standard deviation for NoisyLinear layers (sigma_init in RainbowDQN)

    # --- ICM ---
    use_icm: bool = True  # Flag to enable/disable ICM
    icm_embed_dim: int = 256  # Dimensionality of ICM state encoding
    icm_beta: float = 0.2  # Weight for the forward model loss in ICM
    icm_eta: float = 0.01  # Scaling factor for intrinsic reward
    icm_lr: float = 0.0001  # Learning rate for ICM optimizer
    icm_adam_eps: float = 1.5e-4  # Adam epsilon for ICM

    # --- Logging & Saving ---
    log_interval_steps: int = 10000  # Log progress every N agent steps
    save_interval_steps: int = 100000  # Save checkpoint every N agent steps
    print_episode_summary: bool = True
    checkpoint_dir: str = "mario_rainbow_checkpoints_v4"  # Updated dir name
    load_checkpoint: bool = True  # Set to True to load latest checkpoint if exists

    # --- Derived / Calculated ---
    n_step_gamma: float = field(
        init=False
    )  # Calculated discount factor for n-step returns
    device: torch.device = field(init=False)  # Device for PyTorch
    # processed_resize_dim removed as resize_dim is now always Tuple

    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        self.n_step_gamma = self.gamma**self.n_step
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Validation for resize_dim
        if not (isinstance(self.resize_dim, tuple) and len(self.resize_dim) == 2):
            raise ValueError(
                f"Invalid resize_dim: {self.resize_dim}. Must be Tuple[int, int]."
            )


# === Experience Tuple ===
Experience = namedtuple(
    "Experience", ("state", "action", "reward", "next_state", "done")
)


# === Environment Wrappers ===


class SkipFrame(gym.Wrapper):
    """Skips a specified number of frames, accumulating the reward."""

    def __init__(self, env: gym.Env, skip: int):
        super().__init__(env)
        self._skip = skip
        self._steps_in_skip = 0  # Track steps within the current skip cycle

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Steps the environment and accumulates reward over skipped frames."""
        total_reward = 0.0
        total_original_reward = 0.0  # Track original reward sum
        done = False
        combined_info = {}
        self._steps_in_skip = 0

        for _ in range(self._skip):
            self._steps_in_skip += 1
            obs, reward, step_done, step_info = self.env.step(action)
            combined_info.update(step_info)  # Keep latest info
            total_reward += reward
            total_original_reward += reward  # Sum original reward
            if step_done:
                done = True
                break
        # Store the sum of original rewards before shaping/other wrappers affect it
        combined_info["original_reward_sum"] = total_original_reward
        # Store how many actual steps were taken in this skip cycle
        combined_info["_steps_in_skip"] = self._steps_in_skip
        return obs, total_reward, done, combined_info  # Return accumulated reward

    def reset(self, **kwargs) -> Any:
        """Resets the environment."""
        self._steps_in_skip = 0
        obs = self.env.reset(**kwargs)
        return obs


class GrayScaleResize(gym.ObservationWrapper):
    """Converts observations to grayscale and resizes them."""

    def __init__(self, env: gym.Env, shape: Tuple[int, int]):
        super().__init__(env)
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Grayscale(),
                T.Resize(shape),  # Use shape from params
                T.ToTensor(),  # Output is [0, 1] float tensor (C, H, W)
            ]
        )
        # Output shape is (1, H, W)
        self.observation_space = gym.spaces.Box(
            0.0, 1.0, shape=(1, shape[0], shape[1]), dtype=np.float32
        )

    def observation(self, obs: np.ndarray) -> torch.Tensor:
        """Applies the grayscale and resize transformations to the observation."""
        # transform expects HxWxC numpy array
        tensor_obs = self.transform(obs)
        # Return as numpy array for consistency with other wrappers if needed later
        # RainbowDQN expects tensor, so maybe keep as tensor?
        # Let's return numpy for now, FrameStack will handle concatenation
        return tensor_obs.squeeze(0).numpy()  # Remove channel dim, back to numpy HxW


class FrameStack(gym.Wrapper):
    """Stacks the last k observations along the channel dimension."""

    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        # Input observation space is now HxW after GrayScaleResize squeeze
        shp = env.observation_space.shape
        # Output shape is (k, H, W)
        self.observation_space = gym.spaces.Box(
            0, 1, shape=(k, shp[1], shp[2]), dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """Resets the environment and fills the frame stack with the initial observation."""
        obs = self.env.reset()  # Should be HxW numpy array
        for _ in range(self.k):
            self.frames.append(obs)
        # Stack along the first dimension (new channel dim)
        return np.stack(self.frames, axis=0)  # Output (k, H, W)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Steps the environment and updates the frame stack."""
        obs, reward, done, info = self.env.step(action)  # obs is HxW numpy array
        self.frames.append(obs)
        # Stack along the first dimension
        return np.stack(self.frames, axis=0), reward, done, info  # Output (k, H, W)


# Reward Shaping Wrapper (Keep from first script)
class RewardShapingWrapper(gym.Wrapper):
    """
    Applies custom reward shaping based on game info.
    Refined reset logic based on info dict structure.
    Note: This shapes the *extrinsic* reward. Intrinsic reward is handled separately.
    """

    def __init__(self, env: gym.Env, params: Hyperparameters):
        super().__init__(env)
        self.params = params
        self.prev_x_pos = 0
        self.prev_life = self.params.starting_lives
        self.current_episode_start_lives = self.params.starting_lives

    def reset(self, **kwargs) -> Any:
        """Reset state and internal reward shaping variables."""
        obs = self.env.reset(**kwargs)
        self.prev_x_pos = 0  # Assume starting x_pos is 0
        self.current_episode_start_lives = self.params.starting_lives
        self.prev_life = self.params.starting_lives
        return obs

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Steps the environment and applies reward shaping to the extrinsic reward."""
        obs, extrinsic_reward, done, info = self.env.step(
            action
        )  # Gets accumulated reward from SkipFrame

        custom_extrinsic_reward = extrinsic_reward
        current_x_pos = info.get("x_pos", self.prev_x_pos)
        current_lives = info.get("life", self.prev_life)
        life_lost = current_lives < self.prev_life
        if life_lost:
            custom_extrinsic_reward += self.params.death_penalty

        x_pos_diff = current_x_pos - self.prev_x_pos
        if x_pos_diff > 0:
            custom_extrinsic_reward += self.params.move_reward * x_pos_diff
        elif x_pos_diff == 0 and not done:
            custom_extrinsic_reward += self.params.stuck_penalty

        num_steps_taken = info.get("_steps_in_skip", self.params.skip_frames)
        custom_extrinsic_reward += self.params.step_penalty * num_steps_taken

        effective_done = done or (
            current_lives < self.current_episode_start_lives
            and self.params.life_mode == 1
        )
        info["effective_done"] = effective_done
        info["life_lost"] = life_lost
        info["original_reward_unshaped"] = (
            extrinsic_reward  # Store unshaped reward before shaping
        )

        self.prev_x_pos = current_x_pos
        self.prev_life = current_lives
        return obs, custom_extrinsic_reward, done, info  # Return shaped reward


# === Neural Network Components  ===


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration (factorized Gaussian noise)."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init  # Renamed from sigma_init for consistency

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize mean weights/biases and noise std deviations."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )  # Use out_features for bias

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate factorized Gaussian noise."""
        device = self.weight_mu.device
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Sample new noise vectors for weights and biases."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform linear operation with noisy weights/biases during training."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:  # Evaluation mode: use mean weights and biases (no noise)
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions, noisy_sigma_init):
        super().__init__()
        in_channel, h, w = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(in_channel, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate feature dimension dynamically
        with torch.no_grad():
            # Assume input shape (1, C, H, W) -> (1, 4, 84, 90)
            dummy = torch.zeros(1, in_channel, 84, 90)
            feat_dim = self.features(dummy).shape[1]

        # Value stream
        self.val_noisy = NoisyLinear(feat_dim, 512, std_init=noisy_sigma_init)
        self.val = NoisyLinear(512, 1, std_init=noisy_sigma_init)
        # Advantage stream
        self.adv_noisy = NoisyLinear(feat_dim, 512, std_init=noisy_sigma_init)
        self.adv = NoisyLinear(512, n_actions, std_init=noisy_sigma_init)

    def forward(self, x):
        # Normalize input here
        x = self.features(x / 255.0)
        # Value stream
        v = F.relu(self.val_noisy(x))
        v = self.val(v)
        # Advantage stream
        a = F.relu(self.adv_noisy(x))
        a = self.adv(a)
        # Combine V(s) + (A(s,a) - mean(A(s,a')))
        return v + (a - a.mean(dim=1, keepdim=True))

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers."""
        for m in [self.val_noisy, self.val, self.adv_noisy, self.adv]:
            m.reset_noise()

    def get_feature_dim(self, input_shape):
        """Helper to get feature dim without creating dummy tensor again."""
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            feat_dim = self.features(dummy).shape[1]
            return feat_dim


class ICM(nn.Module):
    """Intrinsic Curiosity Module adapted for RainbowDQN."""

    def __init__(self, feat_dim: int, n_actions: int, embed_dim: int):
        super().__init__()
        self.n_actions = n_actions
        # Encoder takes flattened features from RainbowDQN's feature extractor
        self.encoder = nn.Sequential(
            # nn.Flatten(), # Flattening happens in RainbowDQN.features
            nn.Linear(feat_dim, 512),
            nn.ReLU(),  # Adjusted input layer
            nn.Linear(512, embed_dim),
            nn.ReLU(),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(embed_dim * 2, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )
        self.forward_model = nn.Sequential(
            nn.Linear(embed_dim + n_actions, 512), nn.ReLU(), nn.Linear(512, embed_dim)
        )

    def forward(
        self, feat: torch.Tensor, next_feat: torch.Tensor, action: torch.Tensor
    ):
        """
        Forward pass for ICM. Takes features directly.
        Assumes features are already detached if needed.
        """
        # Encode the provided features
        phi = self.encoder(feat)
        phi_next = self.encoder(next_feat)

        # Inverse model
        inv_in = torch.cat([phi, phi_next], dim=1)
        logits = self.inverse_model(inv_in)
        inv_loss = F.cross_entropy(logits, action.long())  # Ensure action is long

        # Forward model
        a_onehot = F.one_hot(action.long(), self.n_actions).float().to(feat.device)
        # Detach phi for forward model input (as per original logic)
        fwd_in = torch.cat([phi.detach(), a_onehot], dim=1)
        pred_phi_next = self.forward_model(fwd_in)
        # Detach target phi_next for forward loss calculation
        fwd_loss = F.mse_loss(pred_phi_next, phi_next.detach())

        return (
            inv_loss,
            fwd_loss,
            pred_phi_next,
            phi_next.detach(),
        )  # Return detached phi_next

    def intrinsic_reward(
        self, pred_phi_next: torch.Tensor, target_phi_next: torch.Tensor, eta: float
    ) -> torch.Tensor:
        """Calculates the intrinsic reward based on the forward prediction error."""
        # Assumes inputs are detached appropriately from the forward pass
        reward = 0.5 * (pred_phi_next - target_phi_next).pow(2).sum(dim=1)
        return eta * reward


# === Replay Memory (Using simpler version from first script) ===


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer."""

    def __init__(self, capacity: int, alpha: float, params: Hyperparameters):
        self.capacity = capacity
        self.alpha = alpha
        self.params = params
        # Store raw numpy arrays (uint8 for states) to save memory before compression/conversion
        self.buffer: List[Optional[Experience]] = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0
        # N-step buffer stores raw numpy arrays
        self.n_step_accumulator = deque(maxlen=params.n_step)
        self.n_step_gamma = params.n_step_gamma
        self.gamma = params.gamma

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Adds a single-step transition to the n-step buffer."""
        # Store raw numpy states (e.g., (4, 84, 90) float32 [0,1] from FrameStack)
        self.n_step_accumulator.append(
            Experience(state, action, reward, next_state, done)
        )
        if len(self.n_step_accumulator) < self.params.n_step:
            return

        # Calculate N-step return
        n_step_reward = 0.0
        discount = 1.0
        start_exp = self.n_step_accumulator[0]
        final_next_state = self.n_step_accumulator[-1].next_state
        n_step_done = False
        for i in range(self.params.n_step):
            exp = self.n_step_accumulator[i]
            n_step_reward += discount * exp.reward
            discount *= self.gamma
            if exp.done:
                final_next_state = exp.next_state
                n_step_done = True
                break

        # Add N-step experience to main buffer
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        if self.size == 0 and max_prio == 0:
            max_prio = 1.0

        # Store the N-step experience (state_t, action_t, n_reward, state_t+n, done_t+n)
        experience = Experience(
            start_exp.state,
            start_exp.action,
            n_step_reward,
            final_next_state,
            n_step_done,
        )

        self.buffer[self.position] = experience
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float):
        """Samples a batch of experiences based on priorities."""
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")

        priorities_segment = self.priorities[: self.size]
        probs = priorities_segment**self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 1e-8:
            probs = np.ones_like(priorities_segment) / self.size
        else:
            probs /= probs_sum

        indices = np.random.choice(self.size, batch_size, p=probs, replace=True)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max() + 1e-8
        weights = np.array(weights, dtype=np.float32)

        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Stack numpy arrays correctly
        states = np.array(
            states, dtype=np.float32
        )  # Should be (B, C, H, W) float32 [0,1]
        next_states = np.array(
            next_states, dtype=np.float32
        )  # Should be (B, C, H, W) float32 [0,1]
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Updates the priorities of sampled experiences."""
        if not len(indices) == len(priorities):
            raise ValueError("Indices and priorities must have the same length.")
        for idx, priority in zip(indices, priorities):
            if not (0 <= idx < self.size):
                continue  # Skip invalid index
            self.priorities[idx] = abs(priority) + self.params.per_epsilon

    def __len__(self) -> int:
        return self.size


# === Agent Class (Adapted Structure) ===


class Agent:
    """Agent using RainbowDQN and ICM."""

    def __init__(self, state_shape: Tuple, n_actions: int, params: Hyperparameters):
        self.params = params
        self.device = params.device
        self.n_actions = n_actions
        self.steps_done = 0
        print(f"Initializing Agent on device: {self.device}")
        print(f"ICM Enabled: {self.params.use_icm}")

        # --- Initialize DQN Networks (RainbowDQN) ---
        print(f"Initializing RainbowDQN with state_shape: {state_shape}")
        self.online_net = RainbowDQN(
            state_shape, n_actions, params.noisy_sigma_init
        ).to(self.device)
        self.target_net = RainbowDQN(
            state_shape, n_actions, params.noisy_sigma_init
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # --- Initialize ICM (if enabled) ---
        self.icm = None
        self.icm_optimizer = None
        if self.params.use_icm:
            print("Initializing ICM...")
            # Calculate feature dimension from the online network's feature extractor
            with torch.no_grad():
                dummy = torch.zeros(1, *state_shape).to(self.device)
                # Pass dummy through feature extractor part ONLY (input is [0,1])
                feat_dim = self.online_net.features(dummy / 255.0).shape[1]
            self.icm = ICM(feat_dim, n_actions, params.icm_embed_dim).to(self.device)
            self.icm_optimizer = optim.Adam(
                self.icm.parameters(), lr=params.icm_lr, eps=params.icm_adam_eps
            )

        # --- DQN Optimizer ---
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=params.learning_rate, eps=params.adam_eps
        )

        # --- Replay Buffer ---
        self.memory = PrioritizedReplayBuffer(
            params.buffer_size, params.per_alpha, params
        )
        # N-step accumulator is handled inside the buffer class now

    def _current_beta(self) -> float:
        """Calculates the current PER beta value."""
        fraction = min(self.steps_done / self.params.per_beta_frames, 1.0)
        return self.params.per_beta_start + fraction * (
            1.0 - self.params.per_beta_start
        )

    def select_action(self, state: np.ndarray) -> int:
        """Selects action using the online network in evaluation mode."""
        # State is expected to be (C, H, W) numpy array, float32 [0, 1]
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            # RainbowDQN handles normalization internally
            q_values = self.online_net(state_tensor)
            action = q_values.argmax(1).item()
        return action

    def store_transition(
        self, state: Any, action: int, reward: float, next_state: Any, done: bool
    ):
        """Stores a single step transition for N-step calculation in buffer."""
        # Buffer expects numpy arrays (float32 [0,1] for states)
        self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self) -> Tuple[float, float, float]:
        """Performs one optimization step for both DQN and ICM networks."""
        if len(self.memory) < self.params.batch_size:
            return 0.0, 0.0, 0.0

        beta = self._current_beta()
        (
            states,
            actions,
            n_step_rewards_ext,
            next_states,
            n_step_dones,
            indices,
            weights,
        ) = self.memory.sample(self.params.batch_size, beta)

        states = torch.from_numpy(states).to(
            self.device
        )  # Shape (B, C, H, W), float32 [0, 1]
        actions = torch.from_numpy(actions).to(self.device)  # Shape (B,), int64
        n_step_rewards_ext = torch.from_numpy(n_step_rewards_ext).to(self.device)
        next_states = torch.from_numpy(next_states).to(
            self.device
        )  # Shape (B, C, H, W), float32 [0, 1]
        n_step_dones = torch.from_numpy(n_step_dones).to(self.device)
        weights = torch.from_numpy(weights).to(self.device)

        # --- ICM Computations (if enabled) ---
        icm_total_loss_val = 0.0
        intrinsic_reward_val = 0.0
        intrinsic_reward = torch.zeros_like(n_step_rewards_ext)

        if self.params.use_icm and self.icm is not None:
            self.icm.train()
            # Get features from online net's feature extractor (input needs normalization)
            # Detach features before passing to ICM forward
            with torch.no_grad():  # Extract features without tracking gradients for this part
                feat_icm = self.online_net.features(states / 255.0).detach()
                nxt_feat_icm = self.online_net.features(next_states / 255.0).detach()

            # Calculate ICM losses and embeddings
            inv_loss, fwd_loss, pred_phi_next, phi_next = self.icm(
                feat_icm, nxt_feat_icm, actions
            )
            icm_total_loss = (
                1 - self.params.icm_beta
            ) * inv_loss + self.params.icm_beta * fwd_loss
            icm_total_loss_val = icm_total_loss.item()

            # Calculate intrinsic reward
            intrinsic_reward = self.icm.intrinsic_reward(
                pred_phi_next, phi_next, self.params.icm_eta
            )
            intrinsic_reward_val = intrinsic_reward.mean().item()

            # --- Optimize ICM Network ---
            self.icm_optimizer.zero_grad()
            icm_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.icm.parameters(), self.params.gradient_clip_norm
            )
            self.icm_optimizer.step()

        # --- DQN Computations ---
        self.online_net.train()
        self.target_net.eval()

        total_reward = n_step_rewards_ext + intrinsic_reward.detach()

        with torch.no_grad():
            # Double DQN action selection
            next_q_values_online = self.online_net(next_states)  # Input [0, 1]
            next_actions = next_q_values_online.argmax(1)
            # Get Q values from target network
            next_q_values_target = self.target_net(next_states)  # Input [0, 1]
            next_q_target_selected = next_q_values_target.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            # Calculate TD Target
            target_Q = (
                total_reward
                + (1.0 - n_step_dones)
                * self.params.n_step_gamma
                * next_q_target_selected
            )

        # Get current Q values from online network
        current_q_values_online = self.online_net(states)  # Input [0, 1]
        current_Q = current_q_values_online.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate DQN loss (Smooth L1)
        elementwise_loss = F.smooth_l1_loss(
            current_Q, target_Q.detach(), reduction="none"
        )
        dqn_loss = (elementwise_loss * weights).mean()
        dqn_loss_val = dqn_loss.item()

        # --- Optimize DQN Network ---
        self.optimizer.zero_grad()
        dqn_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), self.params.gradient_clip_norm
        )
        self.optimizer.step()

        # --- Update Priorities ---
        new_priorities = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)

        # --- Step Counter & Target Update ---
        self.steps_done += 1
        if self.steps_done % self.params.target_update_freq == 0:
            self.update_target_network()

        # --- Reset Noise ---
        self.reset_noise()

        return dqn_loss_val, icm_total_loss_val, intrinsic_reward_val

    def update_target_network(self):
        """Copies weights from the online network to the target network."""
        print(f"Updating target network at step {self.steps_done}")
        self.target_net.load_state_dict(self.online_net.state_dict())

    def reset_noise(self):
        """Resets noise in the NoisyLinear layers of online and target networks."""
        self.online_net.reset_noise()
        self.target_net.reset_noise()  # Also reset target net noise (though it runs in eval)

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Saving model checkpoint to {filepath}...")
        save_data = {
            "model": self.online_net.state_dict(),  # Key 'model' for online net
            "optimizer": self.optimizer.state_dict(),  # Key 'optimizer' for DQN opt
            "icm_opt": self.icm_optimizer.state_dict()
            if self.params.use_icm and self.icm_optimizer
            else None,  # Key 'icm_opt'
            "steps_done": self.steps_done,  # Keep track of steps
            # Add episode count if needed for resuming logging accurately
            # 'episode': current_episode_count
        }
        try:
            torch.save(save_data, filepath)
            print("Model saved.")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath: str) -> bool:
        if not os.path.isfile(filepath):
            print(f"Checkpoint file not found at {filepath}. Starting from scratch.")
            return False
        print(f"Loading model checkpoint from {filepath}...")
        try:
            checkpoint = torch.load(filepath, map_location=self.device)

            # Load online network state (key 'model')
            if "model" in checkpoint:
                self.online_net.load_state_dict(checkpoint["model"])
                # Load target network immediately after online net
                self.target_net.load_state_dict(self.online_net.state_dict())
                print("Online and Target network states loaded from 'model' key.")
            else:
                print("Warning: Checkpoint missing 'model' key for network state.")
                # Attempt fallback to old key? Or fail? Let's fail for now.
                return False

            # Load DQN optimizer state (key 'optimizer')
            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
                print("DQN optimizer state loaded.")
            else:
                print(
                    "Warning: Checkpoint missing 'optimizer' key. DQN optimizer not loaded."
                )

            # Load ICM optimizer state (key 'icm_opt')
            if self.params.use_icm and self.icm_optimizer:
                if "icm_opt" in checkpoint and checkpoint["icm_opt"]:
                    self.icm_optimizer.load_state_dict(checkpoint["icm_opt"])
                    print("ICM optimizer state loaded.")
                else:
                    print(
                        "Warning: Checkpoint missing 'icm_opt' key or ICM optimizer not initialized. ICM optimizer not loaded."
                    )
            # If ICM parameters are part of 'model', they are loaded above. If they were saved
            # under a separate key (like 'icm_state_dict' in the first script), that logic is needed here.
            # if the ICM module is part of the main Agent class structure during saving, which it isn't here.
            # Let's assume ICM state needs separate loading if it exists.
            if self.params.use_icm and self.icm:
                if (
                    "icm_state_dict" in checkpoint and checkpoint["icm_state_dict"]
                ):  # Check for old key
                    self.icm.load_state_dict(checkpoint["icm_state_dict"])
                    print(
                        "ICM network state loaded from 'icm_state_dict' key (fallback)."
                    )
                # else: No explicit ICM state saved in the target format

            # Load step counter
            self.steps_done = checkpoint.get(
                "steps_done", checkpoint.get("frame_idx", 0)
            )  # Handle both keys
            print(f"Resuming from step {self.steps_done}.")
            # Load episode counter if saved
            # episode_count = checkpoint.get('episode', 0)

            # Ensure networks are on the correct device and modes
            self.online_net.to(self.device)
            self.target_net.to(self.device)
            if self.icm:
                self.icm.to(self.device)
            self.target_net.eval()
            self.online_net.train()
            if self.icm:
                self.icm.train()

            print("Model loaded successfully.")
            self.reset_noise()
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback

            traceback.print_exc()
            return False


# === Environment Creation Function (Adapted) ===


def make_env(params: Hyperparameters) -> gym.Env:
    """Creates and wraps the Super Mario Bros environment."""
    env = gym_super_mario_bros.make(params.env_id)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=params.skip_frames)
    # Use reward shaping wrapper
    env = RewardShapingWrapper(env, params)
    # Use GrayScaleResize wrapper - output is (H, W) numpy array [0, 1]
    env = GrayScaleResize(env, shape=params.resize_dim)
    # Use FrameStack wrapper - output is (k, H, W) numpy array [0, 1]
    env = FrameStack(env, k=params.stack_frames)
    # Apply TimeLimit
    env = TimeLimit(env, max_episode_steps=params.max_episode_steps)
    return env


# === Main Training Script ===

if __name__ == "__main__":
    params = Hyperparameters()

    print("--- Hyperparameters (RainbowDQN Structure) ---")
    param_dict = vars(params)
    derived_keys = ["n_step_gamma", "device"]
    for key, value in param_dict.items():
        if key not in derived_keys:
            print(f"{key}: {value}")
    print(f"device: {params.device}")
    print("--------------------------------------------")

    env = make_env(params)
    state_shape = env.observation_space.shape  # Should be (4, 84, 90)
    print(f"State shape = {state_shape}")
    n_actions = env.action_space.n

    print(f"\n--- Environment Info ---")
    print(f"State shape (after wrappers): {state_shape}")
    print(
        f"Observation space dtype: {env.observation_space.dtype}"
    )  # Should be float32
    print(f"Number of actions: {n_actions}")
    print(f"Running in {params.life_mode}-Life Mode")
    print(f"ICM Enabled: {params.use_icm}")
    print("------------------------\n")

    agent = Agent(state_shape=state_shape, n_actions=n_actions, params=params)

    # --- Load Checkpoint ---
    # Construct path using the new checkpoint_dir
    latest_checkpoint_path = os.path.join(
        params.checkpoint_dir,
        "rainbow_icm_latest.pth",  # Use a consistent name for latest
    )
    start_episode = 0  # Default start
    if params.load_checkpoint:
        if agent.load_model(latest_checkpoint_path):
            # Optionally load episode count from checkpoint if saved
            # checkpoint = torch.load(latest_checkpoint_path, map_location=params.device)
            # start_episode = checkpoint.get('episode', 0)
            pass  # Step count is loaded internally by load_model

    # --- Training Loop ---
    episode_rewards_shaped = deque(maxlen=100)
    episode_rewards_unshaped = deque(maxlen=100)
    episode_rewards_intrinsic = deque(maxlen=100)

    total_agent_steps = agent.steps_done
    episode_count = start_episode  # Start from loaded episode if applicable
    global_start_time = time.time()
    last_log_time = global_start_time
    last_log_step = total_agent_steps

    try:
        while total_agent_steps < params.total_train_steps:
            episode_count += 1
            state = env.reset()  # Returns numpy array (4, 84, 90), float32 [0, 1]

            current_episode_reward_shaped = 0
            current_episode_reward_unshaped = 0
            current_episode_intrinsic_reward_sum = 0
            current_episode_steps = 0
            current_episode_dqn_losses = []
            current_episode_icm_losses = []
            max_x_pos = 0
            episode_status = "RUNNING"

            while True:
                # Agent selects action based on normalized state
                action = agent.select_action(state)

                # Environment steps
                next_state, shaped_extrinsic_reward, done, info = env.step(action)

                # Get original reward if available
                original_reward = info.get(
                    "original_reward_unshaped", shaped_extrinsic_reward
                )

                # Check effective done based on TimeLimit or custom logic
                effective_done = info.get("effective_done", done)
                life_lost = info.get("life_lost", False)  # From RewardShapingWrapper
                current_x_pos = info.get("x_pos", 0)

                # Store transition (states are float32 [0, 1])
                agent.store_transition(
                    state, action, shaped_extrinsic_reward, next_state, done
                )  # Pass gym 'done'

                state = next_state

                # Accumulate stats
                current_episode_reward_shaped += shaped_extrinsic_reward
                current_episode_reward_unshaped += original_reward
                current_episode_steps += 1
                max_x_pos = max(max_x_pos, current_x_pos)

                # Optimize model
                dqn_loss, icm_loss, intrinsic_reward_batch_avg = 0.0, 0.0, 0.0
                if len(agent.memory) >= params.batch_size:
                    dqn_loss, icm_loss, intrinsic_reward_batch_avg = (
                        agent.optimize_model()
                    )
                    if dqn_loss > 0:
                        current_episode_dqn_losses.append(dqn_loss)
                    if icm_loss > 0 and params.use_icm:
                        current_episode_icm_losses.append(icm_loss)
                    current_episode_intrinsic_reward_sum += intrinsic_reward_batch_avg
                    total_agent_steps = agent.steps_done  # Update step count

                    # --- Periodic Logging ---
                    if (
                        total_agent_steps % params.log_interval_steps == 0
                        and total_agent_steps > last_log_step
                    ):
                        avg_reward_shaped_100 = (
                            np.mean(episode_rewards_shaped)
                            if episode_rewards_shaped
                            else 0.0
                        )
                        avg_reward_unshaped_100 = (
                            np.mean(episode_rewards_unshaped)
                            if episode_rewards_unshaped
                            else 0.0
                        )
                        avg_reward_intrinsic_100 = (
                            np.mean(episode_rewards_intrinsic)
                            if episode_rewards_intrinsic and params.use_icm
                            else 0.0
                        )
                        current_time = time.time()
                        elapsed_interval = current_time - last_log_time
                        steps_in_interval = total_agent_steps - last_log_step
                        fps = (
                            steps_in_interval / elapsed_interval
                            if elapsed_interval > 0
                            else 0
                        )
                        print(f"\n--- Progress ---")
                        print(
                            f"Steps: {total_agent_steps}/{params.total_train_steps} ({total_agent_steps/params.total_train_steps*100:.2f}%)"
                        )
                        print(f"Episodes: {episode_count}")
                        print(
                            f"Avg Shaped Reward (Last 100 ep): {avg_reward_shaped_100:.2f}"
                        )
                        print(
                            f"Avg Unshaped Reward (Last 100 ep): {avg_reward_unshaped_100:.2f}"
                        )
                        if params.use_icm:
                            print(
                                f"Avg Intrinsic Reward (Last 100 ep): {avg_reward_intrinsic_100:.4f}"
                            )
                        print(
                            f"Last DQN Loss: {dqn_loss:.4f}" if dqn_loss > 0 else "N/A"
                        )
                        if params.use_icm:
                            print(
                                f"Last ICM Loss: {icm_loss:.4f}"
                                if icm_loss > 0
                                else "N/A"
                            )
                        print(f"Current Beta (PER): {agent._current_beta():.4f}")
                        print(f"FPS (Agent Steps): {fps:.2f}")
                        print(f"Buffer Size: {len(agent.memory)}/{params.buffer_size}")
                        elapsed_total = datetime.timedelta(
                            seconds=int(current_time - global_start_time)
                        )
                        print(f"Elapsed Time: {elapsed_total}")
                        print(f"----------------")
                        last_log_time = current_time
                        last_log_step = total_agent_steps

                    # --- Periodic Checkpoint Saving ---
                    if (
                        total_agent_steps % params.save_interval_steps == 0
                        and total_agent_steps > 0
                        and total_agent_steps > last_log_step
                    ):
                        save_path = os.path.join(
                            params.checkpoint_dir,
                            f"rainbow_icm_step_{total_agent_steps}.pth",
                        )
                        # Save with the new format (keys: 'model', 'optimizer', 'icm_opt', 'steps_done')
                        agent.save_model(save_path)
                        agent.save_model(latest_checkpoint_path)  # Overwrite latest

                # --- Check Episode End ---
                if effective_done:  # Use effective_done from RewardShapingWrapper
                    if info.get("TimeLimit.truncated", False):
                        episode_status = "TIMELIMIT"
                    elif life_lost and params.life_mode == 1:
                        episode_status = "LIFE_LOST (1-Life Mode)"
                    elif done:
                        episode_status = "GAME_DONE"
                    else:
                        episode_status = "UNKNOWN"
                    break

            # --- End of Episode ---
            episode_rewards_shaped.append(current_episode_reward_shaped)
            episode_rewards_unshaped.append(current_episode_reward_unshaped)
            avg_intrinsic_episode = (
                current_episode_intrinsic_reward_sum / current_episode_steps
                if current_episode_steps > 0 and params.use_icm
                else 0.0
            )
            episode_rewards_intrinsic.append(avg_intrinsic_episode)
            avg_dqn_loss_episode = (
                np.mean(current_episode_dqn_losses)
                if current_episode_dqn_losses
                else 0.0
            )
            avg_icm_loss_episode = (
                np.mean(current_episode_icm_losses)
                if current_episode_icm_losses and params.use_icm
                else 0.0
            )

            if params.print_episode_summary:
                print(
                    f"Episode {episode_count} finished after {current_episode_steps} steps. Status: {episode_status}"
                )
                print(f"  Total Shaped Reward: {current_episode_reward_shaped:.2f}")
                print(f"  Total Unshaped Reward: {current_episode_reward_unshaped:.2f}")
                if params.use_icm:
                    print(f"  Avg Intrinsic Reward: {avg_intrinsic_episode:.4f}")
                print(f"  Max X Position: {max_x_pos}")
                print(f"  Avg DQN Loss: {avg_dqn_loss_episode:.4f}")
                if params.use_icm:
                    print(f"  Avg ICM Loss: {avg_icm_loss_episode:.4f}")
                print(f"  Agent Steps: {total_agent_steps}")

            if total_agent_steps >= params.total_train_steps:
                print("\nTarget number of training steps reached.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback

        traceback.print_exc()

    finally:
        print("\nSaving final model...")
        final_save_path = os.path.join(params.checkpoint_dir, "rainbow_icm_final.pth")
        agent.save_model(final_save_path)
        agent.save_model(latest_checkpoint_path)

        print(f"\nTraining finished.")
        print(f"Total episodes: {episode_count}")
        print(f"Total agent steps: {total_agent_steps}")
        env.close()
        print("Environment closed.")

