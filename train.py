# -*- coding: utf-8 -*-
"""
Improved Rainbow DQN Agent for Super Mario Bros. with ICM.

Combines:
- Dueling DQN
- Noisy Nets for exploration
- Prioritized Experience Replay (PER)
- N-step Bootstrapping
- Distributional RL (C51)
- Intrinsic Curiosity Module (ICM) for exploration bonus
- Reward Shaping Wrapper (Refined reset logic)
- Frame Normalization Wrapper (to float32 [0,1])
- Centralized Hyperparameters
- Observation resized to (84, 84)
- Logs extrinsic (shaped/unshaped) and intrinsic rewards
- Fixed device mismatch error during network initialization.
- Integrated ICM using a shared feature extractor.
- Fixed state shape issue caused by GrayScaleObservation keep_dim.
"""

import numpy as np
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers import TimeLimit
from gym.spaces import Box
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from collections import namedtuple, deque
from typing import List, Tuple, Optional, Any, Deque, Dict, Union
import time
import os
from dataclasses import dataclass, field
import datetime # Added for elapsed time formatting

# Optional: For TensorBoard logging
# from torch.utils.tensorboard import SummaryWriter


# === Configuration ===
@dataclass
class Hyperparameters:
    """Centralized hyperparameters for training."""

    # --- Environment ---
    env_id: str = "SuperMarioBros-v0"
    life_mode: int = 3 # Life mode (1 means terminate episode on first life lost)
    starting_lives: int = 3 # Default starting lives in Super Mario Bros
    skip_frames: int = 4
    stack_frames: int = 4
    resize_dim: Union[int, Tuple[int, int]] = (84) # Resize dimension (Height, Width)
    max_episode_steps: int = 4500 # Max steps per episode (via TimeLimit wrapper)

    # --- Reward Shaping ---
    death_penalty: float = 0
    move_reward: float = 0 # Small reward for moving right
    stuck_penalty: float = 0 # Penalty for staying in the same x-pos
    step_penalty: float = 0 # Small penalty per step to encourage progress

    # --- Training ---
    total_train_steps: int = 5_000_000
    batch_size: int = 32
    learning_rate: float = 0.0001 # Adam learning rate for DQN
    adam_eps: float = 1.5e-4 # Adam epsilon for DQN

    gamma: float = 0.8 # Discount factor for Bellman equation (extrinsic)
    target_update_freq: int = 10000 # Steps between target network updates
    gradient_clip_norm: float = 10.0 # Clip gradients to this norm

    # --- Replay Buffer (PER) ---
    buffer_size: int = 10000
    per_alpha: float = 0.5 # Priority exponent
    per_beta_start: float = 0.4 # Initial importance sampling exponent
    per_beta_frames: int = 1_000_000 # Steps to anneal beta to 1.0
    per_epsilon: float = 1e-5 # Small value added to priorities

    # --- N-Step Returns ---
    n_step: int = 5 # Number of steps for N-step returns

    # --- Distributional RL (C51) ---
    num_atoms: int = 51 # Number of atoms in value distribution
    v_min: float = -50.0 # Minimum value for distribution support (adjust if needed for ICM)
    v_max: float = 150.0 # Maximum value for distribution support (adjust if needed for ICM)

    # --- Noisy Nets ---
    noisy_std_init: float = 2.5 # Initial standard deviation for NoisyLinear layers

    # --- ICM ---
    use_icm: bool = True # Flag to enable/disable ICM
    icm_embed_dim: int = 256 # Dimensionality of ICM state encoding
    icm_beta: float = 0.2 # Weight for the forward model loss in ICM
    icm_eta: float = 0.1 # Scaling factor for intrinsic reward
    icm_lr: float = 0.0001 # Learning rate for ICM optimizer
    icm_adam_eps: float = 1.5e-4 # Adam epsilon for ICM

    # --- Logging & Saving ---
    log_interval_steps: int = 10000 # Log progress every N agent steps
    save_interval_steps: int = 100000 # Save checkpoint every N agent steps
    print_episode_summary: bool = True
    checkpoint_dir: str = "mario_rainbow_icm_checkpoints_v1" # Updated dir name
    load_checkpoint: bool = True # Set to True to load latest checkpoint if exists
    # Optional: Path for TensorBoard logs
    # tensorboard_log_dir: str = "logs/mario_rainbow_icm_v1"

    # --- Derived / Calculated ---
    n_step_gamma: float = field(init=False) # Calculated discount factor for n-step returns
    delta_z: float = field(init=False) # Calculated delta_z for distributional RL
    device: torch.device = field(init=False) # Device for PyTorch
    processed_resize_dim: Tuple[int, int] = field(init=False) # Process resize_dim

    def __post_init__(self):
        """Calculate derived parameters after initialization."""
        self.n_step_gamma = self.gamma**self.n_step
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure resize_dim is a tuple (H, W) for ResizeObservation wrapper
        if isinstance(self.resize_dim, int):
            self.processed_resize_dim = (self.resize_dim, self.resize_dim)
        elif isinstance(self.resize_dim, tuple) and len(self.resize_dim) == 2:
            self.processed_resize_dim = self.resize_dim
        else:
            raise ValueError(
                f"Invalid resize_dim: {self.resize_dim}. Must be int or Tuple[int, int]."
            )


# === Experience Tuple ===
Experience = namedtuple(
    "Experience", ("state", "action", "reward", "next_state", "done")
)


# === Environment Wrappers ===

class SkipFrame(gym.Wrapper):
    """
    Return only every `skip`-th frame (frameskipping).
    Sums rewards over skipped frames. Handles info dict aggregation.
    Also stores the sum of *original* rewards in the info dict.
    """
    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        if skip <= 0:
            raise ValueError(f"Frame skip must be > 0, got {skip}")
        self._skip = skip
        self._steps_in_skip = 0 # Track steps within the current skip cycle

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Repeat action, sum reward, and update info."""
        total_reward = 0.0
        total_original_reward = 0.0 # Track original reward sum
        done = False
        combined_info = {}
        self._steps_in_skip = 0

        for i in range(self._skip):
            self._steps_in_skip += 1
            obs, reward, step_done, step_info = self.env.step(action)
            # Update combined_info, prioritizing info from later steps
            combined_info.update(step_info)
            total_reward += reward
            total_original_reward += reward # Sum original reward
            if step_done:
                done = True
                # Pass the final observation and info
                break
        # Store the sum of original rewards before shaping/other wrappers affect it
        combined_info["original_reward_sum"] = total_original_reward
        # Store how many actual steps were taken in this skip cycle
        combined_info["_steps_in_skip"] = self._steps_in_skip
        # The returned observation is from the last step in the skip sequence
        return obs, total_reward, done, combined_info

    def reset(self, **kwargs) -> Any:
        """Resets the environment."""
        self._steps_in_skip = 0
        obs = self.env.reset(**kwargs)
        return obs


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
        self.prev_x_pos = 0 # Assume starting x_pos is 0
        self.current_episode_start_lives = self.params.starting_lives
        self.prev_life = self.params.starting_lives
        return obs

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Steps the environment and applies reward shaping to the extrinsic reward."""
        # The reward received here might already be summed by SkipFrame
        obs, extrinsic_reward, done, info = self.env.step(action)

        # --- Apply Reward Shaping to Extrinsic Reward ---
        custom_extrinsic_reward = extrinsic_reward

        # Safely get current values from info
        current_x_pos = info.get("x_pos", self.prev_x_pos)
        current_lives = info.get("life", self.prev_life)

        # 1. Penalty for losing a life
        life_lost = current_lives < self.prev_life
        if life_lost:
            custom_extrinsic_reward += self.params.death_penalty

        # 2. Reward for moving right (scaled by distance moved)
        x_pos_diff = current_x_pos - self.prev_x_pos
        if x_pos_diff > 0:
            custom_extrinsic_reward += self.params.move_reward * x_pos_diff

        # 3. Penalty for getting stuck (optional)
        elif x_pos_diff == 0 and not done:
            custom_extrinsic_reward += self.params.stuck_penalty

        # 4. Small penalty per step (optional, encourages efficiency)
        num_steps_taken = info.get("_steps_in_skip", self.params.skip_frames)
        custom_extrinsic_reward += self.params.step_penalty * num_steps_taken

        # --- 1-Life Mode Termination Check ---
        effective_done = done or (
            current_lives < self.current_episode_start_lives
            and self.params.life_mode == 1
        )
        info["effective_done"] = effective_done
        info["life_lost"] = life_lost

        # Update state for next step's comparison
        self.prev_x_pos = current_x_pos
        self.prev_life = current_lives

        # Return the original 'done' flag and the SHAPED EXTRINSIC reward
        return obs, custom_extrinsic_reward, done, info


class NormalizeFrame(gym.ObservationWrapper):
    """
    Normalizes frame observations to be float32 in [0, 1].
    Handles LazyFrames from FrameStack.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, obs: Any) -> np.ndarray:
        """Normalizes the observation."""
        # Handle LazyFrames by converting to numpy array first
        if isinstance(obs, gym.wrappers.frame_stack.LazyFrames):
            obs_array = np.array(obs, dtype=np.uint8)
        elif isinstance(obs, np.ndarray):
            obs_array = obs
        else:
            # Fallback if input is not LazyFrames or ndarray
            obs_array = np.array(obs)

        # Ensure it's float32 and in [0, 1]
        if obs_array.dtype == np.uint8:
             normalized_obs = obs_array.astype(np.float32) / 255.0
        elif obs_array.dtype == np.float32:
             # Check if already normalized
             if obs_array.max() > 1.0 or obs_array.min() < 0.0:
                 print(f"Warning: Float32 observation detected outside [0, 1] range (min: {obs_array.min()}, max: {obs_array.max()}). Re-normalizing.")
                 # Clamp and normalize assuming original scale was 0-255
                 normalized_obs = np.clip(obs_array, 0, 255).astype(np.float32) / 255.0
             else:
                 normalized_obs = obs_array # Assume already normalized
        else:
             # Attempt normalization for other types, warn user
             print(f"Warning: Unexpected observation dtype {obs_array.dtype}. Attempting normalization assuming 0-255 scale.")
             normalized_obs = obs_array.astype(np.float32) / 255.0

        # Ensure correct final dtype
        return normalized_obs.astype(np.float32)


# === Neural Network Components ===

# Shared Feature Extractor Module (Used by DQN and ICM)
class FeatureExtractor(nn.Module):
    """Convolutional feature extractor."""
    def __init__(self, input_channels: int):
        super(FeatureExtractor, self).__init__()
        # Standard CNN architecture for Atari-like environments
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes input through convolutional layers.
        Args:
            x: Input tensor (Batch, Channels, Height, Width), expected float32 [0, 1].
        Returns:
            Output feature map tensor.
        """
        # Input x should already be normalized float32 [0, 1] by the wrapper
        return self.conv(x)

    def get_output_dim(self, shape: Tuple[int, int, int]) -> int:
        """Calculate flattened feature size after convolutions.
        Args:
            shape: Input shape (Channels, Height, Width).
        Returns:
            Integer representing the total number of features after flattening.
        """
        with torch.no_grad():
            # Create a dummy input tensor explicitly on CPU
            dummy_input = torch.zeros(1, *shape, device="cpu")
            # Create a temporary features model on CPU to calculate size
            features_cpu = nn.Sequential(
                nn.Conv2d(shape[0], 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            ).to("cpu") # Ensure it's on CPU
            o = features_cpu(dummy_input)
            feature_size = int(np.prod(o.size())) # Calculate product of dimensions
            # print(f"Calculated feature size: {feature_size} from output shape {o.shape}") # Debug print
            return feature_size


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration (factorized Gaussian noise)."""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters for mean weights and biases
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))

        # Learnable parameters for standard deviation of noise
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))

        # Buffers to store noise samples (non-learnable)
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        # Initialize parameters and noise
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize mean weights/biases and noise std deviations."""
        mu_range = 1 / math.sqrt(self.in_features)
        # Initialize means uniformly
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        # Initialize sigmas with a constant value scaled by input size
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features)) # Note: uses out_features for bias sigma

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate factorized Gaussian noise."""
        # Use the layer's device (where parameters are located)
        device = self.weight_mu.device
        x = torch.randn(size, device=device)
        # Apply sign-sqrt transformation for stability
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        """Sample new noise vectors for weights and biases."""
        # Generate noise for input and output dimensions
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # Combine noise using outer product for weights
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in)) # ger is outer product
        # Use output noise directly for biases
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform linear operation with noisy weights/biases during training."""
        if self.training:
            # Calculate noisy weights and biases
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else: # Evaluation mode: use mean weights and biases (no noise)
            weight = self.weight_mu
            bias = self.bias_mu
        # Apply standard linear transformation
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    """Dueling Network with Noisy Layers and Distributional RL (C51)."""
    def __init__(
        self,
        input_shape: Tuple[int, int, int], # Expected (C, H, W) e.g., (4, 84, 84)
        num_actions: int,
        params: Hyperparameters,
        feature_extractor: FeatureExtractor, # Pass the shared feature extractor
    ):
        super(RainbowDQN, self).__init__()
        # *** Check input shape ***
        if len(input_shape) != 3:
            raise ValueError(f"Expected input_shape (C, H, W), got {input_shape}")
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = params.num_atoms
        self.v_min = params.v_min
        self.v_max = params.v_max
        self.device = params.device # Store device from params

        # Use the provided shared feature extractor
        self.features = feature_extractor
        # Calculate feature size based on the shared extractor's output
        feature_size = self.features.get_output_dim(self.input_shape)

        # Dueling Streams: Advantage and Value, using NoisyLinear layers
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_size, 512, std_init=params.noisy_std_init),
            nn.ReLU(),
            # Output: num_actions * num_atoms for the advantage distribution
            NoisyLinear(512, num_actions * self.num_atoms, std_init=params.noisy_std_init),
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(feature_size, 512, std_init=params.noisy_std_init),
            nn.ReLU(),
            # Output: num_atoms for the value distribution
            NoisyLinear(512, self.num_atoms, std_init=params.noisy_std_init),
        )

        # Support atoms for distributional RL (initialized on CPU, moved later)
        self.register_buffer(
            "support", torch.linspace(self.v_min, self.v_max, self.num_atoms)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the probability distribution over atoms for each action.
        Args:
            x: Input tensor (Batch, C, H, W), normalized float32 [0, 1].
        Returns:
            q_probs: Tensor (Batch, Num_Actions, Num_Atoms) representing action-value distributions.
        """
        batch_size = x.size(0)
        # 1. Extract features using the shared CNN
        # Input x should be normalized float32 [0, 1]
        x = self.features(x)
        # 2. Flatten features for linear layers
        x = x.view(batch_size, -1)

        # 3. Pass flattened features through value and advantage streams
        # Reshape outputs to (batch_size, 1, num_atoms) for value
        # and (batch_size, num_actions, num_atoms) for advantage
        value_dist = self.value_stream(x).view(batch_size, 1, self.num_atoms)
        advantage_dist = self.advantage_stream(x).view(
            batch_size, self.num_actions, self.num_atoms
        )

        # 4. Combine streams using dueling architecture formula:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # This is applied to the distributions (logits before softmax)
        q_dist = value_dist + (advantage_dist - advantage_dist.mean(dim=1, keepdim=True))

        # 5. Apply softmax along the atom dimension (-1) to get probabilities
        q_probs = F.softmax(q_dist, dim=-1)
        return q_probs

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """Calculates the expected Q-values from the action-value distributions.
        Args:
            x: Input tensor (Batch, C, H, W), normalized float32 [0, 1].
        Returns:
            q_values: Tensor (Batch, Num_Actions) of expected Q-values.
        """
        # Get the action-value distributions (probabilities)
        q_probs = self.forward(x)
        # Ensure support atoms are on the same device as probabilities
        support = self.support.to(q_probs.device)
        # Calculate expected value: Q(s,a) = sum_i (probability(z_i) * z_i)
        # Reshape support for broadcasting: (1, 1, num_atoms)
        q_values = (q_probs * support.view(1, 1, self.num_atoms)).sum(dim=2)
        return q_values

    def reset_noise(self):
        """Resets noise in all NoisyLinear layers within the network."""
        # Iterate through all modules (including nested ones)
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class ICM(nn.Module):
    """Intrinsic Curiosity Module (ICM).
    Predicts the consequence of actions in a learned feature space
    and generates an intrinsic reward based on prediction error.
    """
    def __init__(
        self,
        input_shape: Tuple[int, int, int], # Expected (C, H, W) e.g., (4, 84, 84)
        num_actions: int,
        params: Hyperparameters,
        feature_extractor: FeatureExtractor, # Pass the shared feature extractor
    ):
        super(ICM, self).__init__()
         # *** Check input shape ***
        if len(input_shape) != 3:
            raise ValueError(f"Expected input_shape (C, H, W), got {input_shape}")
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.embed_dim = params.icm_embed_dim
        self.device = params.device # Store device from params

        # Use the *shared* feature extractor for encoding states
        # Gradients from ICM loss will flow back through this extractor
        self.feature_extractor = feature_extractor
        # Calculate feature size from the shared extractor's output
        feature_size = self.feature_extractor.get_output_dim(self.input_shape)

        # State Encoder (phi) - Maps CNN features to a lower-dimensional embedding space
        # This learned embedding should capture task-relevant information
        self.encoder = nn.Sequential(
            nn.Linear(feature_size, 512), # Intermediate dense layer
            nn.ReLU(),
            nn.Linear(512, self.embed_dim),
            # No final activation often used here, allows embedding to be anywhere in R^embed_dim
        )

        # Inverse Model - Predicts action a_t given state embeddings phi(s_t) and phi(s_{t+1})
        # Helps ensure the learned state embedding phi captures action-relevant information
        self.inverse_model = nn.Sequential(
            nn.Linear(self.embed_dim * 2, 512), # Input is concatenation of two embeddings
            nn.ReLU(),
            nn.Linear(512, self.num_actions), # Output logits for each possible action
        )

        # Forward Model - Predicts next state embedding phi(s_{t+1}) given current embedding phi(s_t) and action a_t
        # The prediction error of this model forms the basis of the intrinsic reward
        self.forward_model = nn.Sequential(
            nn.Linear(self.embed_dim + self.num_actions, 512), # Input is concatenation of embedding and one-hot action
            nn.ReLU(),
            nn.Linear(512, self.embed_dim), # Output is the predicted next state embedding
        )

    def forward(
        self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs forward pass for ICM, calculating losses and embeddings.
        Args:
            state: Batch of current states (normalized float32 [0, 1]). Shape (B, C, H, W).
            next_state: Batch of next states (normalized float32 [0, 1]). Shape (B, C, H, W).
            action: Batch of actions taken (indices, int64). Shape (B,).
        Returns:
            inv_loss: Inverse model loss (scalar).
            fwd_loss: Forward model loss (scalar).
            pred_phi_next: Predicted next state embedding. Shape (B, embed_dim).
            phi_next: Actual next state embedding (detached). Shape (B, embed_dim).
        """
        # 1. Extract features using the shared CNN
        # Gradients *will* flow back from ICM loss to the feature extractor
        feat = self.feature_extractor(state)
        next_feat = self.feature_extractor(next_state)

        # 2. Flatten features
        feat = feat.view(feat.size(0), -1) # Shape (B, feature_size)
        next_feat = next_feat.view(next_feat.size(0), -1) # Shape (B, feature_size)

        # 3. Encode features into embedding space
        phi = self.encoder(feat) # Shape (B, embed_dim)
        phi_next = self.encoder(next_feat) # Shape (B, embed_dim)

        # --- 4. Inverse Model Calculation ---
        # Concatenate current and next state embeddings
        inv_input = torch.cat([phi, phi_next], dim=1) # Shape (B, embed_dim * 2)
        # Predict action logits
        action_logits = self.inverse_model(inv_input) # Shape (B, num_actions)
        # Calculate inverse loss (CrossEntropy between predicted logits and actual action)
        # Ensure action tensor is of type Long for CrossEntropyLoss
        inv_loss = F.cross_entropy(action_logits, action.long())

        # --- 5. Forward Model Calculation ---
        # One-hot encode the action tensor
        action_onehot = F.one_hot(action.long(), num_classes=self.num_actions).float()
        action_onehot = action_onehot.to(self.device) # Ensure one-hot is on the correct device

        # Concatenate *detached* current state embedding and action
        # We detach phi here because the forward model loss should not update
        # the encoder based on how well phi predicts phi_next.
        # The encoder is updated via the inverse loss and the main DQN loss.
        fwd_input = torch.cat([phi.detach(), action_onehot], dim=1) # Shape (B, embed_dim + num_actions)
        # Predict the next state embedding
        pred_phi_next = self.forward_model(fwd_input) # Shape (B, embed_dim)

        # Calculate forward loss (MSE between predicted and actual *detached* next embedding)
        # We detach phi_next here because the forward model's goal is to predict
        # the embedding produced by the (fixed for this calculation) encoder.
        fwd_loss = F.mse_loss(pred_phi_next, phi_next.detach())

        # Return losses and the necessary embeddings for reward calculation
        # Detach phi_next when returning it for reward calculation, as reward shouldn't backpropagate further
        return inv_loss, fwd_loss, pred_phi_next, phi_next.detach()

    def intrinsic_reward(
        self, pred_phi_next: torch.Tensor, target_phi_next: torch.Tensor, eta: float
    ) -> torch.Tensor:
        """
        Calculates the intrinsic reward based on the forward prediction error.
        Args:
            pred_phi_next: Predicted next state embedding from forward(). Shape (B, embed_dim).
            target_phi_next: Target next state embedding (detached) from forward(). Shape (B, embed_dim).
            eta: Scaling factor for the intrinsic reward.
        Returns:
            intrinsic_reward: Tensor of intrinsic rewards for the batch. Shape (B,).
        """
        # Reward is proportional to the squared Euclidean distance between predicted and target embeddings
        # Ensure inputs are detached (should be already from forward())
        reward = 0.5 * (pred_phi_next.detach() - target_phi_next.detach()).pow(2).sum(dim=1)
        # Scale the reward by eta
        return eta * reward


# === Replay Memory ===

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer.
    Stores experiences and samples them based on calculated priorities (TD errors).
    Handles importance sampling weights.
    """
    def __init__(self, capacity: int, alpha: float, params: Hyperparameters):
        """
        Args:
            capacity: Maximum number of experiences to store.
            alpha: Exponent determining how much prioritization is used (0=uniform, 1=full).
            params: Hyperparameters object (used for per_epsilon).
        """
        self.capacity = capacity
        self.alpha = alpha
        self.params = params # Store params for per_epsilon access
        self.buffer: List[Optional[Experience]] = [None] * capacity # Preallocate buffer list
        self.priorities = np.zeros((capacity,), dtype=np.float32) # Stores priority for each experience
        self.position = 0 # Current index to write to in the buffer
        self.size = 0 # Current number of experiences in the buffer

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        """Adds an experience to the buffer.
        New experiences are given the maximum priority currently in the buffer
        to ensure they are sampled at least once.
        Args:
            state: Current state observation.
            action: Action taken.
            reward: N-step reward received.
            next_state: Next state observation (n steps later).
            done: Whether the episode terminated within N steps.
        """
        # Determine max priority: if buffer is not empty, use current max, else use 1.0
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        # Handle edge case where buffer was filled then emptied, max prio could be 0
        if self.size == 0 and max_prio == 0: max_prio = 1.0

        # Create experience tuple
        experience = Experience(state, action, reward, next_state, done)

        # Store experience and its priority
        self.buffer[self.position] = experience
        self.priorities[self.position] = max_prio

        # Update position and size (handle wrap-around)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, beta: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Samples a batch of experiences based on priorities.
        Args:
            batch_size: Number of experiences to sample.
            beta: Exponent for importance sampling correction (annealed externally).
        Returns:
            A tuple containing:
            - states: Batch of states (np.ndarray, float32).
            - actions: Batch of actions (np.ndarray, int64).
            - rewards: Batch of N-step rewards (np.ndarray, float32).
            - next_states: Batch of next states (np.ndarray, float32).
            - dones: Batch of done flags (np.ndarray, float32).
            - indices: Indices of the sampled experiences in the buffer.
            - weights: Importance sampling weights for the sampled experiences.
        """
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")

        # Get priorities of experiences currently in the buffer
        priorities_segment = self.priorities[: self.size]

        # Calculate sampling probabilities: P(i) = p_i^alpha / sum(p_k^alpha)
        probs = priorities_segment**self.alpha
        probs_sum = probs.sum()

        # Handle potential division by zero or negative sum if all priorities are zero
        if probs_sum <= 1e-8:
            print(f"Warning: Sum of probabilities is {probs_sum}. Using uniform sampling.")
            # Fallback to uniform sampling
            probs = np.ones_like(priorities_segment) / self.size
        else:
            probs /= probs_sum # Normalize probabilities

        # Sample indices based on calculated probabilities
        indices = np.random.choice(self.size, batch_size, p=probs, replace=True)

        # Calculate importance sampling (IS) weights: w_i = (N * P(i))^-beta / max(w_k)
        weights = (self.size * probs[indices]) ** (-beta)
        # Normalize weights by the maximum weight for stability
        weights /= (weights.max() + 1e-8) # Add epsilon to avoid division by zero
        weights = np.array(weights, dtype=np.float32)

        # Retrieve the sampled experiences
        batch = [self.buffer[idx] for idx in indices]

        # Unpack batch into separate numpy arrays with correct types
        states, actions, rewards, next_states, dones = zip(*batch)
        # Observations should already be float32 from NormalizeFrame wrapper
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        # Actions need to be int64 for PyTorch embedding lookups/indexing
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        # Dones should be float32 (0.0 or 1.0) for calculations
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Updates the priorities of sampled experiences.
        Args:
            indices: Indices of the experiences whose priorities need updating.
            priorities: New priorities (typically absolute TD errors) for these experiences.
        """
        if not len(indices) == len(priorities):
            raise ValueError("Indices and priorities must have the same length.")
        for idx, priority in zip(indices, priorities):
            # Validate index is within the current buffer size
            if not (0 <= idx < self.size):
                print(f"Warning: Invalid index {idx} for priority update (buffer size {self.size}). Skipping.")
                continue
            # Update priority: use absolute value and add a small epsilon for stability
            self.priorities[idx] = abs(priority) + self.params.per_epsilon

    def __len__(self) -> int:
        """Return the current number of items in the buffer."""
        return self.size


# === Rainbow Agent with ICM ===

class RainbowAgent:
    """Reinforcement Learning Agent combining Rainbow DQN features and ICM.
    Manages networks, optimizers, replay buffer, and the learning process.
    """

    def __init__(self, state_shape: Tuple, n_actions: int, params: Hyperparameters):
        """
        Args:
            state_shape: Shape of the observation space (e.g., (C, H, W)).
            n_actions: Number of possible actions in the environment.
            params: Hyperparameters object.
        """
        self.params = params
        self.device = params.device
        self.n_actions = n_actions
        self.steps_done = 0 # Counter for optimization steps performed
        print(f"Initializing Agent on device: {self.device}")
        print(f"ICM Enabled: {self.params.use_icm}")

        # --- Shared Feature Extractor ---
        # Create one instance of the CNN feature extractor, used by both DQN and ICM
        # Input channels determined by state_shape (e.g., params.stack_frames)
        self.shared_feature_extractor = FeatureExtractor(state_shape[0]).to(self.device)

        # --- Initialize DQN Networks (using shared features) ---
        print(f"Initializing RainbowDQN with state_shape: {state_shape}")
        # Policy network (learns and selects actions)
        self.policy_net = RainbowDQN(state_shape, n_actions, params, self.shared_feature_extractor).to(self.device)
        # Target network (provides stable targets for learning)
        self.target_net = RainbowDQN(state_shape, n_actions, params, self.shared_feature_extractor).to(self.device)

        # --- Initialize ICM (if enabled) ---
        self.icm = None
        self.icm_optimizer = None
        if self.params.use_icm:
            print("Initializing ICM...")
            # ICM module (calculates intrinsic reward and its own losses)
            self.icm = ICM(state_shape, n_actions, params, self.shared_feature_extractor).to(self.device)
            # Separate optimizer for the ICM module
            self.icm_optimizer = optim.Adam(
                self.icm.parameters(), lr=params.icm_lr, eps=params.icm_adam_eps
            )

        # --- Target Network Initialization & Mode ---
        # Copy initial weights from policy net to target net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # Set target network to evaluation mode (no dropout, batchnorm updates, etc.)
        self.target_net.eval()

        # --- DQN Optimizer ---
        # Optimizer for the policy network's parameters (including shared features via policy net)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=params.learning_rate, eps=params.adam_eps
        )

        # --- Replay Buffer ---
        # Prioritized Replay Buffer stores (s, a, R_n, s_n, D_n) tuples
        self.memory = PrioritizedReplayBuffer(
            params.buffer_size, params.per_alpha, params
        )
        # Temporary buffer to accumulate transitions for N-step return calculation
        self.n_step_accumulator = deque(maxlen=params.n_step)

        # --- Distributional RL Support ---
        # Ensure the support tensor (atoms) is on the correct device
        self.policy_net.support = self.policy_net.support.to(self.device)
        self.target_net.support = self.target_net.support.to(self.device)

        # Optional: TensorBoard Writer for logging
        # self.writer = SummaryWriter(log_dir=params.tensorboard_log_dir)

    def _current_beta(self) -> float:
        """Calculates the current PER beta value based on annealing schedule."""
        # Linearly anneal beta from beta_start to 1.0 over beta_frames steps
        fraction = min(self.steps_done / self.params.per_beta_frames, 1.0)
        beta = self.params.per_beta_start + fraction * (1.0 - self.params.per_beta_start)
        return beta


    def select_action(self, state: np.ndarray) -> int:
        """Selects action using the policy network in evaluation mode.
        Noisy layers provide exploration implicitly during training.
        Args:
            state: Current environment observation (np.ndarray, float32 [0, 1]).
        Returns:
            Selected action index (int).
        """
        with torch.no_grad(): # Disable gradient calculations for inference
            # Ensure state is a float32 numpy array
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            elif state.dtype != np.float32:
                # This shouldn't happen if NormalizeFrame works correctly
                print(f"Warning: state dtype is {state.dtype} in select_action. Converting to float32.")
                state = state.astype(np.float32)

            # Convert state to tensor, add batch dimension, move to device
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)

            # Get expected Q-values from the policy network
            # NoisyLinear layers automatically use mean weights in eval mode
            q_values = self.policy_net.get_q_values(state_tensor)

            # Select action with the highest Q-value (argmax)
            action = q_values.argmax(1).item()
        return action

    def store_transition(
        self, state: Any, action: int, reward: float, next_state: Any, done: bool
    ):
        """Stores a single step transition in the N-step accumulator.
        Once the accumulator is full, calculates the N-step return and
        pushes the relevant (s_t, a_t, R_n, s_{t+N}, D_n) tuple to the PER buffer.
        Args:
            state: State observed at time t.
            action: Action taken at time t.
            reward: Shaped extrinsic reward received for taking action a_t.
            next_state: State observed at time t+1.
            done: Whether the episode terminated at time t+1.
        """
        # Note: The 'reward' here is the shaped *extrinsic* reward from the wrapper
        # Store the 1-step transition (s_t, a_t, r_{t+1}, s_{t+1}, d_{t+1})
        self.n_step_accumulator.append(
            Experience(state, action, reward, next_state, done)
        )

        # If accumulator doesn't have enough steps for N-step calculation, return
        if len(self.n_step_accumulator) < self.params.n_step:
            return

        # --- Calculate N-step return ---
        n_step_reward = 0.0
        discount = 1.0
        # The experience we are calculating the N-step return *for* is the oldest one
        start_exp = self.n_step_accumulator[0] # (s_t, a_t, r_{t+1}, s_{t+1}, d_{t+1})
        # The final next state (s_{t+N}) is from the *most recent* experience in the buffer
        final_next_state = self.n_step_accumulator[-1].next_state
        n_step_done = False # Flag indicating if termination occurred within the N steps

        # Iterate through the N steps in the accumulator
        for i in range(self.params.n_step):
            exp = self.n_step_accumulator[i] # (s_{t+i}, a_{t+i}, r_{t+i+1}, s_{t+i+1}, d_{t+i+1})
            # Accumulate discounted reward: R_n = r_{t+1} + gamma*r_{t+2} + ... + gamma^(n-1)*r_{t+n}
            n_step_reward += discount * exp.reward
            # Update discount factor for the next step
            discount *= self.params.gamma # Use extrinsic gamma for N-step calculation

            # Check if this step terminated the episode
            if exp.done:
                # If terminated at step i+1, the effective next state is s_{t+i+1}
                # and the N-step transition is considered "done"
                final_next_state = exp.next_state
                n_step_done = True
                break # Stop accumulating reward past the terminal state

        # --- Push the N-step experience to PER buffer ---
        # We push: (state at time t, action at time t, N-step reward starting from t+1,
        #          state at time t+N (or terminal state if earlier), done flag for N-step)
        self.memory.push(
            start_exp.state, start_exp.action, n_step_reward, final_next_state, n_step_done
        )
        # The accumulator automatically discards the oldest element due to maxlen

    def optimize_model(self) -> Tuple[float, float, float]:
        """Performs one optimization step for both DQN and ICM networks.
        1. Samples a batch from PER.
        2. Calculates ICM loss and intrinsic reward (if ICM enabled).
        3. Optimizes ICM network.
        4. Calculates DQN target distribution using combined reward.
        5. Calculates DQN loss.
        6. Optimizes DQN network (including shared feature extractor).
        7. Updates priorities in PER buffer.
        Returns:
            Tuple[float, float, float]: (DQN loss, ICM loss, Avg Intrinsic Reward in batch)
        """
        # Don't try to optimize if the buffer doesn't have enough samples
        if len(self.memory) < self.params.batch_size:
            return 0.0, 0.0, 0.0 # Return zero losses

        # --- 1. Sample Batch from PER ---
        beta = self._current_beta() # Get current importance sampling exponent
        states, actions, n_step_rewards_ext, next_states, n_step_dones, indices, weights = (
            self.memory.sample(self.params.batch_size, beta)
        )

        # Move sampled data to PyTorch tensors on the correct device
        states = torch.from_numpy(states).to(self.device) # Shape (B, C, H, W)
        actions = torch.from_numpy(actions).to(self.device) # Shape (B,), int64
        n_step_rewards_ext = torch.from_numpy(n_step_rewards_ext).to(self.device) # Shape (B,)
        next_states = torch.from_numpy(next_states).to(self.device) # Shape (B, C, H, W)
        n_step_dones = torch.from_numpy(n_step_dones).to(self.device) # Shape (B,), float32 (0.0 or 1.0)
        weights = torch.from_numpy(weights).to(self.device) # Shape (B,)

        # --- 2. ICM Computations (if enabled) ---
        icm_total_loss_val = 0.0
        intrinsic_reward_val = 0.0
        # Initialize intrinsic reward tensor (defaults to zeros if ICM is off)
        intrinsic_reward = torch.zeros_like(n_step_rewards_ext)

        if self.params.use_icm and self.icm is not None:
            # Set ICM to training mode
            self.icm.train()
            # Calculate ICM losses (inv_loss, fwd_loss) and embeddings (pred_phi_next, phi_next)
            inv_loss, fwd_loss, pred_phi_next, phi_next = self.icm(
                states, next_states, actions
            )

            # Calculate total weighted ICM loss
            icm_total_loss = (1 - self.params.icm_beta) * inv_loss + self.params.icm_beta * fwd_loss
            icm_total_loss_val = icm_total_loss.item() # Store scalar value for logging

            # Calculate intrinsic reward based on forward model prediction error
            intrinsic_reward = self.icm.intrinsic_reward(
                pred_phi_next, phi_next, self.params.icm_eta
            )
            intrinsic_reward_val = intrinsic_reward.mean().item() # Log average intrinsic reward

            # --- 3. Optimize ICM Network ---
            self.icm_optimizer.zero_grad() # Zero gradients for ICM optimizer
            # Calculate gradients for ICM loss (flows through ICM models and shared CNN)
            icm_total_loss.backward()
            # Optional: Clip gradients for ICM parameters if needed
            # torch.nn.utils.clip_grad_norm_(self.icm.parameters(), self.params.gradient_clip_norm)
            self.icm_optimizer.step() # Update ICM parameters

        # --- 4. DQN Computations: Target Distribution ---
        # Set policy/target nets to appropriate modes
        self.policy_net.train() # Policy net needs noise enabled
        self.target_net.eval() # Target net uses mean weights

        # Combine extrinsic N-step reward with intrinsic reward
        # Detach intrinsic reward so DQN loss doesn't optimize ICM parameters directly
        total_reward = n_step_rewards_ext + intrinsic_reward.detach()

        with torch.no_grad(): # Target calculations don't require gradients
            # --- Double DQN: Select best actions for next states using POLICY network ---
            next_q_values_policy = self.policy_net.get_q_values(next_states) # Shape (B, num_actions)
            next_actions = next_q_values_policy.argmax(1) # Shape (B,)

            # --- Get next state distributions from TARGET network ---
            next_q_dist_target = self.target_net(next_states) # Shape (B, num_actions, num_atoms)

            # Select the distribution corresponding to the best action chosen by policy net
            # Gather along action dimension (dim=1)
            next_best_q_dist = next_q_dist_target.gather(
                1, next_actions.view(-1, 1, 1).expand(-1, -1, self.params.num_atoms)
            ).squeeze(1) # Shape: (batch_size, num_atoms)

            # --- Project the target distribution (Categorical/C51 algorithm) ---
            support = self.policy_net.support # Get support atoms (z_i) Shape (num_atoms,)
            # Calculate projected atom locations: Tz = R_n + gamma^n * z
            # Unsqueeze rewards and dones for broadcasting with support
            Tz = total_reward.unsqueeze(1) + (
                1 - n_step_dones.unsqueeze(1) # Use N-step dones here
            ) * self.params.n_step_gamma * support.unsqueeze(0) # Shape (B, num_atoms)

            # Clamp projected atoms to the predefined support range [V_min, V_max]
            Tz = Tz.clamp(self.params.v_min, self.params.v_max)

            # Calculate indices (l, u) and weights for projection onto the original support
            b = (Tz - self.params.v_min) / self.params.delta_z # Normalize Tz to indices
            l = b.floor().long() # Lower bound index
            u = b.ceil().long() # Upper bound index

            # Distribute probability mass (from next_best_q_dist) to l and u bins
            # mass_l = p(a*, z_j) * (u - b)
            # mass_u = p(a*, z_j) * (b - l)
            mass_l = next_best_q_dist * (u.float() - b)
            mass_u = next_best_q_dist * (b - l.float())

            # Initialize target distribution tensor with zeros
            target_dist = torch.zeros_like(next_best_q_dist) # Shape (B, num_atoms)

            # Scatter-add the probability masses to the corresponding indices (l, u)
            # Need to ensure l and u are valid indices [0, num_atoms - 1] after floor/ceil
            l = l.clamp(0, self.params.num_atoms - 1)
            u = u.clamp(0, self.params.num_atoms - 1)
            target_dist.scatter_add_(1, l, mass_l)
            target_dist.scatter_add_(1, u, mass_u)
            # Target_dist now holds the projected target distribution for each state in the batch

        # --- 5. Compute DQN Loss ---
        # Get current state action distributions from POLICY network
        current_q_dist_policy = self.policy_net(states) # Shape (B, num_actions, num_atoms)

        # Gather the distributions for the actions actually taken in the batch
        action_indices = (
            actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.params.num_atoms)
        ) # Shape: (B, 1, num_atoms)
        current_dist = current_q_dist_policy.gather(1, action_indices).squeeze(1) # Shape: (B, num_atoms)

        # Avoid log(0) errors by clamping probabilities to a small positive value
        current_dist = current_dist.clamp(min=1e-8)
        log_p = torch.log(current_dist) # Log probabilities of current distribution

        # Calculate element-wise loss: Cross-entropy between target and current distributions
        # L = - sum_j (target_dist_j * log(current_dist_j))
        elementwise_loss = -(target_dist * log_p).sum(1) # Shape (B,) - These are TD errors for PER

        # --- 6. Optimize DQN Network ---
        # Apply importance sampling weights (from PER) and calculate mean loss
        dqn_loss = (elementwise_loss * weights).mean() # Scalar loss value
        dqn_loss_val = dqn_loss.item() # Store scalar value for logging

        self.optimizer.zero_grad() # Zero gradients for DQN optimizer
        # Calculate gradients for DQN loss (flows through policy net and shared CNN)
        dqn_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.params.gradient_clip_norm
        )
        self.optimizer.step() # Update policy network parameters

        # --- 7. Update Priorities in PER Buffer ---
        # Use the detached element-wise losses (TD errors) to update priorities
        new_priorities = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)

        # --- Agent Step Counter & Target Network Update ---
        self.steps_done += 1 # Increment optimization step counter
        # Periodically update the target network weights
        if self.steps_done % self.params.target_update_freq == 0:
            self.update_target_network()

        # --- Reset Noise for Noisy Layers ---
        # Sample new noise for the next action selection/optimization step
        self.reset_noise()

        # Optional: Log losses and rewards to TensorBoard
        # if hasattr(self, 'writer') and self.writer:
        #     self.writer.add_scalar('Loss/DQN', dqn_loss_val, self.steps_done)
        #     if self.params.use_icm:
        #         self.writer.add_scalar('Loss/ICM', icm_total_loss_val, self.steps_done)
        #         self.writer.add_scalar('Reward/Intrinsic_Avg_Batch', intrinsic_reward_val, self.steps_done)
        #     self.writer.add_scalar('Parameters/beta', beta, self.steps_done)

        return dqn_loss_val, icm_total_loss_val, intrinsic_reward_val

    def update_target_network(self):
        """Copies weights from the policy network to the target network."""
        print(f"Updating target network at step {self.steps_done}")
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_noise(self):
        """Resets noise in the NoisyLinear layers of the policy network."""
        # Target network noise doesn't need resetting as it runs in eval mode
        self.policy_net.reset_noise()

    def save_model(self, filepath: str):
        """Saves the agent's state to a file.
        Includes network state dicts, optimizer states, and current step count.
        Args:
            filepath: Path to save the checkpoint file.
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Saving model checkpoint to {filepath}...")
        save_data = {
            # DQN components
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
            # ICM components (save only if ICM is enabled and initialized)
             "icm_state_dict": self.icm.state_dict() if self.params.use_icm and self.icm else None,
             "icm_optimizer_state_dict": self.icm_optimizer.state_dict() if self.params.use_icm and self.icm_optimizer else None,
        }
        try:
            torch.save(save_data, filepath)
            print("Model saved.")
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, filepath: str) -> bool:
        """Loads the agent's state from a checkpoint file.
        Args:
            filepath: Path to the checkpoint file.
        Returns:
            True if loading was successful, False otherwise.
        """
        if not os.path.isfile(filepath):
            print(f"Checkpoint file not found at {filepath}. Starting from scratch.")
            return False
        print(f"Loading model checkpoint from {filepath}...")
        try:
            # Load checkpoint data onto the agent's current device
            checkpoint = torch.load(filepath, map_location=self.device)

            # Load DQN components
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.steps_done = checkpoint["steps_done"]

            # Load ICM components if ICM is enabled and data exists in checkpoint
            if self.params.use_icm:
                if self.icm and "icm_state_dict" in checkpoint and checkpoint["icm_state_dict"]:
                     self.icm.load_state_dict(checkpoint["icm_state_dict"])
                     print("ICM state loaded.")
                else:
                     print("Warning: ICM enabled but no ICM state found in checkpoint or ICM not initialized.")
                if self.icm_optimizer and "icm_optimizer_state_dict" in checkpoint and checkpoint["icm_optimizer_state_dict"]:
                     self.icm_optimizer.load_state_dict(checkpoint["icm_optimizer_state_dict"])
                     print("ICM optimizer state loaded.")
                else:
                     print("Warning: ICM enabled but no ICM optimizer state found in checkpoint or optimizer not initialized.")

            # Ensure networks are on the correct device and in correct modes after loading
            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            if self.icm: self.icm.to(self.device)

            self.target_net.eval() # Target net always in eval mode
            self.policy_net.train() # Policy net in train mode (for noisy layers)
            if self.icm: self.icm.train() # ICM should be in train mode for updates

            print(f"Model loaded successfully. Resuming from step {self.steps_done}.")
            # Reset noise after loading to start with fresh noise samples
            self.reset_noise()
            return True
        except KeyError as e:
            print(f"Error loading checkpoint: Missing key {e}. Checkpoint might be incompatible or missing required data (e.g., ICM).")
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False


# === Environment Creation Function ===

def make_env(params: Hyperparameters) -> gym.Env:
    """Creates and wraps the Super Mario Bros environment with specified preprocessing.
    Args:
        params: Hyperparameters object containing environment settings.
    Returns:
        Wrapped gym environment ready for the agent.
    """
    # Create the base Super Mario Bros environment
    env = gym_super_mario_bros.make(params.env_id)

    # Apply wrappers in a specific order for correct preprocessing:
    # 1. Action space wrapper (maps agent actions to joystick commands)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    # 2. Frame skipping (repeats action, sums reward)
    env = SkipFrame(env, skip=params.skip_frames)
    # 3. Custom extrinsic reward shaping
    env = RewardShapingWrapper(env, params)
    # 4. Resize observation to specified dimensions (e.g., 84x84)
    env = ResizeObservation(env, shape=params.processed_resize_dim)
    # 5. Convert observation to grayscale (!!! keep_dim=False !!!)
    # This ensures the output shape is (H, W), not (H, W, 1)
    env = GrayScaleObservation(env, keep_dim=False)
    # 6. Stack consecutive frames along a new first dimension (channels)
    # Input shape to FrameStack is (H, W), output shape is (num_stack, H, W)
    env = FrameStack(env, num_stack=params.stack_frames, lz4_compress=True) # Use compression
    # 7. Normalize pixel values to [0, 1] float32
    env = NormalizeFrame(env)
    # 8. Set a maximum number of steps per episode
    env = TimeLimit(env, max_episode_steps=params.max_episode_steps)

    return env


# === Main Training Script ===

if __name__ == "__main__":
    # --- Initialize Hyperparameters ---
    params = Hyperparameters()

    # --- Print Configuration ---
    print("--- Hyperparameters ---")
    param_dict = vars(params)
    # Exclude derived parameters from the main printout
    derived_keys = ["n_step_gamma", "delta_z", "device", "processed_resize_dim"]
    for key, value in param_dict.items():
        if key not in derived_keys:
            print(f"{key}: {value}")
    # Print derived parameters separately
    print(f"processed_resize_dim: {params.processed_resize_dim}")
    print(f"device: {params.device}")
    print("---------------------")

    # --- Environment Setup ---
    env = make_env(params)
    # Get state shape and action count from the wrapped environment
    # Shape should now be (stack_frames, H, W), e.g., (4, 84, 84)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    print(f"\n--- Environment Info ---")
    print(f"State shape (after wrappers): {state_shape}") # Should be (4, 84, 84)
    print(f"Observation space dtype: {env.observation_space.dtype}") # Should be float32
    print(f"Number of actions: {n_actions}")
    print(f"Running in {params.life_mode}-Life Mode")
    print(f"ICM Enabled: {params.use_icm}")
    print("------------------------\n")

    # --- Agent Initialization ---
    # This should now work as state_shape is (4, 84, 84)
    agent = RainbowAgent(state_shape=state_shape, n_actions=n_actions, params=params)

    # --- Load Checkpoint if specified ---
    latest_checkpoint_path = os.path.join(
        params.checkpoint_dir, "rainbow_mario_icm_latest.pth" # Consistent checkpoint name
    )
    if params.load_checkpoint:
        agent.load_model(latest_checkpoint_path)

    # --- Training Loop Initialization ---
    # Deques for storing recent episode rewards for logging averages
    episode_rewards_shaped = deque(maxlen=100) # Extrinsic shaped reward
    episode_rewards_unshaped = deque(maxlen=100) # Extrinsic original reward (from SkipFrame info)
    episode_rewards_intrinsic = deque(maxlen=100) # Average intrinsic reward per episode

    total_agent_steps = agent.steps_done # Start step count from loaded value
    episode_count = 0 # Consider loading episode count from checkpoint if needed for accurate logging
    global_start_time = time.time() # Track total training time
    last_log_time = global_start_time # For calculating FPS
    last_log_step = total_agent_steps # For calculating FPS

    # --- Main Training Loop ---
    try:
        while total_agent_steps < params.total_train_steps:
            episode_count += 1
            # Reset environment, get initial state (normalized, stacked frames)
            state = env.reset() # Returns np.ndarray float32, shape (4, 84, 84)

            # Reset episode statistics
            current_episode_reward_shaped = 0
            current_episode_reward_unshaped = 0
            current_episode_intrinsic_reward_sum = 0 # Sum of batch averages
            current_episode_steps = 0
            current_episode_dqn_losses = []
            current_episode_icm_losses = []
            max_x_pos = 0 # Track furthest progress in the level
            episode_status = "RUNNING" # Track how the episode ended

            # --- Episode Execution Loop ---
            while True:
                # 1. Agent selects action based on current state
                action = agent.select_action(state)

                # 2. Environment steps based on selected action
                # Returns: next observation, shaped extrinsic reward, done flag, info dict
                next_state, shaped_extrinsic_reward, done, info = env.step(action)

                # 3. Retrieve original (unshaped) extrinsic reward sum from info dict
                original_reward = info.get("original_reward_sum", shaped_extrinsic_reward)

                # 4. Determine if episode effectively ended
                # Considers 'done' flag and life loss in 1-life mode
                effective_done = info.get("effective_done", done)
                life_lost = info.get("life_lost", False)
                current_x_pos = info.get("x_pos", 0) # Get current x-position

                # 5. Store transition in N-step buffer (uses shaped extrinsic reward)
                # The accumulator handles N-step calculation and pushing to PER buffer later
                agent.store_transition(state, action, shaped_extrinsic_reward, next_state, done)

                # 6. Update current state for the next iteration
                state = next_state

                # 7. Accumulate episode statistics
                current_episode_reward_shaped += shaped_extrinsic_reward
                current_episode_reward_unshaped += original_reward
                current_episode_steps += 1
                max_x_pos = max(max_x_pos, current_x_pos)

                # 8. Optimize model (DQN and ICM) if buffer is ready
                dqn_loss, icm_loss, intrinsic_reward_batch_avg = 0.0, 0.0, 0.0
                # Only optimize if PER buffer has enough samples for a batch
                if len(agent.memory) >= params.batch_size:
                    # Perform one optimization step
                    dqn_loss, icm_loss, intrinsic_reward_batch_avg = agent.optimize_model()

                    # Store losses if they are valid (non-zero)
                    if dqn_loss > 0:
                        current_episode_dqn_losses.append(dqn_loss)
                    if icm_loss > 0 and params.use_icm:
                        current_episode_icm_losses.append(icm_loss)
                    # Accumulate the average intrinsic reward from the batch
                    current_episode_intrinsic_reward_sum += intrinsic_reward_batch_avg

                    # Update global step count (optimization steps)
                    total_agent_steps = agent.steps_done

                    # --- 9. Periodic Logging ---
                    if (
                        total_agent_steps % params.log_interval_steps == 0
                        and total_agent_steps > last_log_step # Avoid logging at step 0 multiple times
                    ):
                        # Calculate average rewards over the last 100 episodes
                        avg_reward_shaped_100 = np.mean(episode_rewards_shaped) if episode_rewards_shaped else 0.0
                        avg_reward_unshaped_100 = np.mean(episode_rewards_unshaped) if episode_rewards_unshaped else 0.0
                        avg_reward_intrinsic_100 = np.mean(episode_rewards_intrinsic) if episode_rewards_intrinsic and params.use_icm else 0.0

                        # Calculate performance metrics for the interval
                        current_time = time.time()
                        elapsed_interval = current_time - last_log_time
                        steps_in_interval = total_agent_steps - last_log_step
                        fps = steps_in_interval / elapsed_interval if elapsed_interval > 0 else 0

                        # Print progress summary
                        print(f"\n--- Progress ---")
                        print(f"Steps: {total_agent_steps}/{params.total_train_steps} ({total_agent_steps/params.total_train_steps*100:.2f}%)")
                        print(f"Episodes: {episode_count}")
                        print(f"Avg Shaped Reward (Last 100 ep): {avg_reward_shaped_100:.2f}")
                        print(f"Avg Unshaped Reward (Last 100 ep): {avg_reward_unshaped_100:.2f}")
                        if params.use_icm:
                           print(f"Avg Intrinsic Reward (Last 100 ep): {avg_reward_intrinsic_100:.4f}")
                        print(f"Last DQN Loss: {dqn_loss:.4f}" if dqn_loss > 0 else "N/A")
                        if params.use_icm:
                            print(f"Last ICM Loss: {icm_loss:.4f}" if icm_loss > 0 else "N/A")
                        print(f"Current Beta (PER): {agent._current_beta():.4f}")
                        print(f"FPS (Agent Steps): {fps:.2f}")
                        print(f"Buffer Size: {len(agent.memory)}/{params.buffer_size}")
                        elapsed_total = datetime.timedelta(seconds=int(current_time - global_start_time))
                        print(f"Elapsed Time: {elapsed_total}")
                        print(f"----------------")

                        # Update time/step trackers for next log interval
                        last_log_time = current_time
                        last_log_step = total_agent_steps

                    # --- 10. Periodic Checkpoint Saving ---
                    if (
                        total_agent_steps % params.save_interval_steps == 0
                        and total_agent_steps > 0
                        and total_agent_steps > last_log_step # Avoid saving multiple times if log/save intervals match
                    ):
                        # Define path for step-specific checkpoint
                        save_path = os.path.join(
                            params.checkpoint_dir,
                            f"rainbow_mario_icm_step_{total_agent_steps}.pth",
                        )
                        agent.save_model(save_path)
                        # Also save as the latest checkpoint (overwrites previous latest)
                        agent.save_model(latest_checkpoint_path)

                # --- 11. Check Episode End Condition ---
                # Use effective_done which incorporates TimeLimit and life loss mode
                if effective_done:
                    # Determine the reason for episode termination
                    if info.get("TimeLimit.truncated", False):
                        episode_status = "TIMELIMIT"
                    elif life_lost and params.life_mode == 1:
                        episode_status = "LIFE_LOST (1-Life Mode)"
                    elif done: # Actual game over (0 lives) or flagpole reached
                        episode_status = "GAME_DONE"
                    else: # Should not happen if effective_done is True
                        episode_status = "UNKNOWN"
                    break # Exit the inner episode loop

            # --- End of Episode Processing ---
            # Store final episode rewards in deques for averaging
            episode_rewards_shaped.append(current_episode_reward_shaped)
            episode_rewards_unshaped.append(current_episode_reward_unshaped)
            # Calculate average intrinsic reward for this episode
            avg_intrinsic_episode = (
                current_episode_intrinsic_reward_sum / current_episode_steps
                if current_episode_steps > 0 and params.use_icm else 0.0
            )
            episode_rewards_intrinsic.append(avg_intrinsic_episode)

            # Calculate average losses for the episode
            avg_dqn_loss_episode = np.mean(current_episode_dqn_losses) if current_episode_dqn_losses else 0.0
            avg_icm_loss_episode = np.mean(current_episode_icm_losses) if current_episode_icm_losses and params.use_icm else 0.0

            # Print episode summary if enabled
            if params.print_episode_summary:
                print(f"Episode {episode_count} finished after {current_episode_steps} steps. Status: {episode_status}")
                print(f"  Total Shaped Reward: {current_episode_reward_shaped:.2f}")
                print(f"  Total Unshaped Reward: {current_episode_reward_unshaped:.2f}")
                if params.use_icm:
                    print(f"  Avg Intrinsic Reward: {avg_intrinsic_episode:.4f}")
                print(f"  Max X Position: {max_x_pos}")
                print(f"  Avg DQN Loss: {avg_dqn_loss_episode:.4f}")
                if params.use_icm:
                    print(f"  Avg ICM Loss: {avg_icm_loss_episode:.4f}")
                print(f"  Agent Steps: {total_agent_steps}")

            # --- Check if Training Goal Reached ---
            if total_agent_steps >= params.total_train_steps:
                print("\nTarget number of training steps reached.")
                break # Exit the outer training loop

    # --- Handle Exceptions and Cleanup ---
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging

    finally:
        # --- Save Final Model ---
        # Always save the model state upon exit, regardless of reason
        print("\nSaving final model...")
        final_save_path = os.path.join(
            params.checkpoint_dir, "rainbow_mario_icm_final.pth"
        )
        agent.save_model(final_save_path)
        agent.save_model(latest_checkpoint_path) # Save latest one last time

        # --- Clean Up ---
        print(f"\nTraining finished.")
        print(f"Total episodes: {episode_count}")
        print(f"Total agent steps: {total_agent_steps}")
        env.close() # Close the environment
        # Optional: Close TensorBoard writer
        # if hasattr(agent, 'writer') and agent.writer: agent.writer.close()
        print("Environment closed.")
