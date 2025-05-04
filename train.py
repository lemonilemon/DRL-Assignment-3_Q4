# -*- coding: utf-8 -*-
"""
Improved Rainbow DQN Agent for Super Mario Bros.

Combines:
- Dueling DQN
- Noisy Nets for exploration
- Prioritized Experience Replay (PER)
- N-step Bootstrapping
- Distributional RL (C51)
- Reward Shaping Wrapper (Refined reset logic)
- Frame Normalization Wrapper (to float32 [0,1])
- Centralized Hyperparameters
- Observation resized to (84, 84)
- Logs both shaped and unshaped rewards
- Fixed device mismatch error during network initialization.
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
import datetime  # Added for elapsed time formatting

# Optional: For TensorBoard logging
# from torch.utils.tensorboard import SummaryWriter


# === Configuration ===
@dataclass
class Hyperparameters:
    """Centralized hyperparameters for training."""

    # --- Environment ---
    env_id: str = "SuperMarioBros-v0"
    # Life mode (1 means terminate episode on first life lost)
    life_mode: int = 3
    # Default starting lives in Super Mario Bros
    starting_lives: int = 3
    skip_frames: int = 4
    stack_frames: int = 4
    # Resize dimension (Height, Width) - Changed back to 84x84
    resize_dim: Union[int, Tuple[int, int]] = (
        84  # Can be int for square or Tuple for non-square
    )
    # Max steps per episode (via TimeLimit wrapper)
    max_episode_steps: int = 4500

    # --- Reward Shaping ---
    death_penalty: float = -150.0
    move_reward: float = 0  # Small reward for moving right
    stuck_penalty: float = 0  # Penalty for staying in the same x-pos
    step_penalty: float = 0  # Small penalty per step to encourage progress

    # --- Training ---
    total_train_steps: int = 5_000_000
    batch_size: int = 32
    learning_rate: float = 0.0001  # Adam learning rate
    adam_eps: float = 1.5e-4  # Adam epsilon
    gamma: float = 0.8  # Discount factor for Bellman equation
    target_update_freq: int = 10000  # Steps between target network updates
    gradient_clip_norm: float = 10.0  # Clip gradients to this norm

    # --- Replay Buffer (PER) ---
    buffer_size: int = 10000
    per_alpha: float = 0.5  # Priority exponent
    per_beta_start: float = 0.4  # Initial importance sampling exponent
    per_beta_frames: int = 1_000_000  # Steps to anneal beta to 1.0
    per_epsilon: float = 1e-5  # Small value added to priorities

    # --- N-Step Returns ---
    n_step: int = 5  # Number of steps for N-step returns

    # --- Distributional RL (C51) ---
    num_atoms: int = 51  # Number of atoms in value distribution
    v_min: float = -150.0  # Minimum value for distribution support
    v_max: float = 50.0  # Maximum value for distribution support

    # --- Noisy Nets ---
    noisy_std_init: float = 2.5  # Initial standard deviation for NoisyLinear layers

    # --- Logging & Saving ---
    log_interval_steps: int = 10000  # Log progress every N agent steps
    save_interval_steps: int = 100000  # Save checkpoint every N agent steps
    print_episode_summary: bool = True
    # Updated checkpoint directory name to reflect 84x84
    checkpoint_dir: str = "mario_rainbow_checkpoints_v3"
    load_checkpoint: bool = True  # Set to True to load latest checkpoint if exists
    # Optional: Path for TensorBoard logs
    # tensorboard_log_dir: str = "logs/mario_rainbow_v2"

    # --- Derived / Calculated ---
    # Calculated discount factor for n-step returns
    n_step_gamma: float = field(init=False)
    # Calculated delta_z for distributional RL
    delta_z: float = field(init=False)
    # Device for PyTorch
    device: torch.device = field(init=False)
    # Process resize_dim into a tuple if it's an int
    processed_resize_dim: Tuple[int, int] = field(init=False)

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
        self._steps_in_skip = 0  # Track steps within the current skip cycle

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Repeat action, sum reward, and update info."""
        total_reward = 0.0
        total_original_reward = 0.0  # Track original reward sum
        done = False
        combined_info = {}
        self._steps_in_skip = 0

        for i in range(self._skip):
            self._steps_in_skip += 1
            obs, reward, step_done, step_info = self.env.step(action)
            # Update combined_info, prioritizing info from later steps
            combined_info.update(step_info)
            total_reward += reward
            total_original_reward += reward  # Sum original reward
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
    """

    def __init__(self, env: gym.Env, params: Hyperparameters):
        super().__init__(env)
        self.params = params
        self.prev_x_pos = 0
        # Initialize prev_life based on known starting lives
        self.prev_life = self.params.starting_lives
        self.current_episode_start_lives = self.params.starting_lives

    def reset(self, **kwargs) -> Any:
        """Reset state and internal reward shaping variables."""
        obs = self.env.reset(**kwargs)
        # Reset internal state for the new episode
        self.prev_x_pos = 0  # Assume starting x_pos is 0
        # Reset life counts based on known starting lives
        self.current_episode_start_lives = self.params.starting_lives
        self.prev_life = self.params.starting_lives
        return obs

    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """Steps the environment and applies reward shaping."""
        # The reward received here might already be summed by SkipFrame
        obs, reward, done, info = self.env.step(action)

        # --- Apply Reward Shaping ---
        # Start with the potentially summed reward from SkipFrame
        custom_reward = reward

        # Safely get current values from info
        current_x_pos = info.get("x_pos", self.prev_x_pos)
        current_lives = info.get("life", self.prev_life)

        # 1. Penalty for losing a life
        life_lost = current_lives < self.prev_life
        if life_lost:
            custom_reward += self.params.death_penalty

        # 2. Reward for moving right (scaled by distance moved)
        x_pos_diff = current_x_pos - self.prev_x_pos
        if x_pos_diff > 0:
            custom_reward += self.params.move_reward * x_pos_diff

        # 3. Penalty for getting stuck (optional)
        elif x_pos_diff == 0 and not done:
            custom_reward += self.params.stuck_penalty

        # 4. Small penalty per step (optional, encourages efficiency)
        # Use the number of steps actually taken in the skip cycle from info
        num_steps_taken = info.get("_steps_in_skip", self.params.skip_frames)
        custom_reward += self.params.step_penalty * num_steps_taken

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

        # Return the original 'done' flag and the SHAPED reward
        return obs, custom_reward, done, info


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
        if isinstance(obs, gym.wrappers.frame_stack.LazyFrames):
            obs_array = np.array(obs, dtype=np.uint8)
        elif isinstance(obs, np.ndarray):
            obs_array = obs
        else:
            obs_array = np.array(obs)

        if obs_array.dtype == np.uint8:
            normalized_obs = obs_array.astype(np.float32) / 255.0
        elif obs_array.dtype == np.float32:
            normalized_obs = obs_array
        else:
            print(
                f"Warning: Unexpected observation dtype {obs_array.dtype}. Attempting normalization."
            )
            normalized_obs = obs_array.astype(np.float32) / 255.0
        return normalized_obs


# === Neural Network Components ===


class NoisyLinear(nn.Module):
    """Noisy Linear Layer for exploration."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        # Use the device of the parameters, which might be CPU during init
        # but will be correct device after model.to(device)
        device = self.weight_mu.device
        x = torch.randn(size, device=device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    """Dueling Network with Noisy Layers and Distributional RL."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_actions: int,
        params: Hyperparameters,
    ):
        super(RainbowDQN, self).__init__()
        if len(input_shape) != 3:
            raise ValueError(
                f"Expected input_shape with 3 dimensions (C, H, W), got {input_shape}"
            )
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = params.num_atoms
        self.v_min = params.v_min
        self.v_max = params.v_max
        # Store device from params, but layers are initialized on CPU first
        self.device = params.device
        self.register_buffer(
            "support", torch.linspace(self.v_min, self.v_max, self.num_atoms)
        )  # Init support on CPU

        # Feature extraction layers (Convolutional) - Init on CPU
        self.features = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Calculate feature size using CPU computation
        feature_size = self._get_conv_output(self.input_shape)

        # Dueling Streams with Noisy Layers - Init on CPU
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_size, 512, std_init=params.noisy_std_init),
            nn.ReLU(),
            NoisyLinear(
                512, num_actions * self.num_atoms, std_init=params.noisy_std_init
            ),
        )
        self.value_stream = nn.Sequential(
            NoisyLinear(feature_size, 512, std_init=params.noisy_std_init),
            nn.ReLU(),
            NoisyLinear(512, self.num_atoms, std_init=params.noisy_std_init),
        )
        # Move support buffer to the target device after initialization
        self.support = self.support.to(self.device)

    def _get_conv_output(self, shape: Tuple[int, int, int]) -> int:
        """Calculate flattened feature size after convolutions by running on CPU."""
        with torch.no_grad():
            # Create a dummy input tensor explicitly on CPU
            dummy_input = torch.zeros(1, *shape, device="cpu")  # Use CPU here
            # Create a temporary features model on CPU to calculate size
            features_cpu = nn.Sequential(
                nn.Conv2d(shape[0], 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            ).to("cpu")  # Ensure it's on CPU
            o = features_cpu(dummy_input)
            feature_size = int(np.prod(o.size()))
            # print(f"Calculated feature size: {feature_size} from output shape {o.shape}") # Debug print
        return feature_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        # Ensure support is on the same device as input (should be handled by model.to(device))
        if self.support.device != x.device:
            self.support = self.support.to(x.device)

        x = self.features(x)
        x = x.view(batch_size, -1)
        value_dist = self.value_stream(x).view(batch_size, 1, self.num_atoms)
        advantage_dist = self.advantage_stream(x).view(
            batch_size, self.num_actions, self.num_atoms
        )
        q_dist = value_dist + (
            advantage_dist - advantage_dist.mean(dim=1, keepdim=True)
        )
        q_probs = F.softmax(q_dist, dim=-1)
        return q_probs

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        q_probs = self.forward(x)
        # Ensure support is on the correct device before multiplication
        support = self.support.to(x.device)
        q_values = (q_probs * support.view(1, 1, self.num_atoms)).sum(dim=2)
        return q_values

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# === Replay Memory and N-Step Buffer ===


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay (PER) buffer."""

    def __init__(self, capacity: int, alpha: float, params: Hyperparameters):
        self.capacity = capacity
        self.alpha = alpha
        self.params = params
        self.buffer: List[Optional[Experience]] = [None] * capacity
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool):
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        experience = Experience(state, action, reward, next_state, done)
        self.buffer[self.position] = experience
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self, batch_size: int, beta: float
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty buffer")
        priorities_segment = self.priorities[: self.size]
        probs = priorities_segment**self.alpha
        probs_sum = probs.sum()
        if probs_sum <= 0:
            print(
                f"Warning: Sum of probabilities is {probs_sum}. Using uniform sampling."
            )
            if self.size > 0:
                probs = np.ones_like(priorities_segment) / self.size
            else:
                raise ValueError(
                    "Cannot sample, buffer size is zero and sum of probabilities is non-positive."
                )
        else:
            probs /= probs_sum
        indices = np.random.choice(self.size, batch_size, p=probs, replace=True)
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max() + 1e-8
        weights = np.array(weights, dtype=np.float32)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        return (
            states,
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            next_states,
            np.array(dones, dtype=np.float32),
            indices,
            weights,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        if not len(indices) == len(priorities):
            raise ValueError("Indices and priorities must have the same length.")
        for idx, priority in zip(indices, priorities):
            if not (0 <= idx < self.size):
                print(
                    f"Warning: Attempted to update priority for index {idx} outside valid range [0, {self.size-1}]"
                )
                continue
            self.priorities[idx] = abs(priority) + self.params.per_epsilon

    def __len__(self) -> int:
        return self.size


class NStepBuffer:
    """Accumulates transitions to calculate N-step returns."""

    def __init__(self, n_step: int, gamma: float):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer: Deque[Tuple[Any, int, float]] = deque(maxlen=n_step)
        self.done_buffer: Deque[bool] = deque(maxlen=n_step)

    def push(
        self, state: Any, action: int, reward: float, done: bool
    ) -> Optional[Tuple[Any, int, float, bool]]:
        self.buffer.append((state, action, reward))
        self.done_buffer.append(done)
        if len(self.buffer) < self.n_step:
            return None
        n_step_reward = 0.0
        discount = 1.0
        actual_n = 0
        for i in range(self.n_step):
            _, _, r = self.buffer[i]
            n_step_reward += discount * r
            discount *= self.gamma
            actual_n = i + 1
            if self.done_buffer[i]:
                break
        start_state, start_action, _ = self.buffer[0]
        n_step_done = self.done_buffer[actual_n - 1]
        return start_state, start_action, n_step_reward, n_step_done

    def is_full(self) -> bool:
        return len(self.buffer) == self.n_step

    def __len__(self) -> int:
        return len(self.buffer)


# === Rainbow Agent ===


class RainbowAgent:
    """Combines all Rainbow DQN enhancements."""

    def __init__(self, state_shape: Tuple, n_actions: int, params: Hyperparameters):
        self.params = params
        self.device = params.device
        self.n_actions = n_actions
        self.steps_done = 0
        print(f"Using device: {self.device}")

        # Initialize networks (on CPU first, then move)
        print(f"Initializing RainbowDQN with state_shape: {state_shape}")
        self.policy_net = RainbowDQN(state_shape, n_actions, params)
        self.target_net = RainbowDQN(state_shape, n_actions, params)

        # Move networks to the target device
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

        # Initialize target net state and optimizer AFTER moving models
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=params.learning_rate, eps=params.adam_eps
        )

        # Initialize buffers
        self.memory = PrioritizedReplayBuffer(
            params.buffer_size, params.per_alpha, params
        )
        self.n_step_accumulator = deque(maxlen=params.n_step)

        # Optional: TensorBoard Writer
        # self.writer = SummaryWriter(log_dir=params.tensorboard_log_dir)

    def _current_beta(self) -> float:
        return min(
            1.0,
            self.params.per_beta_start
            + self.steps_done
            * (1.0 - self.params.per_beta_start)
            / self.params.per_beta_frames,
        )

    def select_action(self, state: np.ndarray) -> int:
        with torch.no_grad():
            if not isinstance(state, np.ndarray):
                state = np.array(state, dtype=np.float32)
            if state.dtype != np.float32:
                print(f"Warning: state dtype is {state.dtype}, converting to float32.")
                state = state.astype(np.float32)
            # Ensure state tensor is on the correct device
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net.get_q_values(state_tensor)
            action = q_values.argmax(1).item()
        return action

    def store_transition(
        self, state: Any, action: int, reward: float, next_state: Any, done: bool
    ):
        self.n_step_accumulator.append(
            Experience(state, action, reward, next_state, done)
        )
        if len(self.n_step_accumulator) < self.params.n_step:
            return
        n_step_reward = 0.0
        discount = 1.0
        start_state, start_action, _, _, _ = self.n_step_accumulator[0]
        final_next_state = self.n_step_accumulator[-1].next_state
        n_step_done = False
        for i in range(self.params.n_step):
            s, a, r, ns, d = self.n_step_accumulator[i]
            n_step_reward += discount * r
            discount *= self.params.gamma
            if d:
                final_next_state = ns
                n_step_done = True
                break
        self.memory.push(
            start_state, start_action, n_step_reward, final_next_state, n_step_done
        )

    def optimize_model(self) -> float:
        if len(self.memory) < self.params.batch_size:
            return 0.0
        beta = self._current_beta()
        states, actions, n_step_rewards, next_states, n_step_dones, indices, weights = (
            self.memory.sample(self.params.batch_size, beta)
        )

        # Move numpy arrays to tensors on the correct device
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        n_step_rewards = torch.from_numpy(n_step_rewards).float().to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        n_step_dones = torch.from_numpy(n_step_dones).float().to(self.device)
        weights = torch.from_numpy(weights).float().to(self.device)

        # Ensure support is on the correct device (should be handled by model.to(device))
        support = self.policy_net.support.to(self.device)  # Get support from policy net

        with torch.no_grad():  # --- Compute Target Distribution ---
            next_q_values_policy = self.policy_net.get_q_values(next_states)
            next_actions = next_q_values_policy.argmax(1)
            next_q_dist_target = self.target_net(next_states)
            next_best_q_dist = next_q_dist_target[
                range(self.params.batch_size), next_actions
            ]
            Tz = n_step_rewards.unsqueeze(1) + (
                1 - n_step_dones.unsqueeze(1)
            ) * self.params.n_step_gamma * support.unsqueeze(0)
            Tz = Tz.clamp(self.params.v_min, self.params.v_max)
            b = (
                Tz - self.params.delta_z
            ) / self.params.delta_z  # Corrected: (Tz - v_min) / delta_z
            b = (Tz - self.params.v_min) / self.params.delta_z
            l = b.floor().long().clamp(0, self.params.num_atoms - 1)
            u = b.ceil().long().clamp(0, self.params.num_atoms - 1)
            mass_l = next_best_q_dist * (u.float() - b)
            mass_u = next_best_q_dist * (b - l.float())
            target_dist = torch.zeros(
                self.params.batch_size, self.params.num_atoms, device=self.device
            )
            target_dist.scatter_add_(1, l, mass_l)
            target_dist.scatter_add_(1, u, mass_u)

        # --- Compute Loss ---
        current_q_dist_policy = self.policy_net(states)
        action_indices = (
            actions.unsqueeze(1).unsqueeze(2).expand(-1, -1, self.params.num_atoms)
        )
        current_dist = current_q_dist_policy.gather(1, action_indices).squeeze(1)
        current_dist = current_dist.clamp(min=1e-8)
        log_p = torch.log(current_dist)
        elementwise_loss = -(target_dist * log_p).sum(1)
        loss = (elementwise_loss * weights).mean()

        # --- Optimization Step ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), self.params.gradient_clip_norm
        )
        self.optimizer.step()

        # --- Update Priorities ---
        new_priorities = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)

        # --- Update Target Network & Reset Noise ---
        self.steps_done += 1
        if self.steps_done % self.params.target_update_freq == 0:
            self.update_target_network()
        self.reset_noise()

        # Optional: Log loss and beta to TensorBoard
        # if self.writer:
        #     self.writer.add_scalar('Loss/train', loss.item(), self.steps_done)
        #     self.writer.add_scalar('Parameters/beta', beta, self.steps_done)

        return loss.item()

    def update_target_network(self):
        print(f"Updating target network at step {self.steps_done}")
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_noise(self):
        self.policy_net.reset_noise()

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Saving model checkpoint to {filepath}...")
        save_data = {
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
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
            # Load checkpoint onto the correct device
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
            self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.steps_done = checkpoint["steps_done"]
            # Ensure networks are on the correct device after loading state dicts
            self.policy_net.to(self.device)
            self.target_net.to(self.device)
            self.target_net.eval()
            self.policy_net.train()
            print(f"Model loaded successfully. Resuming from step {self.steps_done}.")
            return True
        except KeyError as e:
            print(
                f"Error loading checkpoint: Missing key {e}. Checkpoint might be incompatible."
            )
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            return False


# === Environment Creation Function ===


def make_env(params: Hyperparameters) -> gym.Env:
    """Creates and wraps the Super Mario Bros environment."""
    env = gym_super_mario_bros.make(params.env_id)
    # Apply wrappers in order:
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=params.skip_frames)
    env = RewardShapingWrapper(env, params)
    env = ResizeObservation(env, shape=params.processed_resize_dim)
    env = GrayScaleObservation(env, keep_dim=False)
    env = FrameStack(env, num_stack=params.stack_frames)
    env = NormalizeFrame(env)
    env = TimeLimit(env, max_episode_steps=params.max_episode_steps)
    return env


# === Main Training Script ===

if __name__ == "__main__":
    # --- Initialize Hyperparameters ---
    params = Hyperparameters()

    print("--- Hyperparameters ---")
    for key, value in vars(params).items():
        if key not in ["n_step_gamma", "delta_z", "device", "processed_resize_dim"]:
            print(f"{key}: {value}")
    print(f"processed_resize_dim: {params.processed_resize_dim}")
    print(f"device: {params.device}")
    print("---------------------")

    # --- Environment Setup ---
    env = make_env(params)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    print(f"\n--- Environment Info ---")
    print(f"State shape (after wrappers): {state_shape}")
    print(f"Observation space dtype: {env.observation_space.dtype}")
    print(f"Number of actions: {n_actions}")
    print(f"Running in {params.life_mode}-Life Mode")
    print("------------------------\n")

    # --- Agent Initialization ---
    agent = RainbowAgent(state_shape=state_shape, n_actions=n_actions, params=params)

    # --- Load Checkpoint if specified ---
    latest_checkpoint_path = os.path.join(
        params.checkpoint_dir, "rainbow_mario_model_latest.pth"
    )
    if params.load_checkpoint:
        agent.load_model(latest_checkpoint_path)

    # --- Training Loop ---
    episode_rewards_shaped = deque(maxlen=100)
    episode_rewards_unshaped = deque(maxlen=100)

    total_agent_steps = agent.steps_done
    episode_count = 0
    global_start_time = time.time()
    last_log_time = global_start_time
    last_log_step = total_agent_steps

    try:
        while total_agent_steps < params.total_train_steps:
            episode_count += 1
            state = env.reset()

            current_episode_reward_shaped = 0
            current_episode_reward_unshaped = 0
            current_episode_steps = 0
            current_episode_losses = []
            max_x_pos = 0
            episode_status = "RUNNING"

            # --- Episode Loop ---
            while True:
                action = agent.select_action(state)
                next_state, shaped_reward, done, info = env.step(action)
                original_reward = info.get("original_reward_sum", shaped_reward)
                effective_done = info.get("effective_done", done)
                life_lost = info.get("life_lost", False)
                current_x_pos = info.get("x_pos", 0)

                # Store transition using the SHAPED reward for learning
                agent.store_transition(state, action, shaped_reward, next_state, done)

                state = next_state
                current_episode_reward_shaped += shaped_reward
                current_episode_reward_unshaped += original_reward
                current_episode_steps += 1
                max_x_pos = max(max_x_pos, current_x_pos)

                # Optimize model
                if len(agent.memory) >= params.batch_size:
                    loss = agent.optimize_model()
                    if loss > 0:
                        current_episode_losses.append(loss)
                    total_agent_steps = agent.steps_done

                    # --- Logging ---
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
                        print(f"Last Loss: {loss:.4f}" if loss > 0 else "N/A")
                        print(f"Current Beta (PER): {agent._current_beta():.4f}")
                        print(f"FPS (Agent Steps): {fps:.2f}")
                        print(f"Buffer Size: {len(agent.memory)}/{params.buffer_size}")
                        elapsed_total = datetime.timedelta(
                            seconds=int(current_time - global_start_time)
                        )
                        print(f"Elapsed Time: {elapsed_total}")
                        print(f"----------------")

                        # Optional: Log to TensorBoard
                        # if agent.writer:
                        #    agent.writer.add_scalar('Reward/avg_shaped_reward_100', avg_reward_shaped_100, total_agent_steps)
                        #    agent.writer.add_scalar('Reward/avg_unshaped_reward_100', avg_reward_unshaped_100, total_agent_steps)

                        last_log_time = current_time
                        last_log_step = total_agent_steps

                    # --- Saving Checkpoint ---
                    if (
                        total_agent_steps % params.save_interval_steps == 0
                        and total_agent_steps > 0
                    ):
                        save_path = os.path.join(
                            params.checkpoint_dir,
                            f"rainbow_mario_model_step_{total_agent_steps}.pth",
                        )
                        agent.save_model(save_path)
                        agent.save_model(latest_checkpoint_path)

                # Check episode end condition
                if effective_done:
                    if info.get("TimeLimit.truncated", False):
                        episode_status = "TIMELIMIT"
                    elif life_lost and params.life_mode == 1:
                        episode_status = "LIFE_LOST (1-Life Mode)"
                    elif done:
                        episode_status = "FLAGPOLE/TIMEOUT/DEATH"
                    else:
                        episode_status = "UNKNOWN"
                    break

            # --- End of Episode ---
            episode_rewards_shaped.append(current_episode_reward_shaped)
            episode_rewards_unshaped.append(current_episode_reward_unshaped)
            avg_loss_episode = (
                np.mean(current_episode_losses) if current_episode_losses else 0.0
            )

            if params.print_episode_summary:
                print(
                    f"Episode {episode_count} finished after {current_episode_steps} steps. Status: {episode_status}"
                )
                print(f"  Total Shaped Reward: {current_episode_reward_shaped:.2f}")
                print(f"  Total Unshaped Reward: {current_episode_reward_unshaped:.2f}")
                print(f"  Max X Position: {max_x_pos}")
                print(f"  Avg Loss: {avg_loss_episode:.4f}")
                print(f"  Agent Steps: {total_agent_steps}")
                # Optional: Log episode metrics to TensorBoard
                # if agent.writer:
                #    agent.writer.add_scalar('Reward/episode_shaped_reward', current_episode_reward_shaped, total_agent_steps)
                #    agent.writer.add_scalar('Reward/episode_unshaped_reward', current_episode_reward_unshaped, total_agent_steps)

            # Check training goal
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
        # --- Save Final Model ---
        print("\nSaving final model...")
        final_save_path = os.path.join(
            params.checkpoint_dir, "rainbow_mario_model_final.pth"
        )
        agent.save_model(final_save_path)
        agent.save_model(latest_checkpoint_path)  # Save latest one last time

        # --- Clean Up ---
        print(f"\nTraining finished.")
        print(f"Total episodes: {episode_count}")
        print(f"Total agent steps: {total_agent_steps}")
        env.close()
        # Optional: Close TensorBoard writer
        # if agent.writer: agent.writer.close()
        print("Environment closed.")
