import numpy as np
import gym
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.frame_stack import FrameStack
from gym.wrappers import TimeLimit  # Import TimeLimit
from gym.spaces import Discrete, Box
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from collections import namedtuple, deque
from typing import List, Tuple
import time
import os  # Added for checkpoint directory creation

# Store experience as named tuples
Experience = namedtuple(
    "Experience", ("state", "action", "reward", "next_state", "done")
)

LIFE = 3


# === SkipFrame Wrapper ===
class SkipFrame(gym.Wrapper):
    """
    Return only every `skip`-th frame (frameskipping)
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._initial_lives = 3 - LIFE  # Assume starting lives reported as 2

    def step(self, action):
        """
        Repeat action, and sum reward.
        """
        total_reward = 0.0
        done = False
        info = {}

        for _ in range(self._skip):
            obs, reward, step_done, step_info = self.env.step(action)
            total_reward += reward
            info = step_info
            if step_done:
                done = True
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        """
        Resets the environment.
        """
        obs = self.env.reset(**kwargs)
        return obs


# Noisy Linear Layer for exploration
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=2.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # Initialize weights and biases
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        # Generate and scale noise
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        # Factorized Gaussian noise scaling
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        # Apply noisy linear transformation
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:  # Use mean weights during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# Dueling Network with Noisy Layers and Distributional RL (Rainbow DQN)
class RainbowDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms=51, v_min=-10, v_max=10):
        super(RainbowDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        # Store support on the module, will be moved to device later
        self.register_buffer("support", torch.linspace(v_min, v_max, num_atoms))

        # Feature extraction layers (Convolutional)
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        feature_size = self._get_conv_output(input_shape)

        # Dueling streams (Value and Advantage) with Noisy Layers
        self.value_stream = nn.Sequential(
            NoisyLinear(feature_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_atoms),  # Output distribution for value
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_size, 512),
            nn.ReLU(),
            NoisyLinear(
                512, num_actions * num_atoms
            ),  # Output distribution for advantages
        )

    def _get_conv_output(self, shape):
        # Calculate flattened feature size after convolutions
        with torch.no_grad():
            o = self.features(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size(0)
        # Normalize pixel values
        x = x / 255.0
        # Extract features
        x = self.features(x)
        x = x.view(batch_size, -1)  # Flatten

        # Get value and advantage distributions
        value_dist = self.value_stream(x).view(batch_size, 1, self.num_atoms)
        advantage_dist = self.advantage_stream(x).view(
            batch_size, self.num_actions, self.num_atoms
        )

        # Combine streams (Dueling architecture)
        q_dist = value_dist + (
            advantage_dist - advantage_dist.mean(dim=1, keepdim=True)
        )

        # Convert to probabilities using Softmax
        q_dist = F.softmax(q_dist, dim=-1)
        # Clamp probabilities for numerical stability
        q_dist = q_dist.clamp(min=1e-8)

        return q_dist

    def reset_noise(self):
        # Reset noise in all NoisyLinear layers
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# Prioritized Experience Replay buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.eps = 1e-5  # Small constant for priorities

    def beta(self):
        # Anneal beta from beta_start to 1.0
        return min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames,
        )

    def push(self, state, action, reward, next_state, done):
        # Add experience with max priority
        max_prio = self.priorities.max() if self.buffer else 1.0
        experience = Experience(state, action, reward, next_state, done)

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity
        # Only increment frame count when adding to PER buffer
        # self.frame += 1 # Moved frame increment to Agent's optimize_model

    def sample(self, batch_size, current_beta):
        # Sample experiences based on priority
        buffer_size = len(self.buffer)
        if buffer_size == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.position]

        # Calculate sampling probabilities
        probs = prios**self.alpha
        sum_probs = probs.sum()
        if sum_probs == 0:  # Handle edge case of all zero priorities
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= sum_probs

        # Sample indices
        indices = np.random.choice(buffer_size, batch_size, p=probs, replace=True)

        # Calculate importance sampling weights
        # Use the passed current_beta
        weights = (buffer_size * probs[indices]) ** (-current_beta)
        weights /= weights.max() + 1e-8  # Normalize weights
        weights = np.array(weights, dtype=np.float32)

        # Get batch of experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to numpy arrays (handles LazyFrames)
        states = np.array(states)
        next_states = np.array(next_states)

        return (
            states,
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            next_states,
            np.array(dones, dtype=np.float32),
            indices,
            weights,
        )

    def update_priorities(self, indices, priorities):
        # Update priorities of sampled experiences
        for idx, priority in zip(indices, priorities):
            assert (
                0 <= idx < len(self.buffer)
            ), f"Index {idx} out of range for buffer size {len(self.buffer)}"
            self.priorities[idx] = abs(priority) + self.eps

    def __len__(self):
        return len(self.buffer)


# N-step experience accumulator
class NStepBuffer:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)  # Fixed-size buffer

    def push(self, state, action, reward, next_state, done):
        # Store 1-step transition including its done flag
        self.buffer.append((state, action, reward, done))

        if len(self.buffer) < self.n_step:
            return None  # Not enough steps for an n-step return yet

        # Calculate n-step return
        start_state, start_action, _, _ = self.buffer[0]
        n_step_reward = 0.0
        discount = 1.0
        actual_n = 0
        for i in range(self.n_step):
            _, _, r, d = self.buffer[i]
            n_step_reward += discount * r
            discount *= self.gamma
            actual_n = i + 1
            if d:  # Stop accumulating if a terminal state is reached within n steps
                break

        # The final next_state and done flag are from the perspective of the Nth step (or earlier if terminated)
        # Use the next_state and done passed into this call, as they correspond to the end of the sequence.
        final_next_state = next_state
        final_done = done

        # Return the N-step experience
        return start_state, start_action, n_step_reward, final_next_state, final_done


# Rainbow Agent combining all enhancements
class RainbowAgent:
    def __init__(
        self,
        state_shape,
        n_actions,
        learning_rate=0.0001,
        gamma=0.99,
        n_step=3,
        target_update=10000,
        batch_size=32,
        buffer_size=100000,
        v_min=-10,
        v_max=10,
        num_atoms=51,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=100000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Distributional RL parameters
        self.v_min, self.v_max, self.num_atoms = v_min, v_max, num_atoms
        # Ensure support is created on the correct device from the start
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Policy and Target Networks
        self.policy_net = RainbowDQN(
            state_shape, n_actions, num_atoms, v_min, v_max
        ).to(self.device)
        self.target_net = RainbowDQN(
            state_shape, n_actions, num_atoms, v_min, v_max
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network in evaluation mode

        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=learning_rate, eps=1.5e-4
        )

        # Replay Buffers
        self.memory = PrioritizedReplayBuffer(
            buffer_size, alpha, beta_start, beta_frames
        )
        self.n_step_buffer = NStepBuffer(n_step, gamma)

        # Hyperparameters
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_gamma = gamma**n_step  # Discount for n-step returns
        self.target_update = target_update
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.steps_done = 0  # Counter for target network updates
        self.beta_start = beta_start
        self.beta_frames = beta_frames

    def current_beta(self):
        # Calculate annealed beta based on agent steps_done
        # This ensures beta anneals based on learning steps, not buffer additions
        return min(
            1.0,
            self.beta_start
            + self.steps_done * (1.0 - self.beta_start) / self.beta_frames,
        )

    def select_action(self, state):
        """Selects action based on highest expected Q-value from policy network."""
        with torch.no_grad():
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            dist = self.policy_net(state_tensor)
            # Calculate expected Q-values from the distribution
            # Ensure support is on the correct device (should be via buffer registration)
            expected_value = (
                dist * self.policy_net.support.view(1, 1, self.num_atoms)
            ).sum(2)
            action = expected_value.argmax(1).item()
        return action

    def update_target_network(self):
        """Copies weights from policy_net to target_net."""
        print(f"Updating target network at step {self.steps_done}")
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_noise(self):
        """Resets noise in NoisyLinear layers of the policy network."""
        self.policy_net.reset_noise()

    def store_transition(self, state, action, reward, next_state, done):
        """Processes transition through n-step buffer and stores result in PER."""
        n_step_transition = self.n_step_buffer.push(
            state, action, reward, next_state, done
        )
        if n_step_transition:
            s, a, nr, ns, nd = n_step_transition
            self.memory.push(s, a, nr, ns, nd)  # Store n-step experience

    def optimize_model(self):
        """Performs one optimization step."""
        if len(self.memory) < self.batch_size:
            return 0.0  # Not enough samples yet

        # Calculate current beta for PER sampling
        beta = self.current_beta()

        # Sample n-step experiences from PER
        states, actions, n_step_rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.batch_size, beta)
        )  # Pass beta

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        n_step_rewards = torch.FloatTensor(n_step_rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # --- Compute Target Distribution (Double DQN + N-step + Distributional) ---
        with torch.no_grad():
            # Select actions using policy network
            next_q_dist_policy = self.policy_net(next_states)
            # Use support from the network buffer
            next_expected_values = (
                next_q_dist_policy * self.policy_net.support.view(1, 1, self.num_atoms)
            ).sum(2)
            next_actions = next_expected_values.argmax(1)

            # Get distributions for selected actions from target network
            next_q_dist_target = self.target_net(next_states)
            next_q_dist = next_q_dist_target[
                range(self.batch_size), next_actions
            ]  # Shape: [batch, atoms]

            # Project Bellman update onto support
            # Use support from the network buffer
            Tz = n_step_rewards.unsqueeze(1) + (
                1 - dones.unsqueeze(1)
            ) * self.n_step_gamma * self.target_net.support.view(1, -1)
            Tz = Tz.clamp(self.v_min, self.v_max)  # Clamp to support range
            b = (Tz - self.v_min) / self.delta_z  # Calculate bin indices
            l, u = b.floor().long(), b.ceil().long()
            l = l.clamp(0, self.num_atoms - 1)  # Ensure indices are valid
            u = u.clamp(0, self.num_atoms - 1)

            # Distribute probability mass
            m = torch.zeros(self.batch_size, self.num_atoms, device=self.device)
            mass_l = next_q_dist * (u.float() - b)
            mass_u = next_q_dist * (b - l.float())
            m.scatter_add_(1, l, mass_l)
            m.scatter_add_(1, u, mass_u)  # m is the target distribution

        # --- Compute Loss ---
        current_q_dist_policy = self.policy_net(states)
        current_q_dist = current_q_dist_policy[
            range(self.batch_size), actions
        ]  # Shape: [batch, atoms]
        log_p = torch.log(
            current_q_dist + 1e-8
        )  # Log probabilities of current distribution

        # Cross-entropy loss between target (m) and current (log_p) distributions
        elementwise_loss = -(m * log_p).sum(1)

        # Apply PER weights
        loss = (elementwise_loss * weights).mean()

        # --- Optimization Step ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 10.0
        )  # Clip gradients
        self.optimizer.step()

        # --- Update Priorities ---
        new_priorities = elementwise_loss.detach().cpu().numpy()
        self.memory.update_priorities(indices, new_priorities)

        # --- Update Target Network ---
        # Increment steps_done *after* optimization step
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.update_target_network()

        return loss.item()

    def save_model(self, filepath):
        """Saves model checkpoint."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        print(f"Saving model checkpoint to {filepath}...")
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            filepath,
        )
        print("Model saved.")

    def load_model(self, filepath):
        """Loads model checkpoint."""
        if not os.path.isfile(filepath):
            print(f"Checkpoint file not found at {filepath}. Cannot load.")
            return False  # Indicate loading failed
        print(f"Loading model checkpoint from {filepath}...")
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.steps_done = checkpoint["steps_done"]
        self.target_net.eval()
        self.policy_net.train()  # Ensure policy net is in train mode
        print(f"Model loaded. Resuming from step {self.steps_done}.")
        return True  # Indicate loading succeeded


if __name__ == "__main__":
    # === Training Parameters ===
    total_frames_to_train = 2_000_000
    max_episode_steps = 4500  # Define max steps for TimeLimit wrapper
    log_interval_steps = 10000  # Log based on agent steps (optimize calls)
    save_interval_steps = 100000  # Save based on agent steps
    print_episode_summary = True
    # Reward Shaping Parameters
    death_penalty = -15.0
    move_reward = 1.0

    # === Environment Setup ===
    env_id = "SuperMarioBros-v0"
    skip_frames = 4
    stack_frames = 4
    resize_dim = 84

    env = gym_super_mario_bros.make(env_id)
    # Apply wrappers in order
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = SkipFrame(env, skip=skip_frames)
    env = ResizeObservation(env, shape=resize_dim)
    env = GrayScaleObservation(env, keep_dim=False)
    env = FrameStack(env, num_stack=stack_frames)
    # Add TimeLimit wrapper
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # Get state and action space properties
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n

    print(f"Environment: {env_id}")
    print(f"Frame Skipping: {skip_frames}")
    print(f"Frame Stacking: {stack_frames}")
    print(f"Max Episode Steps (TimeLimit): {max_episode_steps}")
    print(f"State shape (after wrappers): {state_shape}")
    print(f"Number of actions: {n_actions}")
    print(f"--- Running in {LIFE}-Life Mode ---")
    print(
        f"--- Reward Shaping: Death Penalty={death_penalty}, Move Reward={move_reward} ---"
    )  # Log shaping

    # === Agent Initialization ===
    agent = RainbowAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=0.0004,
        gamma=0.9,
        n_step=5,
        target_update=1000,
        batch_size=32,
        buffer_size=10000,
        v_min=-50,
        v_max=150,
        num_atoms=51,
        alpha=0.5,
        beta_start=0.4,
        beta_frames=1000000,
    )

    # Load checkpoint if specified
    load_checkpoint = False
    checkpoint_dir = "mario_rainbow_checkpoints"  # Define checkpoint directory
    checkpoint_path = os.path.join(checkpoint_dir, "rainbow_mario_model_latest.pth")
    if load_checkpoint:
        agent.load_model(checkpoint_path)

    # === Training Loop ===
    episode_rewards_custom = deque(maxlen=100)  # Track custom rewards
    episode_rewards_env = deque(maxlen=100)  # Track original env rewards
    total_agent_steps = agent.steps_done  # Start step count from loaded value if any
    episode_count = 0
    start_time = time.time()

    try:
        while total_agent_steps < total_frames_to_train:  # Loop based on agent steps
            episode_count += 1
            state = env.reset()
            agent.reset_noise()  # Reset noise per episode
            current_episode_reward_custom = 0
            current_episode_reward_env = 0
            current_episode_steps = 0
            current_episode_losses = []
            max_x_pos = 0
            episode_start_lives = 3 - LIFE  # For 1-life mode check
            prev_x_pos = 0  # Initialize for reward shaping

            # Episode loop
            while True:  # Loop until done (effective_done)
                # Select action
                action = agent.select_action(state)

                # Step environment
                next_state, env_reward, done, info = env.step(action)

                # --- 1-Life Modification & Termination Check ---
                current_lives = info.get("life", episode_start_lives)
                life_lost = current_lives < episode_start_lives
                # effective_done determines if the episode loop should break
                effective_done = done or life_lost
                # --- End 1-Life Modification ---

                # --- Reward Shaping ---
                custom_reward = env_reward  # Start with env reward
                current_x_pos = info.get(
                    "x_pos", prev_x_pos
                )  # Use prev if not available
                # 1. Reward for moving right
                if current_x_pos > prev_x_pos:
                    custom_reward += move_reward
                # 2. Penalty for dying
                if life_lost:
                    custom_reward += death_penalty
                # Update previous position
                prev_x_pos = current_x_pos
                # --- End Reward Shaping ---

                # Track progress
                max_x_pos = max(max_x_pos, current_x_pos)

                # Store transition using the *original* done flag from env.step()
                # and the *custom_reward* for learning.
                agent.store_transition(state, action, custom_reward, next_state, done)

                # Update state
                state = next_state
                current_episode_reward_custom += custom_reward
                current_episode_reward_env += (
                    env_reward  # Track original reward separately
                )
                current_episode_steps += 1

                # Optimize model & Increment agent steps
                # Optimize based on agent steps (learning steps)
                if (
                    len(agent.memory) >= agent.batch_size
                ):  # Optimize only when buffer has enough samples
                    loss = (
                        agent.optimize_model()
                    )  # optimize_model now increments agent.steps_done
                    if loss > 0:
                        current_episode_losses.append(loss)
                    total_agent_steps = agent.steps_done  # Update total steps count

                    # Logging based on agent steps
                    if (
                        total_agent_steps % log_interval_steps == 0
                        and total_agent_steps > 0
                    ):
                        avg_reward_custom_100 = (
                            np.mean(episode_rewards_custom)
                            if episode_rewards_custom
                            else 0.0
                        )
                        avg_reward_env_100 = (
                            np.mean(episode_rewards_env) if episode_rewards_env else 0.0
                        )
                        current_loss = (
                            current_episode_losses[-1]
                            if current_episode_losses
                            else 0.0
                        )
                        elapsed_time = time.time() - start_time
                        # Calculate FPS based on agent steps since last log
                        fps = (
                            log_interval_steps / elapsed_time if elapsed_time > 0 else 0
                        )
                        print(f"\n--- Progress ---")
                        print(
                            f"Steps: {total_agent_steps}/{total_frames_to_train} ({total_agent_steps/total_frames_to_train*100:.2f}%)"
                        )
                        print(f"Episodes: {episode_count}")
                        print(
                            f"Avg Custom Reward (Last 100): {avg_reward_custom_100:.2f}"
                        )
                        print(f"Avg Env Reward (Last 100): {avg_reward_env_100:.2f}")
                        print(f"Last Loss: {current_loss:.4f}")
                        print(
                            f"Current Beta: {agent.current_beta():.3f}"
                        )  # Use agent's beta calculation
                        print(f"FPS (Agent Steps): {fps:.2f}")
                        print(f"----------------")
                        start_time = time.time()  # Reset timer for next interval

                    # Save model based on agent steps
                    if (
                        total_agent_steps % save_interval_steps == 0
                        and total_agent_steps > 0
                    ):
                        save_path = os.path.join(
                            checkpoint_dir,
                            f"rainbow_mario_model_step_{total_agent_steps}.pth",
                        )
                        agent.save_model(save_path)
                        # Also save a latest checkpoint
                        latest_save_path = os.path.join(
                            checkpoint_dir, "rainbow_mario_model_latest.pth"
                        )
                        agent.save_model(latest_save_path)

                # Check if episode ended (TimeLimit, life lost, or natural end)
                if effective_done:
                    break

            # End of episode actions
            episode_rewards_custom.append(current_episode_reward_custom)
            episode_rewards_env.append(current_episode_reward_env)  # Log env reward too
            avg_loss_episode = (
                np.mean(current_episode_losses) if current_episode_losses else 0.0
            )
            if print_episode_summary:
                status = (
                    "TIMELIMIT"
                    if info.get("TimeLimit.truncated", False)
                    else ("LIFE_LOST" if life_lost else "FLAGPOLE/TIMEOUT")
                )
                print(
                    f"Episode {episode_count} finished after {current_episode_steps} steps. Status: {status}"
                )
                print(f"  Total Custom Reward: {current_episode_reward_custom:.2f}")
                print(
                    f"  Total Env Reward: {current_episode_reward_env:.2f}"
                )  # Log env reward
                print(f"  Max X Position: {max_x_pos}")
                print(f"  Avg Loss: {avg_loss_episode:.4f}")
                print(f"  Agent Steps: {total_agent_steps}")

            # Check if training goal reached
            if total_agent_steps >= total_frames_to_train:
                print("\nTarget number of training steps reached.")
                break

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    finally:
        # Save final model and close environment
        final_save_path = os.path.join(checkpoint_dir, "rainbow_mario_model_final.pth")
        agent.save_model(final_save_path)
        print(f"\nTraining finished.")
        print(f"Total episodes: {episode_count}")
        print(f"Total agent steps: {total_agent_steps}")
        env.close()
