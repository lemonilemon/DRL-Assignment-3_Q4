import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from gym.wrappers.gray_scale_observation import GrayScaleObservation
from gym.wrappers.resize_observation import ResizeObservation
from gym.wrappers.frame_stack import FrameStack
from gym.spaces import Discrete
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
from collections import namedtuple
from typing import List, Tuple
from tqdm import tqdm, trange
import time

# Store experience as named tuples
Experience = namedtuple(
    "Experience", ("state", "action", "reward", "next_state", "done")
)


# Noisy Linear Layer for exploration
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
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
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


# Improved Dueling Network with Noisy Layers and Distributional RL
class RainbowDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms=51, v_min=-10, v_max=10):
        super(RainbowDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.support = torch.linspace(v_min, v_max, num_atoms)

        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        feature_size = self._get_conv_output()

        # Value stream with noisy linear and distributional output
        self.value_stream = nn.Sequential(
            NoisyLinear(feature_size, 512), nn.ReLU(), NoisyLinear(512, num_atoms)
        )

        # Advantage stream with noisy linear and distributional output
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * num_atoms),
        )

    def _get_conv_output(self):
        # Handle the input shape correctly
        x = torch.zeros(
            1, *self.input_shape[:-1] if len(self.input_shape) > 3 else self.input_shape
        )
        x = self.features(x)
        return x.view(1, -1).size(1)

    def forward(self, x):
        batch_size = x.size(0)

        # Remove extra dimension if present
        if len(x.shape) > 4:
            x = x.squeeze(-1)  # Remove the last dimension if it exists

        # Process through conv layers
        x = self.features(x)
        x = x.view(batch_size, -1)

        # Compute value and advantage
        value = self.value_stream(x).view(batch_size, 1, self.num_atoms)
        advantage = self.advantage_stream(x).view(
            batch_size, self.num_actions, self.num_atoms
        )

        # Combine value and advantage using dueling architecture
        q_dist = value + (advantage - advantage.mean(dim=1, keepdim=True))

        # Apply softmax to get probability distribution
        q_dist = F.softmax(q_dist, dim=-1)

        return q_dist

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


# Prioritized Experience Replay buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=100000):
        self.capacity = capacity
        self.alpha = alpha  # prioritization exponent
        self.beta_start = beta_start  # importance sampling weight
        self.beta_frames = beta_frames
        self.frame = 1  # for beta calculation
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.eps = 1e-5  # small constant to ensure non-zero priority

    def beta(self):
        return min(
            1.0,
            self.beta_start + self.frame * (1.0 - self.beta_start) / self.beta_frames,
        )

    def push(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = Experience(
                state, action, reward, next_state, done
            )

        # New experiences get max priority
        self.priorities[self.position] = max_prio
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.position]

        # Compute sampling probabilities from priorities
        probs = prios**self.alpha
        probs /= probs.sum()

        # Sample indices according to probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Compute importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta())
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        # Extract experiences
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            indices,
            weights,
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + self.eps

    def __len__(self):
        return len(self.buffer)


# N-step experience accumulator
class NStepBuffer:
    def __init__(self, n_step, gamma):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = []

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

        if len(self.buffer) < self.n_step:
            return None

        # Calculate n-step reward
        state, action, reward, _, _ = self.buffer[0]

        for i in range(1, self.n_step):
            _, _, r, next_s, done = self.buffer[i]
            reward += r * (self.gamma**i)

            if done:
                break

        _, _, _, next_state, done = self.buffer[self.n_step - 1]

        # Remove the oldest experience
        self.buffer.pop(0)

        return state, action, reward, next_state, done


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
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parameters for distributional RL
        self.v_min = v_min
        self.v_max = v_max
        self.num_atoms = num_atoms
        self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)
        self.delta_z = (v_max - v_min) / (num_atoms - 1)

        # Main and target networks
        self.policy_net = RainbowDQN(
            state_shape, n_actions, num_atoms, v_min, v_max
        ).to(self.device)
        self.target_net = RainbowDQN(
            state_shape, n_actions, num_atoms, v_min, v_max
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Memory buffers for experience replay
        self.memory = PrioritizedReplayBuffer(buffer_size)
        self.n_step_buffer = NStepBuffer(n_step, gamma)

        # Hyperparameters
        self.gamma = gamma
        self.n_step = n_step
        self.n_step_gamma = gamma**n_step
        self.target_update = target_update
        self.batch_size = batch_size

        self.n_actions = n_actions
        self.steps_done = 0

    def select_action(self, state):
        # No epsilon-greedy because we're using noisy networks for exploration
        with torch.no_grad():
            # Process state to correct format
            if isinstance(state, np.ndarray):
                # Handle numpy array state
                if len(state.shape) > 3 and state.shape[-1] == 1:
                    state = state.squeeze(-1)  # Remove extra dimension if present
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
            else:
                # Handle LazyFrames or other objects
                state = np.array(state)
                if len(state.shape) > 3 and state.shape[-1] == 1:
                    state = state.squeeze(-1)  # Remove extra dimension if present
                state = torch.FloatTensor(state).to(self.device).unsqueeze(0)

            dist = self.policy_net(state)
            # Calculate expected value for each action
            expected_value = dist * self.support.expand_as(dist)
            q_values = expected_value.sum(2)
            action = q_values.argmax(1).item()

        return action

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def reset_noise(self):
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def store_transition(self, state, action, reward, next_state, done):
        # Process through n-step buffer first
        transition = self.n_step_buffer.push(state, action, reward, next_state, done)

        if transition:
            # Add processed n-step transition to main buffer
            self.memory.push(*transition)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0  # Not enough samples

        # Reset noise for this optimization step
        self.reset_noise()

        # Sample from memory
        states, actions, rewards, next_states, dones, indices, weights = (
            self.memory.sample(self.batch_size)
        )

        # Process states and next_states
        # For both states and next_states, we need to remove the last dimension if it exists
        if len(states.shape) > 4 and states.shape[-1] == 1:
            states = states.squeeze(-1)
        if len(next_states.shape) > 4 and next_states.shape[-1] == 1:
            next_states = next_states.squeeze(-1)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Compute current state-action values
        current_q_dist = self.policy_net(states)
        current_q_dist = current_q_dist[range(self.batch_size), actions]

        # Compute next state values (Double DQN)
        with torch.no_grad():
            # Get actions from policy network
            next_q_dist = self.policy_net(next_states)
            next_expected_values = (
                next_q_dist * self.support.expand_as(next_q_dist)
            ).sum(2)
            next_actions = next_expected_values.argmax(1)

            # Get value distribution from target network with those actions
            next_q_dist = self.target_net(next_states)
            next_q_dist = next_q_dist[range(self.batch_size), next_actions]

            # Handle terminal states
            expected_q_dist = torch.zeros(
                (self.batch_size, self.num_atoms), device=self.device
            )

            # Compute Tz (Bellman operator target)
            for atom in range(self.num_atoms):
                # Calculate Tz for each atom in the support
                Tz = rewards + (1 - dones) * self.n_step_gamma * self.support[atom]
                # Clamp Tz to the support range
                Tz = Tz.clamp(self.v_min, self.v_max)

                # Find which bin Tz falls into
                b = (Tz - self.v_min) / self.delta_z

                # Apply lower and upper bounds
                l = b.floor().long()
                u = b.ceil().long()

                # Distribute probability
                offset = torch.zeros(
                    (self.batch_size, self.num_atoms), device=self.device
                )

                # Handle the case where l == u
                equal_indices = l == u
                offset[equal_indices, l[equal_indices]] += next_q_dist[
                    equal_indices, atom
                ]

                # Handle cases where l != u
                ne_indices = ~equal_indices
                offset[ne_indices, l[ne_indices]] += next_q_dist[ne_indices, atom] * (
                    u[ne_indices].float() - b[ne_indices]
                )
                offset[ne_indices, u[ne_indices]] += next_q_dist[ne_indices, atom] * (
                    b[ne_indices] - l[ne_indices].float()
                )

                expected_q_dist += offset

        # Compute KL divergence loss (Cross-entropy works since target is fixed)
        loss = -(expected_q_dist * torch.log(current_q_dist + 1e-10)).sum(1)

        # Apply importance sampling weights from prioritized replay
        weighted_loss = (loss * weights).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # Update priorities in the replay buffer
        self.memory.update_priorities(indices, loss.detach().cpu().numpy())

        # Update target network if needed
        self.steps_done += 1
        if self.steps_done % self.target_update == 0:
            self.update_target_network()

        return loss.mean().item()

    def save_model(self, filepath):
        """Save the model and training state"""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "steps_done": self.steps_done,
            },
            filepath,
        )

    def load_model(self, filepath):
        """Load the model and training state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]


if __name__ == "__main__":
    # Create the environment
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)  # Simplify actions
    env = GrayScaleObservation(env, keep_dim=True)  # Convert to grayscale
    env = ResizeObservation(env, (84, 84))  # Resize to 84x84 pixels
    env = FrameStack(env, 4)  # Stack 4 frames for temporal information

    state_shape = env.observation_space.shape
    if isinstance(env.action_space, Discrete):
        n_actions = env.action_space.n
    else:
        raise ValueError("Action space is not discrete.")

    print(f"State shape: {state_shape}, Number of actions: {n_actions}")
    print(
        f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}"
    )

    # Create the agent
    agent = RainbowAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=0.0001,
        gamma=0.99,
        n_step=3,
        target_update=10000,
        batch_size=32,
        buffer_size=100000,
        v_min=-10,
        v_max=10,
        num_atoms=51,
    )
    # Training parameters
    total_frames = 10_000_000  # 10 million frames total
    max_episode_steps = 10000
    log_interval = 100000  # Print progress every 100k frames
    save_interval = 1000000  # Save model every 1M frames

    # Lists for tracking progress
    episode_rewards = []
    mean_rewards = []
    frame_count = 0
    episode_count = 0

    # Create progress bar for total frames
    pbar_frames = tqdm(total=total_frames, desc="Training Progress")

    try:
        while frame_count < total_frames:
            episode_count += 1
            state = env.reset()
            total_reward = 0
            losses = []
            best_x_pos = 0  # Track furthest Mario has gone
            episode_steps = 0

            # Reset noise for this episode
            agent.reset_noise()

            for step in range(max_episode_steps):
                # Select and perform action
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)

                # Update frame count
                frame_count += 1
                episode_steps += 1

                # Track furthest x position
                x_pos = info.get("x_pos", 0)
                best_x_pos = max(best_x_pos, x_pos)

                # Modify reward for better learning
                # Give positive reward for moving right and gaining score
                modified_reward = reward
                # Add bonus for moving right
                x_pos_diff = info.get("x_pos", 0) - info.get("x_pos_prev", 0)
                if x_pos_diff > 0:
                    modified_reward += 0.1 * x_pos_diff
                # Penalize for time passing (encourages faster completion)
                modified_reward -= 0.01
                # Clip very negative rewards
                if modified_reward < -10:
                    modified_reward = -10

                # Store the transition in memory
                agent.store_transition(state, action, modified_reward, next_state, done)

                # Move to the next state
                state = next_state
                total_reward += reward  # Track original reward for logging

                # Perform optimization step
                loss = agent.optimize_model()
                if loss != 0:
                    losses.append(loss)

                # Update the progress bar
                pbar_frames.update(1)

                # Log at specific frame intervals
                if frame_count % log_interval == 0:
                    # Track reward statistics
                    if len(episode_rewards) > 0:  # Avoid division by zero or empty list
                        avg_loss = np.mean(losses) if losses else 0
                        mean_100_reward = (
                            np.mean(episode_rewards[-100:])
                            if len(episode_rewards) >= 100
                            else np.mean(episode_rewards)
                        )

                        print(
                            f"\nFrame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)"
                        )
                        print(f"  Episodes completed: {episode_count}")
                        print(f"  Current episode reward: {total_reward:.2f}")
                        print(f"  Best x position: {best_x_pos}")
                        print(
                            f"  Average reward (last 100 episodes): {mean_100_reward:.2f}"
                        )
                        print(f"  Average loss: {avg_loss:.4f}")
                        print(f"  Progress: {frame_count/total_frames*100:.1f}%")

                # Save model at specific frame intervals
                if frame_count % save_interval == 0:
                    model_path = f"rainbow_mario_model_frame_{frame_count}.pth"
                    agent.save_model(model_path)
                    print(f"\nSaved model to {model_path}")

                if done:
                    break

            # End of episode tracking
            episode_rewards.append(total_reward)
            mean_reward = (
                np.mean(episode_rewards[-100:])
                if len(episode_rewards) >= 100
                else np.mean(episode_rewards)
            )
            mean_rewards.append(mean_reward)

            # Optional: print brief episode summary (can be removed for less console output)
            print(
                f"Episode {episode_count}: frames={episode_steps}, reward={total_reward:.1f}, x_pos={best_x_pos}"
            )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save model on interrupt
        model_path = f"rainbow_mario_model_interrupted_frame_{frame_count}.pth"
        agent.save_model(model_path)
        print(f"Saved model to {model_path}")

    finally:
        # Save final model
        model_path = f"rainbow_mario_model_final_frame_{frame_count}.pth"
        agent.save_model(model_path)
        print(f"Saved final model to {model_path}")
        print(
            f"Training completed after {episode_count} episodes and {frame_count} frames"
        )
