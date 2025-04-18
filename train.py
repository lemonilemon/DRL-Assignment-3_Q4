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
from collections import deque
import random


class DuelingQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingQNetwork, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        # Convolutional layers
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Fully connected layers
        self.fc = nn.Sequential(nn.Linear(self.feature_size(), 512), nn.ReLU())

        # Value stream
        self.value_stream = nn.Sequential(nn.Linear(512, 1))

        # Advantage stream
        self.advantage_stream = nn.Sequential(nn.Linear(512, num_actions))

    def feature_size(self):
        return self._get_conv_output((1, *self.input_shape))

    def _get_conv_output(self, shape):
        output_feat = self._forward_features(torch.zeros(shape))
        n_size = output_feat.data.view(1, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.features(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)

        # Combine value and advantage
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class RainbowDQNAgent:
    def __init__(self, state_shape, n_actions, learning_rate=0.0001):
        self.q_network = DuelingQNetwork(state_shape, n_actions)
        self.target_network = DuelingQNetwork(state_shape, n_actions)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=100000)  # Placeholder for prioritized replay

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.batch_size = 32
        self.target_update = 10000

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.q_network.num_actions)
        else:
            with torch.no_grad():
                state = np.array(state)
                state = torch.FloatTensor(state).squeeze(-1).unsqueeze(0)
                return self.q_network(state).argmax().item()

    def cache(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.array(states)
        states = torch.FloatTensor(states)
        actions = np.array(actions)
        actions = torch.LongTensor(actions)
        rewards = np.array(rewards)
        rewards = torch.FloatTensor(rewards)
        next_states = np.array(next_states)
        next_states = torch.FloatTensor(next_states)
        dones = np.array(dones)
        dones = torch.FloatTensor(dones)

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if len(self.memory) % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the environment
    env = gym_super_mario_bros.make("SuperMarioBros-v3")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)  # Simplify actions to 7 discrete actions
    env = GrayScaleObservation(env, keep_dim=True)  # Convert to grayscale
    env = ResizeObservation(env, (84, 84))  # Resize to 84x84 pixels
    env = FrameStack(env, 4)  # Stack 4 frames for temporal information

    state_shape = env.observation_space.shape

    if isinstance(env.action_space, Discrete):
        n_actions = env.action_space.n
    else:
        raise (ValueError("Action space is not discrete."))
    print(f"State shape: {state_shape}, Number of actions: {n_actions}")

    agent = RainbowDQNAgent(state_shape, n_actions)
    agent.q_network.to(device)
    agent.target_network.to(device)

    num_episodes = 10000
    max_steps = 10000

    for episode in range(num_episodes):
        state = env.reset()
        state = np.array(state)
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        total_reward = 0
        for step in range(max_steps):
            action = agent.act(np.array(state))
            next_state, reward, done, _ = env.step(action)
            agent.cache(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            if done:
                break
        print(f"Episode {episode}, Total Reward: {total_reward}")
