import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import GrayScaleObservation, ResizeObservation, FrameStack
from collections import deque
import random
from tqdm import tqdm


def conv_output_size(shape, model):
    dummy = torch.zeros(shape)
    out = model(dummy)
    return int(torch.prod(torch.tensor(out.shape[1:])))


class DuelingQNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        c, h, w, _ = input_shape  # e.g. (4,84,84,1)
        self.features = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        conv_out = conv_output_size((1, c, h, w), self.features)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(conv_out, 512), nn.ReLU())
        self.value_stream = nn.Linear(512, 1)
        self.advantage_stream = nn.Linear(512, num_actions)

    def forward(self, x):
        # x: (batch, c, h, w)
        x = self.features(x)
        x = self.fc(x)
        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        q = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q


class RainbowDQNAgent:
    def __init__(
        self,
        state_shape,
        n_actions,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.999,
        memory_size=100000,
        batch_size=32,
        target_update_freq=10000,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = DuelingQNetwork(state_shape, n_actions).to(self.device)
        self.target_network = DuelingQNetwork(state_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.step_count = 0

    def act(self, state):
        # state: raw numpy array from env, shape (4,84,84,1)
        if random.random() < self.epsilon:
            return random.randrange(self.q_network.advantage_stream.out_features)
        state = np.array(state, dtype=np.float32).squeeze(-1)
        state = torch.from_numpy(state).unsqueeze(0).to(self.device)  # (1,4,84,84)
        with torch.no_grad():
            q_vals = self.q_network(state)
        return q_vals.argmax(1).item()

    def cache(self, state, action, reward, next_state, done):
        # store transitions as numpy arrays
        self.memory.append(
            (
                np.array(state, dtype=np.float32),
                action,
                reward,
                np.array(next_state, dtype=np.float32),
                done,
            )
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))

        # preprocess
        states = torch.from_numpy(states).squeeze(-1).to(self.device)  # (B,4,84,84)
        next_states = torch.from_numpy(next_states).squeeze(-1).to(self.device)
        actions = torch.from_numpy(actions).long().to(self.device)
        rewards = torch.from_numpy(rewards).float().to(self.device)
        dones = torch.from_numpy(dones).float().to(self.device)

        # current Q
        q_vals = self.q_network(states)
        q_vals = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        # target Q
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = F.smooth_l1_loss(q_vals, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # update target network
        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())


if __name__ == "__main__":
    # setup env
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = ResizeObservation(env, (84, 84))
    env = FrameStack(env, 4)

    state_shape = env.observation_space.shape  # (4,84,84,1)
    n_actions = env.action_space.n

    agent = RainbowDQNAgent(state_shape, n_actions)

    print(f"Device = {agent.device}")

    num_episodes = 10000
    max_steps = 10000
    for ep in tqdm(range(num_episodes), desc="Episodes"):
        state = env.reset()
        total_reward = 0
        for t in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.cache(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            if done:
                break
        tqdm.write(
            f"Episode {ep} - Reward: {total_reward:.2f} - Epsilon: {agent.epsilon:.4f}"
        )
