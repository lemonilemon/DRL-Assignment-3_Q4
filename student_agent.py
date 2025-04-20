import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pickle
from train import RainbowDQN
from collections import deque


# Frame stacking utility
class FrameStack:
    def __init__(self, k=4):
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        self.frames.clear()

    def add_frame(self, frame):
        self.frames.append(frame)

    def get_stacked_frames(self):
        # Stack frames along the channel dimension
        if len(self.frames) < self.k:
            # If we don't have enough frames, duplicate the last one
            last_frame = (
                self.frames[-1] if self.frames else np.zeros((84, 84), dtype=np.float32)
            )
            while len(self.frames) < self.k:
                self.frames.append(last_frame)

        # Stack frames along first dimension for CNN input
        return np.stack(list(self.frames), axis=0)


# Do not modify the input of the 'act' function and the '__init__' function.
class Agent(object):
    """Agent that acts randomly."""

    def __init__(self):
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Rainbow DQN configuration
        self.input_shape = (4, 84, 84)  # 4 stacked frames, 84x84 each
        self.num_actions = 12
        self.num_atoms = 51
        self.v_min = -10
        self.v_max = 10
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(
            self.device
        )

        # Initialize model
        self.model = RainbowDQN(
            self.input_shape, self.num_actions, self.num_atoms, self.v_min, self.v_max
        ).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Load pre-trained model weights
        self._load_model("rainbow_mario_model_final.pth")

        # Frame stacker for preprocessing
        self.frame_stacker = FrameStack(k=4)

        # Initialize frame stack with empty frames
        self.frame_stacker.reset()
        for _ in range(4):
            self.frame_stacker.add_frame(np.zeros((84, 84), dtype=np.float32))

        print("Rainbow DQN Agent initialized")

    def _load_model(self, filepath):
        """Load the model from a saved file"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint["policy_net"])
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using random weights")

    def _preprocess_observation(self, observation):
        """Convert observation to grayscale and resize to 84x84"""
        # Handle various observation types
        if isinstance(observation, dict) and "pixels" in observation:
            # Handle dictionary observations (e.g., from Atari wrappers)
            observation = observation["pixels"]

        # Convert to grayscale if RGB
        if len(observation.shape) == 3 and observation.shape[2] == 3:
            # Simple RGB to grayscale conversion
            gray = np.mean(observation, axis=2).astype(np.float32)
        else:
            # Already grayscale
            gray = observation.astype(np.float32)

        # Simple resize to 84x84 using numpy (you might want to use a better method)
        # In a real implementation, consider using cv2.resize or similar
        from skimage.transform import resize

        resized = resize(gray, (84, 84), anti_aliasing=True)

        # Normalize to [0, 1]
        normalized = resized / 255.0

        return normalized

    def act(self, observation):
        """Select an action based on the observation"""
        # Preprocess observation
        processed_frame = self._preprocess_observation(observation)

        # Add to frame stack
        self.frame_stacker.add_frame(processed_frame)

        # Get stacked frames
        stacked_frames = self.frame_stacker.get_stacked_frames()

        # Convert to tensor
        state_tensor = torch.FloatTensor(stacked_frames).unsqueeze(0).to(self.device)

        # Get action from model
        with torch.no_grad():
            # Forward pass
            dist = self.model(state_tensor)

            # Calculate expected value for each action
            expected_value = dist * self.support.expand_as(dist)
            q_values = expected_value.sum(2)

            # Select action with highest Q-value
            action = q_values.argmax(1).item()

        return action


# Example usage
if __name__ == "__main__":
    agent = Agent()

    # Example observation (this would come from your environment)
    dummy_observation = np.random.randint(0, 255, (240, 256, 3), dtype=np.uint8)

    # Get action
    action = agent.act(dummy_observation)
    print(f"Selected action: {action}")
