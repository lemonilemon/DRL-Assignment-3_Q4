import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from torchvision import transforms as T
from typing import Tuple, Deque, Optional

# Assuming these are defined in 'train.py' or accessible elsewhere
from train import RainbowDQN, NoisyLinear, COMPLEX_MOVEMENT


class Agent:
    """
    Agent that interacts with the environment using a pre-trained Rainbow DQN model.

    Handles frame preprocessing, frame stacking, action selection based on the model,
    and frame skipping.
    """

    def __init__(
        self,
        checkpoint_path: str = "rainbow_mario_model_final.pth",
        input_shape: Tuple[int, int, int] = (4, 84, 90),  # (Channels, Height, Width)
        n_actions: int = len(COMPLEX_MOVEMENT),
        skip_frames: int = 4,
        device: Optional[str] = None,
    ):
        """
        Initializes the Agent.

        Args:
            checkpoint_path: Path to the saved model checkpoint.
            input_shape: The shape of the stacked input frames (C, H, W).
            n_actions: The number of possible actions.
            skip_frames: The number of frames to repeat the last action for.
            device: The device to run the model on ('cuda', 'cpu', or None for auto-detect).
        """
        self.n_actions = n_actions
        self.skip_frames = max(1, skip_frames)  # Ensure at least 1 frame is processed
        self._frames_to_skip = self.skip_frames - 1  # Internal counter logic

        # Determine device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize the model
        # Note: Ensure RainbowDQN and NoisyLinear are correctly defined/imported
        self.model = RainbowDQN(
            input_shape=input_shape,
            n_actions=self.n_actions,
            noisy_sigma_init=0.5,  # Example value, adjust if needed
        ).to(self.device)

        # Load the checkpoint
        try:
            # Load checkpoint, supporting both full state dict and nested dicts
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            # Check if the checkpoint is a dict containing the model state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model_state = checkpoint["model_state_dict"]
            elif (
                isinstance(checkpoint, dict) and "model" in checkpoint
            ):  # Support original format
                model_state = checkpoint["model"]
            else:
                model_state = checkpoint  # Assume checkpoint is the state_dict itself

            self.model.load_state_dict(model_state)
            print(f"Successfully loaded model from {checkpoint_path}")
        except FileNotFoundError:
            print(f"Error: Checkpoint file not found at {checkpoint_path}.")
            print("Using randomly initialized model instead.")
        except Exception as e:
            print(f"Error loading checkpoint from {checkpoint_path}: {e}")
            print("Using randomly initialized model instead.")

        self.model.eval()  # Set model to evaluation mode (important!)

        # Preprocessing transform: Grayscale -> Resize -> ToTensor
        self.transform = T.Compose(
            [
                T.ToPILImage(),  # Convert numpy array to PIL Image
                T.Grayscale(),  # Convert to grayscale
                T.Resize(
                    (input_shape[1], input_shape[2]),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),  # Resize (H, W)
                T.ToTensor(),  # Convert PIL Image to PyTorch Tensor (scales to [0, 1])
            ]
        )

        # Frame stack (stores the last 4 processed frames)
        self._frame_stack: Deque[np.ndarray] = deque(maxlen=input_shape[0])
        self._state_initialized: bool = False  # Flag to check if frame stack is ready
        self._skip_counter: int = 0
        self._last_action: int = 0  # Default to a safe action (e.g., NOOP)

    def _preprocess_observation(self, observation: np.ndarray) -> torch.Tensor:
        """
        Preprocesses a single observation frame.

        Args:
            observation: The raw observation from the environment (H, W, C).

        Returns:
            A processed frame as a PyTorch tensor (1, H, W).
        """
        # Ensure contiguous array if needed (often helps with PyTorch transforms)
        if not observation.flags["C_CONTIGUOUS"]:
            observation = np.ascontiguousarray(observation)
        # Apply the transformations
        processed_frame = self.transform(observation)  # Output shape: (1, H, W)
        return processed_frame

    def _initialize_state(self, initial_observation: np.ndarray):
        """Initializes the frame stack with the first observation."""
        processed_frame = self._preprocess_observation(initial_observation)
        # Fill the deque with the first frame
        for _ in range(self._frame_stack.maxlen):
            self._frame_stack.append(processed_frame)
        self._state_initialized = True
        print("Frame stack initialized.")

    def act(self, observation: np.ndarray) -> int:
        """
        Selects an action based on the current observation.

        Args:
            observation: The current observation from the environment.

        Returns:
            The selected action index.
        """
        # 1. Initialize frame stack on the very first call
        if not self._state_initialized:
            self._initialize_state(observation)
            # Need to compute the first action immediately after initialization
            # Fall through to action computation

        # 2. Handle frame skipping
        if self._skip_counter > 0:
            self._skip_counter -= 1
            # Repeat the last action
            return self._last_action

        # 3. Preprocess the new frame and update the stack
        processed_frame = self._preprocess_observation(observation)
        self._frame_stack.append(processed_frame)

        # 4. Stack frames and prepare model input
        # Concatenate tensors along the channel dimension (dim=0)
        # The deque stores tensors of shape (1, H, W), stacking makes (C, H, W)
        state_tensor = (
            torch.cat(list(self._frame_stack), dim=0).unsqueeze(0).to(self.device)
        )
        # Final shape: (1, C, H, W) - batch dim added

        # 5. Get action from the model
        with torch.no_grad():  # Disable gradient calculation for inference
            q_values = self.model(state_tensor)
            action = q_values.argmax(dim=1).item()  # Get index of max Q-value

        # 6. Update state for next steps
        self._last_action = action
        self._skip_counter = self._frames_to_skip  # Reset skip counter

        # Optional: Clean up tensor to free memory if needed, though Python's GC usually handles it.
        # del state_tensor, q_values

        return action

    def reset(self):
        """Resets the agent's internal state for a new episode."""
        self._frame_stack.clear()
        self._state_initialized = False
        self._skip_counter = 0
        self._last_action = 0  # Reset to default action
        print("Agent state reset for new episode.")


# Example Usage (requires gym environment)
if __name__ == "__main__":
    # --- Setup Environment (Example: Using a dummy environment) ---
    class DummyEnv:
        def __init__(self):
            # Example observation space (adjust to match Mario)
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=(240, 256, 3), dtype=np.uint8
            )
            # Example action space (matches COMPLEX_MOVEMENT length)
            self.action_space = gym.spaces.Discrete(len(COMPLEX_MOVEMENT))

        def reset(self):
            print("DummyEnv: Resetting environment.")
            # Return a dummy observation (e.g., black screen)
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

        def step(self, action):
            print(f"DummyEnv: Taking action {action}")
            # Return dummy next_state, reward, done, info
            next_obs = np.random.randint(
                0, 256, self.observation_space.shape, dtype=np.uint8
            )
            reward = np.random.rand()
            done = np.random.rand() > 0.95  # Randomly end episode
            info = {}
            return next_obs, reward, done, info

    # --- Initialize Agent and Run ---
    try:
        # Create a dummy checkpoint file if it doesn't exist for testing
        # In a real scenario, this file should contain trained weights
        dummy_checkpoint_path = "rainbow_mario_model_final.pth"
        if not torch.save(
            MockRainbowDQN((4, 84, 90), len(COMPLEX_MOVEMENT)).state_dict(),
            dummy_checkpoint_path,
        ):
            print(f"Created dummy checkpoint at {dummy_checkpoint_path}")

        # Initialize agent (adjust parameters if needed)
        agent = Agent(
            checkpoint_path=dummy_checkpoint_path,
            input_shape=(4, 84, 90),  # Should match model training
            n_actions=len(COMPLEX_MOVEMENT),
            skip_frames=4,
        )

        # Initialize dummy environment
        env = DummyEnv()
        obs = env.reset()

        # Run for a few steps
        total_reward = 0
        for step in range(20):
            action = agent.act(obs)
            print(
                f"Step {step + 1}: Agent selected action {action} ({COMPLEX_MOVEMENT[action]})"
            )

            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            print(f"   Reward: {reward:.4f}, Done: {done}")

            obs = next_obs

            if done:
                print("Episode finished.")
                agent.reset()  # Reset agent state
                obs = env.reset()  # Reset environment

        print(f"\nTotal reward over {step + 1} steps: {total_reward:.4f}")

    except ImportError as e:
        print(
            f"ImportError: {e}. Make sure 'train.py' with RainbowDQN/NoisyLinear/COMPLEX_MOVEMENT is accessible."
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
