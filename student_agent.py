import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from torchvision import transforms as T
import os
from gym_super_mario_bros.actions import (
    COMPLEX_MOVEMENT,
)
from collections import deque

SKIP_FRAMES = 4
STACK_FRAMES = 4


def pil_to_tensor_no_scale(pil_img):
    """
    Converts a PIL Image to a PyTorch tensor (CxHxW) with dtype float32,
    keeping the original [0, 255] value range.
    """
    # Convert PIL image to NumPy array (HxWxC or HxW)
    img_np = np.array(pil_img)  # dtype will be uint8

    # Ensure it has a channel dimension if grayscale (needed for permute)
    if img_np.ndim == 2:
        img_np = np.expand_dims(img_np, axis=2)  # Shape: (H, W, 1)

    # Convert NumPy array to PyTorch Tensor (shares memory)
    # Shape: (H, W, C), dtype: torch.uint8
    img_tensor = torch.from_numpy(img_np)

    # Permute dimensions to CxHxW
    # Shape: (C, H, W), dtype: torch.uint8
    img_tensor = img_tensor.permute(2, 0, 1)

    # Convert to float tensor *without* scaling
    # Shape: (C, H, W), dtype: torch.float32
    img_tensor_float = img_tensor.float()
    # or img_tensor_float = img_tensor.to(torch.float32)

    return img_tensor_float


# --- Import necessary components directly from your training script ---
# Assuming your training script is named 'train.py' and in the same directory
# or accessible via Python's path.
try:
    # We only need the network definitions for the agent class itself
    from train import NoisyLinear, RainbowDQN

    print("Successfully imported NoisyLinear, RainbowDQN, and SkipFrame from train.py")
except ImportError as e:
    print(f"Warning: Error importing from train.py: {e}")
    print(
        "Attempting to proceed, but ensure 'train.py' is accessible and defines "
        "NoisyLinear, RainbowDQN."
    )
    # Define dummy classes if import fails, mainly for the example to potentially run
    # You MUST have the correct classes imported for the agent to function correctly.
    if "NoisyLinear" not in globals():

        class NoisyLinear(nn.Module):
            pass  # Placeholder

    if "RainbowDQN" not in globals():

        class RainbowDQN(nn.Module):
            pass  # Placeholder


# --- Agent Class Definition ---
class Agent:
    """
    Agent that loads a trained Rainbow DQN model and selects actions.
    Designed to be used with an evaluation script expecting an 'act' method.
    Assumes environment preprocessing (wrappers) is handled externally.
    """

    def __init__(
        self,
        v_min=-50,
        v_max=150,
        num_atoms=51,
    ):
        """
        Initializes the agent.

        Args:
            v_min (float): Minimum value of the distributional RL support. MUST match training.
            v_max (float): Maximum value of the distributional RL support. MUST match training.
            num_atoms (int): Number of atoms in the distributional RL support. MUST match training.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent using device: {self.device}")

        # Store config needed for action selection and validation
        self.state_shape = (4, 84, 84)
        self.n_actions = len(COMPLEX_MOVEMENT)  # Number of actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        # --- Network Initialization ---
        # Use the imported RainbowDQN definition
        # Ensure the class was actually imported or defined
        if not issubclass(RainbowDQN, nn.Module) or RainbowDQN is nn.Module:
            raise RuntimeError(
                "RainbowDQN class definition not found or invalid. "
                "Ensure train.py is accessible and defines it correctly."
            )

        self.policy_net = RainbowDQN(
            self.state_shape, self.n_actions, num_atoms, v_min, v_max
        ).to(self.device)

        # --- Load Model Weights ---
        if not self._load_model("rainbow_mario_model_final.pth"):
            raise RuntimeError(
                "Failed to load model from rainbow_mario_model_final.pth"
            )

        # --- Set to Evaluation Mode ---
        # CRITICAL: Disables dropout, batch norm updates, and makes NoisyLinear
        # layers use their mean weights for deterministic actions during inference.
        self.policy_net.eval()
        print("Agent initialized and model set to evaluation mode.")
        self.transform = T.Compose(
            [
                T.ToPILImage(),
                T.Grayscale(num_output_channels=1),
                T.Resize((84, 84)),
                T.Lambda(pil_to_tensor_no_scale),
            ]
        )
        self.last_action = 0
        self.frames = deque(maxlen=STACK_FRAMES)
        self.skip_count = SKIP_FRAMES - 1

    def _load_model(self, filepath):
        """Loads policy network weights from a checkpoint file."""
        if not os.path.isfile(filepath):
            print(f"Error: Checkpoint file not found at {filepath}")
            return False

        print(f"Loading model checkpoint from {filepath}...")
        try:
            # Load the state dict, mapping to the correct device
            checkpoint = torch.load(filepath, map_location=self.device)

            # Determine the state dict to load
            state_dict_to_load = None
            if isinstance(checkpoint, dict) and "policy_net_state_dict" in checkpoint:
                state_dict_to_load = checkpoint["policy_net_state_dict"]
                print("Found 'policy_net_state_dict' in checkpoint.")
            elif isinstance(checkpoint, dict):
                # Check if the checkpoint itself is the state_dict (might happen with older saves)
                # A simple heuristic: check if keys look like model parameters
                if all(
                    isinstance(k, str) and isinstance(v, torch.Tensor)
                    for k, v in checkpoint.items()
                ):
                    state_dict_to_load = checkpoint
                    print("Checkpoint appears to be a state_dict directly.")
                else:
                    print(
                        "Warning: Checkpoint is a dictionary but doesn't contain 'policy_net_state_dict' key "
                        "and doesn't look like a raw state_dict. Attempting to load keys matching the model."
                    )
                    # Fallback: try loading matching keys if it's some other dict format
                    state_dict_to_load = checkpoint

            else:
                # Assume the loaded object *is* the state_dict
                state_dict_to_load = checkpoint
                print("Checkpoint loaded directly as state_dict.")

            if state_dict_to_load is None:
                print(
                    "Error: Could not determine the state dictionary to load from the checkpoint."
                )
                return False

            # --- Load the state dict ---
            # Use strict=False to ignore missing keys like "support".
            # The "support" buffer is initialized in RainbowDQN.__init__ and doesn't
            # strictly need to be loaded from the checkpoint.
            # Other missing/unexpected keys will also be ignored.
            incompatible_keys = self.policy_net.load_state_dict(
                state_dict_to_load, strict=False
            )

            # Print warnings about mismatches (useful for debugging)
            if incompatible_keys.missing_keys:
                print(
                    f"Warning: Missing keys when loading state_dict: {incompatible_keys.missing_keys}"
                )
                # Specifically check if 'support' was missing, which is expected
                if "support" in incompatible_keys.missing_keys:
                    print("('support' key was missing, which is expected and handled.)")
            if incompatible_keys.unexpected_keys:
                print(
                    f"Warning: Unexpected keys when loading state_dict: {incompatible_keys.unexpected_keys}"
                )

            print("Model weights loaded successfully (strict=False).")
            return True

        except Exception as e:
            print(f"Error loading checkpoint from {filepath}: {e}")
            return False

    def act(self, observation: np.ndarray):
        """
        Selects an action based on the given observation.

        Args:
            observation (np.ndarray or LazyFrames): The current environment state observation.
                                                    Expected shape should match `state_shape`
                                                    provided during initialization.

        Returns:
            int: The chosen action index.
        """

        # Convert state to tensor, add batch dimension, move to device
        # Ensure the observation is float32, as expected by Conv2d layers
        # Normalization (dividing by 255) happens inside the RainbowDQN forward pass
        img = self.transform(observation).squeeze(0)

        # Frame stacking
        while len(self.frames) < STACK_FRAMES:
            self.frames.append(img)
        self.frames.append(img)

        # Skip Frames to ensure FPS is consistent with training
        if self.skip_count > 0:
            self.skip_count -= 1
            return self.last_action

        # --- Action Selection ---
        # Disable gradient calculations for inference
        state = np.stack(self.frames, axis=0)  # Shape: [STACK_FRAMES, 84, 84]
        state = (
            torch.tensor(state).unsqueeze(0).to(self.device)
        )  # Shape: [1, STACK_FRAMES, 84, 84]
        with torch.no_grad():
            # Get action distributions from the network
            # The network's forward pass handles normalization and feature extraction
            dist = self.policy_net(state)  # Shape: [1, n_actions, n_atoms]

            # Calculate expected Q-values: sum(probability * support_value)
            # The support vector is registered as a buffer in RainbowDQN and moved to device
            # Ensure support is available and on the correct device
            support = self.policy_net.support.view(1, 1, self.num_atoms)
            expected_value = (dist * support).sum(
                dim=2
            )  # Sum over the atoms dimension. Shape: [1, n_actions]

            # Choose action with the highest expected Q-value
            action = expected_value.argmax(dim=1).item()  # Get the index (action)
        # print(action)
        self.last_action = action
        self.skip_count = SKIP_FRAMES - 1
        return action
