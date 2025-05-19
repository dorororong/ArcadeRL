# dueling_cnn.py
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete # Make sure Discrete is imported
from network_light import MiniCNN # Your CNN definition

class DuelingCNN(nn.Module):
    """
    Dueling Q-Network with MiniCNN as the feature extractor.
    """
    def __init__(self, observation_space: Box, action_space: Discrete, cnn_features_dim: int = 256):
        super().__init__()
        
        # The number of actions from the discrete action space
        self.action_dim = action_space.n

        # CNN feature extractor
        self.cnn = MiniCNN(observation_space, features_dim=cnn_features_dim)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(cnn_features_dim, 128), # Intermediate layer
            nn.ReLU(),
            nn.Linear(128, 1)  # Single value output
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(cnn_features_dim, 128), # Intermediate layer
            nn.ReLU(),
            nn.Linear(128, self.action_dim)  # One advantage value per action
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is expected to be (N, C, H, W)
        # The MiniCNN handles the scaling (division by 255.0)
        features = self.cnn(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine V(s) and A(s,a) using the Dueling DQN formula:
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # The mean is taken over the action dimension (dim=1 for (N, num_actions))
        q_vals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_vals

    # Optional: A method to get only features, can be useful
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)