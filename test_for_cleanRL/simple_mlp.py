import torch
import torch.nn as nn
from gymnasium import spaces

# ① 특징 추출기 (2층 MLP)
class MLPExtractor(nn.Module):
    def __init__(self, input_dim: int, features_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),
            nn.Linear(256, features_dim), nn.ReLU(),
        )
        self.out_dim = features_dim

    def forward(self, x):
        return self.net(x)

# ② 듀얼링 Q‑네트워크
class DuelingMLP(nn.Module):
    """
    - observation: 1‑D 벡터 (e.g., 11개 상태 변수)
    - action_space: gymnasium.spaces.Discrete
    """
    def __init__(self, obs_space: spaces.Space, action_space: spaces.Discrete, features_dim: int = 128):
        super().__init__()
        self.extractor = MLPExtractor(obs_space.shape[0], features_dim)
        self.value = nn.Sequential(
            nn.Linear(features_dim, 128), nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(features_dim, 128), nn.ReLU(),
            nn.Linear(128, action_space.n)
        )

    def forward(self, x):
        x = self.extractor(x)
        v = self.value(x)
        a = self.advantage(x)
        return v + (a - a.mean(dim=1, keepdim=True))
