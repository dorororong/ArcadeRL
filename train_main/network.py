# network.py

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        features_dim=512,
        observation_height=80,
        observation_width=80,
    ):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]

        # CNN 정의
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 4), stride=(2, 4)),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
        )

        # 출력 크기 계산
        with torch.no_grad():
            sample = torch.zeros(
                1, n_channels, observation_height, observation_width
            )
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        x = self.cnn(observations)
        return self.linear(x)