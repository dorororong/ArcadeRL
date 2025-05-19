# network_light.py
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MiniCNN(BaseFeaturesExtractor):
    """
    - 입력: (C, H, W)   예) (3, 24, 48)
    - Conv 3층 + AdaptivePool → Linear(256)
    - 파라미터 ≈ 110 K
    """
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        c, h, w = observation_space.shape

        self.cnn = nn.Sequential(
            # ① stride 2 로 해상도 1/2
            nn.Conv2d(c, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.BatchNorm2d(32),

            # ② stride 2 로 1/4, 수용영역 7×7
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.BatchNorm2d(64),
            # ③ stride 1 로 정보 손실 없이 깊이만 확장, 수용영역 19×19
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),

            # ④ 가변 해상도 대응 용 Average Pooling
            nn.AdaptiveAvgPool2d(output_size=(3, 6)),  # (64, 3, 6) → 1152
            nn.Flatten()
        )

        with torch.no_grad():
            n_flatten = self.cnn(torch.zeros(1, c, h, w)).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # 0‑255 픽셀 → 0‑1 로 스케일
        return self.linear(self.cnn(observations / 255.0))
