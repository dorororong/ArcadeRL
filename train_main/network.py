# network.py
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    화면 크기: (채널, 64, 32)
    - 첫 MaxPool은 (2,1) → 높이만 절반, 너비는 그대로
    - 두 번째부터 (2,2) 풀링
    최종 해상도: 64→32→16→8  (H) / 32→32→16→8  (W)
    """
    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        # 1) observation_space 에서 채널·높이·너비를 동적으로 가져옴
        #    (shape == (C, H, W) 인 경우)
        c, h, w = observation_space.shape

        # 2) 그대로 Conv → Pooling 구조 정의
        self.cnn = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
        )

        # 3) 실제 h, w 를 써서 flatten 차원 계산
        with torch.no_grad():
            sample = torch.zeros(1, c, h, w)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.linear(self.cnn(x))