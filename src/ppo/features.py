"""SB3 feature extractor for Pokemon Red Dict observations."""
from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PokemonFeaturesExtractor(BaseFeaturesExtractor):
    """CNN on screen_tiles + visit_mask, MLP on flat vector features."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        features_dim: int = 256,
        screen_cnn_channels: tuple[int, ...] = (32, 64, 64),
        visit_cnn_channels: tuple[int, ...] = (16, 32),
        vector_mlp_hidden: tuple[int, ...] = (256, 128),
    ):
        super().__init__(observation_space, features_dim=features_dim)

        screen_space = observation_space.spaces["screen_tiles"]
        visit_space = observation_space.spaces["visit_mask"]
        vector_space = observation_space.spaces["vector"]

        # screen (1,18,20) + visit resized/padded path: process separately then fuse.
        self.screen_cnn = self._build_cnn(1, screen_cnn_channels)
        with torch.no_grad():
            screen_dim = self.screen_cnn(
                torch.zeros(1, *screen_space.shape)
            ).shape[1]

        self.visit_cnn = self._build_cnn(1, visit_cnn_channels)
        with torch.no_grad():
            visit_dim = self.visit_cnn(torch.zeros(1, *visit_space.shape)).shape[1]

        vector_dim = int(vector_space.shape[0])
        self.vector_mlp = self._build_mlp(vector_dim, vector_mlp_hidden)

        fused = screen_dim + visit_dim + (vector_mlp_hidden[-1] if vector_mlp_hidden else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fused, features_dim),
            nn.ReLU(),
        )

    @staticmethod
    def _build_cnn(in_channels: int, channels: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_channels
        for i, out_channels in enumerate(channels):
            stride = 1 if i == 0 else 2
            layers.append(
                nn.Conv2d(prev, out_channels, kernel_size=3, stride=stride, padding=1)
            )
            layers.append(nn.ReLU())
            prev = out_channels
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    @staticmethod
    def _build_mlp(in_dim: int, hidden: tuple[int, ...]) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        return nn.Sequential(*layers)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        screen = observations["screen_tiles"].float()
        visit = observations["visit_mask"].float()
        vector = observations["vector"].float()

        s = self.screen_cnn(screen)
        v = self.visit_cnn(visit)
        f = self.vector_mlp(vector)
        return self.fusion(torch.cat([s, v, f], dim=1))
