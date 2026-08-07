"""SB3 feature extractor for Pokemon Red Dict observations."""
from __future__ import annotations

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PokemonFeaturesExtractor(BaseFeaturesExtractor):
    """CNN on screen_tiles + visit_mask + wild_visit_mask, MLP on flat vector features."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)

        screen_space = observation_space.spaces["screen_tiles"]
        visit_space = observation_space.spaces["visit_mask"]
        wild_visit_space = observation_space.spaces["wild_visit_mask"]
        vector_space = observation_space.spaces["vector"]

        # screen (1,18,20) + visit resized/padded path: process separately then fuse.
        self.screen_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            screen_dim = self.screen_cnn(
                torch.zeros(1, *screen_space.shape)
            ).shape[1]

        self.visit_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            visit_dim = self.visit_cnn(torch.zeros(1, *visit_space.shape)).shape[1]

        self.wild_visit_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            wild_visit_dim = self.wild_visit_cnn(
                torch.zeros(1, *wild_visit_space.shape)
            ).shape[1]

        vector_dim = int(vector_space.shape[0])
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        fused = screen_dim + visit_dim + wild_visit_dim + 128
        self.fusion = nn.Sequential(
            nn.Linear(fused, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        screen = observations["screen_tiles"].float()
        visit = observations["visit_mask"].float()
        wild_visit = observations["wild_visit_mask"].float()
        vector = observations["vector"].float()

        s = self.screen_cnn(screen)
        v = self.visit_cnn(visit)
        w = self.wild_visit_cnn(wild_visit)
        f = self.vector_mlp(vector)
        return self.fusion(torch.cat([s, v, w, f], dim=1))
