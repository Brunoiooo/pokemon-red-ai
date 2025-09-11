import torch.nn as nn


class ModelPokemon(nn.Module):
    def __init__(self, inputs: int, outputs: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(inputs, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, outputs),
        )

    def forward(self, x):
        return self.model(x)
