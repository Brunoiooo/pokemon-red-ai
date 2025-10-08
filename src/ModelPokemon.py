import torch
import torch.nn as nn


class ModelPokemon(nn.Module):
    def __init__(self, continuous_dim: int, outputs: int):
        super().__init__()

        self.map_emb = nn.Embedding(256, 32)
        self.dialog_emb = nn.Embedding(256, 16)
        self.pos_emb_x = nn.Embedding(256, 16)
        self.pos_emb_y = nn.Embedding(256, 16)
        self.mode_emb = nn.Embedding(4, 8)

        total_emb_dim = 32 + 16 + 16 + 16 + 8

        self.fc = nn.Sequential(
            nn.Linear(continuous_dim + total_emb_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, outputs),
        )

    def forward(self, x):
        emb = torch.cat(
            [
                self.map_emb(x["map_id"]),
                self.dialog_emb(x["dialog_id"]),
                self.pos_emb_x(x["pos_x"]),
                self.pos_emb_y(x["pos_y"]),
                self.mode_emb(x["mode"]),
                x["continuous"],
            ],
            dim=-1,
        )
        return self.fc(emb)
