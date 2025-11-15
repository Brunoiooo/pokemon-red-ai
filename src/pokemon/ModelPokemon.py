import os
import torch
import torch.nn as nn

from pokemon import Emulator


def get_model(device: str, name: str | None = None):
    emulator = Emulator.Emulator()
    model = ModelPokemon(len(emulator.data.data()), len(emulator.buttons)).to(device)
    emulator.pyboy.stop(False)

    if name is None:
        return model

    ckpt_path = f"models/{name}.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    model.load_state_dict(
        (
            state["model_state"]
            if isinstance(state, dict) and "model_state" in state
            else state
        ),
        strict=True,
    )

    return model


class ModelPokemon(nn.Module):
    def __init__(self, continuous_dim: int, outputs: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.LayerNorm(continuous_dim),
            nn.Linear(continuous_dim, 1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, outputs),
        )

    def _as_float_batch(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.float()

    def forward(self, x):
        return self.fc(self._as_float_batch(x["continuous"]))
