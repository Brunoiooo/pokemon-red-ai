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

        self.map_emb = nn.Embedding(256, 32)
        self.dialog_emb = nn.Embedding(256, 32)
        self.pos_emb_x = nn.Embedding(256, 32)
        self.pos_emb_y = nn.Embedding(256, 32)
        self.mode_emb = nn.Embedding(5, 8)

        total_emb_dim = 32 + 32 + 32 + 32 + 8

        self.fc = nn.Sequential(
            nn.LayerNorm(continuous_dim + total_emb_dim),
            nn.Linear(continuous_dim + total_emb_dim, 1024),
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

    def _as_long_batch(self, t, device):
        t = t.to(device)
        if t.dtype != torch.long:
            t = t.long()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return t

    def _as_float_batch(self, t, device):
        t = t.to(device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.float()

    def forward(self, x):
        device = next(self.parameters()).device

        map_id = self._as_long_batch(x["map_id"], device)
        dialog_id = self._as_long_batch(x["dialog_id"], device)
        pos_x = self._as_long_batch(x["pos_x"], device)
        pos_y = self._as_long_batch(x["pos_y"], device)
        mode = self._as_long_batch(x["mode"], device)

        cont = self._as_float_batch(x["continuous"], device)

        emb = torch.cat(
            [
                self.map_emb(map_id),
                self.dialog_emb(dialog_id),
                self.pos_emb_x(pos_x),
                self.pos_emb_y(pos_y),
                self.mode_emb(mode),
            ],
            dim=-1,
        )

        z = torch.cat([cont, emb], dim=-1)
        return self.fc(z)
