import torch
import torch.nn as nn


class ModelPokemon(nn.Module):
    def __init__(self, continuous_dim: int, outputs: int):
        super().__init__()

        self.map_emb = nn.Embedding(256, 16)
        self.dialog_emb = nn.Embedding(256, 16)
        self.pos_emb_x = nn.Embedding(256, 16)
        self.pos_emb_y = nn.Embedding(256, 16)
        self.mode_emb = nn.Embedding(4, 8)

        # New battle-specific embeddings
        self.battle_type_emb = nn.Embedding(256, 16)  # D057/D05A battle type
        self.pokemon_status_emb = nn.Embedding(256, 16)  # Status conditions
        self.move_type_emb = nn.Embedding(256, 16)  # Move types
        self.pokemon_type_emb = nn.Embedding(256, 16)  # Pokemon types
        self.battle_state_emb = nn.Embedding(256, 32)  # Battle status flags

        total_emb_dim = 32 + 16 + 16 + 16 + 8 + 16 + 16 + 16 + 16 + 32

        self.fc = nn.Sequential(
            nn.Linear(continuous_dim + total_emb_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, outputs),
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

        battle_type = self._as_long_batch(x["battle_type"], device)
        pokemon_status = self._as_long_batch(x["pokemon_status"], device)
        move_type = self._as_long_batch(x["move_type"], device)
        pokemon_type = self._as_long_batch(x["pokemon_type"], device)
        battle_state = self._as_long_batch(x["battle_state"], device)

        cont = self._as_float_batch(x["continuous"], device)

        emb = torch.cat(
            [
                self.map_emb(map_id),
                self.dialog_emb(dialog_id),
                self.pos_emb_x(pos_x),
                self.pos_emb_y(pos_y),
                self.mode_emb(mode),
                self.battle_type_emb(battle_type),
                self.pokemon_status_emb(pokemon_status),
                self.move_type_emb(move_type),
                self.pokemon_type_emb(pokemon_type),
                self.battle_state_emb(battle_state),
            ],
            dim=-1,
        )

        z = torch.cat([cont, emb], dim=-1)
        return self.fc(z)
