import os
import torch
import torch.nn as nn

from pokemon import Emulator


def get_model(device: str, name: str | None = None):
    emulator = Emulator.Emulator()

    inputs = emulator.data.inputs()

    in_dim = len(inputs["continuous"])
    in_dim += 16
    in_dim += 16
    in_dim += 4
    in_dim += 16
    in_dim += 16
    in_dim += 16
    in_dim += 16
    in_dim += 16
    in_dim += 16
    in_dim += 16
    in_dim += 16

    model = ModelPokemon(in_dim, len(emulator.buttons)).to(device)
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
    def __init__(self, dim: int, outputs: int):
        super().__init__()

        self.map_id = nn.Embedding(512, 16)
        self.dialog_id = nn.Embedding(512, 16)
        self.index_of_current_pokemon_send_out = nn.Embedding(7, 4, padding_idx=0)
        self.type_of_battle = nn.Embedding(512, 16)
        self.move_menu_type = nn.Embedding(512, 16)

        self.move_id = nn.Embedding(512, 16)
        self.move_type = nn.Embedding(512, 16)
        self.pokemon_id = nn.Embedding(512, 16)
        self.pokemon_type = nn.Embedding(512, 16)
        self.sprite_id = nn.Embedding(512, 16)
        self.item_id = nn.Embedding(512, 16)

        self.fc = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1024),
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

    def _as_float_batch(self, t, device):
        t = t.to(device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.float()

    def _as_long_batch(self, t, device):
        t = t.to(device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.long()

    def forward(self, x):
        device = next(self.parameters()).device

        cont = self._as_float_batch(x["continuous"], device)

        map_id_emb = self.map_id(self._as_long_batch(x["map_id"], device))
        dialog_id_emb = self.dialog_id(self._as_long_batch(x["dialog_id"], device))
        index_pokemon_emb = self.index_of_current_pokemon_send_out(
            self._as_long_batch(x["index_of_current_pokemon_send_out"], device)
        )
        type_battle_emb = self.type_of_battle(
            self._as_long_batch(x["type_of_battle"], device)
        )
        move_menu_emb = self.move_menu_type(
            self._as_long_batch(x["move_menu_type"], device)
        )
        move_id_emb = self.move_id(self._as_long_batch(x["move_id"], device)).mean(
            dim=1
        )
        move_type_emb = self.move_type(
            self._as_long_batch(x["move_type"], device)
        ).mean(dim=1)
        pokemon_id_emb = self.pokemon_id(
            self._as_long_batch(x["pokemon_id"], device)
        ).mean(dim=1)
        pokemon_type_emb = self.pokemon_type(
            self._as_long_batch(x["pokemon_type"], device)
        ).mean(dim=1)
        sprite_id_emb = self.sprite_id(
            self._as_long_batch(x["sprite_id"], device)
        ).mean(dim=1)
        item_id_emb = self.item_id(self._as_long_batch(x["item_id"], device)).mean(
            dim=1
        )

        return self.fc(
            torch.cat(
                [
                    cont,
                    map_id_emb,
                    dialog_id_emb,
                    index_pokemon_emb,
                    type_battle_emb,
                    move_menu_emb,
                    move_id_emb,
                    move_type_emb,
                    pokemon_id_emb,
                    pokemon_type_emb,
                    sprite_id_emb,
                    item_id_emb,
                ],
                dim=1,
            )
        )
