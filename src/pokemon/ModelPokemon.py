import math
import os
import torch
import torch.nn as nn

from pokemon import Emulator


def get_model(device: str, name: str | None = None):
    emulator = Emulator.Emulator()

    inputs = emulator.data.inputs()

    continuous_dim = len(inputs["continuous"])

    single_embed_dim = 16 + 16 + 4 + 16 + 16 + 16

    multi_embed_dim = (
        len(inputs["move_id"]) * 16
        + len(inputs["move_type"]) * 16
        + len(inputs["pokemon_id"]) * 16
        + len(inputs["pokemon_type"]) * 16
        + len(inputs["sprite_id"]) * 16
        + len(inputs["item_id"]) * 16
        + len(inputs["sprite_data_movement_statuses"]) * 2
        + len(inputs["sprite_data_facing_directions"]) * 4
    )

    total_in_dim = continuous_dim + single_embed_dim + multi_embed_dim

    model = ModelPokemon(
        in_dim=total_in_dim,
        outputs=len(emulator.buttons),
    ).to(device)

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
    def __init__(
        self,
        in_dim: int,
        outputs: int,
    ):
        super().__init__()

        last_action_output = int(math.sqrt(outputs))

        in_dim = in_dim + last_action_output

        self.last_action = nn.Embedding(outputs, last_action_output)
        self.map_id = nn.Embedding(256, 16)
        self.dialog_id = nn.Embedding(256, 16)
        self.index_of_current_pokemon_send_out = nn.Embedding(6, 4)
        self.type_of_battle = nn.Embedding(256, 16)
        self.move_menu_type = nn.Embedding(256, 16)
        self.current_menu_selected_item = nn.Embedding(256, 16)

        self.move_id = nn.Embedding(256, 16, padding_idx=0)
        self.move_type = nn.Embedding(256, 16, padding_idx=0)
        self.pokemon_id = nn.Embedding(256, 16, padding_idx=0)
        self.pokemon_type = nn.Embedding(256, 16, padding_idx=0)
        self.sprite_id = nn.Embedding(256, 16, padding_idx=0)
        self.item_id = nn.Embedding(256, 16, padding_idx=0)
        self.sprite_data_movement_statuses = nn.Embedding(4, 2)
        self.sprite_data_facing_directions = nn.Embedding(13, 4)

        self.trunk = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )

        self.value = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, outputs),
        )

    def _as_float_batch(self, t, device):
        t = t.to(device)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t.float()

    def _as_long_scalar_batch(self, t, device):
        t = t.to(device)
        if t.dtype != torch.long:
            t = t.long()
        if t.dim() == 0:
            t = t.unsqueeze(0)
        return t

    def _as_long_seq_batch(self, t, device):
        t = t.to(device)
        if t.dtype != torch.long:
            t = t.long()
        if t.dim() == 1:
            t = t.unsqueeze(0)
        return t

    def forward(self, x):
        device = next(self.parameters()).device

        cont = self._as_float_batch(x["continuous"], device)

        last_action_emb = self.last_action(
            self._as_long_scalar_batch(x["last_action"], device)
        )
        map_id_emb = self.map_id(self._as_long_scalar_batch(x["map_id"], device))
        dialog_id_emb = self.dialog_id(
            self._as_long_scalar_batch(x["dialog_id"], device)
        )
        index_emb = self.index_of_current_pokemon_send_out(
            self._as_long_scalar_batch(x["index_of_current_pokemon_send_out"], device)
        )
        type_battle_emb = self.type_of_battle(
            self._as_long_scalar_batch(x["type_of_battle"], device)
        )
        move_menu_emb = self.move_menu_type(
            self._as_long_scalar_batch(x["move_menu_type"], device)
        )
        current_menu_selected_item_emb = self.current_menu_selected_item(
            self._as_long_scalar_batch(x["current_menu_selected_item"], device)
        )

        move_id_full = self.move_id(self._as_long_seq_batch(x["move_id"], device))
        move_id_emb = move_id_full.reshape(move_id_full.size(0), -1)

        move_type_full = self.move_type(self._as_long_seq_batch(x["move_type"], device))
        move_type_emb = move_type_full.reshape(move_type_full.size(0), -1)

        pokemon_id_full = self.pokemon_id(
            self._as_long_seq_batch(x["pokemon_id"], device)
        )
        pokemon_id_emb = pokemon_id_full.reshape(pokemon_id_full.size(0), -1)

        pokemon_type_full = self.pokemon_type(
            self._as_long_seq_batch(x["pokemon_type"], device)
        )
        pokemon_type_emb = pokemon_type_full.reshape(pokemon_type_full.size(0), -1)

        sprite_id_full = self.sprite_id(self._as_long_seq_batch(x["sprite_id"], device))
        sprite_id_emb = sprite_id_full.reshape(sprite_id_full.size(0), -1)

        item_id_full = self.item_id(self._as_long_seq_batch(x["item_id"], device))
        item_id_emb = item_id_full.reshape(item_id_full.size(0), -1)

        sprite_data_movement_statuses_full = self.sprite_data_movement_statuses(
            self._as_long_seq_batch(x["sprite_data_movement_statuses"], device)
        )
        sprite_data_movement_statuses_emb = sprite_data_movement_statuses_full.reshape(
            sprite_data_movement_statuses_full.size(0), -1
        )

        sprite_data_facing_directions_full = self.sprite_data_facing_directions(
            self._as_long_seq_batch(x["sprite_data_facing_directions"], device)
        )
        sprite_data_facing_directions_emb = sprite_data_facing_directions_full.reshape(
            sprite_data_facing_directions_full.size(0), -1
        )

        h = torch.cat(
            [
                cont,
                last_action_emb,
                map_id_emb,
                dialog_id_emb,
                index_emb,
                type_battle_emb,
                move_menu_emb,
                current_menu_selected_item_emb,
                move_id_emb,
                move_type_emb,
                pokemon_id_emb,
                pokemon_type_emb,
                sprite_id_emb,
                item_id_emb,
                sprite_data_movement_statuses_emb,
                sprite_data_facing_directions_emb,
            ],
            dim=1,
        )

        z = self.trunk(h)
        v = self.value(z)
        a = self.advantage(z)
        q = v + (a - a.mean(dim=1, keepdim=True))

        return q
