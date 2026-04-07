import math
from multiprocessing.synchronize import RLock
import os
import torch
import torch.nn as nn

from pokemon import Emulator


def get_model(device: str, files_lock: RLock, name: str | None = None):
    emulator = Emulator.Emulator(files_lock=files_lock)

    inputs = emulator.data.inputs()

    single_embed_dim = 16 + 16 + 4 + 16 + 16

    multi_embed_dim = (
        len(inputs["move_id"]) * 16
        + len(inputs["move_type"]) * 16
        + len(inputs["pokemon_id"]) * 16
        + len(inputs["pokemon_type"]) * 16
        + len(inputs["sprite_id"]) * 16
        + len(inputs["item_id"]) * 16
    )

    total_in_dim = single_embed_dim + multi_embed_dim

    model = ModelPokemon(
        in_dim=total_in_dim,
        core_in=len(inputs["core"]),
        battle_in=len(inputs["battle"]),
        menu_battle_dialog_in=len(inputs["menu_battle_dialog"]),
        progress_in=len(inputs["progress"]),
        mode_in=len(inputs["mode"]),
        nav_in=len(inputs["nav"]),
        inv_in=len(inputs["inv"]),
        party_in=len(inputs["party"]),
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
    FLOAT_INPUTS = {
        "core",
        "battle",
        "menu_battle_dialog",
        "progress",
        "nav",
        "inv",
        "party",
        "mode",
    }

    def __init__(
        self,
        in_dim: int,
        core_in: int,
        battle_in: int,
        menu_battle_dialog_in: int,
        progress_in: int,
        mode_in: int,
        nav_in: int,
        inv_in: int,
        party_in: int,
        outputs: int,
    ):
        super().__init__()

        self.screen_out_dim = 32

        total_in_dim = (
            in_dim
            + core_in
            + battle_in
            + menu_battle_dialog_in
            + progress_in
            + mode_in
            + nav_in
            + inv_in
            + party_in
            + self.screen_out_dim
        )

        self.map_id = nn.Embedding(256, 16)
        self.dialog_id = nn.Embedding(256, 16)
        self.index_of_current_pokemon_send_out = nn.Embedding(6, 4)
        self.type_of_battle = nn.Embedding(256, 16)
        self.move_menu_type = nn.Embedding(256, 16)

        self.move_id = nn.Embedding(256, 16, padding_idx=0)
        self.move_type = nn.Embedding(256, 16, padding_idx=0)
        self.pokemon_id = nn.Embedding(256, 16, padding_idx=0)
        self.pokemon_type = nn.Embedding(256, 16, padding_idx=0)
        self.sprite_id = nn.Embedding(256, 16, padding_idx=0)
        self.item_id = nn.Embedding(256, 16, padding_idx=0)

        self.screen_enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(16 * 18 * 20, self.screen_out_dim),
            nn.SiLU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(total_in_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )

        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        self.advantage_head = nn.Sequential(
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

        core = self._as_float_batch(x["core"], device)
        battle = self._as_float_batch(x["battle"], device)
        menu_battle_dialog = self._as_float_batch(x["menu_battle_dialog"], device)
        mode = self._as_float_batch(x["mode"], device)
        progress = self._as_float_batch(x["progress"], device)
        nav = self._as_float_batch(x["nav"], device)
        inv = self._as_float_batch(x["inv"], device)
        party = self._as_float_batch(x["party"], device)

        float_features = torch.cat(
            [
                core,
                battle,
                menu_battle_dialog,
                mode,
                progress,
                nav,
                inv,
                party,
            ],
            dim=1,
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

        screen = self._as_float_batch(x["screen_tiles"], device)
        if screen.dim() == 3:
            screen = screen.unsqueeze(1)
        screen_feat = self.screen_enc(screen)

        h = torch.cat(
            [
                float_features,
                map_id_emb,
                dialog_id_emb,
                index_emb,
                type_battle_emb,
                move_menu_emb,
                move_id_emb,
                move_type_emb,
                pokemon_id_emb,
                pokemon_type_emb,
                sprite_id_emb,
                item_id_emb,
                # sprite_data_movement_statuses_emb,
                # sprite_data_facing_directions_emb,
                screen_feat,
            ],
            dim=1,
        )

        z = self.trunk(h)

        v = self.value_head(z)
        a = self.advantage_head(z)
        q = v + (a - a.mean(dim=1, keepdim=True))

        return q

    def freeze_representation(self):
        modules = [
            self.map_id,
            self.dialog_id,
            self.index_of_current_pokemon_send_out,
            self.type_of_battle,
            self.move_menu_type,
            self.move_id,
            self.move_type,
            self.pokemon_id,
            self.pokemon_type,
            self.sprite_id,
            self.item_id,
            self.screen_enc,
            self.trunk,
        ]

        for module in modules:
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze(self):
        self.freeze_representation()

        for module in [self.value_head, self.advantage_head]:
            for p in module.parameters():
                p.requires_grad = True
