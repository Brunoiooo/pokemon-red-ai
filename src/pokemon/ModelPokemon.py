import math
import os
import torch
import torch.nn as nn

from pokemon import Emulator


def get_model(device: str, name: str | None = None):
    emulator = Emulator.Emulator()

    inputs = emulator.data.inputs()

    single_embed_dim = 16 + 16 + 4 + 16 + 16

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

    total_in_dim = single_embed_dim + multi_embed_dim

    model = ModelPokemon(
        in_dim=total_in_dim,
        core_in=len(inputs["core"]),
        battle_in=len(inputs["battle"]),
        menu_battle_dialog_in=len(inputs["menu_battle_dialog"]),
        dialog_world_in=len(inputs["dialog_world"]),
        progress_in=len(inputs["progress"]),
        mode_in=len(inputs["mode"]),
        nav_in=len(inputs["nav"]),
        inv_in=len(inputs["inv"]),
        party_in=len(inputs["party"]),
        outputs=len(emulator.buttons),
        outputs_max=len(inputs["last_actions"]),
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
        "dialog_world",
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
        dialog_world_in: int,
        progress_in: int,
        mode_in: int,
        nav_in: int,
        inv_in: int,
        party_in: int,
        outputs: int,
        outputs_max: int,
    ):
        super().__init__()

        last_action_output = int(math.sqrt(outputs))

        in_dim = (
            in_dim
            + 128
            + 64
            + 64
            + 64
            + 64
            + 64
            + 32
            + 64
            + 64
            + 128
            + (outputs_max * last_action_output)
        )

        self.last_actions = nn.Embedding(outputs, last_action_output)
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
        self.sprite_data_movement_statuses = nn.Embedding(4, 2)
        self.sprite_data_facing_directions = nn.Embedding(13, 4)

        self.screen_enc = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(16 * 18 * 20, 128),
            nn.SiLU(),
        )

        self.core_enc = nn.Sequential(
            nn.Linear(core_in, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

        self.battle_enc = nn.Sequential(
            nn.Linear(battle_in, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

        self.menu_battle_dialog_enc = nn.Sequential(
            nn.Linear(menu_battle_dialog_in, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

        self.dialog_world_enc = nn.Sequential(
            nn.Linear(dialog_world_in, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

        self.progress_enc = nn.Sequential(
            nn.Linear(progress_in, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

        self.mode_enc = nn.Sequential(
            nn.Linear(mode_in, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
        )

        self.nav_enc = nn.Sequential(
            nn.Linear(nav_in, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

        self.inv_enc = nn.Sequential(
            nn.Linear(inv_in, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
            nn.SiLU(),
        )

        self.party_enc = nn.Sequential(
            nn.Linear(party_in, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
        )

        self.trunk = nn.Sequential(
            nn.Linear(in_dim, 2048),
            nn.LayerNorm(2048),
            nn.SiLU(),
            nn.Dropout(p=0.01),
            nn.Linear(2048, 1024),
            nn.SiLU(),
            nn.Dropout(p=0.01),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(p=0.01),
        )

        self.value = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 1),
        )

        self.advantage = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, outputs),
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
        core = self.core_enc(core)

        battle = self._as_float_batch(x["battle"], device)
        battle = self.battle_enc(battle)

        menu_battle_dialog = self._as_float_batch(x["menu_battle_dialog"], device)
        menu_battle_dialog = self.menu_battle_dialog_enc(menu_battle_dialog)

        dialog_world = self._as_float_batch(x["dialog_world"], device)
        dialog_world = self.dialog_world_enc(dialog_world)

        mode = self._as_float_batch(x["mode"], device)
        mode = self.mode_enc(mode)

        progress = self._as_float_batch(x["progress"], device)
        progress = self.progress_enc(progress)

        nav = self._as_float_batch(x["nav"], device)
        nav = self.nav_enc(nav)

        inv = self._as_float_batch(x["inv"], device)
        inv = self.inv_enc(inv)

        party = self._as_float_batch(x["party"], device)
        party = self.party_enc(party)

        last_actions_emb = self.last_actions(
            self._as_long_seq_batch(x["last_actions"], device)
        )
        last_actions_emb = last_actions_emb.reshape(last_actions_emb.size(0), -1)

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

        screen = self._as_float_batch(x["screen_tiles"], device)
        if screen.dim() == 3:
            screen = screen.unsqueeze(1)
        screen_feat = self.screen_enc(screen)

        h = torch.cat(
            [
                core,
                battle,
                menu_battle_dialog,
                dialog_world,
                mode,
                progress,
                nav,
                inv,
                party,
                last_actions_emb,
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
                sprite_data_movement_statuses_emb,
                sprite_data_facing_directions_emb,
                screen_feat,
            ],
            dim=1,
        )

        z = self.trunk(h)
        v = self.value(z)
        a = self.advantage(z)
        q = v + (a - a.mean(dim=1, keepdim=True))

        return q
