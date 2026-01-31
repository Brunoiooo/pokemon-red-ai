import math
import os
import torch
import torch.nn as nn

from pokemon import Emulator


def get_model(
    device: str, name: str | None = None, gru_hidden: int = 512, gru_layers: int = 1
):
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
        gru_hidden=gru_hidden,
        gru_layers=gru_layers,
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
        gru_hidden: int = 512,
        gru_layers: int = 1,
    ):
        super().__init__()

        last_action_output = int(math.sqrt(outputs))

        per_step_in_dim = (
            in_dim
            + 32  # core_enc
            + 32  # battle_enc
            + 32  # menu_battle_dialog_enc
            + 32  # dialog_world_enc
            + 32  # progress_enc
            + 32  # mode_enc
            + 32  # nav_enc
            + 32  # inv_enc
            + 32  # party_enc
            + (outputs_max * last_action_output)  # last_actions emb (historia w kroku)
            + 16  # map_id
            + 16  # dialog_id
            + 4  # index_of_current_pokemon_send_out
            + 16  # type_of_battle
            + 16  # move_menu_type
            + 32  # screen_feat
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
            nn.Dropout2d(p=0.05),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.SiLU(),
            nn.Dropout2d(p=0.05),
            nn.Flatten(),
            nn.Linear(16 * 18 * 20, 32),
            nn.SiLU(),
            nn.Dropout(p=0.1),
        )

        self.core_enc = self.mlp_enc(core_in)
        self.battle_enc = self.mlp_enc(battle_in)
        self.menu_battle_dialog_enc = self.mlp_enc(menu_battle_dialog_in)
        self.dialog_world_enc = self.mlp_enc(dialog_world_in)
        self.progress_enc = self.mlp_enc(progress_in)
        self.mode_enc = self.mlp_enc(mode_in)
        self.nav_enc = self.mlp_enc(nav_in)
        self.inv_enc = self.mlp_enc(inv_in)
        self.party_enc = self.mlp_enc(party_in)

        self.pre_gru = nn.Sequential(
            nn.Linear(per_step_in_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Dropout(p=0.01),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Dropout(p=0.01),
        )

        self.gru = nn.GRU(
            input_size=512,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=0.0 if gru_layers == 1 else 0.01,
        )

        self.value_heads = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(gru_hidden, 256), nn.SiLU(), nn.Linear(256, 1))
                for _ in range(4)
            ]
        )
        self.advantage_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(gru_hidden, 256), nn.SiLU(), nn.Linear(256, outputs)
                )
                for _ in range(4)
            ]
        )

    def mlp_enc(self, in_f: int):
        return nn.Sequential(
            nn.Linear(in_f, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Dropout(p=0.05),
        )

    def _as_float(self, t, device):
        return t.to(device).float()

    def _as_long(self, t, device):
        t = t.to(device)
        return t.long() if t.dtype != torch.long else t

    def forward(self, x, h0=None, return_seq=False):
        device = next(self.parameters()).device

        B, T = x["core"].shape[0], x["core"].shape[1]

        def enc_float(key, enc):
            t = self._as_float(x[key], device)
            t2 = t.reshape(B * T, t.size(-1))
            e = enc(t2)
            return e.reshape(B, T, -1)

        core = enc_float("core", self.core_enc)
        battle = enc_float("battle", self.battle_enc)
        menu_battle_dialog = enc_float(
            "menu_battle_dialog", self.menu_battle_dialog_enc
        )
        dialog_world = enc_float("dialog_world", self.dialog_world_enc)
        mode = enc_float("mode", self.mode_enc)
        progress = enc_float("progress", self.progress_enc)
        nav = enc_float("nav", self.nav_enc)
        inv = enc_float("inv", self.inv_enc)
        party = enc_float("party", self.party_enc)

        last_actions = self._as_long(x["last_actions"], device)
        la_emb = self.last_actions(last_actions)
        la_emb = la_emb.reshape(B, T, -1)

        def emb_scalar(key, emb):
            ids = self._as_long(x[key], device)
            return emb(ids)

        map_id_emb = emb_scalar("map_id", self.map_id)
        dialog_id_emb = emb_scalar("dialog_id", self.dialog_id)
        index_emb = emb_scalar(
            "index_of_current_pokemon_send_out", self.index_of_current_pokemon_send_out
        )
        type_battle_emb = emb_scalar("type_of_battle", self.type_of_battle)
        move_menu_emb = emb_scalar("move_menu_type", self.move_menu_type)

        def emb_seq(key, emb):
            ids = self._as_long(x[key], device)
            e = emb(ids)
            return e.reshape(B, T, -1)

        move_id_emb = emb_seq("move_id", self.move_id)
        move_type_emb = emb_seq("move_type", self.move_type)
        pokemon_id_emb = emb_seq("pokemon_id", self.pokemon_id)
        pokemon_type_emb = emb_seq("pokemon_type", self.pokemon_type)
        sprite_id_emb = emb_seq("sprite_id", self.sprite_id)
        item_id_emb = emb_seq("item_id", self.item_id)
        sprite_data_movement_statuses_emb = emb_seq(
            "sprite_data_movement_statuses", self.sprite_data_movement_statuses
        )
        sprite_data_facing_directions_emb = emb_seq(
            "sprite_data_facing_directions", self.sprite_data_facing_directions
        )

        screen = self._as_float(x["screen_tiles"], device)
        if screen.dim() == 4:
            screen = screen.unsqueeze(2)
        screen2 = screen.reshape(B * T, 1, screen.size(-2), screen.size(-1))
        screen_feat = self.screen_enc(screen2).reshape(B, T, -1)

        h_step = torch.cat(
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
                la_emb,
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
            dim=2,
        )

        h2 = self.pre_gru(h_step.reshape(B * T, -1)).reshape(B, T, -1)

        z_seq, hT = self.gru(h2, h0)

        raw_mode = self._as_float(x["mode"], device)
        last_raw_mode = raw_mode[:, -1, :]
        mode_sum = last_raw_mode.sum(dim=1)
        mode_idx = last_raw_mode.argmax(dim=1)
        mode_idx = torch.where(mode_sum == 0, torch.full_like(mode_idx, 3), mode_idx)

        def dueling_heads(z_last):
            q = torch.empty(
                (z_last.size(0), self.advantage_heads[0][-1].out_features),
                device=device,
            )
            for i in range(4):
                mask = mode_idx == i
                if mask.any():
                    z_i = z_last[mask]
                    v_i = self.value_heads[i](z_i)
                    a_i = self.advantage_heads[i](z_i)
                    q_i = v_i + (a_i - a_i.mean(dim=1, keepdim=True))
                    q[mask] = q_i
            return q

        if return_seq:
            qs = []
            for t in range(T):
                qs.append(dueling_heads(z_seq[:, t, :]).unsqueeze(1))
            q_all = torch.cat(qs, dim=1)
            return q_all, hT

        q_last = dueling_heads(z_seq[:, -1, :])
        return q_last, hT
