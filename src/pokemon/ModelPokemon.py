import math
from multiprocessing.synchronize import RLock
import os
import torch
import torch.nn as nn

from pokemon import Emulator
from pokemon.TransformerMemory import TransformerMemory


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration (from Rainbow DQN)."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _f(self, x):
        return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-8)

    def sample_noise(self):
        epsilon_in = self._f(torch.randn(self.in_features, device=self.weight_mu.device))
        epsilon_out = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            self.sample_noise()
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            w = self.weight_mu
            b = self.bias_mu
        return torch.nn.functional.linear(x, w, b)


def get_model(device: str, files_lock: RLock, name: str | None = None):
    emulator = Emulator.Emulator(files_lock=files_lock)

    inputs = emulator.data.inputs()

    single_embed_dim = 32 + 32 + 4 + 16 + 16

    multi_embed_dim = (
        len(inputs["move_id"]) * 32
        + len(inputs["move_type"]) * 32
        + len(inputs["pokemon_id"]) * 32
        + len(inputs["pokemon_type"]) * 32
        + len(inputs["sprite_id"]) * 16
        + len(inputs["item_id"]) * 32
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
        use_c51=True,
        use_noisy=True,
        use_transformer=True,
    ).to(device)

    emulator.pyboy.stop(False)

    if name is None:
        return model

    ckpt_path = f"models/{name}.pth"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    state_dict = (
        state["model_state"]
        if isinstance(state, dict) and "model_state" in state
        else state
    )

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        result = model.load_state_dict(state_dict, strict=False)
        missing = len(result.missing_keys)
        unexpected = len(result.unexpected_keys)
        print(
            f"[ModelPokemon] Loaded checkpoint with mismatches "
            f"(missing={missing}, unexpected={unexpected}). "
            f"Architecture may have changed — consider --reset-buffer and fresh training."
        )

    if model.use_c51:
        with torch.no_grad():
            model.support.copy_(
                torch.linspace(model.v_min, model.v_max, model.num_atoms, device=model.support.device)
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
        use_c51: bool = True,
        use_noisy: bool = True,
        use_transformer: bool = True,
    ):
        super().__init__()

        self.outputs = outputs
        self.use_c51 = use_c51
        self.use_noisy = use_noisy
        self.use_transformer = use_transformer

        self.screen_out_dim = 128
        self.transformer_dim = 256 if use_transformer else 0

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
            + self.transformer_dim
        )

        self.map_id = nn.Embedding(256, 32)
        self.dialog_id = nn.Embedding(256, 32)
        self.index_of_current_pokemon_send_out = nn.Embedding(6, 4)
        self.type_of_battle = nn.Embedding(256, 16)
        self.move_menu_type = nn.Embedding(256, 16)

        self.move_id = nn.Embedding(256, 32, padding_idx=0)
        self.move_type = nn.Embedding(256, 32, padding_idx=0)
        self.pokemon_id = nn.Embedding(256, 32, padding_idx=0)
        self.pokemon_type = nn.Embedding(256, 32, padding_idx=0)
        self.sprite_id = nn.Embedding(256, 16, padding_idx=0)
        self.item_id = nn.Embedding(256, 32, padding_idx=0)

        self.screen_enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((9, 10)),
            nn.Flatten(),
            nn.Linear(64 * 9 * 10, 256),
            nn.SiLU(),
            nn.Linear(256, self.screen_out_dim),
            nn.SiLU(),
        )

        if self.use_transformer:
            self.transformer_memory = TransformerMemory(
                state_dim=256, d_model=256, n_heads=4, n_layers=3, ff_dim=512
            )
            self.state_buffer = None

        self.trunk = nn.Sequential(
            nn.Linear(total_in_dim, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, 256),
            nn.SiLU(),
        )

        if self.use_c51:
            self.num_atoms = 51
            self.v_min = -10.0
            self.v_max = 5.0
            self.register_buffer(
                "support",
                torch.linspace(self.v_min, self.v_max, self.num_atoms),
            )

            if self.use_noisy:
                self.value_dist_head = nn.Sequential(
                    NoisyLinear(256, 128),
                    nn.SiLU(),
                    NoisyLinear(128, self.num_atoms),
                )
                self.advantage_dist_head = nn.Sequential(
                    NoisyLinear(256, 128),
                    nn.SiLU(),
                    NoisyLinear(128, outputs * self.num_atoms),
                )
            else:
                self.value_dist_head = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.SiLU(),
                    nn.Linear(128, self.num_atoms),
                )
                self.advantage_dist_head = nn.Sequential(
                    nn.Linear(256, 128),
                    nn.SiLU(),
                    nn.Linear(128, outputs * self.num_atoms),
                )
        else:
            if self.use_noisy:
                self.value_head = nn.Sequential(
                    NoisyLinear(256, 128),
                    nn.SiLU(),
                    NoisyLinear(128, 1),
                )

                self.advantage_head = nn.Sequential(
                    NoisyLinear(256, 128),
                    nn.SiLU(),
                    NoisyLinear(128, outputs),
                )
            else:
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

        if self.use_transformer:
            if "state_sequence" in x:
                transformer_context = self.transformer_memory(
                    x["state_sequence"].to(device)
                )
            else:
                transformer_context = torch.zeros(
                    h.size(0), self.transformer_dim, device=device
                )
            h = torch.cat([h, transformer_context], dim=1)

        z = self.trunk(h)

        if self.use_c51:
            v_logits = self.value_dist_head(z)
            a_logits = self.advantage_dist_head(z).reshape(
                z.size(0), self.outputs, self.num_atoms
            )

            q_logits = v_logits.unsqueeze(1) + a_logits - a_logits.mean(dim=1, keepdim=True)
            q_dist = torch.softmax(q_logits, dim=2)

            q = (q_dist * self.support).sum(dim=2)

            return {"q": q, "dist": q_dist, "z": z}
        else:
            v = self.value_head(z)
            a = self.advantage_head(z)
            q = v + (a - a.mean(dim=1, keepdim=True))

            return {"q": q, "z": z}

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

        if self.use_transformer:
            modules.append(self.transformer_memory)

        for module in modules:
            for p in module.parameters():
                p.requires_grad = False

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def freeze(self):
        self.freeze_representation()

        head_modules = []
        if self.use_c51:
            head_modules = [self.value_dist_head, self.advantage_dist_head]
        else:
            head_modules = [self.value_head, self.advantage_head]

        for module in head_modules:
            for p in module.parameters():
                p.requires_grad = True
