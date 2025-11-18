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

        self.embedding_dim = 16

        self.categorical_keys = [
            "map_id",
            "dialog_id",
            "index_of_current_pokemon_send_out",
            "move_menu_type",
            "your_move_type",
            "player_selected_move",
            "enemy_selected_move",
            "enemy_move_id",
            "enemy_move_type",
            "player_move_id",
            "enemy_type1",
            "enemy_type2",
            "enemy_move1",
            "enemy_move2",
            "enemy_move3",
            "enemy_move4",
            "pokemon_type1",
            "pokemon_type2",
            "pokemon_move_first_slot",
            "pokemon_move_second_slot",
            "pokemon_move_third_slot",
            "pokemon_move_fourth_slot",
            "type_of_battle",
            "sprite_data_ids",
            "poke_mart_items",
            "player_pokemons_ids",
            "player_pokemon_types",
            "items_ids",
            "stored_items_ids",
            "stored_pokemon_ids",
            "stored_pokemon_types",
            "stored_pokemon_moves",
        ]

        self.sequence_keys = {
            "sprite_data_ids",
            "poke_mart_items",
            "player_pokemons_ids",
            "player_pokemon_types",
            "items_ids",
            "stored_items_ids",
            "stored_pokemon_ids",
            "stored_pokemon_types",
            "stored_pokemon_moves",
        }

        self.embeddings = nn.ModuleDict(
            {k: nn.Embedding(256, self.embedding_dim) for k in self.categorical_keys}
        )

        total_dim = continuous_dim + self.embedding_dim * len(self.categorical_keys)

        self.fc = nn.Sequential(
            nn.LayerNorm(total_dim),
            nn.Linear(total_dim, 1024),
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

    def _embed_categoricals(self, x: dict, device: torch.device):
        embs = []

        for k in self.categorical_keys:
            if k not in x:
                continue

            t = x[k].to(device)

            if k in self.sequence_keys:
                if t.dim() == 1:
                    t = t.unsqueeze(0)
            else:
                if t.dim() == 0:
                    t = t.unsqueeze(0)

            t = t.long()
            e = self.embeddings[k](t)

            if k in self.sequence_keys:
                if e.dim() == 2:
                    e = e.unsqueeze(0)
                e = e.mean(dim=1)

            embs.append(e)

        if not embs:
            return None

        return torch.cat(embs, dim=1)

    def forward(self, x):
        device = next(self.parameters()).device

        cont = self._as_float_batch(x["continuous"], device)
        cat = self._embed_categoricals(x, device)

        if cat is not None:
            inp = torch.cat([cont, cat], dim=1)
        else:
            inp = cont

        return self.fc(inp)
