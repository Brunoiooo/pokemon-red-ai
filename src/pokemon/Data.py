from dataclasses import dataclass, field

from pyboy import PyBoy, PyBoyMemoryView
import torch


@dataclass
class Data:
    pyboy: PyBoy

    visited_dialogs_count: dict[int, int] = field(default_factory=dict)
    visited_positions_count: dict[str, int] = field(default_factory=dict)
    visited_maps_count: dict[int, int] = field(default_factory=dict)
    menu_count: dict[int, int] = field(default_factory=dict)
    battle_count: int = 0
    useless_count: int = 0
    __player_pokemon_size = 0x2C
    __pokemon_count = 6

    __stored_pokemon_size = 0x21
    __stored_pokemon_count = 20

    __visited_pokedex_own: list[int] | None = None

    @property
    def visited_pokedex_own(self):
        if self.__visited_pokedex_own is None:
            self.__visited_pokedex_own = self.pokedex_own(self.pyboy.memory)

        return self.__visited_pokedex_own

    @visited_pokedex_own.setter
    def visited_pokedex_own(self, value: list[int]):
        self.__visited_pokedex_own = [
            x | y for x, y in zip(value, self.visited_pokedex_own)
        ]

    __visited_pokedex_seen: list[int] | None = None

    @property
    def visited_pokedex_seen(self):
        if self.__visited_pokedex_seen is None:
            self.__visited_pokedex_seen = self.pokedex_seen(self.pyboy.memory)

        return self.__visited_pokedex_seen

    @visited_pokedex_seen.setter
    def visited_pokedex_seen(self, value: list[int]):
        self.__visited_pokedex_seen = [
            x | y for x, y in zip(value, self.visited_pokedex_seen)
        ]

    def clean(self):
        self.__visited_pokedex_own = None
        self.__visited_pokedex_seen = None
        self.visited_dialogs_count = {}
        self.visited_maps_count = {}
        self.visited_positions_count = {}
        self.menu_count = {}
        self.battle_count = 0
        self.useless_count = 0

    def count(self, memory: bytes, reward: float):
        if self.is_battle(self.pyboy.memory):
            if self.number_of_turns_in_current_battle(
                memory
            ) != self.number_of_turns_in_current_battle(self.pyboy.memory):
                self.battle_count = 0
            else:
                self.battle_count += 1

        if self.is_dialog(self.pyboy.memory):
            self.visited_dialogs_count.setdefault(self.dialog_id(self.pyboy.memory), 0)
            self.visited_dialogs_count[self.dialog_id(self.pyboy.memory)] += 1

        if self.is_world(self.pyboy.memory):
            self.visited_positions_count.setdefault(self.get_position(), 0)
            self.visited_positions_count[self.get_position()] += 1

        if self.is_menu(self.pyboy.memory):
            self.menu_count.setdefault(self.map_id(self.pyboy.memory), 0)
            self.menu_count[self.map_id(self.pyboy.memory)] += 1

        self.visited_pokedex_own = self.pokedex_own(self.pyboy.memory)
        self.visited_pokedex_seen = self.pokedex_seen(self.pyboy.memory)

        self.visited_maps_count.setdefault(self.map_id(self.pyboy.memory), 0)
        self.visited_maps_count[self.map_id(self.pyboy.memory)] += 1

        if 0 < reward:
            self.useless_count = 0
        else:
            self.useless_count += 1

    def inputs(self):
        return {
            "continuous": torch.tensor(self.data(), dtype=torch.float32),
            "map_id": torch.tensor(self.map_id(self.pyboy.memory), dtype=torch.long),
            "dialog_id": torch.tensor(
                self.dialog_id(self.pyboy.memory), dtype=torch.long
            ),
            "index_of_current_pokemon_send_out": torch.tensor(
                self.index_of_current_pokemon_send_out(self.pyboy.memory),
                dtype=torch.long,
            ),
            "type_of_battle": torch.tensor(
                self.type_of_battle(self.pyboy.memory),
                dtype=torch.long,
            ),
            "move_menu_type": torch.tensor(
                self.move_menu_type(self.pyboy.memory),
                dtype=torch.long,
            ),
            "position_x": torch.tensor(
                self.position_x(self.pyboy.memory), dtype=torch.float32
            ),
            "position_y": torch.tensor(
                self.position_y(self.pyboy.memory), dtype=torch.float32
            ),
            "bike_speed": torch.tensor(
                self.bike_speed(self.pyboy.memory), dtype=torch.float32
            ),
            "menu_position_x": torch.tensor(
                self.menu_position_x(self.pyboy.memory), dtype=torch.long
            ),
            "menu_position_y": torch.tensor(
                self.menu_position_y(self.pyboy.memory), dtype=torch.long
            ),
            "current_menu_selected_item": torch.tensor(
                self.current_menu_selected_item(self.pyboy.memory), dtype=torch.long
            ),
            "visited_dialogs_count": torch.tensor(
                (
                    min(
                        self.visited_dialogs_count.get(
                            self.dialog_id(self.pyboy.memory), 0
                        ),
                        8,
                    )
                    if self.is_dialog(self.pyboy.memory)
                    else 0
                ),
                dtype=torch.long,
            ),
            "visited_maps_count": torch.tensor(
                (
                    min(
                        self.visited_maps_count.get(self.map_id(self.pyboy.memory), 0),
                        4,
                    )
                    if self.is_world(self.pyboy.memory)
                    else 0
                ),
                dtype=torch.long,
            ),
            "menu_count": torch.tensor(
                (
                    min(
                        self.menu_count.get(self.map_id(self.pyboy.memory), 0),
                        4,
                    )
                    if not self.is_battle(self.pyboy.memory)
                    else 0
                ),
                dtype=torch.long,
            ),
            "battle_count": torch.tensor(
                (
                    min(
                        self.battle_count,
                        4,
                    )
                    if self.is_battle(self.pyboy.memory)
                    else 0
                ),
                dtype=torch.long,
            ),
            "move_id": torch.tensor(
                [
                    self.player_selected_move(self.pyboy.memory),
                    self.enemy_selected_move(self.pyboy.memory),
                    self.enemy_move1(self.pyboy.memory),
                    self.enemy_move2(self.pyboy.memory),
                    self.enemy_move3(self.pyboy.memory),
                    self.enemy_move4(self.pyboy.memory),
                ]
                + self.stored_pokemon_moves(self.pyboy.memory),
                dtype=torch.long,
            ),
            "move_type": torch.tensor(
                [
                    self.your_move_type(self.pyboy.memory),
                    self.enemy_move_type(self.pyboy.memory),
                    self.pokemon_move_first_slot(self.pyboy.memory),
                    self.pokemon_move_second_slot(self.pyboy.memory),
                    self.pokemon_move_third_slot(self.pyboy.memory),
                    self.pokemon_move_fourth_slot(self.pyboy.memory),
                ],
                dtype=torch.long,
            ),
            "pokemon_id": torch.tensor(
                self.player_pokemons_ids(self.pyboy.memory)
                + self.stored_pokemon_ids(self.pyboy.memory),
                dtype=torch.long,
            ),
            "pokemon_type": torch.tensor(
                [
                    self.enemy_type1(self.pyboy.memory),
                    self.enemy_type2(self.pyboy.memory),
                    self.pokemon_type1(self.pyboy.memory),
                    self.pokemon_type2(self.pyboy.memory),
                ]
                + self.player_pokemon_types(self.pyboy.memory)
                + self.stored_pokemon_types(self.pyboy.memory),
                dtype=torch.long,
            ),
            "sprite_id": torch.tensor(
                self.sprite_data_ids(self.pyboy.memory),
                dtype=torch.long,
            ),
            "item_id": torch.tensor(
                self.poke_mart_items(self.pyboy.memory)
                + self.items_ids(self.pyboy.memory)
                + self.stored_items_ids(self.pyboy.memory),
                dtype=torch.long,
            ),
            "sprite_data_movement_statuses": torch.tensor(
                self.sprite_data_movement_statuses(self.pyboy.memory),
                dtype=torch.long,
            ),
            "sprite_data_facing_directions": torch.tensor(
                self.sprite_data_facing_directions(self.pyboy.memory),
                dtype=torch.long,
            ),
            "sprite_data_y_positions": torch.tensor(
                self.sprite_data_y_positions(self.pyboy.memory),
                dtype=torch.long,
            ),
            "sprite_data_x_positions": torch.tensor(
                self.sprite_data_x_positions(self.pyboy.memory),
                dtype=torch.long,
            ),
        }

    def reward(self, memory: bytes):
        reward = 0.0

        reward += self.reward_core(memory)

        if self.is_battle(self.pyboy.memory):
            reward += self.reward_battle(memory)

        if self.is_dialog(self.pyboy.memory):
            reward += self.reward_dialog()

        if self.is_world(self.pyboy.memory):
            reward += self.reward_position()

        if self.is_menu(self.pyboy.memory):
            reward += self.reward_menu()

        return max(-1.0, min(1.0, reward))

    def reward_core(self, memory: bytes):
        reward = 0.0

        reward += self.reward_milestones(memory)
        reward += self.reward_pokedex(memory)
        reward += self.reward_player_pokemons_current_hps(memory)
        reward += self.reward_player_pokemons_statuses(memory)
        reward += self.reward_player_pokemons_experiences(memory)
        reward += self.reward_player_pokemons_max_hps(memory)
        reward += self.reward_player_pokemons_attacks(memory)
        reward += self.reward_player_pokemons_defenses(memory)
        reward += self.reward_player_pokemons_speeds(memory)
        reward += self.reward_player_pokemons_pps(memory)
        reward += self.reward_map(memory)

        return reward

    def reward_map(self, memory: bytes):
        return 0.01 if self.visited_maps_count.get(self.map_id(memory), 0) < 4 else 0.0

    def reward_player_pokemons_current_hps(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, hp_x, hp_y, max_hp in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_current_hps(memory),
            self.player_pokemons_current_hps(self.pyboy.memory),
            self.player_pokemons_max_hps(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (hp_y - hp_x) / max_hp

        return reward

    def reward_player_pokemons_statuses(self, memory: bytes):
        ids_x = self.player_pokemons_ids(memory)
        ids_y = self.player_pokemons_ids(self.pyboy.memory)
        statuses_x = self.player_pokemons_statuses(memory)
        statuses_y = self.player_pokemons_statuses(self.pyboy.memory)
        statuses_x = [statuses_x[i : i + 7] for i in range(0, len(statuses_x), 7)]
        statuses_y = [statuses_y[i : i + 7] for i in range(0, len(statuses_y), 7)]

        reward = 0.0
        for id_x, id_y, status_x, status_y in zip(ids_x, ids_y, statuses_x, statuses_y):
            if id_x == id_y and id_x != 0:
                for status_x_bit, status_y_bit in zip(status_x, status_y):
                    if status_x_bit == 1 and status_y_bit == 0:
                        reward += 0.3
                    elif status_x_bit == 0 and status_y_bit == 1:
                        reward -= 0.3

        return reward

    def reward_player_pokemons_experiences(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, experience_x, experience_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_experiences(memory),
            self.player_pokemons_experiences(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (experience_y - experience_x) / 0xFFFFFF

        return reward

    def reward_player_pokemons_max_hps(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, max_hp_x, max_hp_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_max_hps(memory),
            self.player_pokemons_max_hps(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (max_hp_y - max_hp_x) / 0xFFFF

        return reward

    def reward_player_pokemons_attacks(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, attack_x, attack_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_attacks(memory),
            self.player_pokemons_attacks(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (attack_y - attack_x) / 0xFFFF

        return reward

    def reward_player_pokemons_defenses(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, defense_x, defense_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_defenses(memory),
            self.player_pokemons_defenses(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (defense_y - defense_x) / 0xFFFF

        return reward

    def reward_player_pokemons_speeds(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, speed_x, speed_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_speeds(memory),
            self.player_pokemons_speeds(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (speed_y - speed_x) / 0xFFFF

        return reward

    def reward_player_pokemons_specials(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, special_x, special_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_specials(memory),
            self.player_pokemons_specials(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (special_y - special_x) / 0xFFFF

        return reward

    def reward_player_pokemons_pps(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, pp_x, pp_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_pps(memory),
            self.player_pokemons_pps(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (pp_y - pp_x) / 0xFF

        return reward

    def terminated(self, memory: bytes):
        return True if 0 < self.reward_badges(memory) else False

    def truncated(self):
        return True if 64 <= self.useless_count else False

    def number_of_turns_in_current_battle(self, memory: bytes):
        return memory[0xCCD5]

    def data(self):
        data = []

        data += self.core_data()
        data += self.battle_data()
        data += self.world_data()

        return data

    def world_data(self):
        return [
            (
                min(self.visited_positions_count.get(self.get_position(), 0), 1)
                if self.is_world(self.pyboy.memory)
                else 0
            )
        ]

    def game_mode_flags_data(self):
        return [
            int(self.is_battle(self.pyboy.memory)),
            int(self.is_dialog(self.pyboy.memory)),
            int(self.is_menu(self.pyboy.memory)),
            int(self.is_world(self.pyboy.memory)),
        ]

    def data_normalizer(self, values: list[int], max=0xFF):
        return [x / max for x in values]

    def core_data(self):
        data = []
        data += [
            0 if self.visited_positions_count.get(self.get_position(), 0) == 0 else 1
        ]
        data += self.game_mode_flags_data()
        data += self.player_data()
        data += self.pokedex_data()
        data += self.data_normalizer(self.items_quantities(self.pyboy.memory))
        data += self.data_normalizer(
            [self.player_money(self.pyboy.memory)], max=0xFFFFFF
        )
        data += self.badges(self.pyboy.memory)
        data += self.data_normalizer(self.stored_items_quantities(self.pyboy.memory))
        data += self.data_normalizer([self.game_coins(self.pyboy.memory)], max=0xFFFF)
        data += self.event_flags_data(self.pyboy.memory)
        data += self.stored_pokemon_data(self.pyboy.memory)

        return data

    def map_id(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD35E]

    def is_dialog(self, memory: PyBoyMemoryView | bytes):
        return (
            True
            if self.is_blocked(memory)
            and self.dialog_id(memory) != 0
            and not self.is_battle(memory)
            else False
        )

    def is_blocked(self, memory: PyBoyMemoryView | bytes):
        return True if memory[0xCFC4] else False

    def dialog_id(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCF13]

    def is_battle(self, memory: PyBoyMemoryView | bytes):
        return True if self.type_of_battle(memory) else False

    def type_of_battle(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD057]

    def get_position(self):
        return f"{self.position_x(self.pyboy.memory)}x{self.position_y(self.pyboy.memory)}x{self.map_id(self.pyboy.memory)}"

    def is_world(self, memory: PyBoyMemoryView | bytes):
        return (
            True
            if not self.is_blocked(memory)
            and not self.is_battle(memory)
            and not self.is_menu(memory)
            else False
        )

    def position_x(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD361]

    def position_y(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD362]

    def is_menu(self, memory: PyBoyMemoryView | bytes):
        return (
            True
            if self.is_blocked(memory)
            and self.dialog_id(memory) == 0
            and not self.is_battle(memory)
            else False
        )

    def sprite_data_ids(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xC100 + 0x10 * x] for x in range(16)]

    def sprite_data_movement_statuses(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xC101 + 0x10 * x] for x in range(16)]

    def sprite_data_facing_directions(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xC109 + 0x10 * x] for x in range(16)]

    def sprite_data_y_positions(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xC204 + 0x10 * x] for x in range(16)]

    def sprite_data_x_positions(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xC205 + 0x10 * x] for x in range(16)]

    def menu_position_x(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCC24]

    def menu_position_y(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCC25]

    def current_menu_selected_item(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCC26]

    def index_of_current_pokemon_send_out(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCC2F]

    def battle_data(self):
        data_bit = (
            self.enemy_status(self.pyboy.memory)
            + self.enemy_base_stats(self.pyboy.memory)
            + self.pokemon_status(self.pyboy.memory)
            + self.battle_status_player(self.pyboy.memory)
            + [
                self.is_gym_leader_battle_music_playing(self.pyboy.memory),
                self.critical_hit_flag(self.pyboy.memory),
                self.one_hit_ko_flag(self.pyboy.memory),
                self.hooked_pokemon_flag(self.pyboy.memory),
            ]
        )
        data_byte = self.data_normalizer(
            [
                self.number_of_turns_in_current_battle(self.pyboy.memory),
                self.players_substitute_hp(self.pyboy.memory),
                self.enemy_substitute_hp(self.pyboy.memory),
                self.enemy_move_power(self.pyboy.memory),
                self.enemy_move_accuracy(self.pyboy.memory),
                self.player_move_power(self.pyboy.memory),
                self.player_move_accuracy(self.pyboy.memory),
                self.enemy_level(self.pyboy.memory),
                self.pokemon_level(self.pyboy.memory),
                self.enemy_pp_first_slot(self.pyboy.memory),
                self.enemy_pp_second_slot(self.pyboy.memory),
                self.enemy_pp_third_slot(self.pyboy.memory),
                self.enemy_pp_fourth_slot(self.pyboy.memory),
                self.pokemon_pp_first_slot(self.pyboy.memory),
                self.pokemon_pp_second_slot(self.pyboy.memory),
                self.pokemon_pp_third_slot(self.pyboy.memory),
                self.pokemon_pp_fourth_slot(self.pyboy.memory),
            ]
        )
        data_2bytes = self.data_normalizer(
            [
                self.enemy_hp(self.pyboy.memory),
                self.enemy_attack(self.pyboy.memory),
                self.enemy_defense(self.pyboy.memory),
                self.enemy_speed(self.pyboy.memory),
                self.enemy_special(self.pyboy.memory),
                self.pokemon_current_hp(self.pyboy.memory),
                self.pokemon_attack(self.pyboy.memory),
                self.pokemon_defense(self.pyboy.memory),
                self.pokemon_speed(self.pyboy.memory),
                self.pokemon_special(self.pyboy.memory),
            ],
            max=65535,
        )

        data = data_bit + data_byte + data_2bytes

        return data if self.is_battle(self.pyboy.memory) else [0] * len(data)

    def enemy_status(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[0xCFE9], end_bit=6)

    def bits_extractor(self, byte: int, start_bit=0, end_bit=7):
        if start_bit < 0 or end_bit > 7 or start_bit > end_bit:
            raise ValueError("Invalid bit range")

        return [1 if (byte & (1 << i)) else 0 for i in range(start_bit, end_bit + 1)]

    def enemy_base_stats(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xD002 + i] for i in range(5)]

    def battle_status_player(self, memory: PyBoyMemoryView | bytes):
        return (
            self.bits_extractor(memory[0xD062])
            + self.bits_extractor(memory[0xD063])
            + self.bits_extractor(memory[0xD064], 0, 3)
        )

    def is_gym_leader_battle_music_playing(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD05C] & 1

    def critical_hit_flag(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD05E] & 1

    def one_hit_ko_flag(self, memory: PyBoyMemoryView | bytes):
        return 1 if memory[0xD05E] & 2 else 0

    def hooked_pokemon_flag(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD05F] & 1

    def number_of_turns_in_current_battle(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCCD5]

    def players_substitute_hp(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCCD7]

    def enemy_substitute_hp(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCCD8]

    def move_menu_type(self, memory: PyBoyMemoryView | bytes):
        return (
            memory[0xCCDB]
            if self.is_battle(memory) or self.is_dialog(memory) or self.is_menu(memory)
            else 0
        )

    def player_selected_move(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCCDC] if self.is_battle(memory) else 0

    def enemy_selected_move(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCCDD] if self.is_battle(memory) else 0

    def your_move_type(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD5] if self.is_battle(memory) else 0

    def enemy_move_power(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFCE]

    def enemy_move_type(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFCF] if self.is_battle(memory) else 0

    def enemy_move_accuracy(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD0]

    def player_move_power(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD4]

    def player_move_accuracy(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD6]

    def enemy_hp(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFE6] | memory[0xCFE7] << 8

    def enemy_level(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFE8]

    def enemy_status(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[0xCFE9], end_bit=6)

    def enemy_type1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFEA] if self.is_battle(memory) else 0

    def enemy_type2(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFEB] if self.is_battle(memory) else 0

    def enemy_move1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFED] if self.is_battle(memory) else 0

    def enemy_move2(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFEE] if self.is_battle(memory) else 0

    def enemy_move3(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFEF] if self.is_battle(memory) else 0

    def enemy_move4(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFF0] if self.is_battle(memory) else 0

    def enemy_max_hp(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFF4] | (memory[0xCFF5] << 8)

    def enemy_attack(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFF6] | (memory[0xCFF7] << 8)

    def enemy_defense(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFF8] | (memory[0xCFF9] << 8)

    def enemy_speed(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFFA] | (memory[0xCFFB] << 8)

    def enemy_special(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFFC] | (memory[0xCFFD] << 8)

    def enemy_pp_first_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFFE]

    def enemy_pp_second_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFFF]

    def enemy_pp_third_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD000]

    def enemy_pp_fourth_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD001]

    def enemy_base_stats(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xD002 + i] for i in range(5)]

    def pokemon_current_hp(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD015] | (memory[0xD016] << 8)

    def pokemon_status(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[0xD018], end_bit=6)

    def pokemon_type1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD019] if self.is_battle(memory) else 0

    def pokemon_type2(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD01A] if self.is_battle(memory) else 0

    def pokemon_move_first_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD01C] if self.is_battle(memory) else 0

    def pokemon_move_second_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD01D] if self.is_battle(memory) else 0

    def pokemon_move_third_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD01E] if self.is_battle(memory) else 0

    def pokemon_move_fourth_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD01F] if self.is_battle(memory) else 0

    def pokemon_level(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD022]

    def pokemon_max_hp(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD023] | (memory[0xD024] << 8)

    def pokemon_attack(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD025] | (memory[0xD026] << 8)

    def pokemon_defense(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD027] | (memory[0xD028] << 8)

    def pokemon_speed(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD029] | (memory[0xD02A] << 8)

    def pokemon_special(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD02B] | (memory[0xD02C] << 8)

    def pokemon_pp_first_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD02D]

    def pokemon_pp_second_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD02E]

    def pokemon_pp_third_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD02F]

    def pokemon_pp_fourth_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD030]

    def poke_mart_items(self, memory: PyBoyMemoryView | bytes):
        data = [memory[i] for i in range(0xCF7C, 0xCF86)]
        return data if self.is_menu(memory) else [0] * len(data)

    def player_data(self):
        data = self.data_normalizer(
            self.player_pokemons_current_hps(self.pyboy.memory),
            max=0xFFFF,
        )

        data += self.player_pokemons_statuses(self.pyboy.memory)

        data += self.data_normalizer(
            self.player_pokemons_experiences(self.pyboy.memory),
            max=0xFFFFFF,
        )

        data += self.data_normalizer(self.player_pokemons_ivs(self.pyboy.memory))

        data += self.data_normalizer(self.player_pokemons_pps(self.pyboy.memory))

        data += self.data_normalizer(
            self.player_pokemons_levels(self.pyboy.memory),
        )

        data += self.data_normalizer(
            self.player_pokemons_max_hps(self.pyboy.memory),
            max=0xFFFF,
        )

        data += self.data_normalizer(
            self.player_pokemons_attacks(self.pyboy.memory),
            max=0xFFFF,
        )

        data += self.data_normalizer(
            self.player_pokemons_defenses(self.pyboy.memory),
            max=0xFFFF,
        )

        data += self.data_normalizer(
            self.player_pokemons_speeds(self.pyboy.memory),
            max=0xFFFF,
        )

        data += self.data_normalizer(
            self.player_pokemons_specials(self.pyboy.memory),
            max=0xFFFF,
        )

        return data

    def player_pokemons_pps(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[i]
            for x in range(self.__pokemon_count)
            for i in range(
                0xD188 + self.__player_pokemon_size * x,
                0xD18C + self.__player_pokemon_size * x,
            )
        ]

    def player_pokemons_ivs(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[i]
            for x in range(self.__pokemon_count)
            for i in range(
                0xD186 + self.__player_pokemon_size * x,
                0xD188 + self.__player_pokemon_size * x,
            )
        ]

    def player_pokemon_types(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[i]
            for x in range(self.__pokemon_count)
            for i in range(
                0xD170 + self.__player_pokemon_size * x,
                0xD172 + self.__player_pokemon_size * x,
            )
        ]

    def player_pokemons_ids(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD16B + self.__player_pokemon_size * x]
            for x in range(self.__pokemon_count)
        ]

    def player_pokemons_current_hps(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD16C + self.__player_pokemon_size * x]
            | memory[0xD16D + self.__player_pokemon_size * x] << 8
            for x in range(self.__pokemon_count)
        ]

    def player_pokemons_statuses(self, memory: PyBoyMemoryView | bytes = None):
        data = []
        for x in range(self.__pokemon_count):
            data += self.bits_extractor(
                memory[0xD16F + self.__player_pokemon_size * x], end_bit=6
            )

        return data

    def player_pokemons_experiences(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD179 + self.__player_pokemon_size * i]
            | memory[0xD17A + self.__player_pokemon_size * i] << 8
            | memory[0xD17B + self.__player_pokemon_size * i] << 16
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_levels(self, memory: PyBoyMemoryView | bytes = None):
        return [memory[0xD18C]]

    def player_pokemons_max_hps(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD18D + self.__player_pokemon_size * i]
            | memory[0xD18E + self.__player_pokemon_size * i] << 8
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_attacks(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD18F + self.__player_pokemon_size * i]
            | memory[0xD190 + self.__player_pokemon_size * i] << 8
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_defenses(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD191 + self.__player_pokemon_size * i]
            | memory[0xD192 + self.__player_pokemon_size * i] << 8
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_speeds(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD193 + self.__player_pokemon_size * i]
            | memory[0xD194 + self.__player_pokemon_size * i] << 8
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_specials(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD195 + self.__player_pokemon_size * i]
            | memory[0xD196 + self.__player_pokemon_size * i] << 8
            for i in range(self.__pokemon_count)
        ]

    def pokedex_data(self):
        return self.pokedex_own(self.pyboy.memory) + self.pokedex_seen(
            self.pyboy.memory
        )

    def pokedex_own(self, memory: PyBoyMemoryView | bytes):
        data = bytes(memory[0xD2F7:0xD30A])

        bits: list[int] = []
        for byte in data:
            bits.extend(self.bits_extractor(byte))

        return bits

    def pokedex_seen(self, memory: PyBoyMemoryView | bytes):
        data = memory[0xD30A:0xD31D]

        bits: list[int] = []
        for byte in data:
            bits.extend(self.bits_extractor(byte))

        return bits

    def items_quantities(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xD31F + i * 2] for i in range(20)]

    def items_ids(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xD31E + i * 2] for i in range(20)]

    def player_money(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD347] | (memory[0xD348] << 8) | (memory[0xD349] << 16)

    def badges(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[0xD356])

    def stored_items_ids(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xD53B + 2 * i] for i in range(50)]

    def stored_items_quantities(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xD53C + 2 * i] for i in range(50)]

    def game_coins(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD5A4] | (memory[0xD5A5] << 8)

    def event_flags_data(self, memory: PyBoyMemoryView | bytes):
        return (
            [
                self.starters_back(memory),
                memory[0xD5C0] & 1,
                self.have_town_map(memory),
                self.have_oaks_parcel(memory),
            ]
            + self.fly_anywhere(memory)
            + [
                self.safari_zone_time(memory),
                self.fossilized_pokemon(memory),
                self.position_in_air(memory),
                self.did_you_get_lapras_yet(memory),
                self.debug_new_game(memory),
                self.fought_giovanni_yet(memory),
                self.fought_brock_yet(memory),
                self.fought_misty_yet(memory),
                self.fought_lt_surge_yet(memory),
                self.fought_erika_yet(memory),
                self.fought_articuno_yet(memory),
                self.fought_koga_yet(memory),
                self.fought_blaine_yet(memory),
                self.fought_sabrina_yet(memory),
                self.fought_zapdos_yet(memory),
                self.fought_snorlax_yet_vermilion(memory),
                self.fought_snorlax_yet_celadon(memory),
                self.fought_moltres_yet(memory),
                self.is_ss_anne_here(memory),
                self.mewtwo_can_be_caught(memory),
            ]
        )

    def starters_back(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD5AB] & 1

    def have_town_map(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD5F3] & 1

    def have_oaks_parcel(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD60D] & 1

    def bike_speed(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD700]

    def fly_anywhere(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[0xD70B]) + self.bits_extractor(memory[0xD70C])

    def safari_zone_time(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD70D] | (memory[0xD70E] << 8)

    def fossilized_pokemon(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD710] & 1

    def position_in_air(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD714] & 1

    def did_you_get_lapras_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD72E] & 1

    def debug_new_game(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD732] & 1

    def fought_giovanni_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD751] & 1

    def fought_brock_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD755] & 1

    def fought_misty_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD75E] & 1

    def fought_lt_surge_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD773] & 1

    def fought_erika_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD77C] & 1

    def fought_articuno_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD782] & 1

    def safari_gameover(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD790] & 0x80

    def fought_koga_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD792] & 1

    def fought_blaine_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD79A] & 1

    def fought_sabrina_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD7B3] & 1

    def fought_zapdos_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD7D4] & 1

    def fought_snorlax_yet_vermilion(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD7D8] & 1

    def fought_snorlax_yet_celadon(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD7E0] & 1

    def fought_moltres_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD7EE] & 1

    def is_ss_anne_here(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD803] & 1

    def mewtwo_can_be_caught(self, memory: PyBoyMemoryView | bytes):
        return 1 if memory[0xD85F] & 2 else 0

    def stored_pokemon_data(self, memory: PyBoyMemoryView | bytes):
        data = []
        data = self.data_normalizer(self.stored_pokemon_hps(memory), max=0xFFFF)
        data += self.data_normalizer(self.stored_pokemon_levels(memory))
        data += self.data_normalizer(self.stored_pokemon_statuses(memory))
        data += self.data_normalizer(
            self.stored_pokemon_experiences(memory), max=0xFFFFFF
        )
        data += self.data_normalizer(self.stored_pokemon_pps(memory))

        return data

    def stored_pokemon_pps(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[x + i * self.__stored_pokemon_size]
            for i in range(self.__stored_pokemon_count)
            for x in range(0xDAB3, 0xDAB7)
        ]

    def stored_pokemon_experiences(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[0xDAA4 + i * self.__stored_pokemon_size]
            | (memory[0xDAA5 + i * self.__stored_pokemon_size] << 8)
            | (memory[0xDAA6 + i * self.__stored_pokemon_size] << 16)
            for i in range(self.__stored_pokemon_count)
        ]

    def stored_pokemon_moves(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[x + i * self.__stored_pokemon_size]
            for i in range(self.__stored_pokemon_count)
            for x in range(0xDA9E, 0xDAA2)
        ]

    def stored_pokemon_types(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[x + i * self.__stored_pokemon_size]
            for i in range(self.__stored_pokemon_count)
            for x in range(0xDA9B, 0xDA9D)
        ]

    def stored_pokemon_statuses(self, memory: PyBoyMemoryView | bytes):
        return [
            bit
            for i in range(self.__stored_pokemon_count)
            for bit in self.bits_extractor(
                memory[0xDA9A + i * self.__stored_pokemon_size]
            )
        ]

    def stored_pokemon_levels(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[0xDA99 + i * self.__stored_pokemon_size]
            for i in range(self.__stored_pokemon_count)
        ]

    def stored_pokemon_hps(self, memory: PyBoyMemoryView | bytes):
        return [
            x | (y << 8)
            for x, y in zip(
                [
                    memory[0xDA97 + i * self.__stored_pokemon_size]
                    for i in range(self.__stored_pokemon_count)
                ],
                [
                    memory[0xDA98 + i * self.__stored_pokemon_size]
                    for i in range(self.__stored_pokemon_count)
                ],
            )
        ]

    def stored_pokemon_ids(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[0xDA96 + i * self.__stored_pokemon_size]
            for i in range(self.__stored_pokemon_count)
        ]

    def reward_dialog(self):
        return (
            0.01
            if self.visited_dialogs_count.get(self.dialog_id(self.pyboy.memory), 0) < 8
            else 0
        )

    def reward_battle(self, memory: bytes):
        reward = 0.01 if self.battle_count < 4 else 0

        reward += self.reward_players_substitute_hp(memory)
        reward += self.reward_enemy_substitute_hp(memory)
        reward += self.reward_enemy_hp(memory)
        reward += self.reward_enemy_status(memory)
        reward += self.reward_pokemon_current_hp(memory)
        reward += self.reward_pokemon_status(memory)
        reward += self.reward_critical_hit_flag(memory)
        reward += self.reward_one_hit_ko_flag(memory)

        return reward

    def reward_position(self):
        return (
            0.01 if self.visited_positions_count.get(self.get_position(), 0) == 0 else 0
        )

    def reward_menu(self):
        return (
            0.01 if self.menu_count.get(self.map_id(self.pyboy.memory), 0) == 0 else 0
        )

    def reward_players_substitute_hp(self, memory: bytes):
        return (
            self.players_substitute_hp(self.pyboy.memory)
            - self.players_substitute_hp(memory)
        ) / 255

    def reward_enemy_substitute_hp(self, memory: bytes):
        return (
            self.enemy_substitute_hp(memory)
            - self.enemy_substitute_hp(self.pyboy.memory)
        ) / 255

    def reward_enemy_hp(self, memory: bytes):
        return (
            (self.enemy_hp(memory) - self.enemy_hp(self.pyboy.memory))
            / self.enemy_max_hp(self.pyboy.memory)
            if self.enemy_max_hp(self.pyboy.memory) != 0
            else 0
        )

    def reward_enemy_status(self, memory: bytes):
        reward = 0

        for bit_before, bit_after in zip(
            self.enemy_status(memory), self.enemy_status(self.pyboy.memory)
        ):
            if bit_before == 0 and bit_after == 1:
                reward += 0.3
            elif bit_before == 1 and bit_after == 0:
                reward -= 0.3

        return reward

    def reward_pokemon_current_hp(self, memory: bytes):
        return (
            (
                self.pokemon_current_hp(self.pyboy.memory)
                - self.pokemon_current_hp(memory)
            )
            / self.pokemon_max_hp(self.pyboy.memory)
            if self.pokemon_max_hp(self.pyboy.memory) != 0
            else 0
        )

    def reward_pokemon_status(self, memory: bytes):
        reward = 0

        for bit_before, bit_after in zip(
            self.pokemon_status(memory), self.pokemon_status(self.pyboy.memory)
        ):
            if bit_before == 0 and bit_after == 1:
                reward -= 0.3
            elif bit_before == 1 and bit_after == 0:
                reward += 0.3

        return reward

    def reward_critical_hit_flag(self, memory: bytes):
        return (
            0.3
            if self.critical_hit_flag(memory) == 0
            and self.critical_hit_flag(self.pyboy.memory) == 1
            else 0.0
        )

    def reward_one_hit_ko_flag(self, memory: bytes):
        return (
            0.3
            if self.one_hit_ko_flag(memory) == 0
            and self.one_hit_ko_flag(self.pyboy.memory) == 1
            else 0.0
        )

    def reward_pokedex(self, memory: bytes):
        return self.reward_pokedex_own(memory) + self.reward_pokedex_seen(memory)

    def reward_pokedex_own(self, memory: bytes):
        reward = 0

        for bit_before, bit_after, visited in zip(
            self.pokedex_own(memory),
            self.pokedex_own(self.pyboy.memory),
            self.visited_pokedex_own,
        ):
            if bit_before == 0 and bit_after == 1 and visited == 0:
                reward += 0.5

        return reward

    def reward_pokedex_seen(self, memory: bytes):
        reward = 0

        for bit_before, bit_after, visited in zip(
            self.pokedex_seen(memory),
            self.pokedex_seen(self.pyboy.memory),
            self.visited_pokedex_seen,
        ):
            if bit_before == 0 and bit_after == 1 and visited == 0:
                reward += 0.4

        return reward

    def reward_badges(self, memory: bytes):
        reward = 0

        for bit_before, bit_after in zip(
            self.badges(memory), self.badges(self.pyboy.memory)
        ):
            if bit_before == 0 and bit_after == 1:
                reward += 1

        return reward

    def reward_event_flags(self, memory: bytes):
        reward = 0

        for flag_x, flag_y in zip(
            self.event_flags_data(memory), self.event_flags_data(self.pyboy.memory)
        ):
            if flag_x == 0 and flag_y == 1:
                reward += 1

        return reward

    def reward_milestones(self, memory: bytes):
        reward = 0.0

        reward += self.reward_badges(memory)

        reward += self.reward_event_flags(memory)

        return reward
