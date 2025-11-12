from dataclasses import dataclass, field

from pyboy import PyBoy, PyBoyMemoryView
import torch


@dataclass
class Data:
    pyboy: PyBoy

    visited_dialogs_count: dict[str, int] = field(default_factory=dict)
    visited_positions_count: dict[str, int] = field(default_factory=dict)
    menu_count: dict[str, int] = field(default_factory=dict)
    battle_count = 0
    __player_pokemon_size = 0x2C
    __pokemon_count = 6

    def clean(self):
        self.visited_dialogs_count = {}
        self.visited_positions_count = {}
        self.menu_count = {}
        self.battle_count = 0

    def count(self, memory: bytes):
        if not self.is_battle() or self.number_of_turns_in_current_battle(
            memory
        ) < self.number_of_turns_in_current_battle(self.pyboy.memory):
            self.battle_count = 0
        else:
            self.battle_count += 1

        if self.is_dialog():
            self.visited_dialogs_count.setdefault(self.dialog_id(self.pyboy.memory), 0)
            self.visited_dialogs_count[self.dialog_id(self.pyboy.memory)] += 1

        if self.is_world():
            self.visited_positions_count.setdefault(self.get_position(), 0)
            self.visited_positions_count[self.get_position()] += 1

        if self.is_menu():
            self.menu_count.setdefault(self.get_position(), 0)
            self.menu_count[self.get_position()] += 1

    def inputs(self):
        return {
            "continuous": torch.tensor(self.data(), dtype=torch.float32),
        }

    def reward(self, memory: bytes):
        reward = -0.001

        reward += self.reward_milestones(memory)
        reward += self.reward_pokedex(memory)
        reward += self.reward_player_pokemons_current_hps(memory)
        reward += self.reward_player_pokemons_statuses(memory)
        reward += self.reward_player_pokemons_experiences(memory)
        reward += self.reward_player_pokemons_hp_evs(memory)
        reward += self.reward_player_pokemons_attack_evs(memory)
        reward += self.reward_player_pokemons_defense_evs(memory)
        reward += self.reward_player_pokemons_speed_evs(memory)
        reward += self.reward_player_pokemons_speed_evs(memory)
        reward += self.reward_player_pokemons_max_hps(memory)
        reward += self.reward_player_pokemons_attacks(memory)
        reward += self.reward_player_pokemons_defenses(memory)
        reward += self.reward_player_pokemons_speeds(memory)
        reward += self.reward_player_pokemons_specials(memory)

        if self.is_battle():
            reward += self.reward_battle(memory)

        if self.is_dialog():
            reward += self.reward_dialog()

        if self.is_world():
            reward += self.reward_position()

        if self.is_menu():
            reward += self.reward_menu()

        return reward

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
                    if status_x_bit != status_y_bit and status_x_bit == 1:
                        reward += 0.05
                    elif status_x_bit != status_y_bit and status_x_bit == 0:
                        reward -= 0.05

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

    def reward_player_pokemons_hp_evs(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, hp_ev_x, hp_ev_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_hp_evs(memory),
            self.player_pokemons_hp_evs(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (hp_ev_y - hp_ev_x) / 0xFFFF

        return reward

    def reward_player_pokemons_attack_evs(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, attack_ev_x, attack_ev_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_attack_evs(memory),
            self.player_pokemons_attack_evs(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (attack_ev_y - attack_ev_x) / 0xFFFF

        return reward

    def reward_player_pokemons_defense_evs(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, defense_ev_x, defense_ev_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_defense_evs(memory),
            self.player_pokemons_defense_evs(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (defense_ev_y - defense_ev_x) / 0xFFFF

        return reward

    def reward_player_pokemons_speed_evs(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, speed_ev_x, speed_ev_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_speed_evs(memory),
            self.player_pokemons_speed_evs(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (speed_ev_y - speed_ev_x) / 0xFFFF

        return reward

    def reward_player_pokemons_speed_evs(self, memory: bytes):
        reward = 0.0

        for id_x, id_y, special_ev_x, special_ev_y in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_special_evs(memory),
            self.player_pokemons_special_evs(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                reward += (special_ev_y - special_ev_x) / 0xFFFF

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

    def terminated(self, memory: bytes):
        return True if 0 < self.reward_milestones(memory) else False

    def truncated(self):
        return (
            True
            if 255
            <= self.visited_dialogs_count.get(self.dialog_id(self.pyboy.memory), 0)
            or 255 <= self.battle_count
            or 255 <= self.visited_positions_count.get(self.get_position(), 0)
            or 255 <= self.menu_count.get(self.get_position(), 0)
            else False
        )

    def number_of_turns_in_current_battle(self, memory: bytes):
        return memory[0xCCD5]

    def data(self):
        data = []

        data += self.game_mode_flags_data()

        data += self.position_data()

        data += self.dialog_data()

        data += self.map_data()

        data += self.sprite_data()

        data += self.menu_data()

        data += self.battle_data()

        data += self.poke_mart_data()

        data += self.player_data()

        data += self.pokedex_data()

        data += self.items_data()

        data += self.money_data()

        data += self.badges(self.pyboy.memory)

        data += self.stored_items_data()

        data += self.game_coins_data()

        data += self.event_flags_data(self.pyboy.memory)

        data += self.opponent_trainers_pokemon_data()

        data += self.stored_pokemon_data()

        return data

    def game_mode_flags_data(self):
        return [
            int(self.is_battle()),
            int(self.is_dialog()),
            int(self.is_menu()),
            int(self.is_world()),
        ]

    def data_normalizer(self, values: list[int], max=0xFF):
        return [x / max for x in values]

    def position_data(self):
        return self.data_normalizer(
            [
                self.map_id(self.pyboy.memory),
                self.position_x(self.pyboy.memory),
                self.position_y(self.pyboy.memory),
            ]
        )

    def dialog_data(self):
        data = [
            self.visited_dialogs_count.get(self.dialog_id(self.pyboy.memory), 0),
            self.dialog_id(self.pyboy.memory),
        ]

        return self.data_normalizer(data) if self.is_dialog() else [0] * len(data)

    def map_id(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD35E]

    def is_dialog(self):
        return (
            True
            if self.is_blocked()
            and self.dialog_id(self.pyboy.memory) != 0
            and not self.is_battle()
            else False
        )

    def is_blocked(self):
        return True if self.pyboy.memory[0xCFC4] else False

    def dialog_id(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCF13]

    def is_battle(self):
        return True if self.type_of_battle(self.pyboy.memory) else False

    def type_of_battle(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD057]

    def map_data(self):
        data = [
            self.visited_positions_count.get(self.get_position(), 0),
            self.bike_speed(self.pyboy.memory),
        ]

        return self.data_normalizer(data) if self.is_world() else [0] * len(data)

    def get_position(self):
        return f"{self.position_x(self.pyboy.memory)}x{self.position_y(self.pyboy.memory)}x{self.map_id(self.pyboy.memory)}"

    def is_world(self):
        return (
            True
            if not self.is_blocked() and not self.is_battle() and not self.is_menu()
            else False
        )

    def position_x(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD361]

    def position_y(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD362]

    def is_menu(self):
        return (
            True
            if self.is_blocked()
            and self.dialog_id(self.pyboy.memory) == 0
            and not self.is_battle()
            else False
        )

    def sprite_data(self):
        data = [
            self.pyboy.memory[i]
            for x in range(16)
            for i in range(0xC100 + 0x10 * x, 0xC10A + 0x10 * x)
        ] + [
            self.pyboy.memory[i]
            for x in range(16)
            for i in range(0xC200 + 0x10 * x, 0xC209 + 0x10 * x)
        ]

        return self.data_normalizer(data) if self.is_world() else [0] * len(data)

    def menu_data(self):
        data = [self.pyboy.memory[i] for i in range(0xCC24, 0xCC36)] + [
            self.menu_count.get(self.get_position(), 0)
        ]

        return (
            self.data_normalizer(data)
            if self.is_menu() or self.is_battle() or self.is_dialog()
            else [0] * len(data)
        )

    def battle_data(self):
        data_bit = (
            self.enemy_status(self.pyboy.memory)
            + self.enemy_base_stats(self.pyboy.memory)
            + self.pokemon_status1(self.pyboy.memory)
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
                self.move_menu_type(self.pyboy.memory),
                self.player_selected_move(self.pyboy.memory),
                self.enemy_selected_move(self.pyboy.memory),
                self.your_move_used(self.pyboy.memory),
                self.your_move_type(self.pyboy.memory),
                self.your_move_effect(self.pyboy.memory),
                self.enemy_move_id(self.pyboy.memory),
                self.enemy_move_effect(self.pyboy.memory),
                self.enemy_move_power(self.pyboy.memory),
                self.enemy_move_type(self.pyboy.memory),
                self.enemy_move_accuracy(self.pyboy.memory),
                self.enemy_move_max_pp(self.pyboy.memory),
                self.player_move_id(self.pyboy.memory),
                self.player_move_power(self.pyboy.memory),
                self.player_move_accuracy(self.pyboy.memory),
                self.player_move_max_pp(self.pyboy.memory),
                self.enemy_pokemon_internal_id1(self.pyboy.memory),
                self.player_pokemon_interna2_id(self.pyboy.memory),
                self.enemy_pokemon_internal_id2(self.pyboy.memory),
                self.enemy_level1(self.pyboy.memory),
                self.enemy_type1(self.pyboy.memory),
                self.enemy_type2(self.pyboy.memory),
                self.enemy_move1(self.pyboy.memory),
                self.enemy_move2(self.pyboy.memory),
                self.enemy_move3(self.pyboy.memory),
                self.enemy_move4(self.pyboy.memory),
                self.enemy_attack_and_defense_ivs(self.pyboy.memory),
                self.enemy_attack_and_special_ivs(self.pyboy.memory),
                self.enemy_level2(self.pyboy.memory),
                self.enemy_pp_first_slot(self.pyboy.memory),
                self.enemy_pp_second_slot(self.pyboy.memory),
                self.enemy_pp_third_slot(self.pyboy.memory),
                self.enemy_pp_fourth_slot(self.pyboy.memory),
                self.enemy_catch_rate(self.pyboy.memory),
                self.enemy_base_experience(self.pyboy.memory),
                self.pokemon_number1(self.pyboy.memory),
                self.pokemon_type11(self.pyboy.memory),
                self.pokemon_type21(self.pyboy.memory),
                self.pokemon_move_first_slot1(self.pyboy.memory),
                self.pokemon_move_second_slot1(self.pyboy.memory),
                self.pokemon_move_third_slot1(self.pyboy.memory),
                self.pokemon_move_fourth_slot1(self.pyboy.memory),
                self.pokemon_attack_and_defense_ivs1(self.pyboy.memory),
                self.pokemon_speed_and_special_ivs1(self.pyboy.memory),
                self.pokemon_level1(self.pyboy.memory),
                self.pokemon_pp_first_slot1(self.pyboy.memory),
                self.pokemon_pp_second_slot1(self.pyboy.memory),
                self.pokemon_pp_third_slot1(self.pyboy.memory),
                self.pokemon_pp_fourth_slot1(self.pyboy.memory),
                self.type_of_battle(self.pyboy.memory),
                self.battle_type(self.pyboy.memory),
                self.battle_count,
            ]
        )
        data_2bytes = self.data_normalizer(
            [
                self.enemy_hp(self.pyboy.memory),
                self.enemy_max_hp(self.pyboy.memory),
                self.enemy_attack(self.pyboy.memory),
                self.enemy_defense(self.pyboy.memory),
                self.enemy_speed(self.pyboy.memory),
                self.enemy_special(self.pyboy.memory),
                self.pokemon_current_hp1(self.pyboy.memory),
                self.pokemon_max_hp1(self.pyboy.memory),
                self.pokemon_attack1(self.pyboy.memory),
                self.pokemon_defense1(self.pyboy.memory),
                self.pokemon_speed1(self.pyboy.memory),
                self.pokemon_special1(self.pyboy.memory),
            ],
            max=65535,
        )

        data = data_bit + data_byte + data_2bytes

        return data if self.is_battle() else [0] * len(data)

    def enemy_status(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[0xCFE9], end_bit=6)

    def bits_extractor(self, byte: int, start_bit=0, end_bit=7):
        if start_bit < 0 or end_bit > 7 or start_bit > end_bit:
            raise ValueError("Invalid bit range")

        return [1 if (byte & (1 << i)) else 0 for i in range(start_bit, end_bit + 1)]

    def enemy_base_stats(self, memory: PyBoyMemoryView | bytes):
        return [memory[0xD002 + i] for i in range(5)]

    def pokemon_status1(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[0xD018], end_bit=6)

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
        return memory[0xCCDB]

    def player_selected_move(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCCDC]

    def enemy_selected_move(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCCDD]

    def your_move_used(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCCDC]

    def your_move_type(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD5]

    def your_move_effect(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD3]

    def enemy_move_id(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFCC]

    def enemy_move_effect(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFCD]

    def enemy_move_power(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFCE]

    def enemy_move_type(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFCF]

    def enemy_move_accuracy(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD0]

    def enemy_move_max_pp(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD1]

    def player_move_id(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD2]

    def player_move_power(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD4]

    def player_move_accuracy(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD6]

    def player_move_max_pp(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD7]

    def enemy_pokemon_internal_id1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD8]

    def player_pokemon_interna2_id(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFD9]

    def enemy_pokemon_internal_id2(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFE5]

    def enemy_hp(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFE6] << 8 | memory[0xCFE7]

    def enemy_level1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFE8]

    def enemy_status(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[0xCFE9], end_bit=6)

    def enemy_type1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFEA]

    def enemy_type2(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFEB]

    def enemy_move1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFED]

    def enemy_move2(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFEE]

    def enemy_move3(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFEF]

    def enemy_move4(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFF0]

    def enemy_attack_and_defense_ivs(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFF1]

    def enemy_attack_and_special_ivs(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFF2]

    def enemy_level2(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCFF3]

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

    def enemy_catch_rate(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD007]

    def enemy_base_experience(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD008]

    def pokemon_number1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD014]

    def pokemon_current_hp1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD015] | (memory[0xD016] << 8)

    def pokemon_status1(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[0xD018], end_bit=6)

    def pokemon_type11(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD019]

    def pokemon_type21(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD01A]

    def pokemon_move_first_slot1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD01C]

    def pokemon_move_second_slot1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD01D]

    def pokemon_move_third_slot1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD01E]

    def pokemon_move_fourth_slot1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD01F]

    def pokemon_attack_and_defense_ivs1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD020]

    def pokemon_speed_and_special_ivs1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD021]

    def pokemon_level1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD022]

    def pokemon_max_hp1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD023] | (memory[0xD024] << 8)

    def pokemon_attack1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD025] | (memory[0xD026] << 8)

    def pokemon_defense1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD027] | (memory[0xD028] << 8)

    def pokemon_speed1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD029] | (memory[0xD02A] << 8)

    def pokemon_special1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD02B] | (memory[0xD02C] << 8)

    def pokemon_pp_first_slot1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD02D]

    def pokemon_pp_second_slot1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD02E]

    def pokemon_pp_third_slot1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD02F]

    def pokemon_pp_fourth_slot1(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD030]

    def battle_type(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD05A]

    def poke_mart_data(self):
        data = [self.pyboy.memory[i] for i in range(0xCF7B, 0xCF86)]

        return self.data_normalizer(data) if self.is_menu() else [0] * len(data)

    def player_data(self):

        data = []

        data += self.data_normalizer(self.player_pokemons_ids(self.pyboy.memory))

        data += self.data_normalizer(
            self.player_pokemons_current_hps(self.pyboy.memory),
            max=0xFFFF,
        )

        data += self.player_pokemons_statuses(self.pyboy.memory)

        data += self.data_normalizer(
            [
                self.pyboy.memory[i]
                for x in range(self.__pokemon_count)
                for i in range(
                    0xD170 + self.__player_pokemon_size * x,
                    0xD177 + self.__player_pokemon_size * x,
                )
            ],
        )

        data += self.data_normalizer(
            [
                self.pyboy.memory[0xD177 + self.__player_pokemon_size * i] << 8
                | self.pyboy.memory[0xD178 + self.__player_pokemon_size * i]
                for i in range(self.__pokemon_count)
            ],
            max=0xFFFF,
        )

        data += self.data_normalizer(
            self.player_pokemons_experiences(self.pyboy.memory),
            max=0xFFFFFF,
        )

        data += self.data_normalizer(
            self.player_pokemons_hp_evs(self.pyboy.memory),
            max=0xFFFF,
        )

        data += self.data_normalizer(
            self.player_pokemons_attack_evs(self.pyboy.memory),
            max=0xFFFF,
        )

        data += self.data_normalizer(
            self.player_pokemons_defense_evs(self.pyboy.memory),
            max=0xFFFF,
        )

        data += self.data_normalizer(
            self.player_pokemons_special_evs(self.pyboy.memory),
            max=0xFFFF,
        )

        data += self.data_normalizer(
            self.player_pokemons_special_evs(self.pyboy.memory),
            max=0xFFFF,
        )

        data += self.data_normalizer(
            [
                self.pyboy.memory[i]
                for x in range(self.__pokemon_count)
                for i in range(
                    0xD186 + self.__player_pokemon_size * x,
                    0xD18C + self.__player_pokemon_size * x,
                )
            ],
        )

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

    def player_pokemons_ids(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD16B + self.__player_pokemon_size * x]
            for x in range(self.__pokemon_count)
        ]

    def player_pokemons_current_hps(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD16C + self.__player_pokemon_size * x] << 8
            | memory[0xD16D + self.__player_pokemon_size * x]
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
            memory[0xD179 + self.__player_pokemon_size * i] << 16
            | memory[0xD17A + self.__player_pokemon_size * i] << 8
            | memory[0xD17B + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_hp_evs(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD17C + self.__player_pokemon_size * i] << 8
            | memory[0xD17D + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_attack_evs(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD17E + self.__player_pokemon_size * i] << 8
            | memory[0xD17F + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_defense_evs(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD180 + self.__player_pokemon_size * i] << 8
            | memory[0xD181 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_speed_evs(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD182 + self.__player_pokemon_size * i] << 8
            | memory[0xD183 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_special_evs(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD184 + self.__player_pokemon_size * i] << 8
            | memory[0xD185 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_levels(self, memory: PyBoyMemoryView | bytes = None):
        return [memory[0xD18C]]

    def player_pokemons_max_hps(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD18D + self.__player_pokemon_size * i] << 8
            | memory[0xD18E + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_attacks(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD18F + self.__player_pokemon_size * i] << 8
            | memory[0xD190 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_defenses(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD191 + self.__player_pokemon_size * i] << 8
            | memory[0xD192 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_speeds(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD193 + self.__player_pokemon_size * i] << 8
            | memory[0xD194 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_specials(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[0xD195 + self.__player_pokemon_size * i] << 8
            | memory[0xD196 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def pokedex_data(self):
        return self.pokedex_own(self.pyboy.memory) + self.pokedex_seen(
            self.pyboy.memory
        )

    def pokedex_own(self, memory: PyBoyMemoryView | bytes):
        data = memory[0xD2F7:0xD30A]

        bits = []
        for byte in data:
            bits.extend(self.bits_extractor(byte))

        return bits

    def pokedex_seen(self, memory: PyBoyMemoryView | bytes):
        data = memory[0xD30A:0xD31D]

        bits = []
        for byte in data:
            bits.extend(self.bits_extractor(byte))

        return bits

    def items_data(self):
        data = [self.pyboy.memory[i] for i in range(0xD31D, 0xD347)]

        return (
            self.data_normalizer(data)
            if self.is_menu() or self.is_battle()
            else [0] * len(data)
        )

    def money_data(self):
        return (
            self.data_normalizer([self.player_money(self.pyboy.memory)], max=0xFFFFFF)
            if self.is_menu() or self.is_dialog()
            else [0]
        )

    def player_money(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD347] + (memory[0xD348] << 8) + (memory[0xD349] << 16)

    def badges(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[0xD356])

    def stored_items_data(self):
        data = self.stored_items(self.pyboy.memory)

        return (
            self.data_normalizer(data)
            if self.is_menu() or self.is_dialog()
            else [0] * len(data)
        )

    def stored_items(self, memory: PyBoyMemoryView | bytes):
        return [memory[i] for i in range(0xD53A, 0xD5A0)]

    def game_coins_data(self):
        return (
            self.data_normalizer([self.game_coins(self.pyboy.memory)], max=0xFFFF)
            if self.is_menu() or self.is_dialog()
            else [0]
        )

    def game_coins(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD5A4] + (memory[0xD5A5] << 8)

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

    def opponent_trainers_pokemon_data(self):
        data = [self.pyboy.memory[i] for i in range(0xD89C, 0xD9AC)]

        return self.data_normalizer(data) if self.is_battle() else [0] * len(data)

    def stored_pokemon_data(self):
        data = [self.pyboy.memory[i] for i in range(0xDA80, 0xDD2A)]

        return (
            self.data_normalizer(data)
            if self.is_menu() or self.is_dialog()
            else [0] * len(data)
        )

    def reward_dialog(self):
        return 0.2 - 0.0025 * self.visited_dialogs_count.get(
            self.dialog_id(self.pyboy.memory), 0
        )

    def reward_battle(self, memory: bytes):
        reward = 0.2 - 0.0025 * self.battle_count

        reward += self.reward_players_substitute_hp(memory)
        reward += self.reward_enemy_substitute_hp(memory)
        reward += self.reward_enemy_hp(memory)
        reward += self.reward_enemy_status(memory)
        reward += self.reward_pokemon_current_hp1(memory)
        reward += self.reward_pokemon_status1(memory)
        reward += self.reward_critical_hit_flag(memory)
        reward += self.reward_one_hit_ko_flag(memory)

        return reward

    def reward_position(self):
        return 0.2 - 0.0025 * self.visited_positions_count.get(self.get_position(), 0)

    def reward_menu(self):
        return 0.2 - 0.0025 * self.menu_count.get(self.get_position(), 0)

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
                reward += 1

        return reward

    def reward_pokemon_current_hp1(self, memory: bytes):
        return (
            (
                self.pokemon_current_hp1(self.pyboy.memory)
                - self.pokemon_current_hp1(memory)
            )
            / self.pokemon_max_hp1(self.pyboy.memory)
            if self.pokemon_max_hp1(self.pyboy.memory) != 0
            else 0
        )

    def reward_pokemon_status1(self, memory: bytes):
        reward = 0

        for bit_before, bit_after in zip(
            self.pokemon_status1(memory), self.pokemon_status1(self.pyboy.memory)
        ):
            if bit_before == 0 and bit_after == 1:
                reward += 1

        return reward

    def reward_critical_hit_flag(self, memory: bytes):
        return (
            0.05
            if self.critical_hit_flag(memory) == 0
            and self.critical_hit_flag(self.pyboy.memory) == 1
            else 0.0
        )

    def reward_one_hit_ko_flag(self, memory: bytes):
        return (
            0.1
            if self.one_hit_ko_flag(memory) == 0
            and self.one_hit_ko_flag(self.pyboy.memory) == 1
            else 0.0
        )

    def reward_pokedex(self, memory: bytes):
        return self.reward_pokedex_own(memory) + self.reward_pokedex_seen(memory)

    def reward_pokedex_own(self, memory: bytes):
        reward = 0

        for bit_before, bit_after in zip(
            self.pokedex_own(memory), self.pokedex_own(self.pyboy.memory)
        ):
            if bit_before == 0 and bit_after == 1:
                reward += 1

        return reward

    def reward_pokedex_seen(self, memory: bytes):
        reward = 0

        for bit_before, bit_after in zip(
            self.pokedex_seen(memory), self.pokedex_seen(self.pyboy.memory)
        ):
            if bit_before == 0 and bit_after == 1:
                reward += 0.5

        return reward

    def reward_badges(self, memory: bytes):
        reward = 0

        for bit_before, bit_after in zip(
            self.badges(memory), self.badges(self.pyboy.memory)
        ):
            if bit_before == 0 and bit_after == 1:
                reward += 10

        return reward

    def reward_event_flags(self, memory: bytes):
        reward = 0

        for flag_x, flag_y in zip(
            self.event_flags_data(memory), self.event_flags_data(self.pyboy.memory)
        ):
            if flag_x == 0 and flag_y == 1:
                reward += 5

        return reward

    def reward_milestones(self, memory: bytes):
        reward = 0.0

        reward += self.reward_badges(memory)

        reward += self.reward_event_flags(memory)

        return reward
