import hashlib
from collections import deque
from multiprocessing.synchronize import RLock
import pickle
from dataclasses import dataclass, field

from pyboy import PyBoy, PyBoyMemoryView
import torch


# Pokemon Red early-game map IDs (used for curriculum milestones).
MAP_PALLET_TOWN = 0
MAP_ROUTE_1 = 12
MAP_REDS_HOUSE_2F = 37
MAP_REDS_HOUSE_1F = 38
MAP_OAKS_LAB = 40
HOUSE_MAPS = frozenset({MAP_REDS_HOUSE_1F, MAP_REDS_HOUSE_2F})

# Emulator button indices — A/B must be mashable to start and advance dialogs.
ACTION_A = 0
ACTION_B = 1
ACTION_NONE = 8
INTERACT_ACTIONS = frozenset({ACTION_A, ACTION_B})

# Curriculum / episode goals (map, event flags, badges).
GOAL_LEFT_HOUSE = "left_house"
GOAL_ROUTE_1 = "route1"
GOAL_OAKS_LAB = "oaks_lab"
GOAL_OAKS_PARCEL = "oaks_parcel"
GOAL_TOWN_MAP = "town_map"
GOAL_FOUGHT_BROCK = "fought_brock"
GOAL_FOUGHT_MISTY = "fought_misty"
GOAL_FOUGHT_SURGE = "fought_surge"
GOAL_FOUGHT_ERIKA = "fought_erika"
GOAL_FOUGHT_KOGA = "fought_koga"
GOAL_FOUGHT_SABRINA = "fought_sabrina"
GOAL_FOUGHT_BLAINE = "fought_blaine"
GOAL_FOUGHT_GIOVANNI = "fought_giovanni"
GOAL_BADGE_1 = "badge1"
GOAL_BADGE_2 = "badge2"
GOAL_BADGE_3 = "badge3"
GOAL_BADGE_4 = "badge4"
GOAL_BADGE_5 = "badge5"
GOAL_BADGE_6 = "badge6"
GOAL_BADGE_7 = "badge7"
GOAL_BADGE_8 = "badge8"
GOAL_SS_ANNE = "ss_anne"
GOAL_LAPRAS = "lapras"
GOAL_SNORLAX = "snorlax"
GOAL_ARTICUNO = "articuno"
GOAL_ZAPDOS = "zapdos"
GOAL_MOLTRES = "moltres"
GOAL_FOSSIL = "fossil"
GOAL_MEWTWO = "mewtwo"
GOAL_ALL_BADGES = "all_badges"

BADGE_GOALS = (
    GOAL_BADGE_1,
    GOAL_BADGE_2,
    GOAL_BADGE_3,
    GOAL_BADGE_4,
    GOAL_BADGE_5,
    GOAL_BADGE_6,
    GOAL_BADGE_7,
    GOAL_BADGE_8,
)


@dataclass
class Data:
    pyboy: PyBoy
    files_lock: RLock

    visited_screens: list[bytes] = field(default_factory=list)
    visited_maps: set[int] = field(default_factory=set)
    visited_dialogs: dict[tuple[int, int], int] = field(default_factory=dict)

    # Hierarchical rewards: macro >> meso >> micro (PokeRL / Whidden style).
    badge_reward: float = 10.0          # macro
    event_reward: float = 2.0           # macro
    left_house_reward: float = 5.0      # macro early milestone
    route1_reward: float = 5.0          # macro early milestone
    oaks_lab_reward: float = 3.0        # meso/macro story
    new_screen_reward: float = 0.1      # meso (new map)
    new_position_reward: float = 0.01   # micro
    new_dialog_reward: float = 0.1      # meso — enter a new dialog
    dialog_advance_reward: float = 0.02 # meso — dialog_id changed while reading
    dialog_exit_reward: float = 0.05    # meso — finished / left a dialog
    new_pokedex_seen_reward: float = 0.5
    new_pokedex_own_reward: float = 1.0
    status_reward: float = 0.02
    base_reward: float = -0.001
    truncated_reward: float = -0.05
    new_item_reward: float = 0.5

    # Anti-loop / anti-spam penalties (PokeRL-style).
    visit_penalty_soft: float = -0.01   # visit count > 3
    visit_penalty_hard: float = -0.05   # visit count > 5
    action_pattern_penalty: float = -0.02
    spatial_loop_penalty: float = -0.03
    menu_spam_penalty: float = -0.02

    # Episode goal for terminated() — early milestones for PPO curriculum.
    goal: str = GOAL_BADGE_1

    in_menu_ticks: float = 0.0
    in_battle_ticks: float = 0.0
    in_dialog_ticks: float = 0.0
    max_useless_ticks: int = 512
    # Hard stuck fuse while a single dialog_id is on screen. Does NOT reset on
    # tile blink / A presses — only on dialog_id change or leaving dialog.
    # 512*4 @ frame_skip 24 ≈ 85 steps; enough for a box, not infinite camp.
    max_useless_dialog_ticks: int = 512 * 4
    __player_pokemon_size: int = 0x2C
    __pokemon_count: int = 6
    buffer_reward: float = 0.0

    __stored_pokemon_size: int = 0x21

    __visited_pokedex_own: list[int] | None = None

    visited_positions: dict[tuple[int, int, int], int] = field(default_factory=dict)
    position_visit_counts: dict[tuple[int, int, int], int] = field(default_factory=dict)
    map_vision_radius: int = 5

    recent_actions: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_positions: deque = field(default_factory=lambda: deque(maxlen=16))
    loop_flag: bool = False
    _milestones_hit: set[str] = field(default_factory=set)
    _start_map_id: int | None = None
    # Distinct dialog screen hashes seen for the current dialog_id. Blink frames
    # revisit old hashes; only a *new* hash counts as text progress.
    _dialog_screens_seen: set[str] = field(default_factory=set)

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

    def save(self, path: str):
        with self.files_lock:
            with open(f"{path}/__visited_pokedex_own.pkl", "wb") as f:
                pickle.dump(self.__visited_pokedex_own, f)
            with open(f"{path}/__visited_pokedex_seen.pkl", "wb") as f:
                pickle.dump(self.__visited_pokedex_seen, f)
            with open(f"{path}/buffer_reward.pkl", "wb") as f:
                pickle.dump(self.buffer_reward, f)
            with open(f"{path}/visited_positions.pkl", "wb") as f:
                pickle.dump(self.visited_positions, f)
            with open(f"{path}/visited_maps.pkl", "wb") as f:
                pickle.dump(self.visited_maps, f)
            with open(f"{path}/visited_dialogs.pkl", "wb") as f:
                pickle.dump(self.visited_dialogs, f)

    def load(self, path: str):
        with self.files_lock:
            with open(f"{path}/__visited_pokedex_own.pkl", "rb") as f:
                self.__visited_pokedex_own = pickle.load(f)
            with open(f"{path}/__visited_pokedex_seen.pkl", "rb") as f:
                self.__visited_pokedex_seen = pickle.load(f)
            with open(f"{path}/buffer_reward.pkl", "rb") as f:
                self.buffer_reward = pickle.load(f)
            with open(f"{path}/visited_positions.pkl", "rb") as f:
                self.visited_positions = pickle.load(f)
            with open(f"{path}/visited_maps.pkl", "rb") as f:
                self.visited_maps = pickle.load(f)
            with open(f"{path}/visited_dialogs.pkl", "rb") as f:
                self.visited_dialogs = pickle.load(f)

    def clean(self):
        self.__visited_pokedex_own = None
        self.__visited_pokedex_seen = None
        self.in_menu_ticks = 0
        self.in_battle_ticks = 0
        self.in_dialog_ticks = 0
        self.buffer_reward = 0.0
        self.visited_positions = {}
        self.position_visit_counts = {}
        self.visited_maps = set()
        self.visited_dialogs = {}
        self.recent_actions.clear()
        self.recent_positions.clear()
        self.loop_flag = False
        self._milestones_hit = set()
        self._dialog_screens_seen = set()
        self._start_map_id = self.map_id(self.pyboy.memory)

    def _dialog_progressed(self, memory: bytes | None, action: int | None = None) -> bool:
        """True when dialog text advanced to a new unique screen.

        Cursor blink oscillates between a few tilemaps — those hashes are
        remembered so they do not reset the stuck fuse or pay advance rewards.
        """
        if not self.is_dialog(self.pyboy.memory):
            return False

        id_changed = (
            memory is not None
            and self.dialog_id(memory) != self.dialog_id(self.pyboy.memory)
        )
        if id_changed:
            self._dialog_screens_seen = set()

        screen_hash = self.screen_tiles_hash(self.pyboy.memory)
        if screen_hash not in self._dialog_screens_seen:
            self._dialog_screens_seen.add(screen_hash)
            return True
        return False

    def count(self, reward: float, action: int, memory: bytes | None = None, duration: int = 16):
        self.visited_pokedex_own = self.pokedex_own(self.pyboy.memory)
        self.visited_pokedex_seen = self.pokedex_seen(self.pyboy.memory)

        self.recent_actions.append(int(action))

        if self.is_world(self.pyboy.memory):
            pos = self.get_position()
            stayed = memory is not None and self.get_position(memory) == pos
            interacting = int(action) in INTERACT_ACTIONS
            # Stuck-fuse time always accumulates — including A/B mash. Otherwise
            # the policy can stand in Oak's lab spamming A forever: sprite does
            # not move, NPCs animate, and truncated() never fires until max_steps.
            # Soft visit *penalties* still skip A/B so a normal talk can open.
            self.visited_positions[pos] = (
                self.visited_positions.get(pos, 0) + duration
            )
            if not (stayed and interacting):
                self.position_visit_counts[pos] = (
                    self.position_visit_counts.get(pos, 0) + 1
                )
                self.recent_positions.append(pos)
            elif pos not in self.position_visit_counts:
                self.position_visit_counts[pos] = 1
        elif (
            not self.is_battle(self.pyboy.memory)
            and not self.is_dialog(self.pyboy.memory)
        ):
            # Script lock / menu with no movement (blocked, NPCs still walk):
            # same stuck fuse as world. Skip during dialog — text uses its own fuse.
            pos = self.get_position()
            self.visited_positions[pos] = (
                self.visited_positions.get(pos, 0) + duration
            )

        if self.is_menu(self.pyboy.memory):
            self.in_menu_ticks += duration
        else:
            self.in_menu_ticks = max(0, self.in_menu_ticks - 0.25 * duration)

        if self.is_battle(self.pyboy.memory):
            if self.number_of_turns_in_current_battle(
                memory
            ) != self.number_of_turns_in_current_battle(self.pyboy.memory):
                self.in_battle_ticks = 0
            else:
                self.in_battle_ticks += duration

        if self.is_dialog(self.pyboy.memory):
            dialog = self.get_dialog()
            self.visited_dialogs[dialog] = self.visited_dialogs.get(dialog, 0) + duration
            if self._dialog_progressed(memory, action):
                self.in_dialog_ticks = 0
            else:
                self.in_dialog_ticks += duration
        else:
            self.in_dialog_ticks = 0
            self._dialog_screens_seen = set()

        self.visited_maps.add(self.map_id(self.pyboy.memory))

    def get_dialog(self, memory: bytes | None = None):
        if memory is None:
            memory = self.pyboy.memory

        return (self.dialog_id(memory), self.map_id(memory))

    def get_position(self, memory: bytes | None = None, offset_x=0, offset_y=0):
        if memory is None:
            memory = self.pyboy.memory

        return (
            self.position_x(memory) + offset_x,
            self.position_y(memory) + offset_y,
            self.map_id(memory),
        )

    def screen_tiles_hash(self, memory: PyBoyMemoryView | bytes | None = None):
        return hashlib.blake2b(
            bytes(self.screen_tiles(memory if memory else self.pyboy.memory)),
            digest_size=16,
        ).hexdigest()

    def visit_mask_grid(self) -> list[list[float]]:
        """Local visit-count mask centered on the player (PokeRL / Whidden style)."""
        r = self.map_vision_radius
        size = 2 * r + 1
        if not self.is_world(self.pyboy.memory):
            return [[0.0] * size for _ in range(size)]
        grid: list[list[float]] = []
        for dy in range(-r, r + 1):
            row: list[float] = []
            for dx in range(-r, r + 1):
                visits = self.position_visit_counts.get(
                    self.get_position(offset_x=dx, offset_y=dy), 0
                )
                row.append(min(visits, 10) / 10.0)
            grid.append(row)
        return grid

    def inputs(self):
        r = self.map_vision_radius
        return {
            "screen_tiles": torch.tensor(
                self.data_normalizer(self.screen_tiles(self.pyboy.memory)),
                dtype=torch.float32,
            ).view(1, 18, 20),
            "visit_mask": torch.tensor(
                self.visit_mask_grid(), dtype=torch.float32
            ).view(1, 2 * r + 1, 2 * r + 1),
            "core": torch.tensor(self.core_data(), dtype=torch.float32),
            "battle": torch.tensor(self.battle_data(), dtype=torch.float32),
            "menu_battle_dialog": torch.tensor(
                self.menu_battle_dialog_data(), dtype=torch.float32
            ),
            "mode": torch.tensor(
                self.game_mode_flags_data(self.pyboy.memory), dtype=torch.float32
            ),
            "progress": torch.tensor(
                self.event_flags_data(self.pyboy.memory)
                + self.badges(self.pyboy.memory),
                dtype=torch.float32,
            ),
            "nav": torch.tensor(self.world_data(), dtype=torch.float32),
            "inv": torch.tensor(
                self.inventory_data(self.pyboy.memory), dtype=torch.float32
            ),
            "party": torch.tensor(
                self.stored_pokemon_data(self.pyboy.memory),
                dtype=torch.float32,
            ),
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
                self.item_ids(),
                dtype=torch.long,
            ),
        }

    def item_ids(self, memory: PyBoyMemoryView | bytes | None = None):
        if memory is None:
            memory = self.pyboy.memory

        data = [
            self.poke_mart_items(memory)[self.real_current_menu_selected_item(memory)],
            self.items_ids(memory)[self.real_current_menu_selected_item(memory)],
            self.stored_items_ids(memory)[self.real_current_menu_selected_item(memory)],
        ]

        return data if self.is_eq_menu(memory) else [0] * len(data)

    def is_eq_menu(self, memory: PyBoyMemoryView | bytes):
        if memory is None:
            memory = self.pyboy.memory

        return (
            True
            if self.menu_position_x(memory) == 4 and self.menu_position_y(memory) == 5
            else False
        )

    def screen_tiles(self, memory: PyBoyMemoryView | bytes):
        return [memory[i] for i in range(0xC3A0, 0xC508)]

    def reward(self, memory: bytes, action: int) -> tuple[float, float]:
        milestone = 0.0
        step = 0.0

        milestone += self.reward_core(memory)
        milestone += self.reward_story_milestones()

        if self.is_battle(self.pyboy.memory):
            milestone += self.reward_battle(memory)

        if self.is_battle(self.pyboy.memory) and self.number_of_turns_in_current_battle(
            memory
        ) == self.number_of_turns_in_current_battle(self.pyboy.memory):
            self.buffer_reward += milestone
            milestone = 0.0
        elif self.is_battle(self.pyboy.memory):
            milestone += self.buffer_reward * (1.0 if 0 < self.buffer_reward else 0.50)
            self.buffer_reward = 0.0

        if self.is_world(self.pyboy.memory):
            step += self.reward_position()
            # Completing a dialog is progress; without this, reading text is pure cost.
            if self.is_dialog(memory):
                milestone += self.dialog_exit_reward
        elif self.is_dialog(self.pyboy.memory):
            m, s = self.reward_dialog(memory, action=action)
            milestone += m
            step += s
        elif self.is_menu(self.pyboy.memory):
            step += self.in_menu_ticks / self.max_useless_ticks * self.base_reward
            if self.is_dialog(memory):
                milestone += self.dialog_exit_reward
        elif self.is_battle(self.pyboy.memory):
            m, s = self.reward_battle_useless_count(memory)
            milestone += m
            step += s

        step += self.reward_anti_loop(action=action, memory=memory)

        return milestone, step

    def reward_story_milestones(self) -> float:
        """One-shot bonuses for story / map progress (order-independent)."""
        reward = 0.0
        mid = self.map_id(self.pyboy.memory)

        checks = [
            (
                GOAL_LEFT_HOUSE,
                bool(self._start_map_id in HOUSE_MAPS and mid not in HOUSE_MAPS),
                self.left_house_reward,
            ),
            (GOAL_ROUTE_1, mid == MAP_ROUTE_1, self.route1_reward),
            (GOAL_OAKS_LAB, mid == MAP_OAKS_LAB, self.oaks_lab_reward),
            (
                GOAL_OAKS_PARCEL,
                bool(self.have_oaks_parcel(self.pyboy.memory)),
                self.event_reward,
            ),
            (
                GOAL_TOWN_MAP,
                bool(self.have_town_map(self.pyboy.memory)),
                self.event_reward,
            ),
        ]
        for name, hit, value in checks:
            if name not in self._milestones_hit and hit:
                self._milestones_hit.add(name)
                reward += value

        badges = self.badges(self.pyboy.memory)
        for i, name in enumerate(BADGE_GOALS):
            if name not in self._milestones_hit and i < len(badges) and badges[i]:
                self._milestones_hit.add(name)
                reward += self.badge_reward

        return reward

    def reward_anti_loop(self, action: int, memory: bytes) -> float:
        """Three-layer anti-loop + menu-spam penalties (PokeRL-style)."""
        penalty = 0.0
        triggered = False
        action = int(action)
        in_dialog = self.is_dialog(self.pyboy.memory)
        interacting = action in INTERACT_ACTIONS

        # 1) Graduated position visit penalties.
        # Skip while pressing A/B in world — that is the talk-to-NPC attempt.
        if self.is_world(self.pyboy.memory) and not interacting:
            visits = self.position_visit_counts.get(self.get_position(), 0)
            if visits > 5:
                penalty += self.visit_penalty_hard
                triggered = True
            elif visits > 3:
                penalty += self.visit_penalty_soft
                triggered = True

        # 2) Action pattern detection (sliding window).
        # Noop / movement loops always count. Prolonged A/B on the same tile
        # without an open dialog is camping (Oak lab idle), not "talking".
        actions = list(self.recent_actions) + [action]
        if len(actions) >= 4:
            a, b, c, d = actions[-4], actions[-3], actions[-2], actions[-1]
            if (
                a == c
                and b == d
                and a != b
                and a not in INTERACT_ACTIONS
                and b not in INTERACT_ACTIONS
            ):
                penalty += self.action_pattern_penalty
                triggered = True
        if len(actions) >= 8 and len(set(actions[-8:])) == 1:
            if not interacting or not in_dialog:
                penalty += self.action_pattern_penalty
                triggered = True

        # Idle in dialog (noop) is never useful — nudge toward A/B.
        if in_dialog and action == ACTION_NONE:
            penalty += self.menu_spam_penalty
            triggered = True

        # 3) Spatial loop: same tile revisited often in recent history.
        # World + non-interact only — standing to talk is not a movement loop.
        if (
            self.is_world(self.pyboy.memory)
            and not interacting
            and len(self.recent_positions) >= 8
        ):
            cur = self.get_position()
            if sum(1 for p in self.recent_positions if p == cur) >= 3:
                penalty += self.spatial_loop_penalty
                triggered = True

        # Menu spam: no menu state change.
        if self.is_menu(self.pyboy.memory) and self.is_menu_illegal_move(memory):
            penalty += self.menu_spam_penalty
            triggered = True

        if triggered:
            self.loop_flag = True

        return penalty

    def reward_battle_useless_count(self, memory: bytes) -> tuple[float, float]:
        if (
            self.number_of_turns_in_current_battle(memory)
            != self.number_of_turns_in_current_battle(self.pyboy.memory)
            or not self.is_battle(memory)
            and self.is_battle(self.pyboy.memory)
        ):
            return self.new_screen_reward, 0.0
        return 0.0, self.in_battle_ticks / self.max_useless_ticks * self.base_reward

    def reward_dialog(self, memory: bytes, action: int) -> tuple[float, float]:
        dialog_changed = self.dialog_id(memory) != self.dialog_id(self.pyboy.memory)
        # Peek whether this screen is new WITHOUT mutating seen-set here;
        # count() owns the set updates after reward.
        screen_hash = self.screen_tiles_hash(self.pyboy.memory)
        new_screen = screen_hash not in self._dialog_screens_seen
        current_dialog = self.get_dialog()
        is_new_dialog = current_dialog not in self.visited_dialogs
        if is_new_dialog:
            dialog_reward = self.new_dialog_reward
        elif dialog_changed:
            dialog_reward = self.dialog_advance_reward
        elif new_screen and int(action) in INTERACT_ACTIONS:
            # New text frame via A/B — not cursor blink (blink hashes already seen).
            dialog_reward = self.dialog_advance_reward * 0.5
        else:
            dialog_reward = 0.0
        progressed = dialog_changed or (new_screen and int(action) in INTERACT_ACTIONS)
        waste = (
            0.0
            if progressed
            else self.in_dialog_ticks / self.max_useless_dialog_ticks * self.base_reward
        )
        return dialog_reward, waste

    def reward_position(self):
        pos = self.get_position()
        ticks_here = self.visited_positions.get(pos, 0)
        visit_count = self.position_visit_counts.get(pos, 0)
        is_new_position = visit_count == 0
        waste_factor = min(ticks_here, self.max_useless_ticks) / self.max_useless_ticks
        exploration_reward = self.new_position_reward if is_new_position else 0.0
        step_penalty = self.base_reward * (1.0 + waste_factor * 9.0)
        return exploration_reward + step_penalty

    def is_menu_illegal_move(self, memory: bytes):
        return (
            self.current_menu_selected_item(self.pyboy.memory)
            == self.current_menu_selected_item(memory)
            and self.menu_position_x(self.pyboy.memory) == self.menu_position_x(memory)
            and self.menu_position_y(self.pyboy.memory) == self.menu_position_y(memory)
            and self.tile_data(self.pyboy.memory) == self.tile_data(memory)
        )

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
        reward += self.reward_map()
        reward += self.reward_player_items(memory)
        reward += self.reward_stored_items(memory)

        return reward

    def reward_player_items(self, memory: bytes):
        return (
            sum(self.items_quantities(self.pyboy.memory))
            - sum(self.items_quantities(memory))
        ) * self.new_item_reward

    def reward_stored_items(self, memory: bytes):
        return (
            (
                sum(self.stored_items_quantities(self.pyboy.memory))
                - sum(self.stored_items_quantities(memory))
            )
            * self.new_item_reward
            * 0.5
        )

    def reward_map(self):
        return (
            self.new_screen_reward
            if self.map_id(self.pyboy.memory) not in self.visited_maps
            else 0.0
        )

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
                        reward += self.status_reward
                    elif status_x_bit == 0 and status_y_bit == 1:
                        reward += -self.status_reward

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

    def is_goal_satisfied(self, goal: str) -> bool:
        """Whether ``goal`` is already true in the current game state.

        Used by curriculum to skip stages when flags/badges were obtained out
        of the recommended STAGE_ORDER.
        """
        mem = self.pyboy.memory
        mid = self.map_id(mem)
        badges = self.badges(mem)

        if goal == GOAL_LEFT_HOUSE:
            return mid not in HOUSE_MAPS
        if goal == GOAL_ROUTE_1:
            return mid == MAP_ROUTE_1
        if goal == GOAL_OAKS_LAB:
            return mid == MAP_OAKS_LAB
        if goal == GOAL_OAKS_PARCEL:
            return bool(self.have_oaks_parcel(mem))
        if goal == GOAL_TOWN_MAP:
            return bool(self.have_town_map(mem))
        if goal == GOAL_FOUGHT_BROCK:
            return bool(self.fought_brock_yet(mem))
        if goal == GOAL_FOUGHT_MISTY:
            return bool(self.fought_misty_yet(mem))
        if goal == GOAL_FOUGHT_SURGE:
            return bool(self.fought_lt_surge_yet(mem))
        if goal == GOAL_FOUGHT_ERIKA:
            return bool(self.fought_erika_yet(mem))
        if goal == GOAL_FOUGHT_KOGA:
            return bool(self.fought_koga_yet(mem))
        if goal == GOAL_FOUGHT_SABRINA:
            return bool(self.fought_sabrina_yet(mem))
        if goal == GOAL_FOUGHT_BLAINE:
            return bool(self.fought_blaine_yet(mem))
        if goal == GOAL_FOUGHT_GIOVANNI:
            return bool(self.fought_giovanni_yet(mem))
        if goal in BADGE_GOALS:
            idx = BADGE_GOALS.index(goal)
            return bool(idx < len(badges) and badges[idx])
        if goal == GOAL_SS_ANNE:
            return bool(self.is_ss_anne_here(mem))
        if goal == GOAL_LAPRAS:
            return bool(self.did_you_get_lapras_yet(mem))
        if goal == GOAL_SNORLAX:
            return bool(
                self.fought_snorlax_yet_vermilion(mem)
                or self.fought_snorlax_yet_celadon(mem)
            )
        if goal == GOAL_ARTICUNO:
            return bool(self.fought_articuno_yet(mem))
        if goal == GOAL_ZAPDOS:
            return bool(self.fought_zapdos_yet(mem))
        if goal == GOAL_MOLTRES:
            return bool(self.fought_moltres_yet(mem))
        if goal == GOAL_FOSSIL:
            return bool(self.fossilized_pokemon(mem))
        if goal == GOAL_MEWTWO:
            return bool(self.mewtwo_can_be_caught(mem))
        if goal == GOAL_ALL_BADGES:
            return bool(badges) and all(badges)
        return False

    def goal_reached(self) -> bool:
        return self.is_goal_satisfied(self.goal)

    def terminated(self, memory: bytes):
        return self.goal_reached()

    def truncated(self, memory: bytes):
        # Stuck fuse uses in_dialog_ticks (resets on progress), not cumulative
        # visited_dialogs — long scripts reuse one dialog_id for many boxes.
        # Also require is_dialog so lingering dialog_id in RAM cannot truncate
        # after the conversation already ended.
        in_dialog = self.is_dialog(self.pyboy.memory)
        stuck_dialog = (
            in_dialog and self.max_useless_dialog_ticks <= self.in_dialog_ticks
        )
        # Tile fuse pauses during dialog so long Oak/NPC text is not cut off.
        stuck_tile = (
            not in_dialog
            and self.max_useless_ticks
            <= self.visited_positions.get(self.get_position(), 0)
        )
        return (
            True
            if stuck_tile
            or stuck_dialog
            or self.max_useless_ticks <= self.in_battle_ticks
            or self.max_useless_ticks <= self.in_menu_ticks
            else False
        )

    def current_milestone(self) -> str:
        # Furthest known goal that is currently satisfied (order-independent).
        priority = (
            GOAL_ALL_BADGES,
            GOAL_MEWTWO,
            GOAL_MOLTRES,
            GOAL_ZAPDOS,
            GOAL_ARTICUNO,
            GOAL_SNORLAX,
            GOAL_LAPRAS,
            GOAL_FOSSIL,
            GOAL_BADGE_8,
            GOAL_FOUGHT_GIOVANNI,
            GOAL_BADGE_7,
            GOAL_FOUGHT_BLAINE,
            GOAL_BADGE_6,
            GOAL_FOUGHT_SABRINA,
            GOAL_BADGE_5,
            GOAL_FOUGHT_KOGA,
            GOAL_BADGE_4,
            GOAL_FOUGHT_ERIKA,
            GOAL_BADGE_3,
            GOAL_FOUGHT_SURGE,
            GOAL_SS_ANNE,
            GOAL_BADGE_2,
            GOAL_FOUGHT_MISTY,
            GOAL_BADGE_1,
            GOAL_FOUGHT_BROCK,
            GOAL_TOWN_MAP,
            GOAL_OAKS_PARCEL,
            GOAL_ROUTE_1,
            GOAL_OAKS_LAB,
            GOAL_LEFT_HOUSE,
        )
        for g in priority:
            if self.is_goal_satisfied(g):
                return g
        if self._milestones_hit:
            return sorted(self._milestones_hit)[-1]
        return "start"

    def is_illegal_world_move(self, memory: bytes, action: int):
        return (
            self.is_world(self.pyboy.memory)
            and self.is_world(memory)
            and self.map_id(self.pyboy.memory) == self.map_id(memory)
            and self.position_x(self.pyboy.memory) == self.position_x(memory)
            and self.position_y(self.pyboy.memory) == self.position_y(memory)
        )

    def menu_battle_dialog_data(self):
        data = self.data_normalizer(
            [
                self.menu_position_x(self.pyboy.memory),
                self.menu_position_y(self.pyboy.memory),
                self.id_of_the_first_displayed_menu_item(self.pyboy.memory),
            ]
        ) + [
            self.current_menu_selected_item(self.pyboy.memory)
            / max(self.id_of_the_last_menu_item(self.pyboy.memory), 1),
        ]

        return (
            data
            if self.is_menu(self.pyboy.memory)
            or self.is_battle(self.pyboy.memory)
            or self.is_dialog(self.pyboy.memory)
            else [0] * len(data)
        )

    def world_data(self):
        data = self.data_normalizer(
            [
                self.position_x(self.pyboy.memory),
                self.position_y(self.pyboy.memory),
                self.bike_speed(self.pyboy.memory),
            ]
            + self.sprite_data_x_positions(self.pyboy.memory)
            + self.sprite_data_y_positions(self.pyboy.memory)
        )

        return data if self.is_world(self.pyboy.memory) else [0] * len(data)

    def game_mode_flags_data(self, memory: PyBoyMemoryView | bytes):
        return [
            int(self.is_battle(memory)),
            int(self.is_dialog(memory)),
            int(self.is_menu(memory)),
            int(self.is_world(memory)),
        ]

    def data_normalizer(self, values: list[int], max=0xFF):
        return [x / max for x in values]

    def core_data(self):
        data = []

        data += self.data_normalizer(
            [self.in_battle_ticks, self.in_menu_ticks],
            max=self.max_useless_ticks,
        )
        data += self.data_normalizer(
            [self.in_dialog_ticks],
            max=self.max_useless_dialog_ticks,
        )
        data += self.player_data()
        data += self.pokedex_data()
        data += self.map_data()

        return data

    def map_data(self, memory: PyBoyMemoryView | bytes | None = None):
        if memory is None:
            memory = self.pyboy.memory

        data = self.data_normalizer(
            [
                self.visited_positions.get(
                    self.get_position(memory, offset_x=dx, offset_y=dy), 0
                )
                for dx in range(-self.map_vision_radius, self.map_vision_radius + 1)
                for dy in range(-self.map_vision_radius, self.map_vision_radius + 1)
            ],
            max=self.max_useless_ticks,
        )

        return data if self.is_world(memory) else [0] * len(data)

    def inventory_data(self, memory: PyBoyMemoryView | bytes):
        data = []

        data += self.data_normalizer(
            [
                self.items_quantities(memory)[
                    self.real_current_menu_selected_item(memory)
                ]
            ]
        )
        data += self.data_normalizer([self.player_money(memory)], max=0xFFFFFF)
        data += self.data_normalizer(
            [
                self.stored_items_quantities(memory)[
                    self.real_current_menu_selected_item(memory)
                ]
            ]
        )
        data += self.data_normalizer([self.game_coins(memory)], max=0xFFFF)

        return data if self.is_eq_menu(memory) else [0] * len(data)

    def id_of_the_last_menu_item(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCC28]

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
        data = [memory[0xC100 + 0x10 * x] for x in range(16)]

        return data if self.is_world(memory) else [0] * len(data)

    def sprite_data_movement_statuses(self, memory: PyBoyMemoryView | bytes):
        data = [memory[0xC101 + 0x10 * x] for x in range(16)]

        return data if self.is_world(memory) else [0] * len(data)

    def sprite_data_facing_directions(self, memory: PyBoyMemoryView | bytes):
        data = [memory[0xC109 + 0x10 * x] for x in range(16)]

        return data if self.is_world(memory) else [0] * len(data)

    def sprite_data_y_positions(self, memory: PyBoyMemoryView | bytes):
        data = [memory[0xC204 + 0x10 * x] for x in range(16)]

        return data if self.is_world(memory) else [0] * len(data)

    def sprite_data_x_positions(self, memory: PyBoyMemoryView | bytes):
        data = [memory[0xC205 + 0x10 * x] for x in range(16)]

        return data if self.is_world(memory) else [0] * len(data)

    def menu_position_x(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCC24]

    def menu_position_y(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCC25]

    def current_menu_selected_item(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCC26]

    def real_current_menu_selected_item(self, memory: PyBoyMemoryView | bytes):
        return self.current_menu_selected_item(
            memory
        ) + self.id_of_the_first_displayed_menu_item(memory)

    def id_of_the_first_displayed_menu_item(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCC36]

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
            self.player_pokemons_level(self.pyboy.memory),
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

    def player_pokemons_level(self, memory: PyBoyMemoryView | bytes = None):
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
        data += self.stored_pokemon_statuses(memory)
        data += self.data_normalizer(
            self.stored_pokemon_experiences(memory), max=0xFFFFFF
        )
        data += self.data_normalizer(self.stored_pokemon_pps(memory))

        return data if self.is_eq_menu(memory) else [0] * len(data)

    def stored_pokemon_pps(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[
                x
                + self.real_current_menu_selected_item(memory)
                * self.__stored_pokemon_size
            ]
            for x in range(0xDAB3, 0xDAB7)
        ]

    def stored_pokemon_experiences(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[
                0xDAA4
                + self.real_current_menu_selected_item(memory)
                * self.__stored_pokemon_size
            ]
            | (
                memory[
                    0xDAA5
                    + self.real_current_menu_selected_item(memory)
                    * self.__stored_pokemon_size
                ]
                << 8
            )
            | (
                memory[
                    0xDAA6
                    + self.real_current_menu_selected_item(memory)
                    * self.__stored_pokemon_size
                ]
                << 16
            )
        ]

    def stored_pokemon_moves(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[
                x
                + self.real_current_menu_selected_item(memory)
                * self.__stored_pokemon_size
            ]
            for x in range(0xDA9E, 0xDAA2)
        ]

    def stored_pokemon_types(self, memory: PyBoyMemoryView | bytes):
        data = [
            memory[
                x
                + self.real_current_menu_selected_item(memory)
                * self.__stored_pokemon_size
            ]
            for x in range(0xDA9B, 0xDA9D)
        ]

        return data if self.is_eq_menu(memory) else [0] * len(data)

    def stored_pokemon_statuses(self, memory: PyBoyMemoryView | bytes):
        return [
            bit
            for bit in self.bits_extractor(
                memory[
                    0xDA9A
                    + self.real_current_menu_selected_item(memory)
                    * self.__stored_pokemon_size
                ]
            )
        ]

    def stored_pokemon_levels(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[
                0xDA99
                + self.real_current_menu_selected_item(memory)
                * self.__stored_pokemon_size
            ]
        ]

    def stored_pokemon_hps(self, memory: PyBoyMemoryView | bytes):
        return [
            x | (y << 8)
            for x, y in zip(
                [
                    memory[
                        0xDA97
                        + self.real_current_menu_selected_item(memory)
                        * self.__stored_pokemon_size
                    ]
                ],
                [
                    memory[
                        0xDA98
                        + self.real_current_menu_selected_item(memory)
                        * self.__stored_pokemon_size
                    ]
                ],
            )
        ]

    def stored_pokemon_ids(self, memory: PyBoyMemoryView | bytes):
        data = [
            memory[
                0xDA96
                + self.real_current_menu_selected_item(memory)
                * self.__stored_pokemon_size
            ]
        ]

        return data if self.is_eq_menu(memory) else [0] * len(data)

    def tile_data(self, memory: PyBoyMemoryView | bytes):
        return [memory[i] for i in range(0xC490, 0xC4F1)]

    def reward_battle(self, memory: bytes):
        reward = 0.0

        reward += self.reward_enemy_hp(memory)
        reward += self.reward_enemy_status(memory)
        reward += self.reward_pokemon_current_hp(memory)
        reward += self.reward_pokemon_status(memory)

        return reward

    def reward_enemy_hp(self, memory: bytes):
        if (
            self.enemy_max_hp(self.pyboy.memory) == 0
            or self.pokemon_max_hp(self.pyboy.memory) == 0
        ):
            return 0

        diff = (
            self.enemy_hp(memory) - self.enemy_hp(self.pyboy.memory)
        ) / self.enemy_max_hp(self.pyboy.memory)

        if self.enemy_max_hp(self.pyboy.memory) < self.pokemon_max_hp(
            self.pyboy.memory
        ):
            diff += min(
                (
                    self.enemy_max_hp(self.pyboy.memory)
                    - self.pokemon_max_hp(self.pyboy.memory)
                )
                / self.enemy_max_hp(self.pyboy.memory),
                1.0,
            )

        return diff

    def reward_enemy_status(self, memory: bytes):
        reward = 0

        for bit_before, bit_after in zip(
            self.enemy_status(memory), self.enemy_status(self.pyboy.memory)
        ):
            if bit_before == 0 and bit_after == 1:
                reward += self.status_reward
            elif bit_before == 1 and bit_after == 0:
                reward += -self.status_reward

        return max(0, reward)

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
                reward += -self.status_reward
            elif bit_before == 1 and bit_after == 0:
                reward += self.status_reward

        return reward

    def reward_pokedex(self, memory: bytes):
        return self.reward_pokedex_own(memory) + self.reward_pokedex_seen(memory)

    def reward_pokedex_own(self, memory: bytes):
        for bit_before, bit_after, visited in zip(
            self.pokedex_own(memory),
            self.pokedex_own(self.pyboy.memory),
            self.visited_pokedex_own,
        ):
            if bit_before == 0 and bit_after == 1 and visited == 0:
                return self.new_pokedex_own_reward

        return 0.0

    def reward_pokedex_seen(self, memory: bytes):
        for bit_before, bit_after, visited in zip(
            self.pokedex_seen(memory),
            self.pokedex_seen(self.pyboy.memory),
            self.visited_pokedex_seen,
        ):
            if bit_before == 0 and bit_after == 1 and visited == 0:
                return self.new_pokedex_seen_reward

        return 0.0

    def reward_badges(self, memory: bytes):
        reward = 0

        for bit_before, bit_after in zip(
            self.badges(memory), self.badges(self.pyboy.memory)
        ):
            if bit_before == 0 and bit_after == 1:
                reward += self.badge_reward

        return reward

    def reward_event_flags(self, memory: bytes):
        reward = 0

        for flag_x, flag_y in zip(
            self.event_flags_data(memory), self.event_flags_data(self.pyboy.memory)
        ):
            if flag_x == 0 and flag_y == 1:
                reward += self.event_reward

        return reward

    def reward_milestones(self, memory: bytes):
        reward = 0.0

        reward += self.reward_badges(memory)

        reward += self.reward_event_flags(memory)

        return reward
