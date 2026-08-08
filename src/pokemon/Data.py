import hashlib
from collections import deque
from multiprocessing.synchronize import RLock
import pickle
from dataclasses import dataclass, field

from pyboy import PyBoy, PyBoyMemoryView
import torch

# Raw WRAM addresses below are documented at:
# https://datacrystal.tcrf.net/wiki/Pokémon_Red_and_Blue/RAM_map


# Pokemon Red early-game map IDs (used for curriculum milestones).
MAP_PALLET_TOWN = 0
MAP_ROUTE_1 = 12
MAP_REDS_HOUSE_2F = 37
MAP_REDS_HOUSE_1F = 38
MAP_RIVALS_HOUSE = 39
MAP_OAKS_LAB = 40
HOUSE_MAPS = frozenset({MAP_REDS_HOUSE_1F, MAP_REDS_HOUSE_2F})

# Map IDs for the rest of the curriculum (GOAL_ALLOWED_MAPS below). wCurMap
# values match the pret/pokered disassembly's map id ordering:
# https://github.com/pret/pokered/blob/master/constants/map_constants.asm
# (cross-checked against the RAM map doc linked above).
MAP_VIRIDIAN_CITY = 1
MAP_PEWTER_CITY = 2
MAP_CERULEAN_CITY = 3
MAP_LAVENDER_TOWN = 4
MAP_VERMILION_CITY = 5
MAP_CELADON_CITY = 6
MAP_FUCHSIA_CITY = 7
MAP_CINNABAR_ISLAND = 8
MAP_INDIGO_PLATEAU = 9
MAP_SAFFRON_CITY = 10

MAP_ROUTE_2 = 13
MAP_ROUTE_3 = 14
MAP_ROUTE_4 = 15
MAP_ROUTE_5 = 16
MAP_ROUTE_6 = 17
MAP_ROUTE_7 = 18
MAP_ROUTE_8 = 19
MAP_ROUTE_9 = 20
MAP_ROUTE_10 = 21
MAP_ROUTE_11 = 22
MAP_ROUTE_12 = 23
MAP_ROUTE_13 = 24
MAP_ROUTE_14 = 25
MAP_ROUTE_15 = 26
MAP_ROUTE_16 = 27
MAP_ROUTE_17 = 28
MAP_ROUTE_18 = 29
MAP_ROUTE_19 = 30
MAP_ROUTE_20 = 31
MAP_ROUTE_21 = 32
MAP_ROUTE_22 = 33
MAP_ROUTE_23 = 34
MAP_ROUTE_24 = 35
MAP_ROUTE_25 = 36

MAP_VIRIDIAN_POKECENTER = 41
MAP_VIRIDIAN_MART = 42
MAP_VIRIDIAN_GYM = 45
MAP_VIRIDIAN_FOREST_NORTH_GATE = 47
MAP_ROUTE_2_GATE = 49
MAP_VIRIDIAN_FOREST_SOUTH_GATE = 50
MAP_VIRIDIAN_FOREST = 51
MAP_PEWTER_GYM = 54
MAP_PEWTER_MART = 56
MAP_PEWTER_POKECENTER = 58
MAP_MT_MOON_1F = 59
MAP_MT_MOON_B1F = 60
MAP_MT_MOON_B2F = 61
MAP_CERULEAN_POKECENTER = 64
MAP_CERULEAN_GYM = 65
MAP_CERULEAN_MART = 67
MAP_MT_MOON_POKECENTER = 68
MAP_ROUTE_5_GATE = 70
MAP_UNDERGROUND_PATH_ROUTE_5 = 71
MAP_ROUTE_6_GATE = 73
MAP_UNDERGROUND_PATH_ROUTE_6 = 74
MAP_ROUTE_7_GATE = 76
MAP_UNDERGROUND_PATH_ROUTE_7 = 77
MAP_ROUTE_8_GATE = 79
MAP_UNDERGROUND_PATH_ROUTE_8 = 80
MAP_ROCK_TUNNEL_POKECENTER = 81
MAP_ROCK_TUNNEL_1F = 82
MAP_POWER_PLANT = 83
MAP_ROUTE_12_GATE_1F = 87
MAP_BILLS_HOUSE = 88
MAP_VERMILION_POKECENTER = 89
MAP_VERMILION_MART = 91
MAP_VERMILION_GYM = 92
MAP_VERMILION_DOCK = 94
MAP_VICTORY_ROAD_1F = 108
MAP_HALL_OF_FAME = 118
MAP_UNDERGROUND_PATH_NORTH_SOUTH = 119
MAP_CHAMPIONS_ROOM = 120
MAP_UNDERGROUND_PATH_WEST_EAST = 121
MAP_CELADON_MART_1F = 122
MAP_CELADON_POKECENTER = 133
MAP_CELADON_GYM = 134
MAP_GAME_CORNER = 135
MAP_LAVENDER_POKECENTER = 141
MAP_MR_FUJIS_HOUSE = 149
MAP_LAVENDER_MART = 150
MAP_FUCHSIA_MART = 152
MAP_FUCHSIA_POKECENTER = 154
MAP_FUCHSIA_GYM = 157
MAP_POKEMON_MANSION_1F = 165
MAP_CINNABAR_GYM = 166
MAP_CINNABAR_LAB = 167
MAP_CINNABAR_POKECENTER = 171
MAP_CINNABAR_MART = 172
MAP_INDIGO_PLATEAU_LOBBY = 174
MAP_FIGHTING_DOJO = 177
MAP_SAFFRON_GYM = 178
MAP_SAFFRON_MART = 180
MAP_SILPH_CO_1F = 181
MAP_SAFFRON_POKECENTER = 182
MAP_ROUTE_15_GATE_1F = 184
MAP_ROUTE_16_GATE_1F = 186
MAP_ROUTE_18_GATE_1F = 190
MAP_SEAFOAM_ISLANDS_1F = 192
MAP_ROUTE_22_GATE = 193
MAP_VICTORY_ROAD_2F = 194
MAP_VICTORY_ROAD_3F = 198
MAP_LANCES_ROOM = 113

# Multi-floor dungeons: grouped by area instead of one constant per floor.
MAP_SS_ANNE = frozenset(range(95, 105))  # 1F..B1F_ROOMS ($5F-$68)
MAP_POKEMON_TOWER = frozenset(range(142, 149))  # 1F-7F ($8E-$94)
MAP_SEAFOAM_ISLANDS_CAVES = frozenset({159, 160, 161, 162})  # B1F-B4F ($9F-$A2)
MAP_ROCKET_HIDEOUT = frozenset({199, 200, 201, 202, 203})  # B1F-B4F + elevator ($C7-$CB)
MAP_CINNABAR_LAB_ROOMS = frozenset({167, 168, 169, 170})  # main + trade/metronome/fossil ($A7-$AA)
MAP_SILPH_CO = frozenset(
    {181, 207, 208, 209, 210, 211, 212, 213, 233, 234, 235, 236}
)  # 1F, 2F-8F, 9F-11F, elevator
MAP_CERULEAN_CAVE = frozenset({226, 227, 228})  # 2F, B1F, 1F ($E2-$E4)
MAP_ELITE_FOUR = frozenset({245, 246, 247})  # Lorelei/Bruno/Agatha rooms ($F5-$F7)

# Emulator button indices — A/B must be mashable to start and advance dialogs.
ACTION_A = 0
ACTION_B = 1
ACTION_NONE = 8
INTERACT_ACTIONS = frozenset({ACTION_A, ACTION_B})

# Curriculum / episode goals (map, event flags, badges).
GOAL_LEFT_HOUSE = "left_house"
# Stepping onto Route 1 before getting a starter auto-triggers Oak's
# "it's dangerous" intercept — this fires as a dialog while map_id is still
# MAP_PALLET_TOWN (the map never actually transitions to Route 1 for this
# pre-starter trigger, confirmed live: map=0 dialog_id=1). Distinct from
# GOAL_ROUTE_1 below (the later, post-starter return trip, a real map change).
GOAL_ROUTE_1_ENTRY = "route1_entry"
ROUTE1_ENTRY_DIALOG_ID = 1
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

# Maps on the critical path for each curriculum goal (excludes distraction
# detours like the rivals' house, Game Corner floor, Safari Zone, etc. unless
# that detour IS the goal). Built up incrementally: each stage's set is the
# previous stage's plus the new ground covered reaching it, because a stage
# can start cold from "start" (no stage checkpoint saved yet — see
# curriculum_config.py) and the agent then has to walk the whole way from
# Pallet Town, not just the final leg.
_MAPS_LEFT_HOUSE = HOUSE_MAPS | {MAP_PALLET_TOWN}
_MAPS_ROUTE_1_ENTRY = frozenset({MAP_PALLET_TOWN, MAP_ROUTE_1, MAP_OAKS_LAB})
# Includes MAP_ROUTE_1: goal auto-advances to oaks_lab the instant the entry
# trigger fires, but the forced walk-into-lab cutscene can still read
# map_id == MAP_ROUTE_1 for a few ticks afterward.
_MAPS_OAKS_LAB = frozenset({MAP_PALLET_TOWN, MAP_OAKS_LAB, MAP_ROUTE_1})
_MAPS_ROUTE_1 = frozenset({MAP_PALLET_TOWN, MAP_OAKS_LAB, MAP_ROUTE_1})

_MAPS_OAKS_PARCEL = _MAPS_ROUTE_1 | {
    MAP_VIRIDIAN_CITY,
    MAP_VIRIDIAN_MART,
    MAP_VIRIDIAN_POKECENTER,
}
_MAPS_TOWN_MAP = _MAPS_OAKS_PARCEL | {MAP_RIVALS_HOUSE}  # Daisy's gift, Pallet Town

_MAPS_FOUGHT_BROCK = _MAPS_TOWN_MAP | {
    MAP_ROUTE_2,
    MAP_ROUTE_2_GATE,
    MAP_VIRIDIAN_FOREST_NORTH_GATE,
    MAP_VIRIDIAN_FOREST_SOUTH_GATE,
    MAP_VIRIDIAN_FOREST,
    MAP_PEWTER_CITY,
    MAP_PEWTER_GYM,
    MAP_PEWTER_MART,
    MAP_PEWTER_POKECENTER,
}
_MAPS_BADGE_1 = _MAPS_FOUGHT_BROCK  # badge is handed over in the same gym room

_MAPS_FOUGHT_MISTY = _MAPS_BADGE_1 | {
    MAP_ROUTE_3,
    MAP_MT_MOON_1F,
    MAP_MT_MOON_B1F,
    MAP_MT_MOON_B2F,
    MAP_MT_MOON_POKECENTER,
    MAP_ROUTE_4,
    MAP_CERULEAN_CITY,
    MAP_CERULEAN_GYM,
    MAP_CERULEAN_POKECENTER,
    MAP_CERULEAN_MART,
}
_MAPS_BADGE_2 = _MAPS_FOUGHT_MISTY

_MAPS_SS_ANNE = (
    _MAPS_BADGE_2
    | {
        MAP_ROUTE_24,
        MAP_ROUTE_25,
        MAP_BILLS_HOUSE,  # S.S. Ticket
        MAP_ROUTE_5,
        MAP_ROUTE_5_GATE,
        MAP_UNDERGROUND_PATH_ROUTE_5,
        MAP_ROUTE_6,
        MAP_ROUTE_6_GATE,
        MAP_UNDERGROUND_PATH_ROUTE_6,
        MAP_VERMILION_CITY,
        MAP_VERMILION_POKECENTER,
        MAP_VERMILION_MART,
        MAP_VERMILION_DOCK,
    }
    | MAP_SS_ANNE
)
_MAPS_FOUGHT_SURGE = _MAPS_SS_ANNE | {MAP_VERMILION_GYM}
_MAPS_BADGE_3 = _MAPS_FOUGHT_SURGE

_MAPS_FOUGHT_ERIKA = _MAPS_BADGE_3 | {
    MAP_ROUTE_7,
    MAP_ROUTE_7_GATE,
    MAP_UNDERGROUND_PATH_ROUTE_7,
    MAP_SAFFRON_CITY,  # just passed through — gym is locked until Silph Co.
    MAP_SAFFRON_POKECENTER,
    MAP_CELADON_CITY,
    MAP_CELADON_GYM,
    MAP_CELADON_POKECENTER,
    MAP_CELADON_MART_1F,
}
_MAPS_BADGE_4 = _MAPS_FOUGHT_ERIKA

# Fuchsia (Koga) is gated behind the sleeping Snorlax on Route 12/16, which
# needs the Poke Flute from Mr. Fuji, which needs the Silph Scope from Team
# Rocket's hideout under the Celadon Game Corner — a hard prerequisite for
# physically reaching Koga, even though Silph Co./Sabrina are the *later*
# curriculum stage (see stage_fought_koga vs stage_fought_sabrina in
# curriculum_config.py).
_MAPS_FOUGHT_KOGA = (
    _MAPS_BADGE_4
    | {
        MAP_GAME_CORNER,
        MAP_ROUTE_8,
        MAP_ROUTE_8_GATE,
        MAP_UNDERGROUND_PATH_ROUTE_8,
        MAP_ROUTE_9,
        MAP_ROUTE_10,
        MAP_ROCK_TUNNEL_POKECENTER,
        MAP_ROCK_TUNNEL_1F,
        MAP_LAVENDER_TOWN,
        MAP_LAVENDER_POKECENTER,
        MAP_LAVENDER_MART,
        MAP_MR_FUJIS_HOUSE,
        MAP_ROUTE_12,
        MAP_ROUTE_12_GATE_1F,
        MAP_ROUTE_13,
        MAP_ROUTE_14,
        MAP_ROUTE_15,
        MAP_ROUTE_15_GATE_1F,
        MAP_ROUTE_16,
        MAP_ROUTE_16_GATE_1F,
        MAP_ROUTE_17,
        MAP_ROUTE_18,
        MAP_ROUTE_18_GATE_1F,
        MAP_FUCHSIA_CITY,
        MAP_FUCHSIA_GYM,
        MAP_FUCHSIA_MART,
        MAP_FUCHSIA_POKECENTER,
    }
    | MAP_POKEMON_TOWER
    | MAP_ROCKET_HIDEOUT
)
_MAPS_BADGE_5 = _MAPS_FOUGHT_KOGA

_MAPS_FOUGHT_SABRINA = (
    _MAPS_BADGE_5
    | {
        MAP_FIGHTING_DOJO,
        MAP_SAFFRON_GYM,
        MAP_SAFFRON_MART,
    }
    | MAP_SILPH_CO
)
_MAPS_BADGE_6 = _MAPS_FOUGHT_SABRINA

_MAPS_FOUGHT_BLAINE = (
    _MAPS_BADGE_6
    | {
        MAP_ROUTE_19,
        MAP_ROUTE_20,
        MAP_SEAFOAM_ISLANDS_1F,
        MAP_ROUTE_21,
        MAP_CINNABAR_ISLAND,
        MAP_CINNABAR_GYM,
        MAP_CINNABAR_MART,
        MAP_CINNABAR_POKECENTER,
        MAP_POKEMON_MANSION_1F,
    }
    | MAP_SEAFOAM_ISLANDS_CAVES
)
_MAPS_BADGE_7 = _MAPS_FOUGHT_BLAINE

_MAPS_FOUGHT_GIOVANNI = _MAPS_BADGE_7 | {MAP_VIRIDIAN_GYM}  # finally unlocked
_MAPS_BADGE_8 = _MAPS_FOUGHT_GIOVANNI

_MAPS_LAPRAS = _MAPS_BADGE_8  # Silph Co. 7F gift — already covered above
_MAPS_SNORLAX = _MAPS_LAPRAS  # both sites (Route 12 / Route 16) already covered

_MAPS_ARTICUNO = _MAPS_SNORLAX  # Seafoam Islands already covered (Blaine leg)

_MAPS_ZAPDOS = _MAPS_ARTICUNO | {MAP_POWER_PLANT}  # reached by Surfing off Route 10

_MAPS_MOLTRES = _MAPS_ZAPDOS | {
    MAP_ROUTE_22,
    MAP_ROUTE_22_GATE,
    MAP_ROUTE_23,
    MAP_VICTORY_ROAD_1F,
    MAP_VICTORY_ROAD_2F,
    MAP_VICTORY_ROAD_3F,
}

# Revived at Cinnabar Lab; the raw fossil pickup itself is Mt Moon B2F,
# already covered by the Misty leg above.
_MAPS_FOSSIL = _MAPS_MOLTRES | MAP_CINNABAR_LAB_ROOMS

# Cerulean Cave only opens after Hall of Fame, so Mewtwo pulls in the
# Elite Four / Champion path too.
_MAPS_MEWTWO = (
    _MAPS_FOSSIL
    | {
        MAP_INDIGO_PLATEAU,
        MAP_INDIGO_PLATEAU_LOBBY,
        MAP_HALL_OF_FAME,
        MAP_UNDERGROUND_PATH_NORTH_SOUTH,
        MAP_UNDERGROUND_PATH_WEST_EAST,
        MAP_LANCES_ROOM,
        MAP_CHAMPIONS_ROOM,
    }
    | MAP_ELITE_FOUR
    | MAP_CERULEAN_CAVE
)

_MAPS_ALL_BADGES = _MAPS_MEWTWO  # by now every badge-path map is covered

GOAL_ALLOWED_MAPS: dict[str, frozenset[int]] = {
    GOAL_LEFT_HOUSE: _MAPS_LEFT_HOUSE,
    GOAL_ROUTE_1_ENTRY: _MAPS_ROUTE_1_ENTRY,
    GOAL_OAKS_LAB: _MAPS_OAKS_LAB,
    GOAL_ROUTE_1: _MAPS_ROUTE_1,
    GOAL_OAKS_PARCEL: _MAPS_OAKS_PARCEL,
    GOAL_TOWN_MAP: _MAPS_TOWN_MAP,
    GOAL_FOUGHT_BROCK: _MAPS_FOUGHT_BROCK,
    GOAL_BADGE_1: _MAPS_BADGE_1,
    GOAL_FOUGHT_MISTY: _MAPS_FOUGHT_MISTY,
    GOAL_BADGE_2: _MAPS_BADGE_2,
    GOAL_SS_ANNE: _MAPS_SS_ANNE,
    GOAL_FOUGHT_SURGE: _MAPS_FOUGHT_SURGE,
    GOAL_BADGE_3: _MAPS_BADGE_3,
    GOAL_FOUGHT_ERIKA: _MAPS_FOUGHT_ERIKA,
    GOAL_BADGE_4: _MAPS_BADGE_4,
    GOAL_FOUGHT_KOGA: _MAPS_FOUGHT_KOGA,
    GOAL_BADGE_5: _MAPS_BADGE_5,
    GOAL_FOUGHT_SABRINA: _MAPS_FOUGHT_SABRINA,
    GOAL_BADGE_6: _MAPS_BADGE_6,
    GOAL_FOUGHT_BLAINE: _MAPS_FOUGHT_BLAINE,
    GOAL_BADGE_7: _MAPS_BADGE_7,
    GOAL_FOUGHT_GIOVANNI: _MAPS_FOUGHT_GIOVANNI,
    GOAL_BADGE_8: _MAPS_BADGE_8,
    GOAL_LAPRAS: _MAPS_LAPRAS,
    GOAL_SNORLAX: _MAPS_SNORLAX,
    GOAL_ARTICUNO: _MAPS_ARTICUNO,
    GOAL_ZAPDOS: _MAPS_ZAPDOS,
    GOAL_MOLTRES: _MAPS_MOLTRES,
    GOAL_FOSSIL: _MAPS_FOSSIL,
    GOAL_MEWTWO: _MAPS_MEWTWO,
    GOAL_ALL_BADGES: _MAPS_ALL_BADGES,
}

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

# Location goals: true only while on the map / outside the house. Leaving undoes
# live progress unless the goal was curriculum-cleared (auto_advance).
REGRESSABLE_GOALS = frozenset(
    {GOAL_LEFT_HOUSE, GOAL_ROUTE_1_ENTRY, GOAL_ROUTE_1, GOAL_OAKS_LAB}
)

# Ordered early→late checklist for live progress counting / regression metrics.
STORY_GOAL_ORDER = (
    GOAL_LEFT_HOUSE,
    GOAL_ROUTE_1_ENTRY,
    GOAL_OAKS_LAB,
    GOAL_ROUTE_1,
    GOAL_OAKS_PARCEL,
    GOAL_TOWN_MAP,
    GOAL_FOUGHT_BROCK,
    GOAL_BADGE_1,
    GOAL_FOUGHT_MISTY,
    GOAL_BADGE_2,
    GOAL_SS_ANNE,
    GOAL_FOUGHT_SURGE,
    GOAL_BADGE_3,
    GOAL_FOUGHT_ERIKA,
    GOAL_BADGE_4,
    GOAL_FOUGHT_KOGA,
    GOAL_BADGE_5,
    GOAL_FOUGHT_SABRINA,
    GOAL_BADGE_6,
    GOAL_FOUGHT_BLAINE,
    GOAL_BADGE_7,
    GOAL_FOUGHT_GIOVANNI,
    GOAL_BADGE_8,
    GOAL_LAPRAS,
    GOAL_SNORLAX,
    GOAL_ARTICUNO,
    GOAL_ZAPDOS,
    GOAL_MOLTRES,
    GOAL_FOSSIL,
    GOAL_MEWTWO,
    GOAL_ALL_BADGES,
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
    # Oak's "dangerous" intercept — reaching Route 1 before the starter.
    # Distinct from route1_reward below (the later, post-starter return trip)
    # so the entry stage gets full active-goal credit instead of borrowing
    # route1_reward's muted off-goal share.
    route1_entry_reward: float = 4.0    # macro early milestone
    route1_reward: float = 8.0          # macro early milestone (scaled further when active goal)
    oaks_lab_reward: float = 3.0        # meso/macro story
    # Goal conditioning: full credit only for the active curriculum goal so the
    # policy cannot farm house/lab returns while the stage target is route1+.
    active_goal_scale: float = 3.0
    off_goal_milestone_scale: float = 0.1
    new_screen_reward: float = 0.1      # meso (new map)
    new_position_reward: float = 0.008  # micro (slightly lower to curb tile farming)
    # Revisit taper: full credit stays a one-shot, but the drop to 0 no longer
    # happens in a single step. Linear ramp-down over this many visits, so a
    # necessary backtrack (e.g. retracing to a room's only exit) isn't an
    # instant cliff from full bonus to the bare step penalty.
    new_position_decay_visits: int = 4
    new_dialog_reward: float = 0.05     # meso — first enter of a (dialog_id, map)
    # Mid-dialog text farming was an exploit (post-rival Oak speech): tiny
    # +0.01 per screen kept the agent camping without leaving for Route 1.
    dialog_advance_reward: float = 0.0
    dialog_exit_reward: float = 0.2     # meso — leaving dialog is real progress
    # Battle exit (wBattleResult @ 0xCF0B): 0=win, 1=lose, 2=fled.
    # Win is mezzo (same scale as event); macro progress stays on event/badge.
    battle_won_reward: float = 2.0
    # No penalty for losing a battle — a negative penalty here was making the
    # agent risk-averse enough to actively avoid fights (and exploit episode
    # truncation to dodge the -1 before it landed) instead of just playing.
    battle_lost_penalty: float = 0.0
    # battle_won_reward and per-hit enemy-HP reward are both scaled by a
    # smoothstep (3r^2-2r^3, r=enemy_lv/active_player_lv clamped to [0,1]) so
    # stomping a far weaker wild Pokemon is no longer free farming (e.g.
    # lvl-10 one-shotting a lvl-2 Pidgey paid the same +2 win bonus and +1.0
    # HP-fraction hit as a genuinely hard fight). Smoothstep is 0 at r=0, 1 at
    # r=1, with zero slope at both ends — no hard floor/kink, it decays all
    # the way to 0 as the level gap grows, and is C1-continuous into the
    # r>=1 cap=1.0 branch. See _battle_difficulty_scale.
    battle_difficulty_invalid_fallback: float = 0.05
    # Successful RUN: compare enemy vs max party level so a lvl-1 sacrificial
    # slot cannot fake a "smart flee" while stronger Pokémon sit in the back.
    flee_smart_reward: float = 0.4
    flee_coward_penalty: float = -1.0
    # Soft-capped party level-ups (Pleines/Whidden): full credit until sum of
    # party levels hits the threshold (~Misty-ready), then /4 to curb grinding.
    level_reward_scale: float = 0.5
    level_reward_threshold: int = 22
    new_pokedex_seen_reward: float = 0.5
    new_pokedex_own_reward: float = 1.0
    status_reward: float = 0.02
    base_reward: float = -0.001
    truncated_reward: float = -0.05
    new_item_reward: float = 0.5

    # Anti-loop / anti-spam penalties (PokeRL-style). Stronger than before so
    # farming a ~17 return without the stage goal is no longer attractive.
    visit_penalty_soft: float = -0.05   # visit count > 3
    visit_penalty_hard: float = -0.15   # visit count > 5
    # Wild encounters are tracked in their own per-tile counter (wild_visit_
    # counts), separate from position_visit_counts — walking through a grass
    # tile is exploration, fighting on it repeatedly is farming, and the two
    # used to be conflated (a wild encounter bumping the walking counter).
    # battle_won_reward decays with repeat wild wins on the same tile the
    # same way new_position_reward decays with repeat walking visits (see
    # reward_battle_exit / _battle_entry_wild_visits), so grinding one grass
    # tile back and forth no longer stays net positive indefinitely.
    wild_visit_decay_visits: int = 4
    action_pattern_penalty: float = -0.08
    spatial_loop_penalty: float = -0.10
    menu_spam_penalty: float = -0.05
    # Cursor oscillating between a couple of menu states (e.g. ITEM <-> CANCEL)
    # changes state every step, so it evades menu_spam_penalty's "no-change"
    # check above. Catch revisits of the same menu state instead.
    menu_loop_penalty: float = -0.10
    # START/SELECT/d-pad while a textbox is open — does not advance story text.
    dialog_wrong_button_penalty: float = -0.08
    # Scale for in-dialog waste when text is not progressing (was ~base_reward).
    dialog_waste_scale: float = -0.04
    # Lingering on a map that cannot complete the active location goal
    # (e.g. rivals' house while targeting Oak's Lab).
    off_goal_camp_penalty: float = -0.12
    # Re-open a dialog that was already exited on this map: 1st = penalty, 2nd = truncate.
    dialog_reopen_penalty: float = -0.5
    # Consecutive anti-loop hits → truncate episode (escape local optima).
    max_loop_streak: int = 48
    # Claw back location-goal payouts when the agent leaves without clearing
    # them via curriculum auto_advance (1.0 = full refund of what was paid).
    goal_regression_scale: float = 1.0

    # Episode goal for terminated() — early milestones for PPO curriculum.
    goal: str = GOAL_BADGE_1

    in_menu_ticks: float = 0.0
    in_battle_ticks: float = 0.0
    in_dialog_ticks: float = 0.0
    max_useless_ticks: int = 512
    # Hard stuck fuse for one dialog_id. Resets only on dialog_id change or
    # leaving dialog — NOT on tilemap blink / partial text frames (those were
    # resetting the fuse forever while farming advance rewards).
    # 512*4 @ frame_skip 16 = 128 steps: enough to A-mash a long script, not
    # camp. Halved from 512*8 — a policy stuck idling in dialog (deterministic
    # eval argmax landing on NONE) was burning a full 256-step episode before
    # truncating; this bounds the damage without touching a legitimate read.
    max_useless_dialog_ticks: int = 512 * 4
    # Battle fuse: same budget as dialog. Turn counter alone is too coarse —
    # intro text, move select, and attack messages all sit on one turn.
    # 512 was 32 steps @ fs 16; *4 gives room mid-fight.
    max_useless_battle_ticks: int = 512 * 4
    __player_pokemon_size: int = 0x2C
    __pokemon_count: int = 6

    __stored_pokemon_size: int = 0x21

    __visited_pokedex_own: list[int] | None = None

    visited_positions: dict[tuple[int, int, int], int] = field(default_factory=dict)
    position_visit_counts: dict[tuple[int, int, int], int] = field(default_factory=dict)
    # Per-tile wild-encounter count, separate from position_visit_counts —
    # fed to the policy as its own mask (see wild_visit_mask_grid) and drives
    # battle_won_reward's decay instead of the walking exploration decay.
    wild_visit_counts: dict[tuple[int, int, int], int] = field(default_factory=dict)
    map_vision_radius: int = 5

    # --heatmap opt-in (set by PokemonRedEnv from its collect_heatmap ctor arg).
    # Gates direction_counts below so plain training never pays for it.
    collect_heatmap: bool = False
    # World-tile the agent left -> {"up"/"down"/"left"/"right": step count},
    # for the --heatmap live window's movement-direction overlay.
    direction_counts: dict[tuple[int, int, int], dict[str, int]] = field(default_factory=dict)
    # (from_map, to_map) -> {(delta_x, delta_y): count} — one sample per step
    # that changes map_id while walking. delta = how much to add to a
    # from_map-local (x, y) to land on the matching to_map-local (x, y),
    # backed out from the step's direction so it's independent of which
    # arrow key border was crossed. Used by --heatmap's combined "all maps"
    # view to auto-stitch adjacent maps into one canvas.
    map_transitions: dict[tuple[int, int], dict[tuple[int, int], int]] = field(
        default_factory=dict
    )
    # Reward earned per (x, y, map_id) tile, for --heatmap's reward-density
    # overlay (spotting farming/exploit hotspots, not just time-spent).
    # Attributed to the last known *world* position even while a battle/
    # dialog/menu is on screen, so e.g. a battle-won reward triggered by
    # walking onto a grass tile lands on that tile, not nowhere.
    reward_sums: dict[tuple[int, int, int], float] = field(default_factory=dict)
    _last_heatmap_pos: tuple[int, int, int] | None = None

    recent_actions: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_positions: deque = field(default_factory=lambda: deque(maxlen=16))
    recent_menu_states: deque = field(default_factory=lambda: deque(maxlen=16))
    loop_flag: bool = False
    loop_streak: int = 0
    _milestones_hit: set[str] = field(default_factory=set)
    # Payout credited when a milestone first hit — used to claw back on regress.
    _milestone_payouts: dict[str, float] = field(default_factory=dict)
    # Regressable milestones that have already been paid-and-clawed-back once
    # this episode — blocks re-payment on re-entry (see reward_goal_regression).
    _milestones_spent: set[str] = field(default_factory=set)
    # Goals cleared by curriculum auto_advance this episode (no clawback on leave).
    _cleared_goals: set[str] = field(default_factory=set)
    # Live story goals satisfied last step (for detecting count drops).
    _prev_live_goals: set[str] = field(default_factory=set)
    # Union of both kinds below — informational only, kept for debug_play.
    _last_regressed: list[str] = field(default_factory=list)
    # Subset of _last_regressed that actually clawed back a payout (goal was
    # NOT yet curriculum-cleared). This is the one that should drive
    # goal_regression_rate — _last_regressed also fires on every ordinary
    # "left a map I already cleared to reach the next stage" step, which is
    # expected curriculum progress, not backsliding.
    _last_hard_regressed: list[str] = field(default_factory=list)
    _peak_live_goals: int = 0
    # Per-step debug trail for eval -vv: (name, payout) actually paid this
    # step, and names that hit their condition but were blocked because
    # they were already regressed-and-spent this episode.
    last_milestone_payouts: list[tuple[str, float]] = field(default_factory=list)
    last_milestone_blocked: list[str] = field(default_factory=list)
    # Set once per reward() call; True on the exact step a full-party wipe
    # exits the battle screen. Shared by reward_core (suppress the resulting
    # free Pokemon-Center-style heal) and reward_battle_exit (classify the
    # exit correctly instead of trusting wBattleResult).
    _just_blacked_out: bool = False
    _start_map_id: int | None = None
    # Distinct dialog screen hashes seen for the current dialog_id. Blink frames
    # revisit old hashes; only a *new* hash counts as text progress.
    _dialog_screens_seen: set[str] = field(default_factory=set)
    # Dialogs cleanly exited this episode → reopen tracking (penalty then truncate).
    _completed_dialogs: set[tuple[int, int]] = field(default_factory=set)
    _dialog_reopen_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    _dialog_reopen_truncate: bool = False
    # Set each step by reward_battle_exit (debug_play / diagnostics).
    last_flee_reward: float = 0.0
    last_flee_info: dict | None = None
    last_battle_exit_info: dict | None = None
    # Set each step by reward_enemy_hp while in battle (debug_play / diagnostics).
    last_enemy_hp_debug: dict | None = None
    # Last known-good (nonzero) enemy/active level this battle, for
    # _battle_difficulty_scale — wEnemyMonLevel/wBattleMonLevel can read back
    # 0 on some frames mid-fight (e.g. during a mon-switch/animation window)
    # even though HP is clearly changing; holding the last real reading is
    # more robust than trusting whatever a single frame happens to show.
    _battle_enemy_level_cache: int = 0
    _battle_active_level_cache: int = 0
    # wild_visit_counts reading for this tile *before* the current fight's own
    # count() bump — captured at battle entry, consumed by reward_battle_exit
    # so the very first encounter on a tile still pays full battle_won_reward
    # (mirrors reward_position reading position_visit_counts pre-increment).
    _battle_entry_wild_visits: int = 0

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
        self.visited_positions = {}
        self.position_visit_counts = {}
        self.wild_visit_counts = {}
        self.direction_counts = {}
        self.map_transitions = {}
        self.reward_sums = {}
        self._last_heatmap_pos = None
        self.visited_maps = set()
        self.visited_dialogs = {}
        self.recent_actions.clear()
        self.recent_positions.clear()
        self.recent_menu_states.clear()
        self.loop_flag = False
        self.loop_streak = 0
        self._milestones_hit = set()
        self._milestone_payouts = {}
        self._milestones_spent = set()
        self._cleared_goals = set()
        self._prev_live_goals = set()
        self._last_regressed = []
        self._last_hard_regressed = []
        self._peak_live_goals = 0
        self._dialog_screens_seen = set()
        self._completed_dialogs = set()
        self._dialog_reopen_counts = {}
        self._dialog_reopen_truncate = False
        self.last_flee_reward = 0.0
        self.last_flee_info = None
        self.last_battle_exit_info = None
        self.last_enemy_hp_debug = None
        self._battle_enemy_level_cache = 0
        self._battle_active_level_cache = 0
        self._battle_entry_wild_visits = 0
        self._start_map_id = self.map_id(self.pyboy.memory)

    def _dialog_id_changed(self, memory: bytes | None) -> bool:
        """True when dialog_id flipped while still in a textbox."""
        if memory is None or not self.is_dialog(self.pyboy.memory):
            return False
        return self.dialog_id(memory) != self.dialog_id(self.pyboy.memory)

    def _dialog_screen_is_new(self) -> bool:
        """Track unique tilemaps for the current dialog_id (blink-safe)."""
        if not self.is_dialog(self.pyboy.memory):
            return False
        screen_hash = self.screen_tiles_hash(self.pyboy.memory)
        if screen_hash not in self._dialog_screens_seen:
            self._dialog_screens_seen.add(screen_hash)
            return True
        return False

    def count(self, reward: float, action: int, memory: bytes | None = None, duration: int = 16):
        self.visited_pokedex_own = self.pokedex_own(self.pyboy.memory)
        self.visited_pokedex_seen = self.pokedex_seen(self.pyboy.memory)

        # --heatmap reward-density overlay: attribute this step's reward to
        # the last known world tile, not just world-state steps — a battle
        # won/lost reward (often the largest single payout) is earned while
        # is_world() is False the whole fight, but should still land on the
        # grass/trainer tile that started it, not vanish from the overlay.
        if self.collect_heatmap:
            if self.is_world(self.pyboy.memory):
                self._last_heatmap_pos = self.get_position()
            if self._last_heatmap_pos is not None:
                self.reward_sums[self._last_heatmap_pos] = (
                    self.reward_sums.get(self._last_heatmap_pos, 0.0) + reward
                )

        # Cutscenes drive the player; do not pollute action-loop history.
        if not self.is_cutscene_locked(self.pyboy.memory):
            self.recent_actions.append(int(action))

        if self.is_cutscene_locked(self.pyboy.memory):
            # Forced walk / joypad ignore (e.g. following Oak): player has no
            # agency — do not accumulate tile stuck fuse or visit counts.
            pass
        elif self.is_world(self.pyboy.memory):
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

            # --heatmap direction overlay: which way the agent actually moved
            # (position delta), not the raw action — a bumped-into-wall A/B
            # mash never fires this. Anchored on the tile it left.
            if self.collect_heatmap and not stayed and memory is not None:
                prev_pos = self.get_position(memory)
                if prev_pos[2] == pos[2]:
                    dx = pos[0] - prev_pos[0]
                    dy = pos[1] - prev_pos[1]
                    direction = {
                        (1, 0): "right",
                        (-1, 0): "left",
                        (0, 1): "down",
                        (0, -1): "up",
                    }.get((dx, dy))
                    if direction is not None:
                        counts = self.direction_counts.setdefault(prev_pos, {})
                        counts[direction] = counts.get(direction, 0) + 1
                elif self.is_world(memory):
                    # Map boundary crossed this step (edge walk or a warp).
                    # The move is still one of the 4 arrow keys (only way
                    # position/map changes outside a cutscene) — use it to
                    # back out the coordinate-system offset between the two
                    # maps' local (x, y) grids, independent of which edge.
                    step = {
                        6: (0, -1),  # up
                        7: (0, 1),  # down
                        4: (-1, 0),  # left
                        5: (1, 0),  # right
                    }.get(int(action))
                    if step is not None:
                        delta = (
                            pos[0] - prev_pos[0] - step[0],
                            pos[1] - prev_pos[1] - step[1],
                        )
                        key = (prev_pos[2], pos[2])
                        votes = self.map_transitions.setdefault(key, {})
                        votes[delta] = votes.get(delta, 0) + 1
        elif (
            self.is_battle(self.pyboy.memory)
            and memory is not None
            and self.is_world(memory)
        ):
            # Stepped onto a grass/trainer tile and the encounter fired on the
            # same step — is_world() is already False here, so the branch
            # above never runs. Register the tile so reward_position() does
            # not see it as brand new on every world<-battle return.
            pos = self.get_position()
            # wIsInBattle==1 is specifically a wild encounter (2 is a scripted
            # trainer, which will not keep re-firing off the same tile). Track
            # it in its own wild_visit_counts instead of inflating the walking
            # counter — position_visit_counts only ever needs a single seed
            # entry here so reward_position() does not see the tile as brand
            # new on every world<-battle return. The pre-bump reading is
            # cached for reward_battle_exit's battle_won_reward decay.
            if self.type_of_battle(self.pyboy.memory) == 1:
                self._battle_entry_wild_visits = self.wild_visit_counts.get(pos, 0)
                self.wild_visit_counts[pos] = self._battle_entry_wild_visits + 1
                if pos not in self.position_visit_counts:
                    self.position_visit_counts[pos] = 1
            elif pos not in self.position_visit_counts:
                self.position_visit_counts[pos] = 1
        elif (
            not self.is_battle(self.pyboy.memory)
            and not self.is_dialog(self.pyboy.memory)
        ):
            # Menu with no movement (blocked, NPCs still walk): same stuck fuse
            # as world. Skip during dialog — text uses its own fuse.
            pos = self.get_position()
            self.visited_positions[pos] = (
                self.visited_positions.get(pos, 0) + duration
            )

        if self.is_menu(self.pyboy.memory):
            self.in_menu_ticks += duration
            self.recent_menu_states.append(
                (
                    self.menu_position_x(self.pyboy.memory),
                    self.menu_position_y(self.pyboy.memory),
                    self.real_current_menu_selected_item(self.pyboy.memory),
                )
            )
        else:
            self.in_menu_ticks = max(0, self.in_menu_ticks - 0.25 * duration)
            self.recent_menu_states.clear()

        if self.is_battle(self.pyboy.memory):
            if self.number_of_turns_in_current_battle(
                memory
            ) != self.number_of_turns_in_current_battle(self.pyboy.memory):
                self.in_battle_ticks = 0
            else:
                self.in_battle_ticks += duration
        else:
            # Do not carry a near-limit fuse into the next encounter.
            self.in_battle_ticks = 0

        if self.is_dialog(self.pyboy.memory):
            dialog = self.get_dialog()
            self.visited_dialogs[dialog] = self.visited_dialogs.get(dialog, 0) + duration
            # Fuse resets only on dialog_id change (or leaving dialog below).
            # New text frames alone used to reset forever → infinite camp.
            if self._dialog_id_changed(memory):
                self._dialog_screens_seen = set()
                self.in_dialog_ticks = 0
            else:
                self.in_dialog_ticks += duration
            self._dialog_screen_is_new()
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

    def _local_count_grid(
        self, counts: dict[tuple[int, int, int], int], cap: int = 10
    ) -> list[list[float]]:
        r = self.map_vision_radius
        size = 2 * r + 1
        if not self.is_world(self.pyboy.memory):
            return [[0.0] * size for _ in range(size)]
        grid: list[list[float]] = []
        for dy in range(-r, r + 1):
            row: list[float] = []
            for dx in range(-r, r + 1):
                v = counts.get(self.get_position(offset_x=dx, offset_y=dy), 0)
                row.append(min(v, cap) / cap)
            grid.append(row)
        return grid

    def visit_mask_grid(self) -> list[list[float]]:
        """Local visit-count mask centered on the player (PokeRL / Whidden style)."""
        return self._local_count_grid(self.position_visit_counts)

    def wild_visit_mask_grid(self) -> list[list[float]]:
        """Local wild-encounter-count mask centered on the player."""
        return self._local_count_grid(self.wild_visit_counts)

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
            "wild_visit_mask": torch.tensor(
                self.wild_visit_mask_grid(), dtype=torch.float32
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

    def _detect_blackout(self, memory: bytes) -> bool:
        """True on the exact step a full-party wipe exits the battle screen.

        wIsInBattle (0xD057) is documented (pokered ram/wram.asm) as -1
        (0xFF) specifically for a lost/blacked-out battle, unlike a normal
        faint-with-a-mon-left-to-switch or a win/flee. Computed once here so
        reward_core and reward_battle_exit agree on the exact same frame.
        """
        left_battle = self.is_battle(memory) and not self.is_battle(self.pyboy.memory)
        return left_battle and self.type_of_battle(memory) == 0xFF

    def reward(self, memory: bytes, action: int) -> tuple[float, float]:
        milestone = 0.0
        step = 0.0
        # Per-step signal; env accumulates into _episode_loop for episode stats.
        self.loop_flag = False
        self._just_blacked_out = self._detect_blackout(memory)
        if self._just_blacked_out:
            # HandlePlayerBlackOut force-warps to the last Pokemon Center —
            # that departure from the goal map is not the agent choosing to
            # abandon it, so don't claw back location milestones already
            # paid for reaching it (mirrors mark_goal_cleared for a
            # legitimate curriculum-advance departure). Without this, a
            # near-death fight on the goal map made finishing the loss cost
            # up to -goal payout (e.g. -24 for route1 at active_goal_scale),
            # far worse than truncated_reward (-0.05) — so the policy
            # learned to mash useless inputs and stall out the battle-stuck
            # fuse instead of ever letting the blackout resolve.
            for name in REGRESSABLE_GOALS:
                if name in self._milestones_hit:
                    self.mark_goal_cleared(name)

        milestone += self.reward_core(memory)
        milestone += self.reward_story_milestones()
        milestone += self.reward_goal_regression()
        # Battle→overworld (or dialog) transition — win/lose/flee uses prev memory.
        milestone += self.reward_battle_exit(memory)

        if self.is_battle(self.pyboy.memory):
            milestone += self.reward_battle(memory)

        if self.is_cutscene_locked(self.pyboy.memory):
            # No step reward/penalty — actions cannot affect the game. Story
            # milestones above still apply when flags/maps change mid-cutscene.
            if self.is_dialog(memory):
                milestone += self.dialog_exit_reward
        elif self.is_world(self.pyboy.memory):
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

        if not self.is_cutscene_locked(self.pyboy.memory):
            step += self.reward_anti_loop(action=action, memory=memory)
        # Always track dialog enter/exit — cutscenes often follow textboxes.
        step += self.reward_dialog_reopen(memory)

        return milestone, step

    def reward_dialog_reopen(self, memory: bytes) -> float:
        """After exiting a dialog, reopening the same (dialog_id, map_id) is a loop.

        1st reopen → penalty; 2nd reopen → truncate (see ``truncated``).
        Staying inside one conversation does not count — only exit then re-enter.
        Reopens that happen while the engine owns input (forced-walk cutscenes,
        e.g. Oak's Route 1 intercept closing/reopening the same textbox) are not
        the player's doing, so they are exempt from the farming check entirely.
        """
        was_dialog = self.is_dialog(memory)
        now_dialog = self.is_dialog(self.pyboy.memory)

        if was_dialog and not now_dialog:
            self._completed_dialogs.add(self.get_dialog(memory))
            return 0.0

        if not was_dialog and now_dialog:
            if self.is_script_locked(self.pyboy.memory):
                return 0.0
            key = self.get_dialog()
            if key not in self._completed_dialogs:
                return 0.0
            n = self._dialog_reopen_counts.get(key, 0) + 1
            self._dialog_reopen_counts[key] = n
            self.loop_flag = True
            if n >= 2:
                self._dialog_reopen_truncate = True
                return 0.0
            return self.dialog_reopen_penalty

        return 0.0

    def _scaled_milestone(self, name: str, base: float) -> float:
        """Full credit for the active goal; muted credit for off-goal story beats."""
        if name == self.goal:
            return base * self.active_goal_scale
        return base * self.off_goal_milestone_scale

    def mark_goal_cleared(self, goal: str) -> None:
        """Curriculum auto_advance: leaving this location is not a regression."""
        if goal:
            self._cleared_goals.add(str(goal))

    def live_story_goals(self) -> list[str]:
        """Story goals that are true in the *current* game state (can shrink)."""
        return [g for g in STORY_GOAL_ORDER if self.is_goal_satisfied(g)]

    def _prereq_cleared(self, goal: str) -> bool:
        """Whether ``goal`` was paid out at some point this episode.

        Used to gate early-game milestone payouts on STORY_GOAL_ORDER so a
        later beat cannot pay out for free by skipping the one before it —
        see reward_story_milestones. Checks _milestones_spent too: a
        regressable goal that was hit and later clawed back (e.g. walked
        back into a house) still genuinely happened, so it should not
        un-satisfy a downstream prerequisite.
        """
        return (
            goal in self._milestones_hit
            or goal in self._milestones_spent
            or goal in self._cleared_goals
        )

    def reward_goal_regression(self) -> float:
        """Claw back location milestones that are no longer true.

        Cleared-via-curriculum goals are exempt so the correct path
        (enter lab → advance → leave toward Route 1) is not punished.
        Unpaid dabbling (off-goal lab visit while training Route 1) is refunded.

        Clawed-back goals are marked ``_milestones_spent`` so re-entering
        cannot earn the payout a second time (see reward_story_milestones) —
        without that, leave-then-return nets a full free payout every trip
        (pay on entry, refund equals the clawback, pay again on re-entry),
        turning the map border into a farmable reward loop.
        """
        self._last_regressed = []
        self._last_hard_regressed = []
        reward = 0.0
        for name in list(self._milestones_hit):
            if name not in REGRESSABLE_GOALS:
                continue
            if name in self._cleared_goals:
                continue
            if self.is_goal_satisfied(name):
                continue
            payout = float(self._milestone_payouts.pop(name, 0.0))
            self._milestones_hit.discard(name)
            self._milestones_spent.add(name)
            reward -= payout * self.goal_regression_scale
            self._last_regressed.append(name)
            self._last_hard_regressed.append(name)

        live = set(self.live_story_goals())
        lost_live = self._prev_live_goals - live
        for name in lost_live:
            if name not in self._last_regressed:
                self._last_regressed.append(name)
        self._peak_live_goals = max(self._peak_live_goals, len(live))
        self._prev_live_goals = live
        return reward

    def reward_story_milestones(self) -> float:
        """One-shot bonuses for story / map progress (order-independent).

        Later-game flags (fought_brock_yet, is_ss_anne_here, ...) used to
        also pay out via a raw prev-vs-current memory bit diff
        (reward_event_flags, now removed) with no "already paid" bookkeeping
        — one of those bits (position_in_air) flips on the pause-menu
        animation, so a policy could farm it indefinitely by spamming START.
        They are folded into the ``checks``/``_milestones_hit``-gated pattern
        below instead, reusing ``is_goal_satisfied`` as the single source of
        truth for each condition.
        """
        reward = 0.0
        mid = self.map_id(self.pyboy.memory)
        self.last_milestone_payouts = []
        self.last_milestone_blocked = []

        checks = [
            (
                GOAL_LEFT_HOUSE,
                bool(self._start_map_id in HOUSE_MAPS and mid not in HOUSE_MAPS),
                self.left_house_reward,
            ),
            (
                GOAL_ROUTE_1_ENTRY,
                # Oak's intercept fires as a dialog while still on Pallet Town
                # (see is_goal_satisfied) — map_id never actually becomes
                # MAP_ROUTE_1 for this pre-starter trigger.
                self._prereq_cleared(GOAL_LEFT_HOUSE)
                and (
                    mid == MAP_ROUTE_1
                    or (
                        mid == MAP_PALLET_TOWN
                        and self.is_dialog(self.pyboy.memory)
                        and self.dialog_id(self.pyboy.memory) == ROUTE1_ENTRY_DIALOG_ID
                    )
                ),
                self.route1_entry_reward,
            ),
            (
                GOAL_ROUTE_1,
                # Guard against the entry-trigger tile also satisfying this
                # check: without it, GOAL_ROUTE_1 fires (muted, off-goal) the
                # instant Route 1 is touched during stage_route1_entry, then
                # gets clawed back by reward_goal_regression() on the forced
                # walk into the Lab — a spurious earn/regress cycle. The
                # oaks_lab prereq below now covers this too, but the explicit
                # goal check is cheap insurance against reordering.
                self._prereq_cleared(GOAL_OAKS_LAB)
                and mid == MAP_ROUTE_1
                and self.goal != GOAL_ROUTE_1_ENTRY,
                self.route1_reward,
            ),
            (
                GOAL_OAKS_LAB,
                self._prereq_cleared(GOAL_ROUTE_1_ENTRY) and mid == MAP_OAKS_LAB,
                self.oaks_lab_reward,
            ),
            (
                GOAL_OAKS_PARCEL,
                self._prereq_cleared(GOAL_ROUTE_1)
                and bool(self.have_oaks_parcel(self.pyboy.memory)),
                self.event_reward,
            ),
            (
                GOAL_TOWN_MAP,
                self._prereq_cleared(GOAL_OAKS_PARCEL)
                and bool(self.have_town_map(self.pyboy.memory)),
                self.event_reward,
            ),
        ]
        # Later-game story flags: each maps 1:1 onto a curriculum goal and
        # is_goal_satisfied() already implements the exact trigger condition
        # (trainer-fought bit, event flag, etc.) — reuse it instead of
        # duplicating the memory reads here.
        checks += [
            (name, self.is_goal_satisfied(name), self.event_reward)
            for name in (
                GOAL_FOUGHT_BROCK,
                GOAL_FOUGHT_MISTY,
                GOAL_FOUGHT_SURGE,
                GOAL_FOUGHT_ERIKA,
                GOAL_FOUGHT_KOGA,
                GOAL_FOUGHT_SABRINA,
                GOAL_FOUGHT_BLAINE,
                GOAL_FOUGHT_GIOVANNI,
                GOAL_SS_ANNE,
                GOAL_LAPRAS,
                GOAL_SNORLAX,
                GOAL_ARTICUNO,
                GOAL_ZAPDOS,
                GOAL_MOLTRES,
                GOAL_FOSSIL,
                GOAL_MEWTWO,
                GOAL_ALL_BADGES,
            )
        ]
        for name, hit, value in checks:
            if not hit or name in self._milestones_hit:
                continue
            if name in self._milestones_spent:
                self.last_milestone_blocked.append(name)
                continue
            payout = self._scaled_milestone(name, value)
            self._milestones_hit.add(name)
            self._milestone_payouts[name] = payout
            reward += payout
            self.last_milestone_payouts.append((name, payout))

        badges = self.badges(self.pyboy.memory)
        for i, name in enumerate(BADGE_GOALS):
            if name not in self._milestones_hit and i < len(badges) and badges[i]:
                payout = self._scaled_milestone(name, self.badge_reward)
                self._milestones_hit.add(name)
                self._milestone_payouts[name] = payout
                reward += payout

        return reward

    def reward_anti_loop(self, action: int, memory: bytes) -> float:
        """Three-layer anti-loop + menu-spam penalties (PokeRL-style)."""
        penalty = 0.0
        triggered = False
        action = int(action)
        in_dialog = self.is_dialog(self.pyboy.memory)
        in_battle = self.is_battle(self.pyboy.memory)
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
        # without an open dialog/battle is camping (Oak lab idle), not
        # "talking" — battle text (attack/effect/faint/EXP messages) forces
        # the same repeated A presses as dialog, so it gets the same pass.
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
            if not interacting or not (in_dialog or in_battle):
                penalty += self.action_pattern_penalty
                triggered = True

        # Idle / wrong buttons in dialog — only A/B advances story text. NONE
        # used to be discounted to menu_spam_penalty (-0.05 vs -0.08), which
        # made "do nothing" the cheapest mistake in a textbox — verified via
        # deterministic-eval action-probs sitting near 50/50 NONE-vs-A on a
        # multi-page Oak's-lab dialog, with argmax landing on NONE and idling
        # until the stuck-dialog fuse truncated the episode. NONE gets the
        # same penalty as any other non-interact button now, closing that gap.
        if in_dialog and action not in INTERACT_ACTIONS:
            penalty += self.dialog_wrong_button_penalty
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

        # Menu loop: cursor oscillating between a small set of states (e.g.
        # ITEM <-> CANCEL) changes state every step, so it slips past the
        # no-change check above. Catch revisits of the same menu state instead
        # (mirrors the spatial_loop_penalty check for the overworld).
        if self.is_menu(self.pyboy.memory) and len(self.recent_menu_states) >= 6:
            cur_menu_state = (
                self.menu_position_x(self.pyboy.memory),
                self.menu_position_y(self.pyboy.memory),
                self.real_current_menu_selected_item(self.pyboy.memory),
            )
            if sum(1 for s in self.recent_menu_states if s == cur_menu_state) >= 3:
                penalty += self.menu_loop_penalty
                triggered = True

        # 4) Off-goal map camping — rivals' house etc. while targeting lab/route.
        allowed = GOAL_ALLOWED_MAPS.get(self.goal)
        if (
            allowed is not None
            and self.is_world(self.pyboy.memory)
            and not self.is_cutscene_locked(self.pyboy.memory)
            and self.map_id(self.pyboy.memory) not in allowed
        ):
            visits = self.position_visit_counts.get(self.get_position(), 0)
            # Punish dwell / A-B mash off the critical path; brief walk-through OK.
            if visits > 2 or (interacting and not in_dialog):
                penalty += self.off_goal_camp_penalty
                triggered = True

        if triggered:
            self.loop_flag = True
            self.loop_streak += 1
        elif not in_battle:
            # Battle structurally skips checks 1/3 (is_world-gated), so
            # resetting here let ducking into a grass encounter wipe an
            # accumulating loop streak for free. Freeze instead of clearing.
            self.loop_streak = 0

        return penalty

    def reward_battle_useless_count(self, memory: bytes) -> tuple[float, float]:
        entered = not self.is_battle(memory) and self.is_battle(self.pyboy.memory)
        turn_changed = (
            self.number_of_turns_in_current_battle(memory)
            != self.number_of_turns_in_current_battle(self.pyboy.memory)
        )
        if entered:
            # No reward for entering a battle, from world or dialog (trainer
            # intros) alike — it was luring the agent into farming grass or
            # trainer sprites for the entry bonus and fleeing immediately after.
            return 0.0, 0.0
        if turn_changed:
            return self.new_screen_reward, 0.0
        return (
            0.0,
            self.in_battle_ticks / self.max_useless_battle_ticks * self.base_reward,
        )

    def reward_dialog(self, memory: bytes, action: int) -> tuple[float, float]:
        dialog_changed = self.dialog_id(memory) != self.dialog_id(self.pyboy.memory)
        current_dialog = self.get_dialog()
        is_new_dialog = current_dialog not in self.visited_dialogs
        # Pay for opening a new conversation and (optionally) dialog_id flips.
        # Do NOT pay for tilemap text frames — that was the post-rival exploit.
        if is_new_dialog:
            dialog_reward = self.new_dialog_reward
        elif dialog_changed:
            dialog_reward = self.dialog_advance_reward
        else:
            dialog_reward = 0.0
        # Waste grows while the same dialog_id sits on screen without flipping.
        waste = (
            0.0
            if dialog_changed
            else (
                self.in_dialog_ticks
                / self.max_useless_dialog_ticks
                * self.dialog_waste_scale
            )
        )
        return dialog_reward, waste

    def reward_position(self):
        pos = self.get_position()
        ticks_here = self.visited_positions.get(pos, 0)
        visit_count = self.position_visit_counts.get(pos, 0)
        waste_factor = min(ticks_here, self.max_useless_ticks) / self.max_useless_ticks
        step_penalty = self.base_reward * (1.0 + waste_factor * 9.0)
        # Tiles that have already produced a wild encounter (wild_visit_counts)
        # pay no walking-exploration bonus — reward on those tiles should only
        # come from the battle itself (battle_won_reward), not from stepping
        # on/off the grass, otherwise the agent can farm exploration_reward by
        # pacing back and forth over a known wild tile.
        if pos in self.wild_visit_counts:
            return step_penalty
        decay = max(0.0, 1.0 - visit_count / self.new_position_decay_visits)
        exploration_reward = self.new_position_reward * decay
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

        reward += self.reward_pokedex(memory)
        # HP/status/PP all get a free full restore as part of the blackout
        # sequence (warp to last Pokemon Center) — that's not something the
        # agent did, so don't pay for it. Without this gate it can nearly
        # cancel out battle_lost_penalty, making whiteout close to free.
        if not self._just_blacked_out:
            reward += self.reward_player_pokemons_current_hps(memory)
            reward += self.reward_player_pokemons_statuses(memory)
            reward += self.reward_player_pokemons_pps(memory)
        reward += self.reward_player_pokemons_experiences(memory)
        reward += self.reward_party_levels(memory)
        reward += self.reward_player_pokemons_max_hps(memory)
        reward += self.reward_player_pokemons_attacks(memory)
        reward += self.reward_player_pokemons_defenses(memory)
        reward += self.reward_player_pokemons_speeds(memory)
        reward += self.reward_map()
        reward += self.reward_player_items(memory)
        reward += self.reward_stored_items(memory)

        return reward

    def reward_player_items(self, memory: bytes):
        bag_delta = sum(self.items_quantities(self.pyboy.memory)) - sum(
            self.items_quantities(memory)
        )
        stored_delta = sum(self.stored_items_quantities(self.pyboy.memory)) - sum(
            self.stored_items_quantities(memory)
        )
        if bag_delta < 0 and stored_delta <= 0:
            # A bag decrease not matched by a PC increase is plain usage/selling,
            # not a deposit — don't punish the agent for using its items.
            return 0.0

        return bag_delta * self.new_item_reward

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
        if self.map_id(self.pyboy.memory) in self.visited_maps:
            return 0.0
        # Don't reward "discovering" a map that's off the active goal's path
        # (see off_goal_camp_penalty / GOAL_ALLOWED_MAPS) — first-visit credit
        # should not offset the penalty for straying off-course.
        allowed = GOAL_ALLOWED_MAPS.get(self.goal)
        if allowed is not None and self.map_id(self.pyboy.memory) not in allowed:
            return 0.0
        return self.new_screen_reward

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

    def reward_party_levels(self, memory: bytes) -> float:
        """Soft-capped reward for party level-ups (Whidden/Pleines style).

        Full ``level_reward_scale`` per level while sum of occupied party levels
        is at or below ``level_reward_threshold``; after that each level pays /4
        so late grinding does not dominate events/badges.
        """
        prev_sum = sum(lv for lv in self.all_party_levels(memory) if lv > 0)
        now_sum = sum(lv for lv in self.all_party_levels(self.pyboy.memory) if lv > 0)
        delta = now_sum - prev_sum
        if delta <= 0:
            return 0.0
        reward = 0.0
        for i in range(delta):
            level_after = prev_sum + i + 1
            if level_after <= self.level_reward_threshold:
                reward += self.level_reward_scale
            else:
                reward += self.level_reward_scale / 4.0
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
        if goal == GOAL_ROUTE_1_ENTRY:
            return mid == MAP_ROUTE_1 or (
                mid == MAP_PALLET_TOWN
                and self.is_dialog(mem)
                and self.dialog_id(mem) == ROUTE1_ENTRY_DIALOG_ID
            )
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
        # Stuck fuse uses in_dialog_ticks (resets on dialog_id change / leave),
        # not tilemap text frames — those no longer clear the fuse.
        # Also require is_dialog so lingering dialog_id in RAM cannot truncate
        # after the conversation already ended.
        in_dialog = self.is_dialog(self.pyboy.memory)
        stuck_dialog = (
            in_dialog and self.max_useless_dialog_ticks <= self.in_dialog_ticks
        )
        # Tile fuse pauses during dialog and cutscenes (forced walks / joy ignore).
        stuck_tile = (
            not in_dialog
            and not self.is_cutscene_locked(self.pyboy.memory)
            and self.max_useless_ticks
            <= self.visited_positions.get(self.get_position(), 0)
        )
        # Tutorial map lock: leaving the one allowed map (GOAL_ALLOWED_MAPS)
        # used to fail the episode outright. It now costs an escalating
        # penalty instead (off_map_ticks / off_map_penalty_scale, tracked in
        # count() and applied in reward()), so a fresh/undertrained policy
        # gets pushed back toward the PC's room rather than losing the
        # episode the instant it steps out.
        return (
            True
            if stuck_tile
            or stuck_dialog
            or self._dialog_reopen_truncate
            or self.loop_streak >= self.max_loop_streak
            or self.max_useless_battle_ticks <= self.in_battle_ticks
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
            GOAL_ROUTE_1_ENTRY,
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
            [self.in_battle_ticks],
            max=self.max_useless_battle_ticks,
        )
        data += self.data_normalizer(
            [self.in_menu_ticks],
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
        # wFontLoaded (CFC4): textbox/menu font occupies walk-anim VRAM.
        return True if memory[0xCFC4] else False

    def is_script_locked(self, memory: PyBoyMemoryView | bytes) -> bool:
        """True when the engine owns player input (cutscene / forced walk).

        Independent of on-screen activity — following Oak still looks like the
        overworld. See pret/pokered JoypadOverworld + wStatusFlags5.
        """
        status5 = int(memory[0xD730])
        # bit7 BIT_SCRIPTED_MOVEMENT_STATE, bit5 BIT_DISABLE_JOYPAD
        if status5 & 0xA0:
            return True
        # wJoyIgnore — bitmask of ignored buttons (often D-pad during scripts)
        if memory[0xCD6B]:
            return True
        # wSimulatedJoypadStatesIndex — remaining forced button presses
        if memory[0xCD38]:
            return True
        return False

    def is_cutscene_locked(self, memory: PyBoyMemoryView | bytes) -> bool:
        """Script lock with no textbox/battle — player cannot usefully act."""
        return (
            self.is_script_locked(memory)
            and not self.is_blocked(memory)
            and not self.is_battle(memory)
        )

    def dialog_id(self, memory: PyBoyMemoryView | bytes):
        return memory[0xCF13]

    def is_battle(self, memory: PyBoyMemoryView | bytes):
        return True if self.type_of_battle(memory) else False

    def type_of_battle(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD057]

    def is_world(self, memory: PyBoyMemoryView | bytes):
        # Cutscene lock looks like overworld on screen but is not agent-controlled.
        # Mode flags become all-zero — distinct obs signal without changing VECTOR_DIM.
        return (
            True
            if not self.is_blocked(memory)
            and not self.is_battle(memory)
            and not self.is_menu(memory)
            and not self.is_script_locked(memory)
            else False
        )

    def position_x(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD362]

    def position_y(self, memory: PyBoyMemoryView | bytes):
        return memory[0xD361]

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
        # Gen1 stores 16-bit stats big-endian (high byte first).
        return (memory[0xCFE6] << 8) | memory[0xCFE7]

    def enemy_level(self, memory: PyBoyMemoryView | bytes):
        # wEnemyMon base is 0xCFE5 (battle_struct layout); 0xCFE8 is
        # BoxLevel (unused trade-display field, always 0 in battle) — the
        # real live Level field is offset 0x0E from base = 0xCFF3. Verified
        # live: 0xCFE8 read 0 for an entire multi-turn fight while 0xCFF3
        # held a stable, sane level matching the opponent's actual strength.
        return memory[0xCFF3]

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
        return (memory[0xCFF4] << 8) | memory[0xCFF5]

    def enemy_attack(self, memory: PyBoyMemoryView | bytes):
        return (memory[0xCFF6] << 8) | memory[0xCFF7]

    def enemy_defense(self, memory: PyBoyMemoryView | bytes):
        return (memory[0xCFF8] << 8) | memory[0xCFF9]

    def enemy_speed(self, memory: PyBoyMemoryView | bytes):
        return (memory[0xCFFA] << 8) | memory[0xCFFB]

    def enemy_special(self, memory: PyBoyMemoryView | bytes):
        return (memory[0xCFFC] << 8) | memory[0xCFFD]

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
        return (memory[0xD015] << 8) | memory[0xD016]

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
        return (memory[0xD023] << 8) | memory[0xD024]

    def pokemon_attack(self, memory: PyBoyMemoryView | bytes):
        return (memory[0xD025] << 8) | memory[0xD026]

    def pokemon_defense(self, memory: PyBoyMemoryView | bytes):
        return (memory[0xD027] << 8) | memory[0xD028]

    def pokemon_speed(self, memory: PyBoyMemoryView | bytes):
        return (memory[0xD029] << 8) | memory[0xD02A]

    def pokemon_special(self, memory: PyBoyMemoryView | bytes):
        return (memory[0xD02B] << 8) | memory[0xD02C]

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
            (memory[0xD16C + self.__player_pokemon_size * x] << 8)
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
            (memory[0xD179 + self.__player_pokemon_size * i] << 16)
            | (memory[0xD17A + self.__player_pokemon_size * i] << 8)
            | memory[0xD17B + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_level(self, memory: PyBoyMemoryView | bytes = None):
        # All 6 party slots (not just the active battler) — the smart/coward
        # flee reward compares the enemy's level against max_party_level, so
        # the model needs bench levels visible to predict its own flee payout.
        return [
            memory[0xD18C + self.__player_pokemon_size * x]
            for x in range(self.__pokemon_count)
        ]

    def player_pokemons_max_hps(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[0xD18D + self.__player_pokemon_size * i] << 8)
            | memory[0xD18E + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_attacks(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[0xD18F + self.__player_pokemon_size * i] << 8)
            | memory[0xD190 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_defenses(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[0xD191 + self.__player_pokemon_size * i] << 8)
            | memory[0xD192 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_speeds(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[0xD193 + self.__player_pokemon_size * i] << 8)
            | memory[0xD194 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_specials(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[0xD195 + self.__player_pokemon_size * i] << 8)
            | memory[0xD196 + self.__player_pokemon_size * i]
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

        if not self.is_battle(memory):
            # Freshly entered — a new enemy is being loaded, so any level
            # cached from a previous fight this episode no longer applies.
            self._battle_enemy_level_cache = 0
            self._battle_active_level_cache = 0

        # Player HP/status are NOT scored here — reward_core's
        # reward_player_pokemons_current_hps / reward_player_pokemons_statuses
        # already cover the whole party unconditionally (every step), and the
        # active battler's wBattleMon HP/status mirror the same party-struct
        # values live during battle. Scoring them again here double-counted
        # every point of damage taken, making fighting look worse than fleeing.
        reward += self.reward_enemy_hp(memory)
        reward += self.reward_enemy_status(memory)

        return reward

    def battle_result(self, memory: PyBoyMemoryView | bytes) -> int:
        """wBattleResult @ 0xCF0B: 0=win, 1=lose, 2=draw (player fled)."""
        return memory[0xCF0B]

    def party_count(self, memory: PyBoyMemoryView | bytes) -> int:
        return memory[0xD163]

    def all_party_levels(self, memory: PyBoyMemoryView | bytes) -> list[int]:
        """Levels of every occupied party slot (not just the active battler)."""
        count = min(int(self.party_count(memory)), self.__pokemon_count)
        return [
            memory[0xD18C + self.__player_pokemon_size * i] for i in range(count)
        ]

    def max_party_level(self, memory: PyBoyMemoryView | bytes) -> int:
        levels = [lv for lv in self.all_party_levels(memory) if lv > 0]
        return max(levels) if levels else 0

    def _battle_difficulty_scale(self, enemy_lv: int, player_lv: int) -> float:
        """Smoothstep(enemy_lv / player_lv) — a same-or-tougher opponent
        (ratio >= 1) pays full credit; below that the reward eases down
        smoothly to 0 as the level gap grows, instead of a hard linear ramp
        with an artificial floor. 3r^2-2r^3 has zero slope at both r=0 and
        r=1, so there is no kink anywhere, including the seam into the
        ratio>=1 cap=1.0 branch.

        Unreadable/invalid levels (<=0) fall back to a small constant, not
        1.0 — this is an anti-farming discount, so a bad read must never
        silently grant full credit (that's exactly the failure mode that let
        the exploit through undetected).
        """
        if player_lv <= 0 or enemy_lv <= 0:
            return self.battle_difficulty_invalid_fallback
        r = min(1.0, enemy_lv / player_lv)
        return 3 * r**2 - 2 * r**3

    def _wild_encounter_decay(self) -> float:
        """Per-tile decay factor for repeat wild encounters this episode.

        Floors at 0 after ``wild_visit_decay_visits`` prior wild fights on
        the current tile (``_battle_entry_wild_visits``, cached at battle
        entry). Applied to every positive reward a wild fight can pay
        (``reward_enemy_hp``, ``reward_enemy_status``, ``battle_won_reward``)
        so grinding one grass tile drains to zero reward, not just the win
        bonus. Trainer battles never call this — they are one-shot per
        sprite, not repeatable on the same tile.
        """
        return max(0.0, 1.0 - self._battle_entry_wild_visits / self.wild_visit_decay_visits)

    def reward_battle_exit(self, memory: bytes) -> float:
        """Reward battle outcome on battle→overworld transition (wBattleResult).

        - 0 win → ``battle_won_reward``
        - 1 lose → ``battle_lost_penalty`` (a mon fainted but the player had
          another to send out — HandlePlayerBlackOut below is the full wipe)
        - 2 fled → smart/coward flee based on enemy vs max party level

        wBattleResult (0xCF0B) is only ever written by the win/faint-switch/
        run paths in pokered's battle engine. A full party wipe goes through
        HandlePlayerBlackOut instead, which never touches wBattleResult — so
        on that exit frame it still holds whatever value was last written by
        an *earlier* battle (often 0/win), and would otherwise be read here
        as a win for the very fight that just wiped the party out. Detect it
        directly from wIsInBattle (0xD057), which pokered documents as being
        set to -1 (0xFF) specifically for a lost/blacked-out battle, and let
        it override wBattleResult.
        """
        self.last_flee_reward = 0.0
        self.last_flee_info = None
        self.last_battle_exit_info = None
        left_battle = self.is_battle(memory) and not self.is_battle(self.pyboy.memory)
        if not left_battle:
            return 0.0

        blacked_out = self._just_blacked_out
        result = self.battle_result(self.pyboy.memory)
        # Fall back to the last known-good in-battle reading — see
        # reward_enemy_hp. The exit frame is exactly where a switch/faint
        # transition is most likely to be mid-flight.
        enemy_lv = self.enemy_level(memory)
        if enemy_lv > 0:
            self._battle_enemy_level_cache = enemy_lv
        else:
            enemy_lv = self._battle_enemy_level_cache
        active_lv = self.pokemon_level(memory)
        if active_lv > 0:
            self._battle_active_level_cache = active_lv
        else:
            active_lv = self._battle_active_level_cache

        party_levels = self.all_party_levels(memory)
        party_max = self.max_party_level(memory)

        difficulty_scale = 1.0
        if blacked_out:
            reward = self.battle_lost_penalty
            kind = "blackout"
        elif result == 0:
            # The active battler's level, not party_max — a weaker party
            # member fighting an appropriately-matched wild Pokemon (i.e.
            # deliberately leveling it) is a fair fight and should pay full
            # credit; only a curbstomp *for the mon that actually fought*
            # (e.g. lvl-10 one-shotting a lvl-2) gets discounted. party_max
            # is reserved for the flee smart/coward check above, where the
            # question is whole-team risk, not this fight's difficulty.
            difficulty_scale = self._battle_difficulty_scale(enemy_lv, active_lv)
            reward = self.battle_won_reward * difficulty_scale
            # Wild wins decay with repeat encounters on the same tile, same
            # shape as reward_position's exploration decay — trainer battles
            # are one-shot per sprite so they are exempt (type_of_battle read
            # from `memory`, the still-in-battle previous frame).
            if self.type_of_battle(memory) == 1:
                reward *= self._wild_encounter_decay()
            kind = "win"
        elif result == 1:
            reward = self.battle_lost_penalty
            kind = "lose"
        elif result == 2:
            # TryRunningFromBattle sets wBattleResult=$02 on successful escape.
            if enemy_lv <= 0 or party_max <= 0:
                return 0.0
            if enemy_lv > party_max:
                reward = self.flee_smart_reward
                kind = "smart"
            else:
                reward = self.flee_coward_penalty
                kind = "coward"
        else:
            return 0.0

        info = {
            "kind": kind,
            "battle_result": result,
            "blacked_out": blacked_out,
            "enemy_level": enemy_lv,
            "active_level": active_lv,
            "party_levels": party_levels,
            "party_max": party_max,
            "difficulty_scale": difficulty_scale,
            "reward": reward,
        }
        self.last_battle_exit_info = info
        if kind in ("smart", "coward"):
            self.last_flee_reward = reward
            self.last_flee_info = info
        return reward

    def reward_battle_flee(self, memory: bytes) -> float:
        """Backward-compatible alias for ``reward_battle_exit``."""
        return self.reward_battle_exit(memory)

    def reward_enemy_hp(self, memory: bytes):
        """Fractional enemy HP lost this step (positive when dealing damage).

        Scaled by ``_battle_difficulty_scale`` — one-shotting a far weaker
        wild Pokemon still costs its full HP bar, but no longer pays the
        full fractional reward every time (see ``battle_won_reward``).
        """
        if (
            self.enemy_max_hp(self.pyboy.memory) == 0
            or self.pokemon_max_hp(self.pyboy.memory) == 0
        ):
            return 0

        frac = (
            self.enemy_hp(memory) - self.enemy_hp(self.pyboy.memory)
        ) / self.enemy_max_hp(self.pyboy.memory)
        # Read levels from the pre-step snapshot, not self.pyboy.memory — see
        # reward_battle_exit. Even so, wEnemyMonLevel/wBattleMonLevel can
        # read back 0 on some frames mid-fight (observed live: a real hit
        # landing with enemy_lv=0). Fall back to the last known-good reading
        # for this battle instead of trusting a single bad frame.
        enemy_lv = self.enemy_level(memory)
        if enemy_lv > 0:
            self._battle_enemy_level_cache = enemy_lv
        else:
            enemy_lv = self._battle_enemy_level_cache

        active_lv = self.pokemon_level(memory)
        if active_lv > 0:
            self._battle_active_level_cache = active_lv
        else:
            active_lv = self._battle_active_level_cache

        scale = self._battle_difficulty_scale(enemy_lv, active_lv)
        scaled = frac * scale
        # Same per-tile decay as battle_won_reward (see _wild_encounter_decay)
        # — only the positive (damage-dealt) side; a negative frac (e.g. an
        # enemy switch-in reading higher HP than the fainted mon it replaced)
        # is not a reward being farmed, so it is left undiscounted.
        if scaled > 0 and self.type_of_battle(self.pyboy.memory) == 1:
            scaled *= self._wild_encounter_decay()
        self.last_enemy_hp_debug = {
            "enemy_level": enemy_lv,
            "active_level": active_lv,
            "difficulty_scale": scale,
            "frac": frac,
            "scaled": scaled,
        }
        return scaled

    def reward_enemy_status(self, memory: bytes):
        reward = 0

        for bit_before, bit_after in zip(
            self.enemy_status(memory), self.enemy_status(self.pyboy.memory)
        ):
            if bit_before == 0 and bit_after == 1:
                reward += self.status_reward
            elif bit_before == 1 and bit_after == 0:
                reward += -self.status_reward

        reward = max(0, reward)
        # Same per-tile wild-encounter decay as reward_enemy_hp/battle_won_reward.
        if reward > 0 and self.type_of_battle(self.pyboy.memory) == 1:
            reward *= self._wild_encounter_decay()
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
