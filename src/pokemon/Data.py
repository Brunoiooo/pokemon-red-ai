import hashlib
from collections import deque
from multiprocessing.synchronize import RLock
import pickle
from dataclasses import dataclass, field

from pyboy import PyBoy, PyBoyMemoryView
import torch

from pokemon import event_constants as _event_constants
from pokemon import event_graph as _event_graph
from pokemon import map_collision as _map_collision
from pokemon import map_constants as _map_constants
from pokemon import map_scripts as _map_scripts
from pokemon import ram_constants as _ram_constants

# Named WRAM/HRAM addresses (RAM.wFoo / RAM.hFoo), generated from pret/pokered
# -- see src/pokemon/ram_constants.py / tools/gen_ram_constants.py.
RAM = _ram_constants.RAM

# Raw WRAM addresses below are documented at:
# https://datacrystal.tcrf.net/wiki/Pokémon_Red_and_Blue/RAM_map


# Emulator button indices — see Emulator.buttons for the canonical list.
ACTION_A = 0
ACTION_B = 1
ACTION_START = 2
ACTION_SELECT = 3
ACTION_LEFT = 4
ACTION_RIGHT = 5
ACTION_UP = 6
ACTION_DOWN = 7
ACTION_NONE = 8
INTERACT_ACTIONS = frozenset({ACTION_A, ACTION_B})

# Named sub-triggers folded into loop_flag by reward_anti_loop — kept as a
# fixed vocabulary so callbacks can log a per-cause breakdown instead of a
# single opaque bool (see EntropyCoefScheduler / MilestoneCallback).
LOOP_CAUSES = (
    "visit_penalty",
    "action_pattern",
    "dialog_wrong_button",
    "spatial_loop",
    "menu_spam",
    "menu_loop",
)

# Which fuse in truncated() ended the episode — "max_steps" is added by
# PokemonRedEnv.step() itself when none of these fired first (see
# TRUNCATE_CAUSES usage in callbacks.py's per-cause breakdown).
TRUNCATE_CAUSES = (
    "stuck_tile",
    "stuck_dialog",
    "loop_streak",
    "stuck_battle",
    "stuck_menu",
    "map_budget",
    "max_steps",
)

# Normalization cap for map_id_visit_grid's dwell-time observation (see
# Data.map_visit_grid_cap). Purely informational input for the model — the
# live per-map truncate budget is size-scaled (see map_truncate_budget), not
# this flat constant.
MAP_DWELL_BUDGET = 256

# position_visit_counts thresholds where reward_anti_loop's graduated visit
# penalty kicks in (soft) and doubles up (hard) -- see Data.reward_anti_loop.
# SOFT also doubles as visit_mask_grid's per-tile observation cap (same
# "meaningful visit count" scale) and, via PositionHeatmap's "visits"
# metric, the reference unit its heatmap color scale is normalized against.
VISIT_PENALTY_SOFT_THRESHOLD = 10
VISIT_PENALTY_HARD_THRESHOLD = 20

# Item / Pokédex IDs needed for goal checks below (constants/item_constants.asm
# and the National Pokédex order, both from pret/pokered — not RAM addresses).
ITEM_ID_OAKS_PARCEL = 0x46  # 70
ITEM_ID_TOWN_MAP = 0x05
MEWTWO_POKEDEX_NUMBER = 150  # fixed since Gen 1, same order as wPokedexOwned

# The 8 badges live in wObtainedBadges (see badges()), not wEventFlags, so
# they can't be expressed as an EVENT_* name from event_constants.py — kept
# as the one hand-picked addition to the otherwise-generic GOAL_CANDIDATES
# pool below.
BADGE_GOALS = (
    "badge1",
    "badge2",
    "badge3",
    "badge4",
    "badge5",
    "badge6",
    "badge7",
    "badge8",
)

# Manual, hand-picked exclusions on top of event_graph.AUTO_BLACKLIST_EVENTS
# (dead / cyclic-toggle / resettable events -- see tools/gen_event_graph.py).
# Extend this set for cases the automatic rules can't catch: e.g. an event
# that is technically one-way and non-cyclic in the disassembly but is known
# (from live play / debug_play -vv) to be an unreliable trigger.
GOAL_MANUAL_BLACKLIST: frozenset[str] = frozenset()

# Generic goal namespace: every named wEventFlags bit from event_constants.py,
# minus anything auto- or manually blacklisted, plus the 8 badges (badges
# live in wObtainedBadges, not wEventFlags, so they can't be expressed as an
# EVENT_* name -- see badges()/is_goal_satisfied()). This is the pool
# curriculum/goal-conditioning code should draw from instead of the old
# hand-picked GOAL_* stage constants above.
EVENT_GOAL_CANDIDATES: frozenset[str] = frozenset(
    name
    for name in _event_constants.EVENTS
    if name not in _event_graph.AUTO_BLACKLIST_EVENTS
    and name not in GOAL_MANUAL_BLACKLIST
)
GOAL_CANDIDATES: frozenset[str] = EVENT_GOAL_CANDIDATES | frozenset(BADGE_GOALS)

# Data.compass_progress's live stall detector (see that method's docstring):
# a goal reported "arrived" (BFS hop distance <= COMPASS_STALL_DIST) for
# COMPASS_STALL_STEPS cumulative steps without ever becoming satisfied gets
# excluded from the compass for the rest of the episode -- a generic
# backstop for gating event_graph's static CheckEvent/SetEvent-proximity
# heuristic structurally cannot see (script-counter state machines, item/
# badge checks, NPC-position checks, ...).
COMPASS_STALL_DIST = 2
COMPASS_STALL_STEPS = 60

# Named reward sub-components tracked as cumulative per-episode sums and
# exposed to the model (see Data.reward_component_vector / _accum_reward).
# Each entry mirrors one addend inside Data.reward(); "total" is the running
# sum of every component, i.e. cumulative episode return so far. Fixed order
# so the observation vector's dimension/layout is stable.
REWARD_COMPONENT_NAMES: tuple[str, ...] = (
    "core",
    "generic_progress",
    "battle_exit",
    "battle",
    "dialog_exit",
    "position",
    "dialog_milestone",
    "dialog_step",
    "menu_useless",
    "battle_useless_milestone",
    "battle_useless_step",
    "anti_loop",
    "dialog_reopen",
    "active_map_presence",
    "new_dialog_presence",
    "new_map_presence",
    "total",
)

# Game-mode buckets a step's *whole* reward (every component combined) gets
# attributed to, based on which mode was active that step -- see
# Data._current_reward_mode / reward_mode_sums. Mirrors the mutually
# exclusive is_world/is_dialog/is_menu/is_battle/is_cutscene_locked states
# (game_mode_flags_data covers the first four; "cutscene" is the fifth,
# script-locked-with-no-textbox state those four all explicitly exclude).
REWARD_MODE_NAMES: tuple[str, ...] = ("world", "dialog", "menu", "battle", "cutscene")


@dataclass
class Data:
    pyboy: PyBoy
    files_lock: RLock

    visited_screens: list[bytes] = field(default_factory=list)

    # Hierarchical rewards: macro >> meso >> micro (PokeRL / Whidden style).
    # badge_reward/event_reward are the only two payout amounts left —
    # reward_generic_progress pays every GOAL_CANDIDATES event/badge flat,
    # once per episode, no per-goal hand-tuned amount or active-goal scaling.
    badge_reward: float = 10.0  # macro
    event_reward: float = 2.0  # macro
    # One-shot new-map/new-dialog payouts (formerly new_screen_reward /
    # new_dialog_reward) are gone — reward_new_map_presence / reward_new_dialog_presence
    # (see reward()) now cover that ground with a decaying-by-visits shape
    # (full rate on a fresh map/dialog, linearly to 0 by
    # new_position_decay_visits), instead of a flat one-time bonus.
    # battle_turn_reward is kept at its old value (0.1) on purpose, not tied
    # to either of those — see reward_battle_useless_count.
    battle_turn_reward: float = 0.1  # micro (turn progressed in battle)
    new_position_reward: float = 0.008  # micro (slightly lower to curb tile farming)
    # Revisit taper: full credit stays a one-shot, but the drop to 0 no longer
    # happens in a single step. Linear ramp-down over this many visits, so a
    # necessary backtrack (e.g. retracing to a room's only exit) isn't an
    # instant cliff from full bonus to the bare step penalty.
    new_position_decay_visits: int = 4
    # Mid-dialog text farming was an exploit (post-rival Oak speech): tiny
    # +0.01 per screen kept the agent camping without leaving for Route 1.
    dialog_advance_reward: float = 0.0
    dialog_exit_reward: float = 0.2  # meso — leaving dialog is real progress
    # Battle exit (wBattleResult @ RAM.wBattleResult): 0=win, 1=lose, 2=fled.
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
    # r=1, with zero slope at both ends. See _battle_difficulty_scale.
    battle_difficulty_invalid_fallback: float = 0.05
    # Floor on the smoothstep above (max(floor, smoothstep(...))) — without
    # it, a big enough level gap scaled reward_enemy_hp/battle_won_reward
    # toward 0 while battle_useless_step's per-tick waste cost (never scaled
    # by anything battle-related) stayed full price, so a clean win against
    # a sufficiently weak wild Pokemon could net negative overall. Same
    # breakeven-point reasoning as wild_encounter_decay_floor (see its own
    # docstring) and reuses its value — a win always clears the per-tick
    # waste cost with margin, however lopsided the level gap, while staying
    # far enough below 1.0 that farming a trivial fight on purpose is still
    # clearly worse than a fair one. Deliberately NOT applied to
    # battle_difficulty_invalid_fallback above — that path is a bad-read
    # anti-exploit guard, not a real difficulty reading, and must stay able
    # to sit below this floor.
    battle_difficulty_scale_floor: float = 0.15
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
    # Bumped from -0.001 -- too weak against gamma=0.99 to discourage pure
    # step-maximizing (wandering to farm small decaying rewards like
    # active_map_event_reward/status_reward instead of pushing forward).
    # Still well under new_position_reward (0.008), so genuine exploration
    # of a fresh tile stays clearly profitable -- only idling/backtracking
    # gets meaningfully more expensive.
    base_reward: float = -0.003
    truncated_reward: float = -0.05
    new_item_reward: float = 0.5

    # Anti-loop / anti-spam penalties (PokeRL-style). Stronger than before so
    # farming a ~17 return without the stage goal is no longer attractive.
    # Thresholds below were tuned against a TensorBoard reading where
    # loop_episode_rate sat at ~1.0 for essentially every episode across
    # every curriculum stage: position_visit_counts is a whole-episode
    # cumulative counter over a handful of small maps, so a doorway/
    # chokepoint tile naturally crosses 3-5 visits on any normal multi
    # thousand-step episode — that's foot traffic, not looping.
    visit_penalty_soft: float = -0.05  # visit count > 10
    visit_penalty_hard: float = -0.15  # visit count > 20
    # Wild-battle rewards (battle_won_reward, and the positive side of
    # reward_enemy_hp/reward_enemy_status) decay with repeat position_visit_
    # counts on the tile a fight started on — same counter as
    # new_position_reward's walking decay (see reward_battle_exit /
    # _wild_encounter_decay / _battle_entry_wild_visits), so grinding one
    # grass tile back and forth no longer stays net positive indefinitely.
    # Set well above new_position_decay_visits (4) — position_visit_counts
    # climbs from ordinary foot traffic (not just fights), so a threshold
    # this low would zero out wild rewards on any well-trodden route tile
    # before the agent ever got its first fight there.
    wild_visit_decay_visits: int = 4
    # Floor for _wild_encounter_decay -- decaying all the way to 0 made every
    # wild-battle reward/cost on a tile past wild_visit_decay_visits net to
    # whatever undecayed cost remains (battle_useless_step's per-tick waste,
    # never decayed), a guaranteed loss forever regardless of outcome. That
    # made re-engaging a well-trodden grass tile strictly worse than walking
    # around it, in a policy that already tends to re-tread the same small
    # footprint (spatial-loop behavior) -- tiles decay past this threshold
    # fast, so most of the agent's wild-encounter exposure ends up here,
    # teaching blanket combat avoidance instead of just curbing farming.
    # 0.15 clears the breakeven point (waste_cost / undecayed_reward_sum,
    # ~0.056 for a representative fair fight) with margin, while staying far
    # enough below 1.0 that grinding a stale tile on purpose is still clearly
    # worse than a fresh fight or fresh exploration -- farming stays
    # unrewarding, it just stops being punished.
    wild_encounter_decay_floor: float = 0.15
    # Brought down from -0.08/-0.10 (previously "stronger than before" per
    # the comment above -- tried against the same ~1.0 loop_episode_rate
    # this is still tuned against, and it didn't move that number) to the
    # same order of magnitude as new_position_reward (0.008). At -0.08/-0.10,
    # a single flagged incident wiped out 10-12 tiles' worth of exploration
    # credit -- interspersed with ordinary interruptions on a long,
    # imperfect traversal (a wild battle, backing off a trainer's sightline)
    # that made genuinely pushing into new territory net-negative far more
    # often than it made pure stalling net-negative, biasing the policy
    # toward not attempting the traversal at all rather than toward doing it
    # more cleanly. Kept above new_position_reward itself so looping in
    # place is still never profitable, just no longer catastrophic relative
    # to real progress happening nearby.
    action_pattern_penalty: float = -0.02
    spatial_loop_penalty: float = -0.02
    # A *single* no-effect menu press (e.g. UP at the top of a list) used to
    # fire this every time — nearly guaranteed at least once in any episode
    # that opens a menu at all. Require menu_spam_streak consecutive no-ops
    # before it counts as spam rather than incidental boundary-bumping.
    menu_spam_penalty: float = -0.05
    menu_spam_streak_threshold: int = 3
    # Cursor oscillating between a couple of menu states (e.g. ITEM <-> CANCEL)
    # changes state every step, so it evades menu_spam_penalty's "no-change"
    # check above. Catch revisits of the same menu state instead.
    menu_loop_penalty: float = -0.10
    # START/SELECT/d-pad while a textbox is open — does not advance story text.
    # A lone stray press mixed into otherwise-correct A/B mashing (near-
    # certain under a stochastic policy) used to flag the whole episode as
    # "looped". Require dialog_wrong_streak consecutive wrong presses before
    # it counts as genuinely stuck rather than one wrong roll.
    dialog_wrong_button_penalty: float = -0.08
    dialog_wrong_streak_threshold: int = 2
    # In-dialog waste now reuses base_reward via the same shape as
    # reward_position's step_penalty (see reward_dialog) — no separate scale.
    # Re-open a dialog that was already exited on this map: 1st = penalty, 2nd = truncate.
    dialog_reopen_penalty: float = -0.5
    # Consecutive anti-loop hits → truncate episode (escape local optima).
    max_loop_streak: int = 48
    # Episode goal for terminated() — a GOAL_CANDIDATES member (an EVENT_*
    # name or one of BADGE_GOALS). Only used by goal_reached()/terminated();
    # reward_generic_progress pays out every GOAL_CANDIDATES event regardless
    # of which one is "the" active goal, so this doesn't gate reward anymore,
    # only episode success/failure and the PPO goal one-hot (curriculum_config.py).
    goal: str = BADGE_GOALS[0]

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
    map_vision_radius: int = 5

    # --heatmap opt-in (set by PokemonRedEnv from its collect_heatmap ctor arg).
    # Gates direction_counts below so plain training never pays for it.
    collect_heatmap: bool = False
    # World-tile the agent left -> {"up"/"down"/"left"/"right": step count},
    # for the --heatmap live window's movement-direction overlay.
    direction_counts: dict[tuple[int, int, int], dict[str, int]] = field(
        default_factory=dict
    )
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
    # Cumulative per-episode sum for each REWARD_COMPONENT_NAMES entry (incl.
    # "total" = sum of every other entry) -- the model-facing "how much of
    # each reward type have I earned so far this episode" observation. See
    # _accum_reward / reward_component_vector.
    reward_component_sums: dict[str, float] = field(default_factory=dict)
    # Cumulative per-episode sum of a step's whole reward (every component
    # combined), bucketed by REWARD_MODE_NAMES -- monitoring only, not fed to
    # the model. See _current_reward_mode / reward_mode_vector.
    reward_mode_sums: dict[str, float] = field(default_factory=dict)
    # Battle outcomes per (x, y, map_id): {"win"/"loss"/"smart"/"coward": n},
    # for --heatmap's win-rate (win vs loss) and flee-rate (smart vs coward)
    # overlays. Attributed to the same last-known-world tile as reward_sums
    # (a battle has no world position of its own). "loss" folds in both a
    # fainted-but-continued loss and a full blackout. "smart"/"coward" are a
    # successful flee's TryRunningFromBattle classification (fled from an
    # over-leveled enemy vs a beatable one) — disjoint from win/loss, so a
    # tile's win-rate denominator never includes a fled fight and vice versa.
    battle_outcome_counts: dict[tuple[int, int, int], dict[str, int]] = field(
        default_factory=dict
    )
    # Story-milestone payouts triggered per (x, y, map_id), for --heatmap's
    # milestone-density overlay — how much of the curriculum's critical path
    # actually lines up with where the agent spends time. Same
    # _last_heatmap_pos attribution as reward_sums/battle_outcome_counts.
    milestone_hit_counts: dict[tuple[int, int, int], int] = field(default_factory=dict)
    # Ticks spent in a dialog per (x, y, map_id) tile, for --heatmap's
    # dialog-recency overlay (episodes since a tile last triggered a
    # dialog). Same _last_heatmap_pos attribution as reward_sums — a dialog
    # has no world position of its own, so it's anchored on the last known
    # world tile.
    dialog_hit_counts: dict[tuple[int, int, int], int] = field(default_factory=dict)
    # Truncation endings per (x, y, map_id) tile, for --heatmap's
    # "truncations" overlay — where episodes actually run out (a stuck
    # fuse, a map/step budget, ...; see TRUNCATE_CAUSES), not just where
    # time is spent. Populated by PokemonRedEnv.step() itself (not here),
    # since the "max_steps" cause is only known at the env level — same
    # _last_heatmap_pos attribution as the other mid-dialog-safe overlays.
    truncate_hit_counts: dict[tuple[int, int, int], int] = field(default_factory=dict)
    # Always-on (not gated by collect_heatmap, unlike battle_outcome_counts
    # above) per-episode-leg tally of reward_battle_exit's `kind` ("win",
    # "lose", "blackout", "smart", "coward") -- for MilestoneCallback's
    # pokemon/battle_*_rate / pokemon/battles_per_episode TensorBoard charts.
    battle_outcome_tally: dict[str, int] = field(default_factory=dict)
    # Env-step count (not raw emulator ticks) spent with is_battle() true
    # this episode leg -- for pokemon/battle_ticks_frac (how much of the
    # episode is spent in battle, the direct measure of combat engagement
    # vs. avoidance, instead of inferring it from map_budget/loop stats).
    battle_step_count: int = 0
    # Sum/count of _wild_encounter_decay() readings taken at wild battle
    # exit (reward_battle_exit), plus how many of those hit
    # wild_encounter_decay_floor -- for pokemon/wild_encounter_decay_mean
    # and pokemon/wild_encounter_decay_floored_rate, so the floor's real
    # in-training effect is measured directly instead of hand-computed from
    # one log excerpt.
    _wild_decay_sum: float = 0.0
    _wild_decay_count: int = 0
    _wild_decay_floored_count: int = 0
    # Set by truncated() the step map_budget fires -- True when the map
    # that tripped it is the same one the episode started on
    # (_start_map_id). Lets MilestoneCallback report what fraction of
    # map_budget truncations happen on the start/home map specifically
    # (pokemon/map_budget_trunc_at_start_rate), instead of guessing which
    # map is responsible from context.
    last_map_budget_trunc_at_start: bool = False
    _last_heatmap_pos: tuple[int, int, int] | None = None
    # Per-(map_id, dialog_id) step counter — dialog_id is read from a single
    # byte (0-255), so this backs a 256-wide per-map histogram exposed to the
    # model (see dialog_id_visit_grid), the dialog analogue of
    # position_visit_counts/visit_mask_grid.
    dialog_id_visit_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    # Per-map_id step counter — map_id is a single byte (0-255), so this backs
    # a 256-wide episode-wide histogram exposed to the model (see
    # map_id_visit_grid): how long (in steps) the agent has spent on each map
    # this episode, the map analogue of dialog_id_visit_counts. Increments
    # every step regardless of mode (world/dialog/battle/menu) — a gym
    # battle or a long dialog on a map is still time spent on that map. Also
    # drives reward_new_map_presence's decay (see below) -- NOT purely
    # informational despite map_visit_grid_cap's comment (that one is about
    # the grid's normalization cap specifically); reset alongside
    # world_map_step_counts/dialog_id_visit_counts in
    # PokemonRedEnv.set_curriculum(clear_visits=True)/debug_play._clear_visits.
    map_id_visit_counts: dict[int, int] = field(default_factory=dict)
    # Normalization cap for map_id_visit_grid — same "distinguish low counts,
    # saturate the tail" shape as visit_mask_grid's fixed 10, but map dwell
    # times run for whole episodes rather than single tile visits, so the cap
    # is much larger. Tied to MAP_DWELL_BUDGET instead of its own constant —
    # this cap itself is purely a display/normalization scale for the grid
    # input, not a reward threshold — but the underlying map_id_visit_counts
    # it normalizes *is* reward-affecting (see reward_new_map_presence).
    map_visit_grid_cap: int = MAP_DWELL_BUDGET
    # Per-map step allowance used to derive curriculum_config's episode
    # max_steps (see MAP_DWELL_BUDGET) and shown by debug_play.py/
    # run_eval_ppo.py's dwell diagnostics — informational only, no reward
    # penalty is tied to it.
    map_dwell_budget: float = MAP_DWELL_BUDGET

    # Size-scaled per-map step budget: steps allowed on one map this episode
    # is that map's own width*height in blocks (map_constants.py, generated
    # from pret/pokered's map_const macro) — no extra multiplier, so the
    # budget is literally the map's block area, scaled by
    # new_position_decay_visits (see map_truncate_budget). Only fills while
    # is_world() (see _tick_map_budget) — dialog/battle/menu time on a map
    # doesn't consume it. No per-step penalty for overstaying; the episode is
    # simply truncated once the budget is exceeded (see map_truncate_budget/
    # truncated).
    world_map_step_counts: dict[int, int] = field(default_factory=dict)

    # Per-step shaping toward whichever map currently has an unfinished,
    # reachable event (see active_map_events): small reward for being on a
    # map with one pending. No penalty for being on a map with none —
    # an active-but-actually-unreachable event (e.g. its parent event_graph
    # edge is wrong) turned that into a tax on every other map, which made
    # the policy camp the falsely-"active" map instead of exploring.
    active_map_event_reward: float = 0.01

    # Generic regression safety net over GOAL_CANDIDATES (see
    # reward_generic_progress). Should rarely fire — event_graph's auto
    # blacklist already excludes resettable/cyclic events from
    # GOAL_CANDIDATES — but a missed case or a manually-added candidate
    # shouldn't be able to farm a false milestone by flip-flopping.
    event_regression_penalty: float = -0.3

    recent_actions: deque = field(default_factory=lambda: deque(maxlen=20))
    recent_positions: deque = field(default_factory=lambda: deque(maxlen=16))
    recent_menu_states: deque = field(default_factory=lambda: deque(maxlen=16))
    loop_flag: bool = False
    loop_causes: set[str] = field(default_factory=set)
    loop_streak: int = 0
    # Consecutive-occurrence counters for the two anti-loop checks that used
    # to fire on a single incident (see menu_spam_streak_threshold /
    # dialog_wrong_streak_threshold).
    menu_noop_streak: int = 0
    dialog_wrong_streak: int = 0
    # Fuse(s) that fired on the most recent truncated() call — see TRUNCATE_CAUSES.
    last_truncate_causes: frozenset[str] = frozenset()
    # Paid-this-episode set for reward_generic_progress — every GOAL_CANDIDATES
    # event/badge pays out at most once per episode.
    _milestones_hit: set[str] = field(default_factory=set)
    # Snapshot of GOAL_CANDIDATES satisfied last step, for the regression
    # diff in reward_generic_progress.
    _prev_satisfied_events: frozenset[str] = field(default_factory=frozenset)
    # compass_progress()'s live stall detector: how many steps (cumulative,
    # not necessarily consecutive -- see compass_progress) the compass has
    # reported each goal as "arrived" (dist <= COMPASS_STALL_DIST) without
    # it ever becoming satisfied. Catches gating event_graph's static
    # CheckEvent/SetEvent-proximity heuristic structurally can't see at all
    # (e.g. a per-map script-counter state machine like OaksLab.asm's
    # wOaksLabCurScript, where later SetEvents are gated purely by earlier
    # SetEvents in the same sequential dispatch, not by any CheckEvent) --
    # complements _goal_parents_satisfied rather than replacing it.
    _compass_near_counts: dict[str, int] = field(default_factory=dict)
    # Goals _compass_near_counts pushed past COMPASS_STALL_STEPS -- excluded
    # from compass_progress's candidate pool for the rest of the episode.
    _compass_excluded: set[str] = field(default_factory=set)
    # Per-step debug trail for eval -vv: (name, payout) actually paid this step.
    last_milestone_payouts: list[tuple[str, float]] = field(default_factory=list)
    # GOAL_CANDIDATES names that un-satisfied this step (see
    # reward_generic_progress's regression penalty). No more soft/hard
    # distinction — every regression is real now (no curriculum-clear
    # exemption bookkeeping left to distinguish "backslide" from "advanced
    # past this on purpose").
    last_regressed: list[str] = field(default_factory=list)
    # Vestigial, always empty now — the old regressed-and-spent blocking
    # concept is gone (see reward_generic_progress). Kept only so external
    # readers (run_eval_ppo.py -v) don't need a hasattr guard.
    last_milestone_blocked: list[str] = field(default_factory=list)
    # Set once per reward() call; True on the exact step a full-party wipe
    # exits the battle screen. Shared by reward_core (suppress the resulting
    # free Pokemon-Center-style heal) and reward_battle_exit (classify the
    # exit correctly instead of trusting wBattleResult).
    _just_blacked_out: bool = False
    # Set alongside _just_blacked_out; consumed the first step is_world() is
    # true again (see count()) once the forced Pokemon-Center warp actually
    # lands, to top up that destination map's world_map_step_counts budget.
    _pending_blackout_recovery: bool = False
    _start_map_id: int | None = None
    # Distinct dialog screen hashes seen for the current dialog_id. Blink frames
    # revisit old hashes; only a *new* hash counts as text progress.
    _dialog_screens_seen: set[str] = field(default_factory=set)
    # wMaxMenuItem snapshotted the instant the current dialog_id started (see
    # count()'s _dialog_id_changed branch) -- pret/pokered does NOT reset
    # wMaxMenuItem when a plain textbox opens, it just keeps whatever value
    # was left over from the last real menu (e.g. 5 after browsing a 6-item
    # START menu), so a raw "!= 0" check reads that stale leftover as a live
    # choice for the rest of the conversation. dialog_has_live_choice()
    # instead treats it as live only once wMaxMenuItem *changes* away from
    # this per-conversation baseline, which only happens when a real
    # HandleMenuInput-backed choice (YesNoChoice etc.) actually runs.
    _dialog_choice_baseline: int | None = None
    # Dialogs cleanly exited this episode → reopen tracking (penalty per reopen).
    _completed_dialogs: set[tuple[int, int]] = field(default_factory=set)
    _dialog_reopen_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    # True once the parcel has actually been observed sitting in the bag/PC
    # (see gave_oaks_parcel) — guards against the have_oaks_parcel event flag
    # (D60D) landing one frame-skip window before the item is actually
    # written into the bag, which would otherwise read as "gone" (=delivered)
    # before it was ever really carried.
    _saw_oaks_parcel_in_bag: bool = False
    # Set each step by reward_battle_exit (debug_play / diagnostics).
    last_flee_reward: float = 0.0
    last_flee_info: dict | None = None
    last_battle_exit_info: dict | None = None
    # Set each step by reward_enemy_hp while in battle (debug_play / diagnostics).
    last_enemy_hp_debug: dict | None = None
    # Set each step by compass_progress (debug_play / diagnostics / eval -vv).
    last_compass_debug: dict | None = None
    # Last known-good (nonzero) enemy/active level this battle, for
    # _battle_difficulty_scale — wEnemyMonLevel/wBattleMonLevel can read back
    # 0 on some frames mid-fight (e.g. during a mon-switch/animation window)
    # even though HP is clearly changing; holding the last real reading is
    # more robust than trusting whatever a single frame happens to show.
    _battle_enemy_level_cache: int = 0
    _battle_active_level_cache: int = 0
    # position_visit_counts reading for this tile *before* the current fight's
    # own count() seed — captured at battle entry, consumed by
    # _wild_encounter_decay so a tile only walked over a handful of times
    # still pays close to full battle_won_reward (mirrors reward_position
    # reading position_visit_counts pre-increment).
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

    def load(self, path: str):
        with self.files_lock:
            with open(f"{path}/__visited_pokedex_own.pkl", "rb") as f:
                self.__visited_pokedex_own = pickle.load(f)
            with open(f"{path}/__visited_pokedex_seen.pkl", "rb") as f:
                self.__visited_pokedex_seen = pickle.load(f)
            with open(f"{path}/visited_positions.pkl", "rb") as f:
                self.visited_positions = pickle.load(f)

    def clean(self):
        self.__visited_pokedex_own = None
        self.__visited_pokedex_seen = None
        self.in_menu_ticks = 0
        self.in_battle_ticks = 0
        self.in_dialog_ticks = 0
        self.visited_positions = {}
        self.position_visit_counts = {}
        self.direction_counts = {}
        self.map_transitions = {}
        self.reward_sums = {}
        self.reward_component_sums = {}
        self.reward_mode_sums = {}
        self.dialog_hit_counts = {}
        self.dialog_id_visit_counts = {}
        self.map_id_visit_counts = {}
        self.world_map_step_counts = {}
        self._just_blacked_out = False
        self._pending_blackout_recovery = False
        self.battle_outcome_counts = {}
        self.milestone_hit_counts = {}
        self.truncate_hit_counts = {}
        self.battle_outcome_tally = {}
        self.battle_step_count = 0
        self._wild_decay_sum = 0.0
        self._wild_decay_count = 0
        self._wild_decay_floored_count = 0
        self.last_map_budget_trunc_at_start = False
        self._last_heatmap_pos = None
        self.recent_actions.clear()
        self.recent_positions.clear()
        self.recent_menu_states.clear()
        self.loop_flag = False
        self.loop_causes = set()
        self.loop_streak = 0
        self.menu_noop_streak = 0
        self.dialog_wrong_streak = 0
        # Seed both from the just-loaded save's actual live state rather than
        # always empty: a checkpoint captured after some GOAL_CANDIDATES
        # events already fired (e.g. saves/EVENT_ENTERED_BLUES_HOUSE) would
        # otherwise have reward_generic_progress's first call this episode
        # see every already-satisfied event as newly hit in one batch --
        # paying out and firing terminated()/goal_success for all of them at
        # once on literally every reset. Seeding _milestones_hit too (not
        # just _prev_satisfied_events) blocks a later regress-then-reflicker
        # of one of these from paying a second time. Reading memory here is
        # safe -- this is a settled save snapshot, not the live frame-skip
        # window where an event flag can flip true a tick before the write
        # lands (same reasoning as _saw_oaks_parcel_in_bag just below).
        self._milestones_hit = {n for n in GOAL_CANDIDATES if self.is_goal_satisfied(n)}
        self._prev_satisfied_events = frozenset(self._milestones_hit)
        self._compass_near_counts = {}
        self._compass_excluded = set()
        self.last_milestone_payouts = []
        self.last_regressed = []
        self._dialog_screens_seen = set()
        self._dialog_choice_baseline = None
        self._completed_dialogs = set()
        self._dialog_reopen_counts = {}
        # Seed from the just-loaded save rather than always False: a
        # checkpoint captured *after* the parcel was delivered has
        # have_oaks_parcel() true with the item already gone from bag/pc, so
        # gave_oaks_parcel() would otherwise wait forever this episode for a
        # "seen in bag" moment that already happened before the save was
        # made (see gave_oaks_parcel). Reading memory here is safe — this is
        # a settled save snapshot, not the live frame-skip window where the
        # D60D event flag can flip true a tick before the item write lands.
        self._saw_oaks_parcel_in_bag = bool(self.have_oaks_parcel(self.pyboy.memory))
        self.last_flee_reward = 0.0
        self.last_flee_info = None
        self.last_battle_exit_info = None
        self.last_enemy_hp_debug = None
        self.last_compass_debug = None
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

    def count(
        self,
        reward: float,
        action: int,
        memory: bytes | None = None,
        duration: int = 16,
    ):
        self.visited_pokedex_own = self.pokedex_own(self.pyboy.memory)
        self.visited_pokedex_seen = self.pokedex_seen(self.pyboy.memory)

        # First step control is back in the world after a blackout's forced
        # Pokemon-Center warp: top up (not reset to 0 -- keeps the fuse a
        # real ceiling, not an open door) the destination map's budget by
        # exactly one map_truncate_budget, clamped at its current floor. The
        # map's own dwell/step count from earlier this episode can otherwise
        # already be near/at budget before the agent ever gets a chance to
        # walk back out, tripping map_budget almost immediately on return --
        # punishing the trip out (e.g. to Route 1) that caused the blackout
        # instead of punishing only the blackout itself (battle_lost_penalty
        # already covers that). Deliberately does NOT touch
        # position_visit_counts/visited_positions -- exploration_reward for
        # this map's tiles stays fully decayed, so there is no reward to
        # farm by blacking out on purpose, only breathing room to leave.
        if self._pending_blackout_recovery and self.is_world(self.pyboy.memory):
            self._pending_blackout_recovery = False
            dest_map = self.map_id(self.pyboy.memory)
            budget = self.map_truncate_budget(dest_map)
            self.world_map_step_counts[dest_map] = max(
                0, self.world_map_step_counts.get(dest_map, 0) - budget
            )

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
            self.visited_positions[pos] = self.visited_positions.get(pos, 0) + duration
            # position_visit_counts always advances too, even on a stationary
            # A/B mash — it drives the decay in reward_position's
            # exploration_reward, and freezing it while "talking" let it pay
            # out a flat per-tick bonus forever instead of decaying to zero
            # (reward_anti_loop's own visit-penalty check has its own
            # independent `not interacting` gate below, so it does not need
            # this counter frozen to spare a
            # normal talk).
            self.position_visit_counts[pos] = self.position_visit_counts.get(pos, 0) + 1
            if not (stayed and interacting):
                self.recent_positions.append(pos)

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
            # trainer, which will not keep re-firing off the same tile). Cache
            # the pre-seed position_visit_counts reading for
            # _wild_encounter_decay before the seed below bumps it, so a tile
            # never walked before this fight still reads 0 (full reward).
            # position_visit_counts only ever needs a single seed entry here
            # so reward_position() does not see the tile as brand new on
            # every world<-battle return.
            if self.type_of_battle(self.pyboy.memory) == 1:
                self._battle_entry_wild_visits = self.position_visit_counts.get(pos, 0)
                self._mark_wild_grass_island(pos)
            if pos not in self.position_visit_counts:
                self.position_visit_counts[pos] = 1
        elif not self.is_battle(self.pyboy.memory) and not self.is_dialog(
            self.pyboy.memory
        ):
            # Menu with no movement (blocked, NPCs still walk): same stuck fuse
            # as world. Skip during dialog — text uses its own fuse.
            pos = self.get_position()
            self.visited_positions[pos] = self.visited_positions.get(pos, 0) + duration

        if self.is_menu(self.pyboy.memory):
            self.in_menu_ticks += duration
        else:
            self.in_menu_ticks = max(0, self.in_menu_ticks - 0.25 * duration)

        # Battle's FIGHT/PKMN/ITEM/RUN and move-select menus reuse the same
        # wCurrentMenuItem/wTopMenuItemX/Y cursor registers as every other
        # menu in the game (see pret/pokered home/window.asm PlaceMenuCursor,
        # shared code for all menus). Track cursor history there too, not
        # just under is_menu() — otherwise this deque stays permanently empty
        # during battle (is_menu() is defined as "blocked and not is_battle")
        # and reward_anti_loop's net_stuck/menu_loop checks below can never
        # tell "cursor actually stuck" from "cursor moving every step", so
        # they default to flagging ordinary battle-menu navigation as a loop.
        # is_battle_menu(), not is_battle(): a plain battle message ("X used
        # TACKLE!") has no real menu open, so these registers just sit at
        # whatever they were last drawn to — appending them here filled this
        # deque with runs of identical "stale" tuples for every multi-message
        # turn (every ordinary battle has several), which made menu_loop
        # below fire almost continuously through normal message-advancing,
        # not actual cursor loops.
        if self.is_menu(self.pyboy.memory) or self.is_battle_menu(self.pyboy.memory):
            self.recent_menu_states.append(
                (
                    self.menu_position_x(self.pyboy.memory),
                    self.menu_position_y(self.pyboy.memory),
                    self.real_current_menu_selected_item(self.pyboy.memory),
                )
            )
        elif not self.is_battle(self.pyboy.memory):
            # Mid-battle-message: leave prior real menu-state history intact
            # instead of clearing it, so the loop detector's memory survives
            # the message pause between menu appearances.
            self.recent_menu_states.clear()

        if self.is_battle(self.pyboy.memory):
            # One env step, not raw ticks -- see battle_step_count's own
            # comment (pokemon/battle_ticks_frac telemetry).
            self.battle_step_count += 1
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
            # Per-map byte-histogram step counter (see dialog_id_visit_grid),
            # same per-step cadence as position_visit_counts.
            did_key = (
                self.map_id(self.pyboy.memory),
                self.dialog_id(self.pyboy.memory),
            )
            self.dialog_id_visit_counts[did_key] = (
                self.dialog_id_visit_counts.get(did_key, 0) + 1
            )
            # --heatmap dialog-recency overlay: same _last_heatmap_pos
            # attribution as reward_sums (a dialog can be mid-cutscene, off
            # is_world(), so it's anchored on the last known world tile).
            if self.collect_heatmap and self._last_heatmap_pos is not None:
                self.dialog_hit_counts[self._last_heatmap_pos] = (
                    self.dialog_hit_counts.get(self._last_heatmap_pos, 0) + duration
                )
            # Fuse resets only on dialog_id change (or leaving dialog below).
            # New text frames alone used to reset forever → infinite camp.
            if self._dialog_id_changed(memory):
                self._dialog_screens_seen = set()
                self.in_dialog_ticks = 0
                # New conversation/topic -- re-baseline dialog_has_live_choice
                # against whatever wMaxMenuItem happens to be right now
                # (stale or not; only a *change* from here reads as a live
                # choice, see the field comment).
                self._dialog_choice_baseline = int(self.pyboy.memory[RAM.wMaxMenuItem])
            else:
                self.in_dialog_ticks += duration
            self._dialog_screen_is_new()
        else:
            self.in_dialog_ticks = 0
            self._dialog_screens_seen = set()
            self._dialog_choice_baseline = None

        mid = self.map_id(self.pyboy.memory)
        self.map_id_visit_counts[mid] = self.map_id_visit_counts.get(mid, 0) + 1
        self._tick_map_budget(self.pyboy.memory)

    def _mark_wild_grass_island(self, start_pos: tuple[int, int, int]) -> None:
        """BFS-flood the grass-tile patch a wild encounter's tile sits in,
        over the full static per-map grass bitmap (see pokemon.map_collision,
        generated straight from pret/pokered's own map/tileset data by
        tools/gen_map_collision.py) and bump position_visit_counts by 1 on
        every reachable tile -- so one wild battle credits the whole
        connected grass "island" the tile sits in, not just that one square,
        discouraging camping a single tile of a patch the agent has
        effectively already scouted.

        Grass-only, not "any walkable tile": on an overworld map the grass
        directly abuts the path with no wall between them, so flooding over
        is_walkable() would swallow the entire connected route instead of
        stopping at the grass patch's actual edge. Empty for a non-grass
        encounter tile (cave floor, water, Safari Zone building, ...) --
        those tilesets have no grass tile at all, so flood_grass_island()
        returns nothing and the single-tile seed in the caller is all that
        tile gets, same as before this only ever covered grass anyway.

        Unlike a live on-screen read, this covers the map's actual full
        connected region (not just whatever currently fits on the GB
        screen around the player) and needs no per-step caching -- it's
        static data, looked up directly by (map_id, x, y).
        """
        x0, y0, map_id = start_pos
        for x, y in _map_collision.flood_grass_island(map_id, x0, y0):
            world_pos = (x, y, map_id)
            self.position_visit_counts[world_pos] = (
                self.position_visit_counts.get(world_pos, 0) + 1
            )

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

    def current_map_script_state(self, memory: bytes | None = None) -> int | None:
        """Live wFooCurScript byte for the CURRENT map (pokemon.map_scripts.
        MAP_SCRIPTS), if this map has its own dedicated CurScript variable --
        which exact dispatch state its story is in RIGHT NOW, ground truth
        from RAM instead of a static "assume state 0" guess. This matters
        whenever an episode resumes mid-sequence (a checkpoint captured
        partway through a map's cutscene chain) -- the static analysis has
        no way to know which state is actually active, only RAM does.

        None if this map has no def_script_pointers table at all, no
        dedicated variable (shares the single wCurMapScript byte instead,
        which only reflects *some* map's state -- meaningless to read
        unless we already know it's this one, and pokemon.map_scripts
        doesn't record that reliably enough to trust here), or the byte
        read doesn't match any state index this module actually parsed for
        that map (a table this static analysis didn't fully capture, or a
        transient/invalid mid-transition value).
        """
        if memory is None:
            memory = self.pyboy.memory
        ms = _map_scripts.MAP_SCRIPTS.get(self.map_id(memory))
        if ms is None or ms["cur_script_ram"] is None:
            return None
        addr = getattr(RAM, ms["cur_script_ram"], None)
        if addr is None:
            return None
        idx = memory[addr]
        return idx if idx in ms["states"] else None

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
                row.append(
                    min(visits, VISIT_PENALTY_SOFT_THRESHOLD)
                    / VISIT_PENALTY_SOFT_THRESHOLD
                )
            grid.append(row)
        return grid

    def dialog_id_visit_grid(self) -> list[float]:
        """Per-map histogram of step counts for every possible dialog_id byte
        value (0-255), the dialog analogue of visit_mask_grid."""
        map_id = self.map_id(self.pyboy.memory)
        return [
            min(self.dialog_id_visit_counts.get((map_id, d), 0), 10) / 10.0
            for d in range(256)
        ]

    def map_id_visit_grid(self) -> list[float]:
        """Episode-wide histogram of step counts for every possible map_id
        byte value (0-255) — how long the agent has spent on each map this
        episode, the map analogue of dialog_id_visit_grid."""
        cap = self.map_visit_grid_cap
        return [min(self.map_id_visit_counts.get(m, 0), cap) / cap for m in range(256)]

    def map_budget_progress(self) -> list[float]:
        """Episode-wide histogram, one entry per possible map_id byte value
        (0-255), of world_map_step_counts[m] / map_truncate_budget(m) -- how
        close each map the agent has spent time on is to the map_budget
        truncate cause, the same per-map_id histogram shape as
        map_id_visit_grid/dialog_id_visit_grid. Unclamped per entry (no
        min()-cap) -- truncated()'s check runs every step regardless of
        mode, so the *current* map's entry can only ever land marginally
        over 1.0 for the single step before the episode actually ends; maps
        no longer occupied are frozen wherever they were left.
        """
        return [
            self.world_map_step_counts.get(m, 0) / self.map_truncate_budget(m)
            for m in range(256)
        ]

    def stuck_tile_progress(self) -> list[float]:
        """Current tile's visited_positions duration as a fraction of the
        stuck_tile truncate fuse (max_useless_ticks) -- the same "distance to
        the fuse" signal core_data already gives for in_battle_ticks/
        in_menu_ticks/in_dialog_ticks, for the one fuse (stuck_tile) that
        previously had no direct normalized input; visit_mask_grid/
        position_visit_counts is a different counter (revisit frequency, cap
        10) and does not track this. Unclamped like core_data's ratios.
        """
        return self.data_normalizer(
            [self.visited_positions.get(self.get_position(), 0)],
            max=self.max_useless_ticks,
        )

    def loop_streak_progress(self) -> list[float]:
        """loop_streak as a fraction of the loop_streak truncate fuse
        (max_loop_streak) -- previously not exposed to the model at all.
        Unclamped like core_data's tick ratios.
        """
        return self.data_normalizer([self.loop_streak], max=self.max_loop_streak)

    def _goal_parents_satisfied(self, goal: str) -> bool:
        """Whether every event_graph parent of ``goal`` is already true in
        THIS episode's live game state (self.is_goal_satisfied), not just
        physically walkable to. BFS reachability (pokemon.navigation) only
        catches *geographic* gating (a Cut tree/badge/HM blocking the
        route) -- it has no way to know a tile's flag is additionally
        gated on an unrelated story flag with no physical obstacle at all
        (e.g. EVENT_DAISY_WALKING requires EVENT_GOT_TOWN_MAP +
        EVENT_ENTERED_BLUES_HOUSE first per PalletTown.asm's
        PalletTownDaisyScript, even though the tile itself is reachable
        from minute one). event_graph is still just a static-analysis
        hint, not proof (see event_graph.py's own caveat) -- but checked
        live against this episode's actual flags, a false positive here
        only means "compass skips a goal that was secretly fine", never
        "compass points at something wrong", so it's a safe filter to
        layer on top of BFS, not a replacement for it. True (no gate) for
        badges (no EVENT_GRAPH entry of their own -- wObtainedBadges isn't
        a named event) and any goal event_graph has no parents for.
        """
        info = _event_graph.EVENT_GRAPH.get(goal)
        if info is None:
            return True
        return all(self.is_goal_satisfied(p) for p in info["parents"])

    def compass_progress(self) -> list[float]:
        """[dx, dy, has_target] toward the nearest currently-walkable,
        not-yet-satisfied, not-currently-story-gated, not-stalled
        GOAL_CANDIDATES tile (pokemon.navigation.nearest_objective,
        filtered by _goal_parents_satisfied and _compass_excluded) -- a
        deterministic BFS "compass", not a learned signal. dx/dy are the
        *next graph hop's* direction, clipped to [-1, 1], not a
        straight-line offset to the target -- see
        navigation.nearest_objective's docstring for why (pos and a
        cross-map target don't share a coordinate frame). [0, 0, 0]
        (has_target=0) whenever no candidate is reachable within
        navigation.DEFAULT_MAX_HOPS, or outside world mode (position is
        meaningless mid-battle/dialog/menu) -- a clean, learnable "no
        signal" state, deliberately never a guessed direction. Imports
        pokemon.navigation lazily (inside this method, not at module level)
        because pokemon.goal_positions -> curriculum_config -> pokemon.Data
        is already a real import chain; importing navigation at Data.py's
        own module level would be circular (see navigation.py's docstring).

        A goal the compass reports as "arrived" (dist <= COMPASS_STALL_DIST)
        for COMPASS_STALL_STEPS cumulative steps without ever becoming
        satisfied is added to _compass_excluded for the rest of the episode
        -- a live, generic backstop for gating _goal_parents_satisfied's
        static event_graph data structurally cannot see (e.g. a per-map
        script-counter state machine, an item/badge check, an NPC-position
        check -- anything that isn't a CheckEvent near the SetEvent). This
        is deliberately slow to trigger (COMPASS_STALL_STEPS is generous):
        an early, barely-trained policy can also sit near a perfectly
        achievable target for a long time simply because it hasn't learned
        the right button yet, and wrongly excluding a fine goal for the
        rest of that one episode is a far cheaper mistake than the
        alternative (a target that can genuinely never fire this episode
        eating the compass slot indefinitely).
        """
        if not self.is_world(self.pyboy.memory):
            self.last_compass_debug = None
            return [0.0, 0.0, 0.0]
        from pokemon import navigation as _navigation

        candidates = frozenset(
            g
            for g in GOAL_CANDIDATES - self._prev_satisfied_events - self._compass_excluded
            if self._goal_parents_satisfied(g)
        )
        result = _navigation.nearest_objective(self.get_position(), candidates)
        if result is None:
            self.last_compass_debug = {
                "goal": None,
                "dx": 0.0,
                "dy": 0.0,
                "dist": None,
                "candidates": len(candidates),
                "near_steps": None,
            }
            return [0.0, 0.0, 0.0]
        dx, dy, dist, goal = result
        if dist <= COMPASS_STALL_DIST:
            near = self._compass_near_counts.get(goal, 0) + 1
            self._compass_near_counts[goal] = near
            if near >= COMPASS_STALL_STEPS:
                self._compass_excluded.add(goal)
        self.last_compass_debug = {
            "goal": goal,
            "dx": float(max(-1, min(1, dx))),
            "dy": float(max(-1, min(1, dy))),
            "dist": dist,
            "candidates": len(candidates),
            "near_steps": self._compass_near_counts.get(goal, 0),
        }
        return [float(max(-1, min(1, dx))), float(max(-1, min(1, dy))), 1.0]

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
            "dialog_id_visit_counts": torch.tensor(
                self.dialog_id_visit_grid(), dtype=torch.float32
            ),
            "map_id_visit_counts": torch.tensor(
                self.map_id_visit_grid(), dtype=torch.float32
            ),
            "reward_component_sums": torch.tensor(
                self.reward_component_vector(), dtype=torch.float32
            ),
            "map_budget_progress": torch.tensor(
                self.map_budget_progress(), dtype=torch.float32
            ),
            "stuck_tile_progress": torch.tensor(
                self.stuck_tile_progress(), dtype=torch.float32
            ),
            "loop_streak_progress": torch.tensor(
                self.loop_streak_progress(), dtype=torch.float32
            ),
            "compass_progress": torch.tensor(
                self.compass_progress(), dtype=torch.float32
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

        selected_item = self.real_current_menu_selected_item(memory)
        data = [
            self._safe_index(self.poke_mart_items(memory), selected_item),
            self._safe_index(self.items_ids(memory), selected_item),
            self._safe_index(self.stored_items_ids(memory), selected_item),
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
        return [memory[i] for i in range(RAM.wTileMap, RAM.wTileMapBackup)]

    def _detect_blackout(self, memory: bytes) -> bool:
        """True on the exact step a full-party wipe exits the battle screen.

        wIsInBattle (RAM.wIsInBattle) is documented (pokered ram/wram.asm) as -1
        (0xFF) specifically for a lost/blacked-out battle, unlike a normal
        faint-with-a-mon-left-to-switch or a win/flee. Computed once here so
        reward_core and reward_battle_exit agree on the exact same frame.
        """
        left_battle = self.is_battle(memory) and not self.is_battle(self.pyboy.memory)
        return left_battle and self.type_of_battle(memory) == 0xFF

    def _accum_reward(self, name: str, value: float) -> float:
        """Add ``value`` to this episode's running total for ``name`` (see
        REWARD_COMPONENT_NAMES / reward_component_sums) and pass it through
        unchanged, so call sites can wrap an addend in place."""
        self.reward_component_sums[name] = (
            self.reward_component_sums.get(name, 0.0) + value
        )
        return value

    def reward_component_vector(self) -> list[float]:
        """Cumulative per-episode sum of each REWARD_COMPONENT_NAMES entry —
        the model-facing "how much of each reward type have I earned so far
        this episode" observation (see reward() / _accum_reward)."""
        return [
            self.reward_component_sums.get(name, 0.0) for name in REWARD_COMPONENT_NAMES
        ]

    def _current_reward_mode(self) -> str:
        """Which REWARD_MODE_NAMES bucket the current (post-tick) state falls
        into. Mutually exclusive by construction (is_dialog/is_menu/is_world
        each already require not is_battle, and is_cutscene_locked requires
        not is_battle/not is_blocked), so check order doesn't matter."""
        mem = self.pyboy.memory
        if self.is_battle(mem):
            return "battle"
        if self.is_dialog(mem):
            return "dialog"
        if self.is_menu(mem):
            return "menu"
        if self.is_world(mem):
            return "world"
        return "cutscene"

    def reward_mode_vector(self) -> list[float]:
        """Cumulative per-episode sum of whole-step reward bucketed by
        REWARD_MODE_NAMES — monitoring only (TensorBoard), not fed to the
        model. See _current_reward_mode / reward()."""
        return [self.reward_mode_sums.get(name, 0.0) for name in REWARD_MODE_NAMES]

    def reward(self, memory: bytes, action: int) -> tuple[float, float]:
        milestone = 0.0
        step = 0.0
        # Per-step signal; env accumulates into _episode_loop for episode stats.
        self.loop_flag = False
        self.loop_causes = set()
        self.last_milestone_payouts = []
        self._just_blacked_out = self._detect_blackout(memory)
        if self._just_blacked_out:
            self._pending_blackout_recovery = True

        milestone += self._accum_reward("core", self.reward_core(memory))
        # One-shot payout for any newly-satisfied GOAL_CANDIDATES event/badge
        # + regression penalty for any that un-satisfy (see
        # reward_generic_progress). Replaces the old named-goal
        # reward_story_milestones/reward_goal_regression pair — no clawback
        # exemption needed for HandlePlayerBlackOut's forced warp anymore,
        # since every surviving candidate is a permanent event bit (the auto
        # blacklist already excludes map-presence-style resettable checks),
        # not a "currently on this map" location goal that a warp could undo.
        milestone += self._accum_reward(
            "generic_progress", self.reward_generic_progress(memory)
        )
        # Battle→overworld (or dialog) transition — win/lose/flee uses prev memory.
        milestone += self._accum_reward("battle_exit", self.reward_battle_exit(memory))

        if self.is_battle(self.pyboy.memory):
            milestone += self._accum_reward("battle", self.reward_battle(memory))

        if self.is_cutscene_locked(self.pyboy.memory):
            # No step reward/penalty — actions cannot affect the game. Story
            # milestones above still apply when flags/maps change mid-cutscene.
            if self._is_fresh_dialog_exit(memory):
                milestone += self._accum_reward("dialog_exit", self.dialog_exit_reward)
        elif self.is_world(self.pyboy.memory):
            step += self._accum_reward("position", self.reward_position())
            # Completing a dialog is progress; without this, reading text is pure cost.
            if self._is_fresh_dialog_exit(memory):
                milestone += self._accum_reward("dialog_exit", self.dialog_exit_reward)
        elif self.is_dialog(self.pyboy.memory):
            m, s = self.reward_dialog(memory, action=action)
            milestone += self._accum_reward("dialog_milestone", m)
            step += self._accum_reward("dialog_step", s)
        elif self.is_menu(self.pyboy.memory):
            # Same waste shape and tempo as reward_position/reward_dialog/
            # reward_battle_useless_count (base_reward scaled up to 10x at
            # saturation over max_useless_ticks) -- this branch was missed
            # when those three got upgraded off the old flat linear ramp
            # (topped out at just -0.001, 10x weaker), leaving menu idling
            # the only blocked-state waste cost still too weak to teach the
            # agent to back out before stuck_menu's hard truncate fires.
            menu_waste_factor = self.in_menu_ticks / self.max_useless_ticks
            step += self._accum_reward(
                "menu_useless", self.base_reward * (1.0 + menu_waste_factor * 9.0)
            )
            if self._is_fresh_dialog_exit(memory):
                milestone += self._accum_reward("dialog_exit", self.dialog_exit_reward)
        elif self.is_battle(self.pyboy.memory):
            m, s = self.reward_battle_useless_count(memory)
            milestone += self._accum_reward("battle_useless_milestone", m)
            step += self._accum_reward("battle_useless_step", s)

        if not self.is_cutscene_locked(self.pyboy.memory):
            step += self._accum_reward(
                "anti_loop", self.reward_anti_loop(action=action, memory=memory)
            )
        # Always track dialog enter/exit — cutscenes often follow textboxes.
        step += self._accum_reward("dialog_reopen", self.reward_dialog_reopen(memory))
        # active_map_presence removed (was: nudge toward maps with unfinished
        # reachable progress, world-mode only, see reward_active_map_presence
        # -- now deleted). "active_map_presence" stays in
        # REWARD_COMPONENT_NAMES / reward_component_sums as a permanently-0
        # slot rather than being removed from the tuple -- shrinking it would
        # change VECTOR_DIM and break the observation shape of any
        # checkpoint trained before this change.
        # Same decay curve/magnitude, nudging toward fresh dialog (dialog-mode
        # only) and fresh maps (world-mode only) instead of active events.
        step += self._accum_reward(
            "new_dialog_presence", self.reward_new_dialog_presence(self.pyboy.memory)
        )
        step += self._accum_reward(
            "new_map_presence", self.reward_new_map_presence(self.pyboy.memory)
        )

        total = self._accum_reward("total", milestone + step)
        mode = self._current_reward_mode()
        self.reward_mode_sums[mode] = self.reward_mode_sums.get(mode, 0.0) + total
        return milestone, step

    def _is_fresh_dialog_exit(self, memory: bytes) -> bool:
        """True when exiting a dialog that has never been completed before.

        Guards dialog_exit_reward against the reopen/exit farm: reward_dialog_reopen
        only adds to _completed_dialogs on exit, so a dialog already in that set here
        means this is a repeat exit of a previously-finished conversation, not progress.
        """
        return (
            self.is_dialog(memory)
            and self.get_dialog(memory) not in self._completed_dialogs
        )

    def reward_dialog_reopen(self, memory: bytes) -> float:
        """After exiting a dialog, reopening the same (dialog_id, map_id) is a loop.

        Every reopen pays dialog_reopen_penalty — no truncation. Staying inside
        one conversation does not count — only exit then re-enter. Reopens that
        happen while the engine owns input (forced-walk cutscenes, e.g. Oak's
        Route 1 intercept closing/reopening the same textbox) are not the
        player's doing, so they are exempt from the farming check entirely.
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
            self._dialog_reopen_counts[key] = self._dialog_reopen_counts.get(key, 0) + 1
            self.loop_flag = True
            return self.dialog_reopen_penalty

        return 0.0

    def mark_goal_cleared(self, goal: str) -> None:
        """Vestigial no-op, kept for API compat with pokemon_red_env.py /
        debug_play.py call sites. The old curriculum-advance-exempts-
        regression bookkeeping this backed (_cleared_goals) was removed with
        reward_story_milestones/reward_goal_regression — see
        reward_generic_progress, which has no such exemption because its
        regression check only fires for GOAL_CANDIDATES (auto-blacklist
        already excludes resettable/cyclic events, so a real regression
        should be rare regardless of curriculum movement).
        """

    def seed_cleared_goals(self, active_goal: str) -> None:
        """Vestigial no-op, kept for API compat — see mark_goal_cleared."""

    def live_story_goals(self) -> list[str]:
        """GOAL_CANDIDATES currently true in game state (can shrink)."""
        return sorted(g for g in GOAL_CANDIDATES if self.is_goal_satisfied(g))

    def reward_generic_progress(
        self, memory: PyBoyMemoryView | bytes | None = None
    ) -> float:
        """One-shot bonus for every GOAL_CANDIDATES event/badge newly
        satisfied this step, plus a penalty for any that regress (see
        event_regression_penalty) — the generic replacement for the old
        hand-picked reward_story_milestones/reward_goal_regression pair.

        No prereq chains, no active-goal scaling, no clawback-then-repay
        bookkeeping: every candidate pays the same flat amount once per
        episode (badges pay badge_reward, everything else pays
        event_reward), tracked by self._milestones_hit (paid-this-episode
        set) so a flicker (regress then re-satisfy) cannot double-pay.
        """
        if memory is None:
            memory = self.pyboy.memory
        current = frozenset(n for n in GOAL_CANDIDATES if self.is_goal_satisfied(n))
        reward = 0.0

        newly_hit = 0
        for name in current - self._prev_satisfied_events:
            if name in self._milestones_hit:
                continue
            payout = self.badge_reward if name in BADGE_GOALS else self.event_reward
            self._milestones_hit.add(name)
            self.last_milestone_payouts.append((name, payout))
            reward += payout
            newly_hit += 1

        regressed = self._prev_satisfied_events - current
        self.last_regressed = sorted(regressed)
        if regressed:
            self.loop_flag = True
            reward += self.event_regression_penalty * len(regressed)

        self._prev_satisfied_events = current

        # --heatmap milestone-density overlay: same _last_heatmap_pos
        # attribution as reward_sums/battle_outcome_counts (a milestone can
        # fire mid-dialog, off is_world(), so it's anchored on the last
        # known world tile rather than nowhere).
        if self.collect_heatmap and newly_hit and self._last_heatmap_pos is not None:
            self.milestone_hit_counts[self._last_heatmap_pos] = (
                self.milestone_hit_counts.get(self._last_heatmap_pos, 0) + newly_hit
            )

        return reward

    def reward_anti_loop(self, action: int, memory: bytes) -> float:
        """Three-layer anti-loop + menu-spam penalties (PokeRL-style)."""
        penalty = 0.0
        causes: set[str] = set()
        action = int(action)
        in_dialog = self.is_dialog(self.pyboy.memory)
        in_battle = self.is_battle(self.pyboy.memory)
        interacting = action in INTERACT_ACTIONS

        # 1) Graduated position visit penalties.
        # Skip while pressing A/B in world — that is the talk-to-NPC attempt.
        if self.is_world(self.pyboy.memory) and not interacting:
            visits = self.position_visit_counts.get(self.get_position(), 0)
            if visits > VISIT_PENALTY_HARD_THRESHOLD:
                penalty += self.visit_penalty_hard
                causes.add("visit_penalty")
            elif visits > VISIT_PENALTY_SOFT_THRESHOLD:
                penalty += self.visit_penalty_soft
                causes.add("visit_penalty")

        # 2) Action pattern detection (sliding window).
        # Noop / movement loops always count. Prolonged A/B on the same tile
        # without an open dialog/battle is camping (Oak lab idle), not
        # "talking" — battle text (attack/effect/faint/EXP messages) forces
        # the same repeated A presses as dialog, so it gets the same pass.
        # In the overworld, a repeated button *pattern* only means "stuck" if
        # it didn't actually go anywhere: a Down,Left,Down,Left staircase (or
        # walking Right for 8 straight tiles down a corridor) repeats the same
        # button(s) while still making net progress, unlike a real
        # Left,Right,Left,Right ping-pong or bumping into a wall. Verified via
        # debug-play: efficient human navigation tripped this rule purely from
        # ordinary zigzag/straight-line movement. Position is meaningless
        # outside the world (battle/menu cursor navigation), so fall back to
        # the pure button-pattern check there.
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
                net_stuck = True
                if self.is_world(self.pyboy.memory) and len(self.recent_positions) >= 2:
                    net_stuck = self.recent_positions[-2] == self.get_position()
                if net_stuck:
                    penalty += self.action_pattern_penalty
                    causes.add("action_pattern")
        if len(actions) >= 8 and len(set(actions[-8:])) == 1:
            # Same talk-to-NPC exemption as check 1: is_world() implies
            # neither in_dialog nor in_battle, so "not (in_dialog or
            # in_battle)" was always True there regardless of `interacting` —
            # repeated A/B while still in world mode (waiting for a textbox
            # to open, or for a trainer's sight-triggered forced walk to
            # kick in) got flagged as a stuck loop exactly like holding a
            # movement button into a wall, even though the agent has no way
            # to speed either of those up. Once a real dialog/battle opens,
            # is_world() goes False and this check applies normally again
            # (see the dialog_wrong / in-battle turn-spam handling below).
            world_interact_wait = interacting and self.is_world(self.pyboy.memory)
            if not world_interact_wait and (
                not interacting or not (in_dialog or in_battle)
            ):
                net_stuck = True
                if self.is_world(self.pyboy.memory) and len(self.recent_positions) >= 8:
                    net_stuck = self.recent_positions[-8] == self.get_position()
                elif (self.is_menu(self.pyboy.memory) or in_battle) and len(
                    self.recent_menu_states
                ) >= 8:
                    # Same fix as the world branch: holding one direction to
                    # scroll a long menu list (or a battle FIGHT/PKMN/ITEM/RUN
                    # / move-select menu) repeats the button 8x while the
                    # cursor keeps moving — only flag it if the menu state
                    # actually stopped changing too. in_battle included here
                    # since is_menu() is defined as "not in battle" but battle
                    # menus share the exact same cursor registers.
                    cur_menu_state = (
                        self.menu_position_x(self.pyboy.memory),
                        self.menu_position_y(self.pyboy.memory),
                        self.real_current_menu_selected_item(self.pyboy.memory),
                    )
                    net_stuck = list(self.recent_menu_states)[-8] == cur_menu_state
                if net_stuck:
                    penalty += self.action_pattern_penalty
                    causes.add("action_pattern")

        # Idle / wrong buttons in dialog — only A/B advances story text. NONE
        # used to be discounted to menu_spam_penalty (-0.05 vs -0.08), which
        # made "do nothing" the cheapest mistake in a textbox — verified via
        # deterministic-eval action-probs sitting near 50/50 NONE-vs-A on a
        # multi-page Oak's-lab dialog, with argmax landing on NONE and idling
        # until the stuck-dialog fuse truncated the episode. NONE gets the
        # same penalty as any other non-interact button now, closing that gap.
        # Arrows are exempted while dialog_has_live_choice (a Yes/No inside
        # dialog_id != 0, e.g. OaksLab's starter picker) -- legal_action_mask
        # unmasks them there for exactly the same reason, and punishing a
        # legal, sometimes-mandatory action (there is no other way to answer
        # NO) would fight the mask instead of agreeing with it.
        dialog_wrong = (
            in_dialog
            and action not in INTERACT_ACTIONS
            and not (
                self.dialog_has_live_choice(self.pyboy.memory)
                and action in (ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_DOWN)
            )
        )
        if dialog_wrong:
            penalty += self.dialog_wrong_button_penalty
            self.dialog_wrong_streak += 1
            if self.dialog_wrong_streak >= self.dialog_wrong_streak_threshold:
                causes.add("dialog_wrong_button")
        else:
            self.dialog_wrong_streak = 0

        # 3) Spatial loop: same tile revisited often in recent history.
        # World + non-interact only — standing to talk is not a movement loop.
        if (
            self.is_world(self.pyboy.memory)
            and not interacting
            and len(self.recent_positions) >= 8
        ):
            cur = self.get_position()
            # >= 3 matches within the last 16 world-steps used to catch
            # perfectly ordinary chokepoint traffic (doorways, staircases) —
            # >= 6 means the same tile came up in more than a third of the
            # recent window, which is genuine pacing rather than a pass-through.
            if sum(1 for p in self.recent_positions if p == cur) >= 6:
                penalty += self.spatial_loop_penalty
                causes.add("spatial_loop")

        # Menu spam: no menu state change, sustained for menu_spam_streak_
        # threshold consecutive presses — a single boundary-bump (e.g. UP at
        # the top of a list) is not spam.
        if self.is_menu(self.pyboy.memory) and self.is_menu_illegal_move(memory):
            self.menu_noop_streak += 1
            if self.menu_noop_streak >= self.menu_spam_streak_threshold:
                penalty += self.menu_spam_penalty
                causes.add("menu_spam")
        else:
            self.menu_noop_streak = 0

        # Menu loop: cursor oscillating between a small set of states (e.g.
        # ITEM <-> CANCEL, or FIGHT <-> PKMN in a battle menu) changes state
        # every step, so it slips past the no-change check above. Catch
        # revisits of the same menu state instead (mirrors the
        # spatial_loop_penalty check for the overworld). is_battle_menu(),
        # not in_battle: a plain battle message has no real menu/cursor to
        # loop on (see the matching fix in count()'s recent_menu_states
        # append) -- gating on in_battle here fired this on every ordinary
        # multi-message battle turn instead of an actual stuck cursor.
        if (
            self.is_menu(self.pyboy.memory) or self.is_battle_menu(self.pyboy.memory)
        ) and len(self.recent_menu_states) >= 6:
            cur_menu_state = (
                self.menu_position_x(self.pyboy.memory),
                self.menu_position_y(self.pyboy.memory),
                self.real_current_menu_selected_item(self.pyboy.memory),
            )
            if sum(1 for s in self.recent_menu_states if s == cur_menu_state) >= 3:
                penalty += self.menu_loop_penalty
                causes.add("menu_loop")

        # 4) Off-goal map camping used to compare self.map_id against a
        # hand-curated GOAL_ALLOWED_MAPS[goal] set, then later against the
        # generic active_map_events()-driven reward_active_map_presence
        # (since removed — see reward()). No replacement currently; nothing
        # penalizes off-goal map camping here anymore.

        self.loop_causes = causes
        if causes:
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
        turn_changed = self.number_of_turns_in_current_battle(
            memory
        ) != self.number_of_turns_in_current_battle(self.pyboy.memory)
        if entered:
            # No reward for entering a battle, from world or dialog (trainer
            # intros) alike — it was luring the agent into farming grass or
            # trainer sprites for the entry bonus and fleeing immediately after.
            return 0.0, 0.0
        if turn_changed:
            reward = self.battle_turn_reward
            # Same per-tile decay as reward_enemy_hp / battle_won_reward (see
            # _wild_encounter_decay) — without it, mashing through turns on a
            # repeat wild encounter paid full price forever regardless of
            # position_visit_counts, even after enemy-HP/win rewards had
            # already decayed to zero. Trainer battles are exempt (one-shot
            # per sprite, type_of_battle read from `memory` like the other
            # decay call sites).
            if self.type_of_battle(memory) == 1:
                reward *= self._wild_encounter_decay()
            return reward, 0.0
        # Same waste shape and tempo as reward_position/reward_dialog (base_reward
        # scaled up to 10x at saturation over max_useless_ticks) — idling on a
        # stuck turn now costs exactly what idling on a tile costs, instead of
        # the old flat linear ramp that topped out at just -0.001 (10x weaker).
        # Ramped against max_useless_ticks (512), NOT max_useless_battle_ticks
        # (2048) — the latter stays the longer stuck fuse for truncated().
        waste_factor = (
            min(self.in_battle_ticks, self.max_useless_ticks) / self.max_useless_ticks
        )
        return 0.0, self.base_reward * (1.0 + waste_factor * 9.0)

    def reward_dialog(self, memory: bytes, action: int) -> tuple[float, float]:
        dialog_changed = self.dialog_id(memory) != self.dialog_id(self.pyboy.memory)
        # Pay for dialog_id flips (dialog_advance_reward). Do NOT pay for
        # tilemap text frames — that was the post-rival exploit. The old
        # one-time "first time ever seeing this (dialog_id, map)" bonus is
        # gone — reward_new_dialog_presence (see reward()) now covers fresh
        # conversations with a decaying-by-visits shape instead of a flat
        # one-shot payout.
        dialog_reward = self.dialog_advance_reward if dialog_changed else 0.0
        # Same per-tick waste shape AND same ramp tempo as reward_position's
        # step_penalty (base_reward scaled up to 10x at saturation over
        # max_useless_ticks) — idling in dialog now costs exactly what idling on
        # a tile costs at every tick, not just at the (old, 4x-slower) cap.
        # Deliberately ramped against max_useless_ticks (512), NOT
        # max_useless_dialog_ticks (2048) — the latter stays the longer stuck
        # fuse for truncated() so a legit long script still isn't cut short,
        # it just no longer controls the reward ramp's speed.
        waste_factor = (
            min(self.in_dialog_ticks, self.max_useless_ticks) / self.max_useless_ticks
        )
        waste = self.base_reward * (1.0 + waste_factor * 9.0)
        return dialog_reward, waste

    def reward_position(self):
        pos = self.get_position()
        ticks_here = self.visited_positions.get(pos, 0)
        visit_count = self.position_visit_counts.get(pos, 0)
        waste_factor = min(ticks_here, self.max_useless_ticks) / self.max_useless_ticks
        step_penalty = self.base_reward * (1.0 + waste_factor * 9.0)
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

    def reward_player_pokemons_current_hps(self, memory: bytes):
        """Fractional HP change this step, summed over the party.

        Damage *taken* while in battle is scaled by ``_battle_difficulty_scale``
        (enemy_lv vs active_lv), the same discount ``reward_enemy_hp``/
        ``battle_won_reward`` already apply to damage *dealt* -- getting hit
        by a same-level opponent pays full cost, but eating a big hit from
        something far weaker (a bad roll on an otherwise easy, winnable
        fight) no longer nearly cancels out the eventual win reward. Wild
        battles further decay this cost by ``_wild_encounter_decay`` (same
        per-tile repeat-visit decay as every other wild-battle reward
        component, see its docstring) -- once a tile's win/HP-dealt rewards
        have decayed to ~0 from repetition, the HP-taken cost decays with
        them instead of staying full price forever, so a heavily-farmed
        tile's battles trend toward ~0 net, not net-negative. Trainer
        battles are exempt (one-shot per sprite, same as everywhere else
        this decay applies). Healing (positive delta) and any HP change
        outside battle (poison ticks, centre heals -- already separately
        gated by _just_blacked_out in reward_core) are left at full,
        undiscounted magnitude.
        """
        reward = 0.0
        in_battle = self.is_battle(self.pyboy.memory)
        scale = 1.0
        wild = False
        if in_battle:
            scale = self._battle_difficulty_scale(
                self.enemy_level(self.pyboy.memory),
                self.pokemon_level(self.pyboy.memory),
            )
            wild = self.type_of_battle(self.pyboy.memory) == 1

        for id_x, id_y, hp_x, hp_y, max_hp in zip(
            self.player_pokemons_ids(memory),
            self.player_pokemons_ids(self.pyboy.memory),
            self.player_pokemons_current_hps(memory),
            self.player_pokemons_current_hps(self.pyboy.memory),
            self.player_pokemons_max_hps(self.pyboy.memory),
        ):
            if id_x == id_y and id_x != 0:
                delta = (hp_y - hp_x) / max_hp
                if in_battle and delta < 0:
                    delta *= scale
                    if wild:
                        delta *= self._wild_encounter_decay()
                reward += delta

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

        ``goal`` is either one of BADGE_GOALS ("badge1".."badge8", read off
        wObtainedBadges — see badges()) or a raw EVENT_* name from
        event_constants.py, read generically by is_event_satisfied(). Used by
        curriculum to skip goals already true in the live save (see
        curriculum_config.pick_new_goal's is_satisfied callback).
        """
        if goal in BADGE_GOALS:
            badges = self.badges(self.pyboy.memory)
            idx = BADGE_GOALS.index(goal)
            return bool(idx < len(badges) and badges[idx])
        return self.is_event_satisfied(goal, self.pyboy.memory)

    def is_event_satisfied(
        self, event_name: str, memory: PyBoyMemoryView | bytes | None = None
    ) -> bool:
        """Generic wEventFlags bit read for any name known to event_constants.py.

        This is the fallback every unrecognized ``goal`` string falls through
        to in is_goal_satisfied(), and the single source of truth for the new
        generic goal pool (GOAL_CANDIDATES / EVENT_GOAL_CANDIDATES) — no
        per-event hand-written method, so no per-event chance of a wrong-bit
        bug like the fought_X_yet cluster this replaces.
        """
        if event_name not in _event_constants.EVENTS:
            return False
        if memory is None:
            memory = self.pyboy.memory
        addr = _event_constants.event_address(event_name)
        bit = _event_constants.event_bit(event_name)
        return bool(memory[addr] & (1 << bit))

    def map_truncate_budget(self, map_id: int) -> int:
        """Hard step cutoff that ends the episode (see truncated()) -- the
        map's own width*height in blocks, floor of 64 so a degenerate/
        UNUSED_MAP entry with width=height=0 in map_constants.py still gets a
        usable budget.

        No longer scaled by new_position_decay_visits (was max(64, area) * 4,
        i.e. "enough to visit every tile 4 times"). That multiplier was sized
        for _tick_map_budget's old meaning -- a raw per-episode dwell total --
        and stopped making sense once _tick_map_budget switched to counting
        stall since the map's last brand-new tile (reset to 0 on every fresh
        one, see its own docstring): with a resettable counter, x4 let a
        policy that pings a single new tile every few hundred stalled steps
        wander indefinitely without ever tripping this fuse (confirmed live --
        truncate_cause_map_budget sat at 0.0000 for an entire run under the
        resettable counter with this multiplier still in place). Area alone
        is generous enough for a genuine detour around an obstacle without
        multiplying out a stall streak's tolerance to unreachable lengths. No
        per-step penalty ramps up before this -- truncated() simply ends the
        episode once it's passed.
        """
        area = _map_constants.map_area_blocks(map_id)
        return max(64, area)

    def _tick_map_budget(self, memory: PyBoyMemoryView | bytes) -> None:
        """Advance the current map's stall counter. Called once per step from
        count() — only fills while actually walking the world; dialog/
        battle/menu time on a map is free (see map_truncate_exceeded, which
        is checked every mode regardless).

        Counts world-steps *since this map last gained a brand-new tile*
        (position_visit_counts == 1, checked after count()'s own increment
        earlier this step — see its call site), not a raw per-episode total.
        A raw total meant leaving an overstayed map and coming back later
        resumed adding to the same running sum, so a handful of honest,
        exploratory forays that each retreated partway (e.g. into a wild
        battle, or just turning back) could burn through the whole budget
        between them, leaving no room for the one attempt that would have
        actually crossed the map. Resetting on every genuinely new tile makes
        the fuse measure actual stalling instead of cumulative dwell time —
        a policy that keeps making net-new progress, however many times it
        steps away and back, never trips it.
        """
        if not self.is_world(memory):
            return
        mid = self.map_id(memory)
        if self.position_visit_counts.get(self.get_position(memory), 0) <= 1:
            self.world_map_step_counts[mid] = 0
        else:
            self.world_map_step_counts[mid] = self.world_map_step_counts.get(mid, 0) + 1

    def map_truncate_exceeded(
        self, memory: PyBoyMemoryView | bytes | None = None
    ) -> bool:
        """True once the current map's step count has passed
        map_truncate_budget -- the hard cutoff checked by truncated().
        Checked every step regardless of mode: a policy stuck cycling through
        a menu or a battle on an already-overstayed map should not be able to
        dodge it just because _tick_map_budget itself only fills during
        is_world().
        """
        if memory is None:
            memory = self.pyboy.memory
        mid = self.map_id(memory)
        return self.world_map_step_counts.get(mid, 0) > self.map_truncate_budget(mid)

    def active_map_events(
        self, map_id: int | None = None, memory: PyBoyMemoryView | bytes | None = None
    ) -> list[str]:
        """GOAL_CANDIDATES events whose home map is ``map_id`` (default: the
        current map), not yet satisfied, but whose parents (per
        event_graph.py's static-analysis edges) are all already satisfied —
        i.e. plausibly "next in line" to happen right here, right now.

        Events with no inferred parent (event_graph ROOT_EVENTS) are active
        as soon as their map is entered, since there's nothing to wait on.
        """
        if memory is None:
            memory = self.pyboy.memory
        if map_id is None:
            map_id = self.map_id(memory)

        active: list[str] = []
        for name in EVENT_GOAL_CANDIDATES:
            info = _event_graph.EVENT_GRAPH.get(name)
            if info is None or info["map_id"] != map_id:
                continue
            if self.is_event_satisfied(name, memory):
                continue
            if all(
                self.is_event_satisfied(p, memory)
                for p in info["parents"]
                if p in _event_constants.EVENTS
            ):
                active.append(name)
        return active

    def has_active_map_event(
        self, memory: PyBoyMemoryView | bytes | None = None
    ) -> bool:
        return bool(self.active_map_events(memory=memory))

    def reward_new_dialog_presence(
        self, memory: PyBoyMemoryView | bytes | None = None
    ) -> float:
        """Decaying presence bonus keyed by (map_id, dialog_id): a fresh
        conversation pays active_map_event_reward, decaying to 0 by
        new_position_decay_visits dialog-steps in. Reuses both constants
        rather than its own so this and reward_new_map_presence can't drift
        apart in length or value.
        """
        if memory is None:
            memory = self.pyboy.memory
        if not self.is_dialog(memory):
            return 0.0
        did_key = (self.map_id(memory), self.dialog_id(memory))
        visit_count = self.dialog_id_visit_counts.get(did_key, 0)
        decay = max(0.0, 1.0 - visit_count / self.new_position_decay_visits)
        return self.active_map_event_reward * decay

    def reward_new_map_presence(
        self, memory: PyBoyMemoryView | bytes | None = None
    ) -> float:
        """Decaying presence bonus keyed by map_id_visit_counts: first
        setting foot on a map not visited (much) yet this episode pays
        active_map_event_reward, decaying to 0 by new_position_decay_visits
        steps in.
        """
        if memory is None:
            memory = self.pyboy.memory
        if not self.is_world(memory):
            return 0.0
        mid = self.map_id(memory)
        visit_count = self.map_id_visit_counts.get(mid, 0)
        decay = max(0.0, 1.0 - visit_count / self.new_position_decay_visits)
        return self.active_map_event_reward * decay

    def goal_reached(self) -> bool:
        return self.is_goal_satisfied(self.goal)

    def terminated(self, memory: bytes):
        """Episode/curriculum-leg end signal.

        True when the specifically-assigned self.goal is satisfied, OR when
        ANY other GOAL_CANDIDATES event/badge was newly satisfied this exact
        step (see reward_generic_progress -- last_milestone_payouts, already
        computed by the time this runs since reward() calls it first). In-
        game event order does not reliably match STAGE_ORDER's heuristic
        sort (see curriculum_config.py's module docstring) -- whichever
        goal actually lands first should end this leg and hand training a
        fresh, still-unsatisfied goal, not just the one curriculum happened
        to assign. self.goal is itself always a GOAL_CANDIDATES member, so
        goal_reached() is normally already covered by last_milestone_payouts;
        it's kept as a fallback for a custom/off-list --goal.

        goal_reached() is level-triggered (true for as long as the event
        flag stays set), unlike last_milestone_payouts which is a one-step
        edge. Since _advance_after_goal no longer reassigns self.goal away
        from a satisfied one (see its docstring), gating on goal_reached()
        unconditionally would fire every single step for the rest of the
        episode once the assigned goal's flag is set -- re-entering the
        auto-advance branch in PokemonRedEnv.step() every step and wiping
        visit counts (set_curriculum(clear_visits=True)) each time instead
        of just once. Only fall back to it for a goal outside GOAL_CANDIDATES
        (the one case last_milestone_payouts truly cannot see).
        """
        if self.goal not in GOAL_CANDIDATES:
            return self.goal_reached()
        return bool(self.last_milestone_payouts)

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
        stuck_loop = self.loop_streak >= self.max_loop_streak
        stuck_battle = self.max_useless_battle_ticks <= self.in_battle_ticks
        stuck_menu = self.max_useless_ticks <= self.in_menu_ticks
        # Size-scaled per-map hard cutoff (see map_truncate_exceeded/
        # map_truncate_budget), checked in every mode. No per-step penalty
        # before this; the episode is simply truncated once it's passed.
        map_budget = self.map_truncate_exceeded(self.pyboy.memory)
        if map_budget:
            self.last_map_budget_trunc_at_start = (
                self.map_id(self.pyboy.memory) == self._start_map_id
            )

        causes: set[str] = set()
        if stuck_tile:
            causes.add("stuck_tile")
        if stuck_dialog:
            causes.add("stuck_dialog")
        if stuck_loop:
            causes.add("loop_streak")
        if stuck_battle:
            causes.add("stuck_battle")
        if stuck_menu:
            causes.add("stuck_menu")
        if map_budget:
            causes.add("map_budget")
        self.last_truncate_causes = frozenset(causes)

        return bool(causes)

    def current_milestone(self) -> str:
        """Most-advanced GOAL_CANDIDATES event/badge paid out this episode
        (see reward_generic_progress / self._milestones_hit), using each
        event's wEventFlags bit index as a rough story-order proxy (events
        are declared in event_constants.asm roughly city-by-city in
        progression order — see tools/gen_event_constants.py). Badges have
        no bit index (wObtainedBadges, not wEventFlags) so they're anchored
        to a fixed pseudo-index band between event sections instead.
        """
        if not self._milestones_hit:
            return "start"

        def order_key(name: str) -> int:
            if name in BADGE_GOALS:
                return -1_000_000 + (BADGE_GOALS.index(name) + 1) * 1000
            return _event_constants.EVENTS.get(name, -1)

        return max(self._milestones_hit, key=order_key)

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

        selected_item = self.real_current_menu_selected_item(memory)
        data += self.data_normalizer(
            [self._safe_index(self.items_quantities(memory), selected_item)]
        )
        data += self.data_normalizer([self.player_money(memory)], max=0xFFFFFF)
        data += self.data_normalizer(
            [self._safe_index(self.stored_items_quantities(memory), selected_item)]
        )
        data += self.data_normalizer([self.game_coins(memory)], max=0xFFFF)

        return data if self.is_eq_menu(memory) else [0] * len(data)

    def id_of_the_last_menu_item(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wMaxMenuItem]

    def map_id(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wCurMap]

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
        return True if memory[RAM.wFontLoaded] else False

    def is_script_locked(self, memory: PyBoyMemoryView | bytes) -> bool:
        """True when the engine owns player input (cutscene / forced walk).

        Independent of on-screen activity — following Oak still looks like the
        overworld. See pret/pokered JoypadOverworld + wStatusFlags5.
        """
        status5 = int(memory[RAM.wStatusFlags5])
        # bit7 BIT_SCRIPTED_MOVEMENT_STATE, bit5 BIT_DISABLE_JOYPAD
        if status5 & 0xA0:
            return True
        # wJoyIgnore — bitmask of ignored buttons (often D-pad during scripts)
        if memory[RAM.wJoyIgnore]:
            return True
        # wSimulatedJoypadStatesIndex — remaining forced button presses
        if memory[RAM.wSimulatedJoypadStatesIndex]:
            return True
        return False

    def is_cutscene_locked(self, memory: PyBoyMemoryView | bytes) -> bool:
        """Script lock with no textbox/battle — player cannot usefully act."""
        return (
            self.is_script_locked(memory)
            and not self.is_blocked(memory)
            and not self.is_battle(memory)
        )

    def cutscene_skip_frames(self, memory: PyBoyMemoryView | bytes) -> int:
        """Frames still guaranteed input-locked, read straight from pokered's
        own countdown registers instead of guessing a fixed hold length.

        wIgnoreInputCounter (D13A) ticks down to 0 once per frame whenever
        BIT_DISABLE_JOYPAD is set — the "ignore input for half a second"
        window the engine sets on every door/warp transition (see
        pret/pokered engine/play_time.asm CountDownIgnoreInputBitReset,
        home/overworld.asm IgnoreInputForHalfSecond).

        wSimulatedJoypadStatesIndex (CD38) ticks down once per frame
        whenever BIT_SCRIPTED_MOVEMENT_STATE is set — the remaining length
        of a forced-walk script (e.g. following Oak into the lab); see
        pret/pokered home/overworld.asm's simulated-button-press handling.

        Both are exact remaining-frame counts, not booleans, so a caller can
        tick straight through the whole locked window in one go instead of
        polling it away frame_skip frames at a time across many wasted
        env.step() calls. Returns 0 when neither applies (still locked for
        another reason, e.g. wJoyIgnore/link state) — caller should fall
        back to single-frame ticks in that case.
        """
        status5 = int(memory[RAM.wStatusFlags5])
        remaining = 0
        if status5 & 0x20:  # BIT_DISABLE_JOYPAD
            remaining = max(remaining, int(memory[RAM.wIgnoreInputCounter]))
        if status5 & 0x80:  # BIT_SCRIPTED_MOVEMENT_STATE
            remaining = max(remaining, int(memory[RAM.wSimulatedJoypadStatesIndex]))
        return remaining

    def dialog_id(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wSpriteIndex]

    def is_battle(self, memory: PyBoyMemoryView | bytes):
        return True if self.type_of_battle(memory) else False

    def type_of_battle(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wIsInBattle]

    # pret/pokered constants/menu_constants.asm MESSAGE_BOX ($01) -- the
    # wTextBoxID value PrintText (home/window.asm) writes before every
    # single message it prints, battle included.
    _MESSAGE_BOX_ID = 1

    def is_battle_message(self, memory: PyBoyMemoryView | bytes) -> bool:
        """True while a plain battle message ("X used TACKLE!", "It's super
        effective!", "Wild RATTATA fainted!", ...) is on screen with no
        real menu open -- the battle-mode analog of is_dialog().

        is_dialog()/is_menu() key off wFontLoaded (is_blocked()), but that
        register is never touched by battle's message path: PrintText's
        MESSAGE_BOX case draws its box via TextBoxBorder directly (see
        pret/pokered engine/menus/text_box.asm's .coordTableMatch branch),
        never through DisplayTextIDInit (the overworld-only routine that
        actually sets wFontLoaded) -- so is_blocked() can't distinguish
        anything during battle and is_dialog()/is_menu() correctly exclude
        it entirely rather than guess wrong.

        wTextBoxID can, and doesn't have wMaxMenuItem's staleness problem
        either: PrintText unconditionally writes MESSAGE_BOX to it before
        every message (home/window.asm), and every battle menu writes its
        own distinct template constant to that *same* register right
        before opening -- BATTLE_MENU_TEMPLATE for FIGHT/PKMN/ITEM/RUN,
        TWO_OPTION_MENU for the run-away Yes/No prompt,
        SWITCH_STATS_CANCEL_MENU_TEMPLATE for the party-switch menu (all in
        pret/pokered engine/battle/core.asm). So unlike wMaxMenuItem, which
        pokered never resets and which would misread a message straight
        after a closed menu as "still live," wTextBoxID is refreshed on
        every single textbox/menu draw -- message or not -- and can't go
        stale between them.

        One known gap: for the handful of frames after wIsInBattle first
        flips before the battle's first PrintText call actually runs,
        wTextBoxID still holds whatever it was left at pre-battle (e.g. the
        last overworld dialog), so this can misread briefly right at battle
        start. Self-corrects the instant the "wild X appeared" /
        trainer-intro message prints, which is essentially immediate.
        """
        return (
            self.is_battle(memory)
            and int(memory[RAM.wTextBoxID]) == self._MESSAGE_BOX_ID
        )

    def is_battle_menu(self, memory: PyBoyMemoryView | bytes) -> bool:
        """True while a real battle menu/list is reading input --
        FIGHT/PKMN/ITEM/RUN, move select, party switch, the run-away Yes/No
        prompt, ... -- the battle-mode analog of a live dialog choice (see
        dialog_has_live_choice). See is_battle_message for why wTextBoxID,
        not wFontLoaded/wMaxMenuItem, is the signal used here.
        """
        return self.is_battle(memory) and not self.is_battle_message(memory)

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

    def legal_action_mask(self, memory: PyBoyMemoryView | bytes) -> list[bool]:
        """Which of the N_ACTIONS buttons can actually change the game this tick.

        Hardens two invariants reward_anti_loop already proves out via
        reward shaping (dialog_wrong_button_penalty, menu_spam_penalty) into
        a hard constraint for action-masked PPO (see sb3-contrib MaskablePPO
        / PokemonRedEnv.action_masks):

        - Dialog only advances on A/B (INTERACT_ACTIONS) — pret/pokered's
          text engine ignores every other button while a textbox with
          dialog_id != 0 is open, so anything else (including NONE) is a
          wasted tick by construction, not just empirically. EXCEPT: a Yes/No
          (or other list) choice can appear *while dialog_id stays nonzero*
          — e.g. OaksLab.asm's starter picker calls YesNoChoice right after
          PrintText without any wSpriteIndex reset in between (dialog_id is
          set from the interacted object's sprite, not by PrintText, and
          nothing clears it for the Yes/No). is_dialog() alone can't tell
          "plain text page" from "a live cursor choice" apart, so it would
          otherwise mask out the arrows the player needs to ever answer NO.
          wMaxMenuItem is pokered's own live item-count for whichever
          cursor list is currently reading input (PlaceMenuCursor /
          HandleMenuInput, shared by every menu including YesNoChoice) —
          nonzero only while such a list is genuinely active, so it doubles
          as the "a choice beyond A/B is live" signal. It can still read
          stale-nonzero for one tick right as a plain textbox opens (leftover
          from whatever menu was open before), which would wrongly leave
          arrows legal for that one tick — harmless (an unused legal action,
          not a forbidden necessary one), so this deliberately errs toward
          over- rather than under-including arrows.
        - NONE has no legitimate use outside is_world. Standing still on a
          world tile is a real decision; "doing nothing" in a menu or battle
          never changes menu/turn state, so it's structurally the same as
          the illegal-move case is_menu_illegal_move already penalizes, just
          guaranteed on every tick instead of only after a streak.
        - A plain battle message ("X used TACKLE!", "It's super effective!")
          is the battle-mode analog of plain dialog text: only A/B do
          anything (see is_battle_message), everything else -- including
          NONE -- is wasted the same way it is mid-conversation.

        Cutscene-lock ticks are not special-cased here: Emulator._press_action
        already discards every action during is_cutscene_locked, so whatever
        this returns for that window is moot -- left all-True rather than
        adding a distinction with no observable effect.
        """
        mask = [True] * (ACTION_NONE + 1)
        if self.is_dialog(memory):
            mask = [False] * (ACTION_NONE + 1)
            mask[ACTION_A] = True
            mask[ACTION_B] = True
            if self.dialog_has_live_choice(memory):
                mask[ACTION_LEFT] = True
                mask[ACTION_RIGHT] = True
                mask[ACTION_UP] = True
                mask[ACTION_DOWN] = True
        elif self.is_battle(memory):
            if self.is_battle_message(memory):
                mask = [False] * (ACTION_NONE + 1)
                mask[ACTION_A] = True
                mask[ACTION_B] = True
            else:
                mask[ACTION_NONE] = False
        elif self.is_menu(memory):
            mask[ACTION_NONE] = False
        return mask

    def dialog_has_live_choice(self, memory: PyBoyMemoryView | bytes) -> bool:
        """True while a Yes/No (or other list) cursor is actually reading
        input during a dialog — see legal_action_mask's docstring for why
        is_dialog() alone can't tell this apart from a plain text page.

        Compares against _dialog_choice_baseline (wMaxMenuItem snapshotted
        the instant the current dialog_id started, see count()) instead of
        checking wMaxMenuItem != 0 directly: pret/pokered never resets
        wMaxMenuItem when a plain textbox opens, it just leaves whatever a
        previous, unrelated menu (e.g. a 6-item START menu) last wrote there
        -- a raw nonzero check reads that leftover as "a choice is live" for
        the rest of the conversation, permanently hiding the arrows from
        legal_action_mask. Only an actual *change* away from the baseline --
        which only a real HandleMenuInput-backed choice (YesNoChoice etc.)
        produces -- counts.
        """
        if self._dialog_choice_baseline is None:
            return False
        return int(memory[RAM.wMaxMenuItem]) != self._dialog_choice_baseline

    def position_x(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wXCoord]

    def position_y(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wYCoord]

    def is_menu(self, memory: PyBoyMemoryView | bytes):
        return (
            True
            if self.is_blocked(memory)
            and self.dialog_id(memory) == 0
            and not self.is_battle(memory)
            else False
        )

    def sprite_data_ids(self, memory: PyBoyMemoryView | bytes):
        data = [
            memory[RAM.wSpritePlayerStateData1PictureID + 0x10 * x] for x in range(16)
        ]

        return data if self.is_world(memory) else [0] * len(data)

    def sprite_data_movement_statuses(self, memory: PyBoyMemoryView | bytes):
        data = [
            memory[RAM.wSpritePlayerStateData1MovementStatus + 0x10 * x]
            for x in range(16)
        ]

        return data if self.is_world(memory) else [0] * len(data)

    def sprite_data_facing_directions(self, memory: PyBoyMemoryView | bytes):
        data = [
            memory[RAM.wSpritePlayerStateData1FacingDirection + 0x10 * x]
            for x in range(16)
        ]

        return data if self.is_world(memory) else [0] * len(data)

    def sprite_data_y_positions(self, memory: PyBoyMemoryView | bytes):
        data = [memory[RAM.wSpritePlayerStateData2MapY + 0x10 * x] for x in range(16)]

        return data if self.is_world(memory) else [0] * len(data)

    def sprite_data_x_positions(self, memory: PyBoyMemoryView | bytes):
        data = [memory[RAM.wSpritePlayerStateData2MapX + 0x10 * x] for x in range(16)]

        return data if self.is_world(memory) else [0] * len(data)

    def menu_position_x(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wTopMenuItemY]

    def menu_position_y(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wTopMenuItemX]

    def current_menu_selected_item(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wCurrentMenuItem]

    def real_current_menu_selected_item(self, memory: PyBoyMemoryView | bytes):
        return self.current_menu_selected_item(
            memory
        ) + self.id_of_the_first_displayed_menu_item(memory)

    def id_of_the_first_displayed_menu_item(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wListScrollOffset]

    @staticmethod
    def _safe_index(data: list, index: int):
        # real_current_menu_selected_item is a cursor shared across menus of
        # different lengths (mart/bag/PC) — it can exceed a given list's
        # bounds when read outside that specific menu, so wrap instead of
        # crashing; the value is discarded downstream unless relevant.
        return data[index % len(data)] if data else 0

    def index_of_current_pokemon_send_out(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wPlayerMonNumber]

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
        return [memory[RAM.wEnemyMonBaseStats + i] for i in range(5)]

    def battle_status_player(self, memory: PyBoyMemoryView | bytes):
        return (
            self.bits_extractor(memory[RAM.wPlayerBattleStatus1])
            + self.bits_extractor(memory[RAM.wPlayerBattleStatus2])
            + self.bits_extractor(memory[RAM.wPlayerBattleStatus3], 0, 3)
        )

    def is_gym_leader_battle_music_playing(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wGymLeaderNo] & 1

    def critical_hit_flag(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wCriticalHitOrOHKO] & 1

    def one_hit_ko_flag(self, memory: PyBoyMemoryView | bytes):
        return 1 if memory[RAM.wCriticalHitOrOHKO] & 2 else 0

    def hooked_pokemon_flag(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wMoveMissed] & 1

    def number_of_turns_in_current_battle(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wAILayer2Encouragement]

    def players_substitute_hp(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wPlayerSubstituteHP]

    def enemy_substitute_hp(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemySubstituteHP]

    def move_menu_type(self, memory: PyBoyMemoryView | bytes):
        return (
            memory[RAM.wMoveMenuType]
            if self.is_battle(memory) or self.is_dialog(memory) or self.is_menu(memory)
            else 0
        )

    def player_selected_move(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wPlayerSelectedMove] if self.is_battle(memory) else 0

    def enemy_selected_move(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemySelectedMove] if self.is_battle(memory) else 0

    def your_move_type(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wPlayerMoveType] if self.is_battle(memory) else 0

    def enemy_move_power(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMovePower]

    def enemy_move_type(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMoveType] if self.is_battle(memory) else 0

    def enemy_move_accuracy(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMoveAccuracy]

    def player_move_power(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wPlayerMovePower]

    def player_move_accuracy(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wPlayerMoveAccuracy]

    def enemy_hp(self, memory: PyBoyMemoryView | bytes):
        # Gen1 stores 16-bit stats big-endian (high byte first).
        return (memory[RAM.wEnemyMonHP] << 8) | memory[RAM.wEnemyMonHP + 1]

    def enemy_level(self, memory: PyBoyMemoryView | bytes):
        # wEnemyMon base is 0xCFE5 (battle_struct layout); 0xCFE8 is
        # BoxLevel (unused trade-display field, always 0 in battle) — the
        # real live Level field is offset 0x0E from base = RAM.wEnemyMonLevel. Verified
        # live: 0xCFE8 read 0 for an entire multi-turn fight while RAM.wEnemyMonLevel
        # held a stable, sane level matching the opponent's actual strength.
        return memory[RAM.wEnemyMonLevel]

    def enemy_status(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[RAM.wEnemyMonStatus], end_bit=6)

    def enemy_type1(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMonType1] if self.is_battle(memory) else 0

    def enemy_type2(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMonType2] if self.is_battle(memory) else 0

    def enemy_move1(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMonMoves] if self.is_battle(memory) else 0

    def enemy_move2(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMonMoves + 1] if self.is_battle(memory) else 0

    def enemy_move3(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMonMoves + 2] if self.is_battle(memory) else 0

    def enemy_move4(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMonMoves + 3] if self.is_battle(memory) else 0

    def enemy_max_hp(self, memory: PyBoyMemoryView | bytes):
        return (memory[RAM.wEnemyMonMaxHP] << 8) | memory[RAM.wEnemyMonMaxHP + 1]

    def enemy_attack(self, memory: PyBoyMemoryView | bytes):
        return (memory[RAM.wEnemyMonAttack] << 8) | memory[RAM.wEnemyMonAttack + 1]

    def enemy_defense(self, memory: PyBoyMemoryView | bytes):
        return (memory[RAM.wEnemyMonDefense] << 8) | memory[RAM.wEnemyMonDefense + 1]

    def enemy_speed(self, memory: PyBoyMemoryView | bytes):
        return (memory[RAM.wEnemyMonSpeed] << 8) | memory[RAM.wEnemyMonSpeed + 1]

    def enemy_special(self, memory: PyBoyMemoryView | bytes):
        return (memory[RAM.wEnemyMonSpecial] << 8) | memory[RAM.wEnemyMonSpecial + 1]

    def enemy_pp_first_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMonPP]

    def enemy_pp_second_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMonPP + 1]

    def enemy_pp_third_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMonPP + 2]

    def enemy_pp_fourth_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wEnemyMonPP + 3]

    def enemy_base_stats(self, memory: PyBoyMemoryView | bytes):
        return [memory[RAM.wEnemyMonBaseStats + i] for i in range(5)]

    def pokemon_current_hp(self, memory: PyBoyMemoryView | bytes):
        return (memory[RAM.wBattleMonHP] << 8) | memory[RAM.wBattleMonHP + 1]

    def pokemon_status(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[RAM.wBattleMonStatus], end_bit=6)

    def pokemon_type1(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wBattleMonType1] if self.is_battle(memory) else 0

    def pokemon_type2(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wBattleMonType2] if self.is_battle(memory) else 0

    def pokemon_move_first_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wBattleMonMoves] if self.is_battle(memory) else 0

    def pokemon_move_second_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wBattleMonMoves + 1] if self.is_battle(memory) else 0

    def pokemon_move_third_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wBattleMonMoves + 2] if self.is_battle(memory) else 0

    def pokemon_move_fourth_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wBattleMonMoves + 3] if self.is_battle(memory) else 0

    def pokemon_level(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wBattleMonLevel]

    def pokemon_max_hp(self, memory: PyBoyMemoryView | bytes):
        return (memory[RAM.wBattleMonMaxHP] << 8) | memory[RAM.wBattleMonMaxHP + 1]

    def pokemon_attack(self, memory: PyBoyMemoryView | bytes):
        return (memory[RAM.wBattleMonAttack] << 8) | memory[RAM.wBattleMonAttack + 1]

    def pokemon_defense(self, memory: PyBoyMemoryView | bytes):
        return (memory[RAM.wBattleMonDefense] << 8) | memory[RAM.wBattleMonDefense + 1]

    def pokemon_speed(self, memory: PyBoyMemoryView | bytes):
        return (memory[RAM.wBattleMonSpeed] << 8) | memory[RAM.wBattleMonSpeed + 1]

    def pokemon_special(self, memory: PyBoyMemoryView | bytes):
        return (memory[RAM.wBattleMonSpecial] << 8) | memory[RAM.wBattleMonSpecial + 1]

    def pokemon_pp_first_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wBattleMonPP]

    def pokemon_pp_second_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wBattleMonPP + 1]

    def pokemon_pp_third_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wBattleMonPP + 2]

    def pokemon_pp_fourth_slot(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wBattleMonPP + 3]

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
                RAM.wPartyMon1PP + self.__player_pokemon_size * x,
                RAM.wPartyMon1Level + self.__player_pokemon_size * x,
            )
        ]

    def player_pokemons_ivs(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[i]
            for x in range(self.__pokemon_count)
            for i in range(
                RAM.wPartyMon1DVs + self.__player_pokemon_size * x,
                RAM.wPartyMon1PP + self.__player_pokemon_size * x,
            )
        ]

    def player_pokemon_types(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[i]
            for x in range(self.__pokemon_count)
            for i in range(
                RAM.wPartyMon1Type1 + self.__player_pokemon_size * x,
                RAM.wPartyMon1CatchRate + self.__player_pokemon_size * x,
            )
        ]

    def player_pokemons_ids(self, memory: PyBoyMemoryView | bytes = None):
        return [
            memory[RAM.wPartyMon1Species + self.__player_pokemon_size * x]
            for x in range(self.__pokemon_count)
        ]

    def player_pokemons_current_hps(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[RAM.wPartyMon1HP + self.__player_pokemon_size * x] << 8)
            | memory[RAM.wPartyMon1HP + 1 + self.__player_pokemon_size * x]
            for x in range(self.__pokemon_count)
        ]

    def player_pokemons_statuses(self, memory: PyBoyMemoryView | bytes = None):
        data = []
        for x in range(self.__pokemon_count):
            data += self.bits_extractor(
                memory[RAM.wPartyMon1Status + self.__player_pokemon_size * x], end_bit=6
            )

        return data

    def player_pokemons_experiences(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[RAM.wPartyMon1Exp + self.__player_pokemon_size * i] << 16)
            | (memory[RAM.wPartyMon1Exp + 1 + self.__player_pokemon_size * i] << 8)
            | memory[RAM.wPartyMon1Exp + 2 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_level(self, memory: PyBoyMemoryView | bytes = None):
        # All 6 party slots (not just the active battler) — the smart/coward
        # flee reward compares the enemy's level against max_party_level, so
        # the model needs bench levels visible to predict its own flee payout.
        return [
            memory[RAM.wPartyMon1Level + self.__player_pokemon_size * x]
            for x in range(self.__pokemon_count)
        ]

    def player_pokemons_max_hps(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[RAM.wPartyMon1MaxHP + self.__player_pokemon_size * i] << 8)
            | memory[RAM.wPartyMon1MaxHP + 1 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_attacks(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[RAM.wPartyMon1Attack + self.__player_pokemon_size * i] << 8)
            | memory[RAM.wPartyMon1Attack + 1 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_defenses(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[RAM.wPartyMon1Defense + self.__player_pokemon_size * i] << 8)
            | memory[RAM.wPartyMon1Defense + 1 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_speeds(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[RAM.wPartyMon1Speed + self.__player_pokemon_size * i] << 8)
            | memory[RAM.wPartyMon1Speed + 1 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def player_pokemons_specials(self, memory: PyBoyMemoryView | bytes = None):
        return [
            (memory[RAM.wPartyMon1Special + self.__player_pokemon_size * i] << 8)
            | memory[RAM.wPartyMon1Special + 1 + self.__player_pokemon_size * i]
            for i in range(self.__pokemon_count)
        ]

    def pokedex_data(self):
        return self.pokedex_own(self.pyboy.memory) + self.pokedex_seen(
            self.pyboy.memory
        )

    def pokedex_own(self, memory: PyBoyMemoryView | bytes):
        data = bytes(memory[RAM.wPokedexOwned : RAM.wPokedexSeen])

        bits: list[int] = []
        for byte in data:
            bits.extend(self.bits_extractor(byte))

        return bits

    def pokedex_seen(self, memory: PyBoyMemoryView | bytes):
        data = memory[RAM.wPokedexSeen : RAM.wPokedexSeenEnd]

        bits: list[int] = []
        for byte in data:
            bits.extend(self.bits_extractor(byte))

        return bits

    def items_quantities(self, memory: PyBoyMemoryView | bytes):
        return [memory[RAM.wBagItems + 1 + i * 2] for i in range(20)]

    def items_ids(self, memory: PyBoyMemoryView | bytes):
        return [memory[RAM.wBagItems + i * 2] for i in range(20)]

    def player_money(self, memory: PyBoyMemoryView | bytes):
        return (
            memory[RAM.wPlayerMoney]
            | (memory[RAM.wPlayerMoney + 1] << 8)
            | (memory[RAM.wPlayerMoney + 2] << 16)
        )

    def badges(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[RAM.wObtainedBadges])

    def stored_items_ids(self, memory: PyBoyMemoryView | bytes):
        return [memory[RAM.wBoxItems + 2 * i] for i in range(50)]

    def stored_items_quantities(self, memory: PyBoyMemoryView | bytes):
        return [memory[RAM.wBoxItems + 1 + 2 * i] for i in range(50)]

    def game_coins(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wPlayerCoins] | (memory[RAM.wPlayerCoins + 1] << 8)

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

    def have_town_map(self, memory: PyBoyMemoryView | bytes) -> bool:
        """True once the Town Map is actually held: in the bag or in PC item
        storage. The wD5F3 event byte alone is not trustworthy as a trigger
        (see GOAL_TOWN_MAP), so this checks real inventory contents instead."""
        if self._has_item_in_slots(
            self.items_ids(memory), self.items_quantities(memory), ITEM_ID_TOWN_MAP
        ):
            return True
        return self._has_item_in_slots(
            self.stored_items_ids(memory),
            self.stored_items_quantities(memory),
            ITEM_ID_TOWN_MAP,
        )

    def have_oaks_parcel(self, memory: PyBoyMemoryView | bytes):
        """D60D packs two adjacent wEventFlags bits from the Viridian Mart
        parcel hand-off script: bit 0 is a transient flag that's set mid-
        dialog and cleared again the instant the parcel actually lands in
        the bag, while bit 1 (EVENT_GOT_OAKS_PARCEL) is the real persistent
        "player has received the parcel" flag. Reading bit 0 (as this used
        to) reads True only for the few frames mid-dialog and then reads
        False for the rest of the game — including for the entire walk to
        Oak's Lab — which starved GOAL_OAKS_PARCEL and, transitively,
        GOAL_GAVE_PARCEL (gave_oaks_parcel() bails out immediately when this
        reads False) of ever firing. Confirmed by stepping a fresh emulator
        from the saves/stage_gave_parcel/ checkpoint and dumping D60D frame
        by frame across the mart hand-off."""
        return bool(memory[RAM.wViridianMartCurScript] & 2)

    def _has_item_in_slots(
        self, ids: list[int], quantities: list[int], item_id: int
    ) -> bool:
        for slot_id, qty in zip(ids, quantities):
            if slot_id in (0, 255):
                break
            if slot_id == item_id and qty > 0:
                return True
        return False

    def gave_oaks_parcel(self, memory: PyBoyMemoryView | bytes) -> bool:
        """Delivered to Oak: had the parcel at some point (persistent event
        flag), it was actually seen carried in the bag/PC at some point, and
        it's now gone from both. The "actually seen carried" part matters:
        the D60D event flag is set by the same script that hands you the
        item, and on the frame_skip window that catches the flag flipping
        true, the item write into the bag may not have landed yet — without
        _saw_oaks_parcel_in_bag that transient state reads as "gone" (i.e.
        delivered) before the parcel was ever really carried, which skips
        stage_gave_parcel entirely (goes straight from oaks_parcel to
        town_map)."""
        if not self.have_oaks_parcel(memory):
            return False
        in_bag = self._has_item_in_slots(
            self.items_ids(memory), self.items_quantities(memory), ITEM_ID_OAKS_PARCEL
        )
        in_pc = self._has_item_in_slots(
            self.stored_items_ids(memory),
            self.stored_items_quantities(memory),
            ITEM_ID_OAKS_PARCEL,
        )
        if in_bag or in_pc:
            self._saw_oaks_parcel_in_bag = True
            return False
        return self._saw_oaks_parcel_in_bag

    def bike_speed(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wWalkBikeSurfState]

    def fly_anywhere(self, memory: PyBoyMemoryView | bytes):
        return self.bits_extractor(memory[RAM.wTownVisitedFlag]) + self.bits_extractor(
            memory[RAM.wTownVisitedFlag + 1]
        )

    def safari_zone_time(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wSafariSteps] | (memory[RAM.wSafariSteps + 1] << 8)

    def fossilized_pokemon(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wFossilMon] & 1

    def position_in_air(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wPlayerJumpingYScreenCoordsIndex] & 1

    def did_you_get_lapras_yet(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wStatusFlags4] & 1

    def debug_new_game(self, memory: PyBoyMemoryView | bytes):
        return memory[RAM.wStatusFlags6] & 1

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
        """Whether Mewtwo can currently be encountered in Cerulean Cave.

        Kept as-is for the observation vector (event_flags_data) — it's
        genuinely different information from "caught": see has_caught_mewtwo
        for the actual GOAL_MEWTWO completion check.
        """
        return 1 if memory[0xD85F] & 2 else 0

    def has_caught_mewtwo(self, memory: PyBoyMemoryView | bytes) -> bool:
        """wPokedexOwned bit for Mewtwo (National Pokédex #150, fixed since
        Gen 1) — set once it's caught, independent of internal species IDs."""
        return bool(self.pokedex_own(memory)[MEWTWO_POKEDEX_NUMBER - 1])

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
            for x in range(RAM.wBoxMon1PP, RAM.wBoxMon2)
        ]

    def stored_pokemon_experiences(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[
                RAM.wBoxMon1Exp
                + self.real_current_menu_selected_item(memory)
                * self.__stored_pokemon_size
            ]
            | (
                memory[
                    RAM.wBoxMon1Exp
                    + 1
                    + self.real_current_menu_selected_item(memory)
                    * self.__stored_pokemon_size
                ]
                << 8
            )
            | (
                memory[
                    RAM.wBoxMon1Exp
                    + 2
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
            for x in range(RAM.wBoxMon1Moves, RAM.wBoxMon1OTID)
        ]

    def stored_pokemon_types(self, memory: PyBoyMemoryView | bytes):
        data = [
            memory[
                x
                + self.real_current_menu_selected_item(memory)
                * self.__stored_pokemon_size
            ]
            for x in range(RAM.wBoxMon1Type1, RAM.wBoxMon1CatchRate)
        ]

        return data if self.is_eq_menu(memory) else [0] * len(data)

    def stored_pokemon_statuses(self, memory: PyBoyMemoryView | bytes):
        return [
            bit
            for bit in self.bits_extractor(
                memory[
                    RAM.wBoxMon1Status
                    + self.real_current_menu_selected_item(memory)
                    * self.__stored_pokemon_size
                ]
            )
        ]

    def stored_pokemon_levels(self, memory: PyBoyMemoryView | bytes):
        return [
            memory[
                RAM.wBoxMon1BoxLevel
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
                        RAM.wBoxMon1HP
                        + self.real_current_menu_selected_item(memory)
                        * self.__stored_pokemon_size
                    ]
                ],
                [
                    memory[
                        RAM.wBoxMon1HP
                        + 1
                        + self.real_current_menu_selected_item(memory)
                        * self.__stored_pokemon_size
                    ]
                ],
            )
        ]

    def stored_pokemon_ids(self, memory: PyBoyMemoryView | bytes):
        data = [
            memory[
                RAM.wBoxMon1Species
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
        """wBattleResult @ RAM.wBattleResult: 0=win, 1=lose, 2=draw (player fled)."""
        return memory[RAM.wBattleResult]

    def party_count(self, memory: PyBoyMemoryView | bytes) -> int:
        return memory[RAM.wPartyCount]

    def all_party_levels(self, memory: PyBoyMemoryView | bytes) -> list[int]:
        """Levels of every occupied party slot (not just the active battler)."""
        count = min(int(self.party_count(memory)), self.__pokemon_count)
        return [
            memory[RAM.wPartyMon1Level + self.__player_pokemon_size * i]
            for i in range(count)
        ]

    def max_party_level(self, memory: PyBoyMemoryView | bytes) -> int:
        levels = [lv for lv in self.all_party_levels(memory) if lv > 0]
        return max(levels) if levels else 0

    def _battle_difficulty_scale(self, enemy_lv: int, player_lv: int) -> float:
        """max(battle_difficulty_scale_floor, smoothstep(enemy_lv / player_lv))
        — a same-or-tougher opponent (ratio >= 1) pays full credit; below
        that the reward eases down smoothly toward the floor as the level
        gap grows, instead of a hard linear ramp or decaying all the way to
        0. 3r^2-2r^3 has zero slope at both r=0 and r=1, so there is no kink
        anywhere, including the seam into the ratio>=1 cap=1.0 branch.

        Floored (not left to decay to 0) so a clean win against a far
        weaker wild Pokemon can't net negative once battle_useless_step's
        per-tick waste cost (never discounted by this scale) is netted
        against it — see battle_difficulty_scale_floor's own docstring for
        the reasoning, identical to wild_encounter_decay_floor's.

        Unreadable/invalid levels (<=0) fall back to a small constant below
        this floor, not 1.0 — this is an anti-farming discount, so a bad
        read must never silently grant full (or even floor-level) credit
        (that's exactly the failure mode that let the exploit through
        undetected).
        """
        if player_lv <= 0 or enemy_lv <= 0:
            return self.battle_difficulty_invalid_fallback
        r = min(1.0, enemy_lv / player_lv)
        smoothstep = 3 * r**2 - 2 * r**3
        return max(self.battle_difficulty_scale_floor, smoothstep)

    def _wild_encounter_decay(self) -> float:
        """Per-tile decay factor for wild-battle rewards, keyed off ordinary
        tile traffic rather than a dedicated encounter counter.

        Decays toward (not to) ``wild_encounter_decay_floor`` after
        ``wild_visit_decay_visits`` prior position_visit_counts on the
        current tile (``_battle_entry_wild_visits``, the pre-seed reading
        cached at battle entry) — walking a grass tile and fighting on it
        both spend down the same budget, so pacing on/off a known wild tile
        without even triggering a fight already burns its reward down.
        Applied to every reward/cost a wild fight can pay (``reward_enemy_hp``,
        ``reward_enemy_status``, ``battle_won_reward``, ``battle_turn_reward``,
        the HP-taken cost in ``reward_player_pokemons_current_hps``, and the
        exit-outcome rewards in ``reward_battle_exit``) so grinding one grass
        tile drains toward the floor, not just the win bonus. A hard floor
        of 0 used to mean every one of those undecays to nothing while
        battle_useless_step's per-tick waste cost (never decayed, not part
        of this budget) stays full price — a stale tile's battles net a
        guaranteed loss forever regardless of outcome, which taught blanket
        combat avoidance rather than just curbing farming (see
        wild_encounter_decay_floor's own docstring for the math). Trainer
        battles never call this — they are one-shot per sprite, not
        repeatable on the same tile.
        """
        return max(
            self.wild_encounter_decay_floor,
            1.0 - self._battle_entry_wild_visits / self.wild_visit_decay_visits,
        )

    def reward_battle_exit(self, memory: bytes) -> float:
        """Reward battle outcome on battle→overworld transition (wBattleResult).

        - 0 win → ``battle_won_reward``
        - 1 lose → ``battle_lost_penalty`` (a mon fainted but the player had
          another to send out — HandlePlayerBlackOut below is the full wipe)
        - 2 fled → smart/coward flee based on enemy vs max party level

        wBattleResult (RAM.wBattleResult) is only ever written by the win/faint-switch/
        run paths in pokered's battle engine. A full party wipe goes through
        HandlePlayerBlackOut instead, which never touches wBattleResult — so
        on that exit frame it still holds whatever value was last written by
        an *earlier* battle (often 0/win), and would otherwise be read here
        as a win for the very fight that just wiped the party out. Detect it
        directly from wIsInBattle (RAM.wIsInBattle), which pokered documents as being
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

        # Every battle-exit outcome (win/lose/blackout/smart-flee/coward-flee)
        # decays with repeat encounters on the same tile, same shape as
        # reward_position's exploration decay and every other wild-battle
        # reward component (reward_enemy_hp, reward_enemy_status,
        # battle_turn_reward, reward_player_pokemons_current_hps) — a
        # heavily-farmed tile stops mattering symmetrically on every outcome,
        # not just the win path, instead of only the reward side decaying
        # while penalties (or a repeat coward-flee) stay full price forever.
        # Trainer battles are one-shot per sprite so they are exempt
        # (type_of_battle read from `memory`, the still-in-battle previous
        # frame — self.pyboy.memory is already back in the overworld here).
        if self.type_of_battle(memory) == 1:
            decay = self._wild_encounter_decay()
            reward *= decay
            # Always-on (not collect_heatmap-gated) telemetry for
            # pokemon/wild_encounter_decay_mean / _floored_rate — see
            # _wild_decay_sum's own comment.
            self._wild_decay_sum += decay
            self._wild_decay_count += 1
            if decay <= self.wild_encounter_decay_floor + 1e-9:
                self._wild_decay_floored_count += 1

        # Always-on (not collect_heatmap-gated) outcome tally for
        # pokemon/battle_*_rate / pokemon/battles_per_episode — see
        # battle_outcome_tally's own comment.
        self.battle_outcome_tally[kind] = self.battle_outcome_tally.get(kind, 0) + 1

        # --heatmap win-rate/flee-rate overlays, both keyed off the same
        # per-tile counts dict: win/loss feed the win-rate metric, smart/
        # coward feed the flee-rate metric — they're disjoint outcomes of
        # the same battle so one dict with four buckets avoids a second
        # snapshot channel through PokemonRedEnv. Anchored on the same
        # _last_heatmap_pos reward_sums uses — is_world() has been False for
        # the whole fight, so it still holds the pre-battle world tile.
        if self.collect_heatmap and self._last_heatmap_pos is not None:
            bucket = {"win": "win", "lose": "loss", "blackout": "loss"}.get(kind, kind)
            if bucket in ("win", "loss", "smart", "coward"):
                counts = self.battle_outcome_counts.setdefault(
                    self._last_heatmap_pos,
                    {"win": 0, "loss": 0, "smart": 0, "coward": 0},
                )
                counts[bucket] = counts.get(bucket, 0) + 1

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
