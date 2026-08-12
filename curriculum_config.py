"""
Curriculum Learning Configuration for Pokemon Red AI (PPO).

A "stage" is now literally a goal name — see pokemon.Data.GOAL_CANDIDATES:
either an EVENT_* name from event_constants.py (a wEventFlags bit, minus
whatever event_graph.py's static analysis auto-blacklisted as dead/cyclic/
resettable) or one of BADGE_GOALS ("badge1".."badge8", read off
wObtainedBadges). There is no more hand-picked STAGE_ORDER/CURRICULUM dict
distinct from the goal list.

STAGE_ORDER sorts GOAL_CANDIDATES by each event's wEventFlags bit index --
originally a rough story-progression proxy, since event_constants.asm
declares events roughly city-by-city in game order (see
tools/gen_event_constants.py). That turned out not to hold reliably: in-game
event order does not actually follow STAGE_ORDER's heuristic sort, and goal
completion should not be order-dependent anyway (an episode may legitimately
reach a "later" goal before an "earlier" one). STAGE_ORDER is therefore kept
ONLY as a stable enumeration for two purposes that need *a* fixed order, not
a *correct* one: (1) GOAL_ORDER/GOAL_INDEX's one-hot indexing for the
policy's goal observation, and (2) cosmetic listings (--list-stages,
create_stage_save.py --list). Nothing picks "the next goal" by walking
STAGE_ORDER anymore -- see pick_new_goal(), which is a random, order-free
pick instead. Badges (no bit index of their own) are still anchored next to
their corresponding "beat gym leader" event purely for STAGE_ORDER's list
position; that anchoring has no bearing on pick_new_goal().

Save directories under saves/<goal_name>/checkpoint.state, written
dynamically as episodes reach them (see
PokemonRedEnv._save_milestone_checkpoints) -- whichever goal an episode
actually clears first gets a checkpoint, regardless of order. If a goal has
no checkpoint of its own yet, falls back to "start" (see resolve_checkpoint).
"""

from __future__ import annotations

import os
import random

from pokemon import event_constants as _event_constants
from pokemon.Data import BADGE_GOALS, GOAL_CANDIDATES

# Reassign the training goal when this fraction of recent episodes hit
# *some* goal (any goal -- see pick_new_goal).
ADVANCE_SUCCESS_THRESHOLD = 0.70
ADVANCE_MIN_EPISODES = 40
ADVANCE_CHECK_EVERY = 2048

# Generous flat safety-net ceiling on top of Data.py's live, size-scaled
# per-map budget (map_step_budget) — that budget grows as new maps are
# visited instead of being precomputed per goal here, so this constant only
# guards against a policy that never triggers any map's budget/other fuses
# at all (e.g. standing still on the very first map forever).
DEFAULT_MAX_STEPS = 20_000

# Badges have no wEventFlags bit of their own (wObtainedBadges instead — see
# badges()/BADGE_GOALS), so they need a stand-in event to sort next to.
_BADGE_ANCHOR_EVENT: dict[str, str] = {
    "badge1": "EVENT_BEAT_BROCK",
    "badge2": "EVENT_BEAT_MISTY",
    "badge3": "EVENT_BEAT_LT_SURGE",
    "badge4": "EVENT_BEAT_ERIKA",
    "badge5": "EVENT_BEAT_KOGA",
    "badge6": "EVENT_BEAT_SABRINA",
    "badge7": "EVENT_BEAT_BLAINE",
    "badge8": "EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI",
}


def _order_key(goal: str) -> tuple[int, int]:
    anchor = _BADGE_ANCHOR_EVENT.get(goal)
    if anchor is not None:
        # Sort a badge immediately after the event it's anchored to.
        return (_event_constants.EVENTS.get(anchor, 0), 1)
    return (_event_constants.EVENTS.get(goal, 0), 0)


def _description(goal: str) -> str:
    if goal in BADGE_GOALS:
        return f"Earn badge #{BADGE_GOALS.index(goal) + 1}"
    name = goal[len("EVENT_"):] if goal.startswith("EVENT_") else goal
    return name.replace("_", " ").title()


def _stage(goal: str) -> dict:
    return {
        "checkpoint": goal,
        "goal": goal,
        "max_steps": DEFAULT_MAX_STEPS,
        "description": _description(goal),
        "enabled": True,
        "earlier": [],
    }


# Stable enumeration only -- see module docstring. Not used to decide what
# goal comes next (see pick_new_goal); only for one-hot indexing (GOAL_ORDER)
# and cosmetic listings.
STAGE_ORDER: list[str] = sorted(GOAL_CANDIDATES, key=_order_key)

CURRICULUM: dict[str, dict] = {goal: _stage(goal) for goal in STAGE_ORDER}

# Best-effort aliases from the old hand-picked stage ids (pre-generic-goal
# system) to a new goal name, so an old --stage flag doesn't hard-crash.
# Not exhaustive: several old stages (stage_left_house, stage_route1_entry,
# stage_route1, stage_lapras, ...) were map-presence or known-broken-bit
# checks with no wEventFlags equivalent at all (see the fought_X_yet audit
# in Data.py's git history) — those fall through unresolved to
# resolve_stage_name's caller-side fallback instead of a fabricated mapping.
_LEGACY_STAGE_ALIASES: dict[str, str] = {
    "stage_oaks_parcel": "EVENT_GOT_OAKS_PARCEL",
    "stage_gave_parcel": "EVENT_OAK_GOT_PARCEL",
    "stage_town_map": "EVENT_GOT_TOWN_MAP",
    "stage_fought_brock": "EVENT_BEAT_BROCK",
    "stage_badge1": "badge1",
    "stage_fought_misty": "EVENT_BEAT_MISTY",
    "stage_badge2": "badge2",
    "stage_fought_surge": "EVENT_BEAT_LT_SURGE",
    "stage_badge3": "badge3",
    "stage_fought_erika": "EVENT_BEAT_ERIKA",
    "stage_badge4": "badge4",
    "stage_fought_koga": "EVENT_BEAT_KOGA",
    "stage_badge5": "badge5",
    "stage_fought_sabrina": "EVENT_BEAT_SABRINA",
    "stage_badge6": "badge6",
    "stage_fought_blaine": "EVENT_BEAT_BLAINE",
    "stage_badge7": "badge7",
    "stage_fought_giovanni": "EVENT_BEAT_VIRIDIAN_GYM_GIOVANNI",
    "stage_badge8": "badge8",
}
# Goal one-hot for policy observations (order matches STAGE_ORDER).
GOAL_ORDER: list[str] = STAGE_ORDER
GOAL_INDEX: dict[str, int] = {g: i for i, g in enumerate(GOAL_ORDER)}
N_GOALS: int = len(GOAL_ORDER)


def resolve_stage_name(stage: str) -> str:
    """Normalize a stage id: a live goal name resolves to itself; a
    recognized legacy 'stage_*' id (pre-generic-goal system) maps to its
    nearest new equivalent (see _LEGACY_STAGE_ALIASES); anything else is
    returned unchanged (callers already fall back to STAGE_ORDER[0] on a
    CURRICULUM miss, e.g. get_goal_for_stage)."""
    if stage in CURRICULUM:
        return stage
    return _LEGACY_STAGE_ALIASES.get(stage, stage)


def _save_exists(name: str) -> bool:
    return os.path.isfile(f"saves/{name}/checkpoint.state")


def resolve_checkpoint(name: str) -> str:
    if _save_exists(name):
        return name
    return "start"


def get_goal_for_stage(stage: str) -> str:
    stage = resolve_stage_name(stage)
    cfg = CURRICULUM.get(stage, CURRICULUM[STAGE_ORDER[0]])
    return cfg["goal"]


def get_stage_max_steps(stage: str) -> int:
    stage = resolve_stage_name(stage)
    cfg = CURRICULUM.get(stage, CURRICULUM[STAGE_ORDER[0]])
    return int(cfg["max_steps"])


def get_curriculum_saves(stage: str) -> list[str]:
    """Ordered list of save dirs for curriculum mix (earlier … current)."""
    stage = resolve_stage_name(stage)
    cfg = CURRICULUM.get(stage, CURRICULUM[STAGE_ORDER[0]])
    saves: list[str] = []
    for name in cfg.get("earlier", []):
        saves.append(resolve_checkpoint(name))
    saves.append(resolve_checkpoint(cfg["checkpoint"]))
    named = resolve_checkpoint(stage)
    if named not in saves and named != "start":
        saves.append(named)
    seen: set[str] = set()
    out: list[str] = []
    for s in saves:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out or ["start"]


def stage_index(stage: str) -> int:
    """STAGE_ORDER position -- informational only (TensorBoard's
    curriculum_stage_idx), not used to pick what comes next anymore."""
    stage = resolve_stage_name(stage)
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return 0


def pick_new_goal(
    *,
    is_satisfied=None,
    prefer_discovered: bool = True,
    rng: random.Random | None = None,
) -> str | None:
    """Random, order-free pick of a goal to work toward next.

    Replaces the old next_stage()/prev_stage() STAGE_ORDER walk: there is no
    "next" goal, only "some goal not yet satisfied". If ``is_satisfied`` is
    given (a live, single-playthrough check -- e.g.
    Data.is_goal_satisfied), candidates it reports as already true are
    excluded; omit it when there's no single live state to check against
    (e.g. MilestoneCallback picking a base goal across many independent
    parallel workers).

    When ``prefer_discovered`` is True (default) and at least one candidate
    already has its own saves/<goal>/checkpoint.state -- i.e. some episode
    has actually reached it before, see
    PokemonRedEnv._save_milestone_checkpoints -- the pick is restricted to
    those, so training isn't handed a goal with no known path to it yet.
    Falls back to the full candidate pool otherwise (e.g. very early on,
    before anything's been discovered).
    """
    rng = rng or random
    candidates = [
        g
        for g in STAGE_ORDER
        if CURRICULUM.get(g, {}).get("enabled", True)
        and (is_satisfied is None or not is_satisfied(g))
    ]
    if not candidates:
        return None
    if prefer_discovered:
        discovered = [g for g in candidates if _save_exists(g)]
        if discovered:
            candidates = discovered
    return rng.choice(candidates)


def stage_for_goal(goal: str) -> str:
    """First curriculum stage that targets ``goal`` (fallback: STAGE_ORDER[0])."""
    if goal in CURRICULUM:
        return goal
    return STAGE_ORDER[0]
