"""
Curriculum Learning Configuration for Pokemon Red AI (PPO).

A "stage" is now literally a goal name — see pokemon.Data.GOAL_CANDIDATES:
either an EVENT_* name from event_constants.py (a wEventFlags bit, minus
whatever event_graph.py's static analysis auto-blacklisted as dead/cyclic/
resettable) or one of BADGE_GOALS ("badge1".."badge8", read off
wObtainedBadges). There is no more hand-picked STAGE_ORDER/CURRICULUM dict
distinct from the goal list.

STAGE_ORDER sorts GOAL_CANDIDATES by each event's wEventFlags bit index as a
rough story-progression proxy: event_constants.asm declares events roughly
city-by-city in game order (see tools/gen_event_constants.py), so bit index
correlates with progression more often than not. Badges (no bit index of
their own) are anchored next to their corresponding "beat gym leader" event.
This is a static approximation, not hand-verified game knowledge — e.g.
Viridian Gym/Giovanni's events sit early in bit-index space because
Viridian is a very early city, even though badge8 is earned last in normal
play. next_stage()'s is_satisfied skip-ahead and the advance/demote
threshold logic in callbacks.py both tolerate an imperfect order (a
goal already true just gets skipped); a wrong position mostly costs
early-training sample efficiency, not correctness.

Save directories under saves/<goal_name>/checkpoint.state. If a stage save
is missing, falls back to "start" (see resolve_checkpoint) — as of this
writing only saves/start exists; create per-goal checkpoints with
create_stage_save.py as training progresses (the checkpoint dir name is
just the goal name, so this works for any goal, not a fixed hand-picked
set).
"""

from __future__ import annotations

import os
from typing import Callable

from pokemon import event_constants as _event_constants
from pokemon.Data import BADGE_GOALS, GOAL_CANDIDATES

# Auto-advance when this fraction of recent episodes hit the stage goal.
ADVANCE_SUCCESS_THRESHOLD = 0.70
ADVANCE_MIN_EPISODES = 40
ADVANCE_CHECK_EVERY = 2048

# Auto-demote one stage back if the agent is fully stalled (near-zero success)
# for this many consecutive advance checks. Guards against catastrophic
# forgetting: the policy drifts on an earlier goal while training grinds on a
# later stage, and there is otherwise no way back down.
DEMOTE_STALL_CHECKS = 60
DEMOTE_SUCCESS_CEILING = 0.05

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


# Recommended soft order — see module docstring on how it's derived and its
# known imprecision. Goals already true are skipped at advance time.
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


def next_stage(
    stage: str,
    *,
    is_satisfied: Callable[[str], bool] | None = None,
) -> str | None:
    """Next enabled stage after ``stage``.

    If ``is_satisfied(goal)`` is provided, skip stages whose goals are already
    true (handles out-of-order event flags during in-place eval).
    """
    stage = resolve_stage_name(stage)
    try:
        idx = STAGE_ORDER.index(stage)
    except ValueError:
        return None
    for name in STAGE_ORDER[idx + 1 :]:
        cfg = CURRICULUM.get(name)
        if not cfg or not cfg.get("enabled", True):
            continue
        if is_satisfied is not None and is_satisfied(cfg["goal"]):
            continue
        return name
    return None


def prev_stage(stage: str) -> str | None:
    """Nearest enabled stage before ``stage`` (for stall demotion)."""
    stage = resolve_stage_name(stage)
    try:
        idx = STAGE_ORDER.index(stage)
    except ValueError:
        return None
    for name in reversed(STAGE_ORDER[:idx]):
        cfg = CURRICULUM.get(name)
        if not cfg or not cfg.get("enabled", True):
            continue
        return name
    return None


def stage_index(stage: str) -> int:
    stage = resolve_stage_name(stage)
    try:
        return STAGE_ORDER.index(stage)
    except ValueError:
        return 0


def get_checkpoint_for_episode(total_steps: int) -> str | None:
    """Legacy step-range curriculum for Rainbow DQN workers."""
    chunk = 500_000
    for i, stage_name in enumerate(STAGE_ORDER):
        cfg = CURRICULUM.get(stage_name)
        if not cfg or not cfg.get("enabled"):
            continue
        lo = i * chunk
        hi = (i + 1) * chunk if i < len(STAGE_ORDER) - 1 else 50_000_000
        if lo <= total_steps < hi:
            return resolve_checkpoint(cfg["checkpoint"])
    return None


def get_current_stage(total_steps: int) -> str | None:
    chunk = 500_000
    for i, stage_name in enumerate(STAGE_ORDER):
        cfg = CURRICULUM.get(stage_name)
        if not cfg or not cfg.get("enabled"):
            continue
        lo = i * chunk
        hi = (i + 1) * chunk if i < len(STAGE_ORDER) - 1 else 50_000_000
        if lo <= total_steps < hi:
            return stage_name
    return None


def stage_for_goal(goal: str) -> str:
    """First curriculum stage that targets ``goal`` (fallback: STAGE_ORDER[0])."""
    if goal in CURRICULUM:
        return goal
    return STAGE_ORDER[0]
