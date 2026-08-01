"""
Curriculum Learning Configuration for Pokemon Red AI (PPO).

Stages cover navigation, story flags, gym fights, and all 8 badges.
``STAGE_ORDER`` is a *recommended* soft order — many event flags can be
obtained in different sequences. Auto-curriculum / eval therefore skip any
stage whose goal is already satisfied in the live save (see ``next_stage``).

Save directories under saves/<checkpoint>/checkpoint.state.
If a stage save is missing, falls back to "start".
"""
from __future__ import annotations

import os
from typing import Callable

from pokemon.Data import (
    GOAL_ALL_BADGES,
    GOAL_ARTICUNO,
    GOAL_BADGE_1,
    GOAL_BADGE_2,
    GOAL_BADGE_3,
    GOAL_BADGE_4,
    GOAL_BADGE_5,
    GOAL_BADGE_6,
    GOAL_BADGE_7,
    GOAL_BADGE_8,
    GOAL_FOSSIL,
    GOAL_FOUGHT_BLAINE,
    GOAL_FOUGHT_BROCK,
    GOAL_FOUGHT_ERIKA,
    GOAL_FOUGHT_GIOVANNI,
    GOAL_FOUGHT_KOGA,
    GOAL_FOUGHT_MISTY,
    GOAL_FOUGHT_SABRINA,
    GOAL_FOUGHT_SURGE,
    GOAL_LAPRAS,
    GOAL_LEFT_HOUSE,
    GOAL_MEWTWO,
    GOAL_MOLTRES,
    GOAL_OAKS_LAB,
    GOAL_OAKS_PARCEL,
    GOAL_ROUTE_1,
    GOAL_ROUTE_1_ENTRY,
    GOAL_SNORLAX,
    GOAL_SS_ANNE,
    GOAL_TOWN_MAP,
    GOAL_ZAPDOS,
)

# Auto-advance when this fraction of recent episodes hit the stage goal.
ADVANCE_SUCCESS_THRESHOLD = 0.70
ADVANCE_MIN_EPISODES = 40
ADVANCE_CHECK_EVERY = 2048

# Auto-demote one stage back if the agent is fully stalled (near-zero success)
# for this many consecutive advance checks. Guards against catastrophic
# forgetting: the policy drifts on an earlier goal while training grinds on a
# later stage, and there is otherwise no way back down.
DEMOTE_STALL_CHECKS = 15
DEMOTE_SUCCESS_CEILING = 0.05


def _stage(
    goal: str,
    *,
    max_steps: int,
    description: str,
    checkpoint: str | None = None,
    earlier: list[str] | None = None,
) -> dict:
    name_checkpoint = checkpoint or "start"
    return {
        "checkpoint": name_checkpoint,
        "goal": goal,
        "max_steps": max_steps,
        "description": description,
        "enabled": True,
        "earlier": list(earlier or []),
    }


# Recommended soft order. Flags/badges already true are skipped at advance time.
STAGE_ORDER = [
    "stage_left_house",
    "stage_route1_entry",
    "stage_oaks_lab",
    "stage_route1",
    "stage_oaks_parcel",
    "stage_town_map",
    "stage_fought_brock",
    "stage_badge1",
    "stage_fought_misty",
    "stage_badge2",
    "stage_ss_anne",
    "stage_fought_surge",
    "stage_badge3",
    "stage_fought_erika",
    "stage_badge4",
    "stage_fought_koga",
    "stage_badge5",
    "stage_fought_sabrina",
    "stage_badge6",
    "stage_fought_blaine",
    "stage_badge7",
    "stage_fought_giovanni",
    "stage_badge8",
    "stage_lapras",
    "stage_snorlax",
    "stage_articuno",
    "stage_zapdos",
    "stage_moltres",
    "stage_fossil",
    "stage_mewtwo",
    "stage_all_badges",
]

CURRICULUM: dict[str, dict] = {
    "stage_left_house": _stage(
        GOAL_LEFT_HOUSE, max_steps=2048, description="Leave Red's house"
    ),
    "stage_route1_entry": _stage(
        GOAL_ROUTE_1_ENTRY,
        max_steps=2048,
        description="Head for Route 1 — triggers Oak's intercept into the Lab",
        checkpoint="stage_route1_entry",
        earlier=["start"],
    ),
    "stage_oaks_lab": _stage(
        GOAL_OAKS_LAB,
        max_steps=4096,
        description="Reach Oak's Lab",
        checkpoint="stage_oaks_lab",
        earlier=["start", "stage_route1_entry"],
    ),
    "stage_route1": _stage(
        GOAL_ROUTE_1,
        max_steps=4096,
        description="Reach Route 1 / tall grass",
        checkpoint="stage_route1",
        earlier=["start", "stage_oaks_lab"],
    ),
    "stage_oaks_parcel": _stage(
        GOAL_OAKS_PARCEL,
        max_steps=8192,
        description="Obtain Oak's Parcel",
        checkpoint="stage_oaks_parcel",
        earlier=["start", "stage_route1"],
    ),
    "stage_town_map": _stage(
        GOAL_TOWN_MAP,
        max_steps=8192,
        description="Obtain the Town Map",
        checkpoint="stage_town_map",
        earlier=["start", "stage_oaks_parcel"],
    ),
    "stage_fought_brock": _stage(
        GOAL_FOUGHT_BROCK, max_steps=12288, description="Fight Brock (Pewter Gym)"
    ),
    "stage_badge1": _stage(
        GOAL_BADGE_1, max_steps=12288, description="Earn Boulder Badge"
    ),
    "stage_fought_misty": _stage(
        GOAL_FOUGHT_MISTY, max_steps=16384, description="Fight Misty (Cerulean)"
    ),
    "stage_badge2": _stage(
        GOAL_BADGE_2, max_steps=16384, description="Earn Cascade Badge"
    ),
    "stage_ss_anne": _stage(
        GOAL_SS_ANNE, max_steps=16384, description="SS Anne present / progress"
    ),
    "stage_fought_surge": _stage(
        GOAL_FOUGHT_SURGE, max_steps=16384, description="Fight Lt. Surge"
    ),
    "stage_badge3": _stage(
        GOAL_BADGE_3, max_steps=16384, description="Earn Thunder Badge"
    ),
    "stage_fought_erika": _stage(
        GOAL_FOUGHT_ERIKA, max_steps=20480, description="Fight Erika"
    ),
    "stage_badge4": _stage(
        GOAL_BADGE_4, max_steps=20480, description="Earn Rainbow Badge"
    ),
    "stage_fought_koga": _stage(
        GOAL_FOUGHT_KOGA, max_steps=24576, description="Fight Koga"
    ),
    "stage_badge5": _stage(
        GOAL_BADGE_5, max_steps=24576, description="Earn Soul Badge"
    ),
    "stage_fought_sabrina": _stage(
        GOAL_FOUGHT_SABRINA, max_steps=24576, description="Fight Sabrina"
    ),
    "stage_badge6": _stage(
        GOAL_BADGE_6, max_steps=24576, description="Earn Marsh Badge"
    ),
    "stage_fought_blaine": _stage(
        GOAL_FOUGHT_BLAINE, max_steps=28672, description="Fight Blaine"
    ),
    "stage_badge7": _stage(
        GOAL_BADGE_7, max_steps=28672, description="Earn Volcano Badge"
    ),
    "stage_fought_giovanni": _stage(
        GOAL_FOUGHT_GIOVANNI, max_steps=32768, description="Fight Giovanni"
    ),
    "stage_badge8": _stage(
        GOAL_BADGE_8, max_steps=32768, description="Earn Earth Badge"
    ),
    "stage_lapras": _stage(
        GOAL_LAPRAS, max_steps=24576, description="Obtain Lapras"
    ),
    "stage_snorlax": _stage(
        GOAL_SNORLAX, max_steps=24576, description="Encounter Snorlax (either site)"
    ),
    "stage_articuno": _stage(
        GOAL_ARTICUNO, max_steps=32768, description="Fight Articuno"
    ),
    "stage_zapdos": _stage(
        GOAL_ZAPDOS, max_steps=32768, description="Fight Zapdos"
    ),
    "stage_moltres": _stage(
        GOAL_MOLTRES, max_steps=32768, description="Fight Moltres"
    ),
    "stage_fossil": _stage(
        GOAL_FOSSIL, max_steps=24576, description="Obtain a fossil"
    ),
    "stage_mewtwo": _stage(
        GOAL_MEWTWO, max_steps=32768, description="Mewtwo becomes catchable"
    ),
    "stage_all_badges": _stage(
        GOAL_ALL_BADGES, max_steps=32768, description="Hold all 8 badges"
    ),
}


def resolve_stage_name(stage: str) -> str:
    """Normalize stage id (no legacy aliases)."""
    return stage


def _save_exists(name: str) -> bool:
    return os.path.isfile(f"saves/{name}/checkpoint.state")


def resolve_checkpoint(name: str) -> str:
    if _save_exists(name):
        return name
    return "start"


def get_goal_for_stage(stage: str) -> str:
    stage = resolve_stage_name(stage)
    cfg = CURRICULUM.get(stage, CURRICULUM["stage_left_house"])
    return cfg["goal"]


def get_stage_max_steps(stage: str) -> int:
    stage = resolve_stage_name(stage)
    cfg = CURRICULUM.get(stage, CURRICULUM["stage_left_house"])
    return int(cfg["max_steps"])


def get_curriculum_saves(stage: str) -> list[str]:
    """Ordered list of save dirs for curriculum mix (earlier … current)."""
    stage = resolve_stage_name(stage)
    cfg = CURRICULUM.get(stage, CURRICULUM["stage_left_house"])
    saves: list[str] = []
    for name in cfg.get("earlier", []):
        saves.append(resolve_checkpoint(name))
    saves.append(resolve_checkpoint(cfg["checkpoint"]))
    # Also allow a same-named mid-game save if the user created one.
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
    """First curriculum stage that targets ``goal`` (fallback: left_house)."""
    for name in STAGE_ORDER:
        cfg = CURRICULUM.get(name)
        if cfg and cfg.get("goal") == goal:
            return name
    return "stage_left_house"


# Goal one-hot for policy observations (order matches STAGE_ORDER).
GOAL_ORDER = [CURRICULUM[s]["goal"] for s in STAGE_ORDER]
GOAL_INDEX = {g: i for i, g in enumerate(GOAL_ORDER)}
N_GOALS = len(GOAL_ORDER)
