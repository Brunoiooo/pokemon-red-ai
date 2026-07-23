"""
Curriculum Learning Configuration for Pokemon Red AI (PPO).

Stages progress through early-game milestones:
  stage_0 — leave the player's house / reach Pallet Town
  stage_1 — reach Route 1 / tall grass
  stage_2 — Oak's lab / rival path, push toward Badge 1

Save directories under saves/<checkpoint>/checkpoint.state.
If a stage save is missing, falls back to "start".
"""
from __future__ import annotations

import os

from pokemon.Data import (
    GOAL_BADGE_1,
    GOAL_LEFT_HOUSE,
    GOAL_OAKS_LAB,
    GOAL_ROUTE_1,
)

CURRICULUM = {
    "stage_0": {
        "checkpoint": "start",
        "goal": GOAL_LEFT_HOUSE,
        "max_steps": 2048,
        "description": "Leave Red's house and enter Pallet Town",
        "enabled": True,
        "earlier": [],
    },
    "stage_1": {
        "checkpoint": "stage_1",
        "goal": GOAL_ROUTE_1,
        "max_steps": 4096,
        "description": "Reach Route 1 / tall grass",
        "enabled": True,
        "earlier": ["start"],
    },
    "stage_2": {
        "checkpoint": "stage_2",
        "goal": GOAL_BADGE_1,
        "max_steps": 8192,
        "description": "Progress toward Pewter Gym / Badge 1",
        "enabled": True,
        "earlier": ["start", "stage_1"],
    },
}


def _save_exists(name: str) -> bool:
    return os.path.isfile(f"saves/{name}/checkpoint.state")


def resolve_checkpoint(name: str) -> str:
    if _save_exists(name):
        return name
    return "start"


def get_goal_for_stage(stage: str) -> str:
    cfg = CURRICULUM.get(stage, CURRICULUM["stage_0"])
    return cfg["goal"]


def get_stage_max_steps(stage: str) -> int:
    cfg = CURRICULUM.get(stage, CURRICULUM["stage_0"])
    return int(cfg["max_steps"])


def get_curriculum_saves(stage: str) -> list[str]:
    """Ordered list of save dirs for curriculum mix (earlier … current)."""
    cfg = CURRICULUM.get(stage, CURRICULUM["stage_0"])
    saves: list[str] = []
    for name in cfg.get("earlier", []):
        saves.append(resolve_checkpoint(name))
    saves.append(resolve_checkpoint(cfg["checkpoint"]))
    # Deduplicate while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for s in saves:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out or ["start"]


# ---------------------------------------------------------------------------
# Legacy DQN helpers (still imported by ExperienceWorker if used)
# ---------------------------------------------------------------------------
def get_checkpoint_for_episode(total_steps: int) -> str | None:
    """Legacy step-range curriculum for Rainbow DQN workers."""
    ranges = {
        "stage_0": (0, 500_000),
        "stage_1": (500_000, 2_000_000),
        "stage_2": (2_000_000, 10_000_000),
    }
    for stage_name, (lo, hi) in sorted(ranges.items(), key=lambda x: x[1][0], reverse=True):
        cfg = CURRICULUM.get(stage_name)
        if not cfg or not cfg.get("enabled"):
            continue
        if lo <= total_steps < hi:
            return resolve_checkpoint(cfg["checkpoint"])
    return None


def get_current_stage(total_steps: int) -> str | None:
    ranges = {
        "stage_0": (0, 500_000),
        "stage_1": (500_000, 2_000_000),
        "stage_2": (2_000_000, 10_000_000),
    }
    for stage_name, (lo, hi) in ranges.items():
        cfg = CURRICULUM.get(stage_name)
        if not cfg or not cfg.get("enabled"):
            continue
        if lo <= total_steps < hi:
            return stage_name
    return None
