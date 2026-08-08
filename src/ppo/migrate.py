"""Weight-surgery helper for resuming PPO training after the observation
vector's goal one-hot block changes size (see curriculum_config.GOAL_ORDER).

SB3's PPO.load() requires an exact match between a checkpoint's saved tensors
and the live policy's parameter shapes. Growing GOAL_ORDER (a new curriculum
goal) breaks that for exactly one input dimension of the *_features_extractor
.vector_mlp.0 layer — the goal one-hot is appended last to the flat "vector"
observation (see env.pokemon_red_env._torch_inputs_to_obs), so every feature
before it keeps its old column index; only the goal block's width and the
positions of pre-existing goals inside it change.

migrate_state_dict() copies every parameter whose shape is unchanged as-is,
and remaps just that one layer's goal columns by *name* (not position), so
where the new goal was inserted into GOAL_ORDER doesn't matter. Columns for
goals that did not exist in the checkpoint keep the freshly-initialized
model's random init — the policy has to learn those from scratch, same as
any brand-new goal always would.
"""
from __future__ import annotations

import torch

# Goals introduced after older checkpoints were trained. Extend this set
# (never remove past entries) whenever GOAL_ORDER grows, so --migrate keeps
# working against checkpoints saved at any earlier point in training history.
# The internal consistency check in _remap_goal_columns raises loudly if this
# set doesn't fully account for a checkpoint's actual goal count, rather than
# silently mis-mapping columns.
#
# "pc_tutorial" is a plain string, not a live GOAL_* constant: the stage was
# later removed from curriculum_config entirely, so any checkpoint that still
# carries that column now hits the "goal no longer in curriculum, weights
# discarded" path in _remap_goal_columns instead of being remapped.
NEW_GOALS_SINCE: dict[str, str] = {
    "pc_tutorial": "2026-08-06: pc_tutorial stage inserted at STAGE_ORDER[0]",
}

_VECTOR_MLP_IN_KEYS = (
    "features_extractor.vector_mlp.0.weight",
    "pi_features_extractor.vector_mlp.0.weight",
    "vf_features_extractor.vector_mlp.0.weight",
)


def _old_goal_order(goal_order: list[str]) -> list[str]:
    return [g for g in goal_order if g not in NEW_GOALS_SINCE]


def _remap_goal_columns(
    old_weight: torch.Tensor,
    new_weight: torch.Tensor,
    old_goal_order: list[str],
    goal_index: dict[str, int],
) -> torch.Tensor:
    old_dim = old_weight.shape[1]
    new_dim = new_weight.shape[1]
    base_old = old_dim - len(old_goal_order)
    base_new = new_dim - len(goal_index)
    if base_old != base_new or base_old < 0:
        raise ValueError(
            f"Non-goal feature width doesn't line up (old base={base_old}, "
            f"new base={base_new}) — NEW_GOALS_SINCE in ppo/migrate.py is "
            f"probably missing an entry for this checkpoint's vintage, or "
            f"the mismatch isn't just the goal one-hot growing."
        )
    out = new_weight.clone()
    out[:, :base_new] = old_weight[:, :base_old]
    dropped = []
    for i, goal_name in enumerate(old_goal_order):
        j = goal_index.get(goal_name)
        if j is None:
            dropped.append(goal_name)
            continue
        out[:, base_new + j] = old_weight[:, base_old + i]
    if dropped:
        print(f"  (goal(s) no longer in curriculum, weights discarded: {dropped})")
    return out


def migrate_state_dict(
    old_sd: dict[str, torch.Tensor],
    new_sd: dict[str, torch.Tensor],
    goal_order: list[str],
    goal_index: dict[str, int],
) -> dict[str, torch.Tensor]:
    """Best-effort transplant of ``old_sd`` onto ``new_sd``'s shapes.

    Returns a new dict shaped like ``new_sd``; unmatched / brand-new params
    keep new_sd's (freshly-initialized) values. Raises on any shape mismatch
    this module doesn't know how to reconcile (i.e. not the goal one-hot).
    """
    old_goal_order = _old_goal_order(goal_order)
    merged = dict(new_sd)
    copied = 0
    remapped = 0
    skipped_new = 0
    skipped_shape: list[str] = []

    for key, new_tensor in new_sd.items():
        if key not in old_sd:
            skipped_new += 1
            continue
        old_tensor = old_sd[key]
        if old_tensor.shape == new_tensor.shape:
            merged[key] = old_tensor.clone()
            copied += 1
        elif key in _VECTOR_MLP_IN_KEYS:
            merged[key] = _remap_goal_columns(
                old_tensor, new_tensor, old_goal_order, goal_index
            )
            remapped += 1
        else:
            skipped_shape.append(
                f"{key}: old={tuple(old_tensor.shape)} new={tuple(new_tensor.shape)}"
            )

    if skipped_shape:
        raise ValueError(
            "Unhandled shape mismatch(es) — this checkpoint differs by more "
            "than the goal one-hot, can't auto-migrate:\n  "
            + "\n  ".join(skipped_shape)
        )

    print(
        f"Migrated policy weights: {copied} params copied unchanged, "
        f"{remapped} goal-layer(s) remapped by name, "
        f"{skipped_new} new param(s) left at fresh init."
    )
    return merged
