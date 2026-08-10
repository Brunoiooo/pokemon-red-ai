"""Weight-surgery helper for resuming PPO training after the observation
vector's goal one-hot block changes size (see curriculum_config.GOAL_ORDER).

SB3's PPO.load() requires an exact match between a checkpoint's saved tensors
and the live policy's parameter shapes. Growing or shrinking GOAL_ORDER (a
curriculum goal added or removed) breaks that for exactly one input dimension
of the *_features_extractor.vector_mlp.0 layer — the goal one-hot is appended
last to the flat "vector" observation (see env.pokemon_red_env
._torch_inputs_to_obs), so every feature before it keeps its old column
index; only the goal block's width and the positions of goals inside it
change.

migrate_state_dict() copies every parameter whose shape is unchanged as-is,
and remaps just that one layer's goal columns by *name* (not position), so
where a goal was inserted into (or dropped from) GOAL_ORDER doesn't matter.
Columns for goals that did not exist in the checkpoint keep the
freshly-initialized model's random init — the policy has to learn those from
scratch, same as any brand-new goal always would. Columns for goals the
checkpoint has but the live curriculum no longer does are simply discarded.
"""
from __future__ import annotations

import itertools

import torch

# Goals introduced after older checkpoints were trained. Extend this set
# (never remove past entries) whenever GOAL_ORDER grows, so --migrate keeps
# working against checkpoints saved at any earlier point in training history.
# A checkpoint may or may not already have any given entry's column — there's
# no stored per-checkpoint timestamp, so _old_goal_order_candidates tries
# both ways for every entry and _resolve_old_goal_order picks whichever
# combination reproduces the checkpoint's actual tensor width, raising
# loudly if none do rather than silently mis-mapping columns.
#
# "pc_tutorial" is a plain string, not a live GOAL_* constant: the stage was
# later removed from curriculum_config entirely, so any checkpoint that still
# carries that column now hits the "goal no longer in curriculum, weights
# discarded" path in _remap_goal_columns instead of being remapped.
NEW_GOALS_SINCE: dict[str, str] = {
    "pc_tutorial": "2026-08-06: pc_tutorial stage inserted at STAGE_ORDER[0]",
    "gave_parcel": (
        "2026-08-09: gave_parcel stage inserted between "
        "oaks_parcel and town_map"
    ),
    "champion": "2026-08-09: champion stage appended after all_badges",
}

# Goals dropped from the live GOAL_ORDER after older checkpoints were
# trained — the mirror image of NEW_GOALS_SINCE, same "may or may not have
# it" handling. Extend this (never remove past entries) whenever a goal is
# deleted from curriculum_config: a candidate that keeps a dropped goal
# reinserts it immediately after ``after``, so ``after`` must name a goal
# that is still (and was already) in GOAL_ORDER.
REMOVED_GOALS_SINCE: dict[str, dict[str, str]] = {
    "oaks_lab": {
        "after": "route1_entry",
        "note": (
            "2026-08-10: oaks_lab stage folded away — reaching the Lab is a "
            "forced walk-in cutscene right after route1_entry, not something "
            "the agent navigates, so it never needed its own goal/reward."
        ),
    },
}

_VECTOR_MLP_IN_KEYS = (
    "features_extractor.vector_mlp.0.weight",
    "pi_features_extractor.vector_mlp.0.weight",
    "vf_features_extractor.vector_mlp.0.weight",
)


def _old_goal_order_candidates(goal_order: list[str]) -> list[list[str]]:
    """Every plausible reconstruction of an older checkpoint's goal order.

    A given checkpoint may postdate any subset of NEW_GOALS_SINCE (already
    has that goal's column) and predate any subset of REMOVED_GOALS_SINCE
    (still has a column for a goal the live curriculum has since dropped) —
    there's no stored per-checkpoint timestamp to resolve this precisely
    (e.g. a checkpoint trained after gave_parcel/champion were added but
    before oaks_lab was removed needs BOTH kept, not the blanket "predates
    everything" assumption a single fixed order would make). Try every
    combination and let _resolve_old_goal_order pick whichever reproduces
    the checkpoint's actual tensor width.
    """
    added = list(NEW_GOALS_SINCE)
    removed = list(REMOVED_GOALS_SINCE)
    seen: set[tuple[str, ...]] = set()
    candidates: list[list[str]] = []
    for strip_mask in itertools.product([True, False], repeat=len(added)):
        stripped = {name for name, strip in zip(added, strip_mask) if strip}
        base = [g for g in goal_order if g not in stripped]
        for keep_mask in itertools.product([True, False], repeat=len(removed)):
            order = list(base)
            for name, keep in zip(removed, keep_mask):
                if not keep:
                    continue
                anchor = REMOVED_GOALS_SINCE[name]["after"]
                if anchor not in order:
                    continue
                order.insert(order.index(anchor) + 1, name)
            key = tuple(order)
            if key not in seen:
                seen.add(key)
                candidates.append(order)
    return candidates


def _resolve_old_goal_order(
    old_dim: int, new_dim: int, goal_order: list[str], goal_index: dict[str, int]
) -> list[str]:
    """Pick the candidate old goal order whose width matches this checkpoint.

    Raises loudly (rather than silently mis-mapping columns) if no candidate
    reproduces the checkpoint's actual non-goal feature width, which means
    NEW_GOALS_SINCE / REMOVED_GOALS_SINCE don't fully account for it.
    """
    base_new = new_dim - len(goal_index)
    matches = [
        cand
        for cand in _old_goal_order_candidates(goal_order)
        if old_dim - len(cand) == base_new
    ]
    if not matches:
        raise ValueError(
            f"Non-goal feature width doesn't line up for any reconstruction "
            f"of this checkpoint's goal order (old_dim={old_dim}, expected "
            f"non-goal width={base_new}) — NEW_GOALS_SINCE / "
            f"REMOVED_GOALS_SINCE in ppo/migrate.py probably don't fully "
            f"account for this checkpoint's vintage, or the mismatch isn't "
            f"just the goal one-hot changing."
        )
    if len({tuple(m) for m in matches}) > 1:
        print(
            f"  (goal-order reconstruction is ambiguous — {len(matches)} "
            f"distinct candidates match this checkpoint's width; using "
            f"{matches[0]})"
        )
    return matches[0]


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
    merged = dict(new_sd)
    copied = 0
    remapped = 0
    skipped_new = 0
    skipped_shape: list[str] = []
    old_goal_order: list[str] | None = None

    for key, new_tensor in new_sd.items():
        if key not in old_sd:
            skipped_new += 1
            continue
        old_tensor = old_sd[key]
        if old_tensor.shape == new_tensor.shape:
            merged[key] = old_tensor.clone()
            copied += 1
        elif key in _VECTOR_MLP_IN_KEYS:
            if old_goal_order is None:
                old_goal_order = _resolve_old_goal_order(
                    old_tensor.shape[1], new_tensor.shape[1], goal_order, goal_index
                )
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
