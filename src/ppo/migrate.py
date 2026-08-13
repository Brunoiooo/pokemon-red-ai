"""Weight-surgery helper for resuming PPO training after the observation
vector's shape changes: either the goal one-hot block (see
curriculum_config.GOAL_ORDER) or a named feature block inserted elsewhere
in the flat "vector" observation (see NEW_BASE_BLOCKS_SINCE below).

SB3's PPO.load() requires an exact match between a checkpoint's saved tensors
and the live policy's parameter shapes. Two things can break that for the
*_features_extractor.vector_mlp.0 layer's input dimension:

1. GOAL_ORDER growing or shrinking (a curriculum goal added/removed) — the
   goal one-hot is always appended last to the vector (see
   env.pokemon_red_env._torch_inputs_to_obs), so only the tail shifts.
2. A new named feature block spliced into the *base* (pre-goal) portion of
   the vector, e.g. Data.dialog_id_visit_grid's per-map histogram — this
   shifts every column after its insertion point.

migrate_state_dict() copies every parameter whose shape is unchanged as-is,
and remaps just that one layer's columns by *name* (not raw position), so
neither where a goal sits in GOAL_ORDER nor where a base block was spliced in
matters. Columns for goals/blocks that did not exist in the checkpoint keep
the freshly-initialized model's random init — the policy has to learn those
from scratch, same as any brand-new feature always would. Columns for goals
the checkpoint has but the live curriculum no longer does are discarded.
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

# Fixed-width feature blocks spliced into the *base* (pre-goal) portion of
# the flat "vector" observation since older checkpoints were trained. Extend
# this (never remove past entries) whenever env.pokemon_red_env's
# _VECTOR_FLOAT_KEYS/_ID_SCALAR_KEYS/_ID_SEQ_KEYS gains a new entry that
# isn't just appended after everything else.
#
# "offset" is this block's insertion point measured in *skeleton* columns —
# i.e. the width of every _VECTOR_FLOAT_KEYS/_ID_SCALAR_KEYS/_ID_SEQ_KEYS
# entry that existed before any block in this list was ever added (798 as of
# 2026-08-10, matching env.pokemon_red_env's old _BASE_VECTOR_DIM). Blocks do
# not consume skeleton columns, so later blocks' offsets are also measured
# against that same fixed skeleton, not against each other's shifted output
# position — see _skeleton_abs_start/_block_abs_start.
#
# A checkpoint may or may not already have any given entry — same "may or
# may not have it" combinatorics as NEW_GOALS_SINCE below, resolved the same
# way (try every subset, keep whichever reproduces the checkpoint's actual
# tensor width).
NEW_BASE_BLOCKS_SINCE: list[dict] = [
    {
        "name": "dialog_id_visit_counts",
        "width": 256,
        "offset": 733,  # end of the "party" block, right before "map_id"
        "note": (
            "2026-08-10: per-map dialog_id byte-histogram added to the "
            "observation (see Data.dialog_id_visit_grid)."
        ),
    },
    {
        "name": "map_id_visit_counts",
        "width": 256,
        # Same skeleton offset as dialog_id_visit_counts (both splice in
        # right before "map_id"); declared after it in this list so the
        # offset tie-break in _block_abs_start orders it right after.
        "offset": 733,
        "note": (
            "2026-08-11: episode-wide map_id byte-histogram added to the "
            "observation (see Data.map_id_visit_grid)."
        ),
    },
    {
        "name": "reward_component_sums",
        # Fixed width matching len(Data.REWARD_COMPONENT_NAMES) as of
        # 2026-08-13. Hardcoded rather than imported: this must stay pinned
        # to the width this block actually had in checkpoints from that date,
        # not track future changes to REWARD_COMPONENT_NAMES (a further
        # change there needs its own new NEW_BASE_BLOCKS_SINCE entry, same as
        # any other block here).
        "width": 18,
        # Same skeleton offset as dialog_id_visit_counts/map_id_visit_counts
        # (all three splice in right before "map_id"); declared last so the
        # offset tie-break orders it after both.
        "offset": 733,
        "note": (
            "2026-08-13: cumulative per-episode sum of each named reward "
            "sub-component (incl. total) added to the observation (see "
            "Data.reward_component_vector / REWARD_COMPONENT_NAMES)."
        ),
    },
    {
        "name": "map_budget_progress",
        # Episode-wide per-map_id histogram, same 256-wide byte-range shape
        # as dialog_id_visit_counts/map_id_visit_counts above (not a single
        # scalar -- see Data.map_budget_progress).
        "width": 256,
        # Same skeleton offset as the three blocks above (all four splice in
        # right before "map_id"); declared after them so the offset
        # tie-break orders it right after, matching env.pokemon_red_env's
        # _VECTOR_FLOAT_KEYS tuple order.
        "offset": 733,
        "note": (
            "2026-08-13: per-map_id world_map_step_counts / "
            "map_truncate_budget histogram added to the observation (see "
            "Data.map_budget_progress)."
        ),
    },
    {
        "name": "stuck_tile_progress",
        "width": 1,
        "offset": 733,
        "note": (
            "2026-08-13: current tile's visited_positions / max_useless_ticks "
            "(stuck_tile truncate fuse) ratio added to the observation (see "
            "Data.stuck_tile_progress)."
        ),
    },
    {
        "name": "loop_streak_progress",
        "width": 1,
        "offset": 733,
        "note": (
            "2026-08-13: loop_streak / max_loop_streak (loop_streak truncate "
            "fuse) ratio added to the observation (see "
            "Data.loop_streak_progress)."
        ),
    },
]


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


def _base_block_subsets() -> list[list[dict]]:
    """Every subset of NEW_BASE_BLOCKS_SINCE a checkpoint might already have."""
    blocks = NEW_BASE_BLOCKS_SINCE
    return [
        [b for b, keep in zip(blocks, mask) if keep]
        for mask in itertools.product([True, False], repeat=len(blocks))
    ]


def _skeleton_abs_start(skel_pos: int, blocks: list[dict]) -> int:
    """Absolute column where skeleton coordinate `skel_pos` lands once
    `blocks` are spliced in — counts every block at or before this point,
    since a skeleton run resuming at `skel_pos` comes after them."""
    return skel_pos + sum(b["width"] for b in blocks if b["offset"] <= skel_pos)


def _block_abs_start(block: dict, blocks: list[dict]) -> int:
    """Absolute start column of `block` itself once `blocks` are spliced in —
    counts only strictly-earlier blocks (by offset, then declaration order),
    not `block` or same-offset blocks declared after it."""
    idx = NEW_BASE_BLOCKS_SINCE.index(block)
    extra = sum(
        b["width"]
        for b in blocks
        if b is not block
        and (
            b["offset"] < block["offset"]
            or (
                b["offset"] == block["offset"]
                and NEW_BASE_BLOCKS_SINCE.index(b) < idx
            )
        )
    )
    return block["offset"] + extra


def _resolve_old_layout(
    old_dim: int, new_dim: int, goal_order: list[str], goal_index: dict[str, int]
) -> tuple[list[str], list[dict]]:
    """Pick the candidate old goal order + base-block set matching this checkpoint.

    Raises loudly (rather than silently mis-mapping columns) if no candidate
    reproduces the checkpoint's actual tensor width, which means
    NEW_GOALS_SINCE / REMOVED_GOALS_SINCE / NEW_BASE_BLOCKS_SINCE don't fully
    account for it.
    """
    skeleton_width = (
        new_dim - len(goal_index) - sum(b["width"] for b in NEW_BASE_BLOCKS_SINCE)
    )
    matches = []
    for goal_cand in _old_goal_order_candidates(goal_order):
        for block_cand in _base_block_subsets():
            base_old = skeleton_width + sum(b["width"] for b in block_cand)
            if base_old + len(goal_cand) == old_dim:
                matches.append((goal_cand, block_cand))
    if not matches:
        raise ValueError(
            f"Feature width doesn't line up for any reconstruction of this "
            f"checkpoint's layout (old_dim={old_dim}, skeleton width="
            f"{skeleton_width}) — NEW_GOALS_SINCE / REMOVED_GOALS_SINCE / "
            f"NEW_BASE_BLOCKS_SINCE in ppo/migrate.py probably don't fully "
            f"account for this checkpoint's vintage, or the mismatch isn't "
            f"just those known changes."
        )
    distinct = {
        (tuple(g), tuple(b["name"] for b in blk)) for g, blk in matches
    }
    if len(distinct) > 1:
        print(
            f"  (layout reconstruction is ambiguous — {len(matches)} "
            f"distinct candidates match this checkpoint's width; using "
            f"{matches[0]})"
        )
    return matches[0]


def _remap_vector_mlp_columns(
    old_weight: torch.Tensor,
    new_weight: torch.Tensor,
    old_goal_order: list[str],
    old_base_blocks: list[dict],
    goal_index: dict[str, int],
) -> torch.Tensor:
    skeleton_width = (
        new_weight.shape[1]
        - len(goal_index)
        - sum(b["width"] for b in NEW_BASE_BLOCKS_SINCE)
    )
    out = new_weight.clone()

    # Skeleton columns (every _VECTOR_FLOAT_KEYS/_ID_SCALAR_KEYS/_ID_SEQ_KEYS
    # entry that predates NEW_BASE_BLOCKS_SINCE) copy straight across, split
    # only at points where a base block was spliced in.
    boundaries = sorted(
        {0, skeleton_width} | {b["offset"] for b in NEW_BASE_BLOCKS_SINCE}
    )
    for s0, s1 in zip(boundaries, boundaries[1:]):
        old_start = _skeleton_abs_start(s0, old_base_blocks)
        new_start = _skeleton_abs_start(s0, NEW_BASE_BLOCKS_SINCE)
        width = s1 - s0
        out[:, new_start : new_start + width] = old_weight[
            :, old_start : old_start + width
        ]

    # Base blocks the checkpoint already had copy across too; ones it
    # predates are left at the freshly-initialized model's random values.
    dropped_blocks = []
    for block in NEW_BASE_BLOCKS_SINCE:
        new_start = _block_abs_start(block, NEW_BASE_BLOCKS_SINCE)
        if block in old_base_blocks:
            old_start = _block_abs_start(block, old_base_blocks)
            out[:, new_start : new_start + block["width"]] = old_weight[
                :, old_start : old_start + block["width"]
            ]
        else:
            dropped_blocks.append(block["name"])
    if dropped_blocks:
        print(
            f"  (base feature(s) new to this checkpoint, left at fresh init: "
            f"{dropped_blocks})"
        )

    # Goal one-hot tail — same by-name remap as before, just at the
    # (possibly base-block-shifted) tail offset.
    base_old = skeleton_width + sum(b["width"] for b in old_base_blocks)
    base_new = skeleton_width + sum(b["width"] for b in NEW_BASE_BLOCKS_SINCE)
    dropped_goals = []
    for i, goal_name in enumerate(old_goal_order):
        j = goal_index.get(goal_name)
        if j is None:
            dropped_goals.append(goal_name)
            continue
        out[:, base_new + j] = old_weight[:, base_old + i]
    if dropped_goals:
        print(f"  (goal(s) no longer in curriculum, weights discarded: {dropped_goals})")
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
    this module doesn't know how to reconcile (i.e. not a known goal or base
    block change — see NEW_GOALS_SINCE / REMOVED_GOALS_SINCE /
    NEW_BASE_BLOCKS_SINCE).
    """
    merged = dict(new_sd)
    copied = 0
    remapped = 0
    skipped_new = 0
    skipped_shape: list[str] = []
    old_layout: tuple[list[str], list[dict]] | None = None

    for key, new_tensor in new_sd.items():
        if key not in old_sd:
            skipped_new += 1
            continue
        old_tensor = old_sd[key]
        if old_tensor.shape == new_tensor.shape:
            merged[key] = old_tensor.clone()
            copied += 1
        elif key in _VECTOR_MLP_IN_KEYS:
            if old_layout is None:
                old_layout = _resolve_old_layout(
                    old_tensor.shape[1], new_tensor.shape[1], goal_order, goal_index
                )
            old_goal_order, old_base_blocks = old_layout
            merged[key] = _remap_vector_mlp_columns(
                old_tensor, new_tensor, old_goal_order, old_base_blocks, goal_index
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
