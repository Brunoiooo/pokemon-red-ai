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
pick instead, optionally biased by pick_new_goal's ``weights`` arg (see
MilestoneCallback._goal_hit_counts) so goals the policy rarely lands on get
sampled more than ones it already clears constantly -- a practice order that
emerges from what's been learned, still with no fixed sequence behind it.
Badges (no bit index of their own) are still anchored next to their
corresponding "beat gym leader" event purely for STAGE_ORDER's list
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
import warnings

from pokemon import event_constants as _event_constants
from pokemon import event_graph as _event_graph
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

# Fallback stage/goal for any --stage value that resolve_stage_name can't
# place (not a live goal name, not a recognized _LEGACY_STAGE_ALIASES key --
# e.g. the stale "stage_left_house" that used to be every CLI's --stage
# default: it has no wEventFlags equivalent at all, see
# _LEGACY_STAGE_ALIASES' docstring). Previously this silently substituted
# STAGE_ORDER[0] -- an arbitrary "lowest wEventFlags bit index" pick that
# resolved to EVENT_GOT_TOWN_MAP, a goal with no saves/ checkpoint and no
# geographic relationship to a fresh "start" save, so every episode run
# under it wasted its whole length wandering toward an unreachable goal and
# eating reward_active_map_presence's off-goal-map penalty the entire time.
# EVENT_GOT_STARTER is the actual earliest real milestone (picking a starter
# in Oak's Lab), already checkpointed from minutes into the game, so an
# episode assigned this can actually make progress instead of just being
# punished for existing.
DEFAULT_STAGE = "EVENT_GOT_STARTER"

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


def _graph_key(goal: str) -> str:
    """EVENT_GRAPH lookup key for ``goal`` -- badges have no entry of their
    own (event_graph.py only knows about wEventFlags bits), so they borrow
    their anchor event's (see _BADGE_ANCHOR_EVENT / _order_key)."""
    return _BADGE_ANCHOR_EVENT.get(goal, goal)


def _goal_map_id(goal: str) -> int | None:
    """Map a goal is set on, per event_graph.py's static analysis -- None if
    ``goal`` (or its badge anchor) has no EVENT_GRAPH entry at all."""
    info = _event_graph.EVENT_GRAPH.get(_graph_key(goal))
    return info["map_id"] if info else None


def _parents_satisfied(goal: str) -> bool:
    """Whether every EVENT_GRAPH parent of ``goal`` that's itself a tracked
    GOAL_CANDIDATE has already been reached by some episode this training
    run (i.e. has its own saves/<parent>/checkpoint.state -- see
    _save_exists). Parents outside GOAL_CANDIDATES (blacklisted/dead/cyclic,
    see event_graph.AUTO_BLACKLIST_EVENTS) can't be checked this way and are
    treated as satisfied, same as goals with no EVENT_GRAPH entry (badges
    with no anchor, or events event_graph never saw) -- there's nothing to
    gate on. A goal with no parents at all (event_graph ROOT_EVENTS) is
    trivially satisfied too.
    """
    info = _event_graph.EVENT_GRAPH.get(_graph_key(goal))
    if info is None:
        return True
    return all(_save_exists(p) for p in info["parents"] if p in GOAL_CANDIDATES)


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
# in Data.py's git history) — those fall through to resolve_stage_name's own
# DEFAULT_STAGE fallback (with a warning) instead of a fabricated mapping.
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
    nearest new equivalent (see _LEGACY_STAGE_ALIASES). Anything else --
    e.g. a dead pre-generic-goal id with no equivalent at all, like the old
    "stage_left_house" default -- previously fell through unchanged and let
    every caller's own ``CURRICULUM.get(stage, CURRICULUM[DEFAULT_STAGE])``
    silently substitute an arbitrary goal (whatever happens to sort first by
    wEventFlags bit index). That goal had no checkpoint and no geographic
    relationship to a fresh save, so an episode assigned it wasted its whole
    length chasing an unreachable target. Warn loudly and resolve to
    DEFAULT_STAGE instead, so callers no longer need their own fallback."""
    if stage in CURRICULUM:
        return stage
    resolved = _LEGACY_STAGE_ALIASES.get(stage)
    if resolved is not None:
        return resolved
    warnings.warn(
        f"resolve_stage_name: unrecognized stage {stage!r}, falling back to "
        f"DEFAULT_STAGE ({DEFAULT_STAGE!r}) instead of an arbitrary goal",
        stacklevel=2,
    )
    return DEFAULT_STAGE


def _save_exists(name: str) -> bool:
    return os.path.isfile(f"saves/{name}/checkpoint.state")


def resolve_checkpoint(name: str) -> str:
    if _save_exists(name):
        return name
    return "start"


def goal_status(goal: str) -> str:
    """"mastered" (has its own saves/<goal>/checkpoint.state), "practicable"
    (no checkpoint yet, but every EVENT_GRAPH parent that's a goal candidate
    already is mastered -- the same eligibility prefer_discovered/
    _parents_satisfied grants a goal even at 0 hits), or "locked" (neither --
    pick_new_goal won't hand this one out yet). Used by the --heatmap side
    panel (PositionHeatmap.py's goal saturation tree) to mark each goal with
    a green/yellow dot."""
    if _save_exists(goal):
        return "mastered"
    if _parents_satisfied(goal):
        return "practicable"
    return "locked"


def get_goal_for_stage(stage: str) -> str:
    stage = resolve_stage_name(stage)
    cfg = CURRICULUM.get(stage, CURRICULUM[DEFAULT_STAGE])
    return cfg["goal"]


def get_stage_max_steps(stage: str) -> int:
    stage = resolve_stage_name(stage)
    cfg = CURRICULUM.get(stage, CURRICULUM[DEFAULT_STAGE])
    return int(cfg["max_steps"])


def get_curriculum_saves(stage: str) -> list[str]:
    """Ordered list of save dirs for curriculum mix (earlier … current).

    "start" is always the first (earlier) entry, even once the stage's own
    checkpoint exists -- see PokemonRedEnv._pick_save / --curriculum-mix,
    which resets from curriculum_saves[:-1] with probability curriculum_mix
    and from curriculum_saves[-1] (the frontier checkpoint) otherwise.
    Without this, curriculum_mix had nothing earlier than the frontier to
    ever mix in, so episodes never reset from the true game start once a
    stage's checkpoint existed. If the stage isn't mastered yet, every entry
    resolves to "start" and dedup collapses this to a single-element list
    (no mixing -- see PokemonRedEnv._pick_save's len(...) > 1 guard).
    """
    stage = resolve_stage_name(stage)
    cfg = CURRICULUM.get(stage, CURRICULUM[DEFAULT_STAGE])
    saves: list[str] = ["start"]
    for name in cfg.get("earlier", []):
        saves.append(resolve_checkpoint(name))
    saves.append(resolve_checkpoint(cfg["checkpoint"]))
    saves.append(resolve_checkpoint(stage))
    seen: set[str] = set()
    out: list[str] = []
    for s in saves:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


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
    weights: dict[str, float] | None = None,
    rng: random.Random | None = None,
) -> str | None:
    """Random pick of an already-mastered goal (has its own
    saves/<goal>/checkpoint.state -- see _save_exists) to start a fresh
    episode/leg from.

    This is a "which checkpoint do we practice forward from" pick, not "aim
    for this specific target": reward_generic_progress already pays out for
    *every* GOAL_CANDIDATES event/badge newly satisfied in an episode,
    regardless of which goal was assigned, and get_stage_max_steps is a flat
    constant for every goal too -- the assigned goal has never actually
    gated reward or episode length. Restricting candidates to _save_exists
    goals keeps the pick meaningful: every candidate is somewhere a real
    episode has actually stood, so resolve_checkpoint(nxt) always resolves
    to a real, reachable state, never silently falling back to "start".

    Checkpoints are written by deterministic eval rollouts only (see
    train_ppo.py's eval_env / MaskableEvalCallback and
    PokemonRedEnv._save_milestone_checkpoints), not noisy exploratory
    training episodes -- so "mastered" means the policy demonstrably reaches
    it under its current (near-)greedy behavior, not that one lucky training
    rollout stumbled into it once.

    Used to gate on EVENT_GRAPH parents / a training run's visited maps to
    guess whether an unreached goal was "probably reachable" -- dropped
    because event_graph's parent derivation only sees wEventFlags gates, not
    the geography needed to actually walk there (e.g. a goal three towns
    away with no in-game event dependency at all was "eligible" from step
    one). Restricting to _save_exists sidesteps that guesswork entirely.

    ``weights`` optionally biases the pick within the candidate pool -- e.g.
    MilestoneCallback passes a per-goal inverse-hit-count so goals the
    policy rarely starts from get sampled more often than ones it's already
    drilled constantly. This is what lets a practice order emerge from what
    the model has actually learned, instead of every mastered goal being
    equally likely forever. A candidate missing from ``weights`` (never seen
    yet) defaults to weight 1.0, same as an already-seen-once goal, so
    novel goals aren't starved relative to ones with one recorded hit. Omit
    for a plain uniform pick.

    None if nothing is mastered yet (very early in a fresh run, before the
    first eval pass has confirmed any goal) -- callers should leave the
    current assignment alone in that case.
    """
    rng = rng or random
    candidates = [
        g
        for g in STAGE_ORDER
        if CURRICULUM.get(g, {}).get("enabled", True) and _save_exists(g)
    ]
    if not candidates:
        return None
    if weights:
        w = [weights.get(g, 1.0) for g in candidates]
        if sum(w) > 0:
            return rng.choices(candidates, weights=w, k=1)[0]
    return rng.choice(candidates)


def stage_for_goal(goal: str) -> str:
    """First curriculum stage that targets ``goal`` (fallback: DEFAULT_STAGE)."""
    if goal in CURRICULUM:
        return goal
    warnings.warn(
        f"stage_for_goal: unrecognized goal {goal!r}, falling back to "
        f"DEFAULT_STAGE ({DEFAULT_STAGE!r})",
        stacklevel=2,
    )
    return DEFAULT_STAGE
