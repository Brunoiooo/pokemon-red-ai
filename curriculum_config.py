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
    visited_maps: set[int] | None = None,
    weights: dict[str, float] | None = None,
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

    ``visited_maps`` optionally restricts candidates to goals whose home map
    (event_graph.py's per-event map_id, or a badge's anchor event's -- see
    _goal_map_id) is in the set -- e.g. MilestoneCallback passes every map_id
    any episode has ever touched this training run, so the pick doesn't hand
    out a goal on a map nothing has reached yet. Falls back to the
    unrestricted pool if that would leave nothing (e.g. the set is still
    empty very early on).

    When ``prefer_discovered`` is True (default), candidates are further
    restricted to goals that are either already discovered -- i.e. have
    their own saves/<goal>/checkpoint.state, meaning some episode has
    actually reached it before, see
    PokemonRedEnv._save_milestone_checkpoints -- or whose EVENT_GRAPH
    parents are all already discovered (see _parents_satisfied), meaning
    whatever leads up to this goal has already happened at least once even
    if the goal itself hasn't been hit yet. This keeps training from being
    handed a goal with no known path to it (or into it) yet. Falls back to
    the (map-filtered) candidate pool otherwise (e.g. very early on, before
    anything's been discovered).

    ``weights`` optionally biases the pick within the (already-filtered)
    candidate pool -- e.g. MilestoneCallback passes a per-goal
    inverse-hit-count so goals the policy rarely lands on (still hard / not
    yet mastered) get sampled more often than ones it clears constantly.
    This is what lets a practice order emerge from what the model has
    actually learned, instead of every not-yet-satisfied goal being equally
    likely forever. A candidate missing from ``weights`` (never seen yet)
    defaults to weight 1.0, same as an already-seen-once goal, so novel
    goals aren't starved relative to ones with one recorded hit. Omit for a
    plain uniform pick.
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
    if visited_maps is not None:
        on_visited_map = [g for g in candidates if _goal_map_id(g) in visited_maps]
        if on_visited_map:
            candidates = on_visited_map
    if prefer_discovered:
        eligible = [
            g for g in candidates if _save_exists(g) or _parents_satisfied(g)
        ]
        if eligible:
            candidates = eligible
    if weights:
        w = [weights.get(g, 1.0) for g in candidates]
        if sum(w) > 0:
            return rng.choices(candidates, weights=w, k=1)[0]
    return rng.choice(candidates)


def stage_for_goal(goal: str) -> str:
    """First curriculum stage that targets ``goal`` (fallback: STAGE_ORDER[0])."""
    if goal in CURRICULUM:
        return goal
    return STAGE_ORDER[0]
