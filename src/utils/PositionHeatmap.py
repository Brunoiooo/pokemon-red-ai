"""Live exploration heatmap: rolling window of visited (map, x, y) ticks,
averaged per run, rendered in its own process so it never blocks
training/eval. Opt-in via --heatmap on train_ppo.py / run_eval_ppo.py.

Data flow: PokemonRedEnv snapshots Data.visited_positions / direction_counts
/ map_transitions / reward_sums / battle_outcome_counts / milestone_hit_counts
/ dialog_hit_counts / truncate_hit_counts / position_visit_counts at every
"run" boundary (episode end, or a mid-episode curriculum-leg clear) into
info["heatmap_positions"/"heatmap_directions"/"heatmap_transitions"/
"heatmap_rewards"/"heatmap_battle_outcomes"/"heatmap_milestones"/
"heatmap_dialogs"/"heatmap_truncations"/"heatmap_visit_counts"/
"heatmap_steps"].
A callback (training) or the eval loop (eval) forwards those snapshots
through a multiprocessing.Queue to _run_window, which runs in a separate
process and owns the matplotlib window.

`run_eval_ppo.py --batch` always feeds RollingHeatmapAggregator(s) directly
(window_frames=None — an unbounded pooled total, not a rolling window) and
calls save_heatmap_images() once at the end to render static PNGs headlessly
— a run that size needs a permanent artifact, not just a window. Pass
--heatmap alongside --batch and it *also* opens the same live _run_window as
plain eval, fed the same snapshots as they arrive; the difference from plain
eval is only that it's left open at the end instead of auto-closed, since a
300k-run batch is exactly the case where you want to sit and read the result
afterward.

Two view modes (toggle with 'c'):
  - single map: one map_id's grid + direction arrows, cycled with left/right.
  - combined: every reachable map auto-stitched into one canvas. Geometry
    for outdoor Kanto comes from pokemon.map_connections.overworld_offsets —
    exact (x, y) world-cell offsets derived straight from reference/pokered's
    `connection north/south/west/east` map-header data (the same alignment
    the game engine itself uses to keep wXCoord/wYCoord continuous across an
    edge), so the *whole* connected outdoor world is fully built and shown
    the instant the window opens or combined mode is selected — no episode
    data needed for the stitching itself, only for which tiles then get
    colored in. Interiors/caves/etc. have no `connection` data at all (they
    only ever warp, which isn't a geometric relationship — see
    tools/render_route.py's module docstring), so those still fall back to
    RollingHeatmapAggregator._empirical_offsets: inferred from observed
    step deltas across map_id boundaries, best-effort (a door warp can
    *look* like a consistent edge connection and get stitched in at a
    nonsense position — there's no memory-flag distinguishing "walked
    across an edge" from "used a door"). The static graph always wins on
    conflict (see global_offsets). Maps that still couldn't be placed
    either way aren't dropped from view: up to _MAX_SIDE_PANELS of them get
    their own small thumbnail in a side column, sharing the main image's
    color scale. Beyond that cap, the remainder are just named in the last
    slot's text instead of rendered.

Map background: every tile grid (single map, combined view, and side-panel
thumbnails alike) is drawn on top of the actual map pixel art — decoded
straight from reference/pokered's tile sheets/blocksets/.blk layouts by
pokemon.map_render, at _OVERLAY_ALPHA so the underlying walls/paths/grass/
buildings stay legible through the metric color, with NaN (unvisited)
tiles fully transparent so they show the plain map with no tint at all.
The combined view stitches per-map art using the same offsets (static
where available, empirical otherwise — see the view-modes section above)
as the data grid, so both align pixel-for-pixel. A map with no resolvable
source assets (a handful of unused "_COPY" placeholder map_ids) just falls
back to the plain dark _BG_FACECOLOR, same as every metric looked before
this background existed.

Eight color metrics (cycle with 'r'), independent of the view mode:
  - ticks: avg ticks/run per tile (time spent) — the original view.
  - reward: avg reward/run per tile — where reward is actually earned, not
    just where time is spent. A battle-won/lost payout is attributed to the
    grass/trainer tile that started the fight even though the fight itself
    has no world position, so farming loops (grind a tile for repeat battle
    reward, spam a menu for event-flag reward, etc.) show up as a hot tile
    here even when the ticks view looks unremarkable.
  - winrate: battle win-rate per tile (wins / (wins + losses), a fled battle
    counts as neither), same attribution as reward. Separates "this tile pays
    well" from "this tile pays well because it's an easy fight" — a grass
    tile can be reward-hot and win-rate-low at once (rare wins pay a lot on
    net) or the reverse (frequent small wins).
  - fleerate: successful-flee quality per tile (smart / (smart + coward)) —
    smart means the enemy outleveled the whole party, coward means it didn't.
    Disjoint from winrate (a fled fight is neither a win nor a loss), same
    _MIN_BATTLE_SAMPLES floor.
  - recency: episodes since a tile was last touched, within the current
    window. Not an average — a raw "how long ago" count, for spotting tiles
    the policy has drifted away from recently without waiting for them to
    fall out of the window entirely.
  - dialog_recency: episodes since a tile last triggered a dialog (NPC talk,
    sign, cutscene text), NaN on tiles that have never triggered one even if
    heavily visited. Same "how long ago" shape as recency, but scoped to
    dialog-bearing tiles only — separates "the policy walked through here
    recently" from "the policy actually talked to something here recently".
  - milestones: avg story-milestone hits/run per tile — how much of the
    curriculum's critical path actually lines up with where the agent
    spends time, same attribution as reward.
  - truncations: avg episode-ending truncations/run per tile — where runs
    actually run out (a stuck fuse, a map/step budget, ...; see
    Data.TRUNCATE_CAUSES), not just where time is spent. Same attribution
    as reward/milestones. A camping/farming spot that never gets punished
    stays quiet here even if it's hot on the ticks view; a spot where the
    policy reliably gets stuck shows up bright regardless of how much
    total time it accumulates.
  - visits: avg Data.position_visit_counts/run per tile, divided by
    Data.VISIT_PENALTY_SOFT_THRESHOLD (see _visit_metric_scale) — unlike
    ticks (raw time spent), this is the exact counter reward_anti_loop's
    graduated visit penalty reads, so a value of 1.0 is "at the soft-penalty
    threshold" and 2.0 is "at the hard threshold". Fixed [0, 2] color scale
    like winrate/fleerate (doesn't rescale to the data) since those
    threshold multiples, not the window's hottest tile, are what's worth
    reading off the color directly. Includes the flood-filled "island" bump
    _mark_wild_grass_island adds around a wild encounter's tile, so a grass
    patch discovered via one battle reads as visited even before it's been
    walked.
  Rate metrics (winrate/fleerate) are left blank on tiles with too few
  recorded battles (< _MIN_BATTLE_SAMPLES) rather than shown at a noisy
  100%/0%.

Off-goal map cue (combined and single views, all metrics): any map outside
the active stage goal's small event_graph neighborhood (its own home map
plus inferred parent/child events' home maps — see _allowed_maps_lookup) is
labeled and tinted red. A map-level classification, not a new per-tile
signal, and skipped for the "All" stage pool (no single goal to check
against) or a goal with no event_graph entry (e.g. a badge).

Per-tile numeric value labels (toggle with 'v'), independent of view mode
and metric: prints the exact per-tile value on top of each cell, for
reading precise values instead of eyeballing color. Auto-hidden above
_MAX_VALUE_LABELS cells since that many text artists stalls redraws.

Interactive view: scroll-wheel zooms centered on the cursor, left-click-drag
pans. Both views auto-fit to the currently-discovered extent on every redraw
(so a map's visible area grows as more of it gets explored, and switching
map/stage/mode re-fits) until the user scrolls or drags, at which point the
manual viewport sticks — even as new tiles are discovered elsewhere — until
they switch context again or press '0' to reset the view.

Goal saturation side panel: a scrollable table, embedded next to the stage
dropdown, listing curriculum goals -- restricted to the ones pick_new_goal
could actually hand out right now (parents satisfied and home map already
visited by some episode this run -- see _refresh_goal_saturation; a fully
"locked" goal, or one on a map never reached yet, is left off entirely) --
and how many completed training episodes have hit each so far
(most-practiced first, zero-hit goals greyed out). Reads
saves/goal_hit_counts.json and saves/visited_maps.json directly off disk on
a timer instead of through the queue -- those are MilestoneCallback's
whole-run state (see ppo/callbacks.py), not a per-episode heatmap snapshot,
so this panel stays live even when HeatmapCallback itself has nothing new
to push.
"""
from __future__ import annotations

import queue as _queue_mod
from collections import deque
from multiprocessing import Process, Queue

import numpy as np

# Sentinel telling the visualizer process to shut down cleanly.
STOP = None

_DIRS = ("up", "down", "left", "right")
# Cells need at least this many recorded steps before an arrow is drawn —
# a single pass-through is noise, not a "most common direction".
_MIN_DIRECTION_SAMPLES = 3
# Tiles need at least this many recorded battles (per the relevant pair of
# buckets below) before a rate metric shows a value — one lucky/unlucky
# fight would otherwise read as a solid 100%/0% hotspot.
_MIN_BATTLE_SAMPLES = 3
# metric name -> (numerator bucket, denominator-partner bucket) into
# battle_outcome_counts. winrate and fleerate are both a ratio over one
# disjoint pair of that dict's four buckets (win/loss, smart/coward).
_RATE_METRICS = {"winrate": ("win", "loss"), "fleerate": ("smart", "coward")}
# Above this many cells, per-tile value labels are skipped — matplotlib text
# artists are expensive enough that a big combined view would stall redraws.
_MAX_VALUE_LABELS = 800
# A (map_a, map_b) pair needs at least this many crossing samples, with the
# modal offset winning at least this fraction of them, before it's trusted as
# a real connection for the combined view. A bare >=0.5 majority let a dead
# -even split (e.g. one real edge-crossing delta and one unrelated door-warp
# delta, from as few as 2 samples) get "confirmed" off an arbitrary tie-break
# — raised well past 0.5 so a genuinely mixed pair stays unconfirmed instead
# of picking a winner at random.
_MIN_TRANSITION_VOTES = 3
_MIN_TRANSITION_MAJORITY = 0.75
# Stricter bar for a pair with votes in only ONE direction (see
# _confirmed_transitions) — no reverse measurement exists to cross-check
# against, so demand more samples and a near-unanimous majority instead.
_MIN_TRANSITION_VOTES_SOLO = 6
_MIN_TRANSITION_MAJORITY_SOLO = 0.9
# Max number of unconnected-map thumbnails shown in the combined view's side
# column (live window and batch PNG alike). Kept small since each is a full
# axes/imshow — past this, the last slot becomes a text list of the rest
# instead of one more thumbnail, so an "All"-stage pool with dozens of
# not-yet-connected maps doesn't stall the redraw or bury the window.
_MAX_SIDE_PANELS = 6
# Episodes kept for the rolling "avg party size / level" indicator — a plain
# count-based window (see RollingHeatmapAggregator._party_history), not tied
# to the frame-based eviction everything else here uses.
_PARTY_HISTORY = 50
# Alpha for the metric-color overlay drawn on top of the real map pixel art
# (see pokemon.map_render) — low enough that the underlying tile art (walls,
# paths, grass, buildings) stays readable through the color, high enough
# that the metric itself is still the dominant thing on screen. NaN cells
# use each cmap's set_bad(alpha=0) instead (fully transparent), so unvisited
# tiles show the plain map with no tint at all.
_OVERLAY_ALPHA = 0.6
# Background fill behind/around the map art — shows through on map_ids
# whose source assets couldn't be resolved (pokemon.map_render.render_map_rgb
# returned None; rare, only a handful of unused "_COPY" placeholder maps)
# instead of leaving a blank white gap.
_BG_FACECOLOR = "#111111"


def _visit_metric_scale() -> int:
    """Data.VISIT_PENALTY_SOFT_THRESHOLD -- the "visits" metric divides avg
    visits/run by this, so a value of 1.0 reads as "at the anti-loop soft
    penalty threshold" and 2.0 as "at the hard threshold" (see
    RollingHeatmapAggregator.average_grid/combined_view), instead of an
    unbounded raw count like ticks. Imported lazily (like _map_name_lookup's
    Data import below) so importing this module doesn't drag pyboy/torch
    into the lightweight heatmap-viewer process just to read one constant.
    """
    from pokemon.Data import VISIT_PENALTY_SOFT_THRESHOLD

    return VISIT_PENALTY_SOFT_THRESHOLD


class RollingHeatmapAggregator:
    """Average ticks-spent-per-run, per (map_id, x, y), over the last
    ``window_frames`` steps pooled across every run that fed it. Also tracks
    the dominant walking direction per tile, and (permanently — map
    connectivity doesn't age out like traffic does) empirical map-to-map
    coordinate offsets for the combined view.

    One ``add_episode`` call = one "run": a full episode, or one curriculum
    leg when auto-curriculum clears visited_positions mid-episode. That's
    the unit "average per run" is taken over.

    ``window_frames=None`` disables the rolling window entirely: runs are
    never evicted and never even kept around (no ``_runs`` deque growth), so
    the aggregator becomes a plain unbounded accumulator. Used by the batch
    eval mode, which wants a total over e.g. 300k runs rather than a "recent
    traffic" window — keeping every run's raw position dict for a batch that
    size would be a lot of wasted memory for eviction bookkeeping nothing
    will ever use.
    """

    def __init__(self, window_frames: int | None):
        self.window_frames = None if window_frames is None else int(window_frames)
        self._runs: deque[
            tuple[
                int,
                dict[tuple[int, int, int], int],
                dict[tuple[int, int, int], dict[str, int]],
                dict[tuple[int, int, int], float],
                dict[tuple[int, int, int], dict[str, int]],
                dict[tuple[int, int, int], int],
                dict[tuple[int, int, int], int],
                dict[tuple[int, int, int], int],
            ]
        ] = deque()
        self.total_frames = 0
        # Total add_episode() calls that weren't dropped for being empty —
        # unlike total_frames/run_count this never gets decremented by
        # eviction, so it's the right "N runs" figure for a batch summary.
        self.total_episodes = 0
        # map_id -> {(x, y): summed ticks across runs in the window}
        self.sum_ticks: dict[int, dict[tuple[int, int], int]] = {}
        # map_id -> {(x, y): summed reward across runs in the window}
        self.sum_rewards: dict[int, dict[tuple[int, int], float]] = {}
        # map_id -> {(x, y): {"win"/"loss": count summed across runs in window}}
        self.sum_battle_outcomes: dict[int, dict[tuple[int, int], dict[str, int]]] = {}
        # map_id -> {(x, y): summed story-milestone hit count across runs in window}
        self.sum_milestones: dict[int, dict[tuple[int, int], int]] = {}
        # map_id -> {(x, y): summed truncation-ending count across runs in
        # window} — same piggyback-on-ticks eviction as sum_milestones.
        self.sum_truncations: dict[int, dict[tuple[int, int], int]] = {}
        # map_id -> {(x, y): summed Data.position_visit_counts across runs in
        # window} — same piggyback-on-ticks eviction as sum_milestones. Backs
        # the "visits" metric, normalized against VISIT_PENALTY_SOFT_THRESHOLD
        # (see average_grid/combined_view) so its color scale reads directly
        # as "how many multiples of the anti-loop soft-penalty threshold",
        # not just a raw unbounded count like ticks.
        self.sum_visit_counts: dict[int, dict[tuple[int, int], int]] = {}
        # map_id -> {(x, y): total_episodes value as of the last run that
        # touched this tile} — permanent, never evicted (like
        # transition_votes), so "recency" can keep pointing at how long ago
        # a tile was last touched even after its ticks/reward eviction age
        # out; the "recency" metric's own footprint is the ticks grid, so an
        # entry only actually renders while its tile is still in-window.
        self.last_seen: dict[int, dict[tuple[int, int], int]] = {}
        # map_id -> {(x, y): total_episodes value as of the last run that
        # triggered a dialog on this tile} — same permanent, never-evicted
        # shape as last_seen, but populated only from dialog attribution
        # (see Data.dialog_hit_counts), not every visit. Backs the
        # "dialog_recency" metric.
        self.last_dialog_seen: dict[int, dict[tuple[int, int], int]] = {}
        # map_id -> number of runs in the window that touched this map
        self.run_count: dict[int, int] = {}
        # map_id -> {(x, y): {"up"/"down"/"left"/"right": step count}}
        self.direction_sums: dict[int, dict[tuple[int, int], dict[str, int]]] = {}
        # (from_map, to_map) -> {(dx, dy): vote count} — permanent, never evicted.
        self.transition_votes: dict[tuple[int, int], dict[tuple[int, int], int]] = {}
        # Most recent episode's party size / average level, for the "right
        # now" indicator — and a small bounded history (count-based, not
        # frame-windowed like everything else here: a party snapshot is one
        # scalar per episode, not a per-tile grid, so there's nothing to
        # evict tile-by-tile) for a steadier running average.
        self.last_party_count: int | None = None
        self.last_party_avg_level: float | None = None
        self._party_history: deque[tuple[int, float]] = deque(maxlen=_PARTY_HISTORY)

    def add_episode(
        self,
        positions: dict[tuple[int, int, int], int],
        directions: dict[tuple[int, int, int], dict[str, int]] | None,
        transitions: dict[tuple[int, int], dict[tuple[int, int], int]] | None,
        rewards: dict[tuple[int, int, int], float] | None,
        battle_outcomes: dict[tuple[int, int, int], dict[str, int]] | None,
        milestones: dict[tuple[int, int, int], int] | None,
        dialogs: dict[tuple[int, int, int], int] | None,
        truncations: dict[tuple[int, int, int], int] | None,
        visit_counts: dict[tuple[int, int, int], int] | None,
        steps: int,
        party_count: int | None = None,
        party_avg_level: float | None = None,
    ) -> None:
        if not positions or steps <= 0:
            return
        self.total_episodes += 1
        directions = directions or {}
        rewards = rewards or {}
        battle_outcomes = battle_outcomes or {}
        milestones = milestones or {}
        dialogs = dialogs or {}
        truncations = truncations or {}
        visit_counts = visit_counts or {}
        for (x, y, map_id) in dialogs:
            self.last_dialog_seen.setdefault(map_id, {})[(x, y)] = self.total_episodes
        if party_count is not None and party_avg_level is not None:
            self.last_party_count = party_count
            self.last_party_avg_level = party_avg_level
            self._party_history.append((party_count, party_avg_level))
        if self.window_frames is not None:
            self._runs.append(
                (
                    steps, positions, directions, rewards, battle_outcomes,
                    milestones, truncations, visit_counts,
                )
            )
        self.total_frames += steps
        touched_maps = set()
        for (x, y, map_id), ticks in positions.items():
            touched_maps.add(map_id)
            grid = self.sum_ticks.setdefault(map_id, {})
            grid[(x, y)] = grid.get((x, y), 0) + ticks
            self.last_seen.setdefault(map_id, {})[(x, y)] = self.total_episodes
        for map_id in touched_maps:
            self.run_count[map_id] = self.run_count.get(map_id, 0) + 1
        for (x, y, map_id), dcounts in directions.items():
            cell = self.direction_sums.setdefault(map_id, {}).setdefault(
                (x, y), {d: 0 for d in _DIRS}
            )
            for d, c in dcounts.items():
                cell[d] = cell.get(d, 0) + c
        for (x, y, map_id), r in rewards.items():
            grid = self.sum_rewards.setdefault(map_id, {})
            grid[(x, y)] = grid.get((x, y), 0.0) + r
        for (x, y, map_id), counts in battle_outcomes.items():
            cell = self.sum_battle_outcomes.setdefault(map_id, {}).setdefault(
                (x, y), {"win": 0, "loss": 0}
            )
            for k, c in counts.items():
                cell[k] = cell.get(k, 0) + c
        for (x, y, map_id), n in milestones.items():
            grid = self.sum_milestones.setdefault(map_id, {})
            grid[(x, y)] = grid.get((x, y), 0) + n
        for (x, y, map_id), n in truncations.items():
            grid = self.sum_truncations.setdefault(map_id, {})
            grid[(x, y)] = grid.get((x, y), 0) + n
        for (x, y, map_id), n in visit_counts.items():
            grid = self.sum_visit_counts.setdefault(map_id, {})
            grid[(x, y)] = grid.get((x, y), 0) + n
        for key, votes in (transitions or {}).items():
            dest = self.transition_votes.setdefault(key, {})
            for delta, c in votes.items():
                dest[delta] = dest.get(delta, 0) + c
        if self.window_frames is not None:
            self._evict()

    def _evict(self) -> None:
        while self._runs and self.total_frames > self.window_frames:
            (
                steps, positions, directions, rewards, battle_outcomes,
                milestones, truncations, visit_counts,
            ) = self._runs.popleft()
            self.total_frames -= steps
            touched_maps = set()
            for (x, y, map_id), ticks in positions.items():
                touched_maps.add(map_id)
                grid = self.sum_ticks.get(map_id)
                if grid is None:
                    continue
                remaining = grid.get((x, y), 0) - ticks
                if remaining <= 0:
                    grid.pop((x, y), None)
                else:
                    grid[(x, y)] = remaining
                if not grid:
                    self.sum_ticks.pop(map_id, None)
            for map_id in touched_maps:
                n = self.run_count.get(map_id, 0) - 1
                if n <= 0:
                    self.run_count.pop(map_id, None)
                else:
                    self.run_count[map_id] = n
            for (x, y, map_id), dcounts in directions.items():
                dmap = self.direction_sums.get(map_id)
                cell = dmap.get((x, y)) if dmap is not None else None
                if cell is None:
                    continue
                for d, c in dcounts.items():
                    cell[d] = cell.get(d, 0) - c
                if not any(cell.values()):
                    dmap.pop((x, y), None)
                    if not dmap:
                        self.direction_sums.pop(map_id, None)
            for (x, y, map_id), r in rewards.items():
                grid = self.sum_rewards.get(map_id)
                if grid is None:
                    continue
                # Unlike ticks, reward can legitimately net to ~0 without the
                # tile being "stale" (equal positive and negative visits), so
                # it can't use its own value to decide eviction — piggyback
                # on the ticks grid (already evicted above this same pass):
                # once a tile has no ticks left in the window, drop its
                # reward too, since nothing referencing it remains.
                still_ticked = (x, y) in self.sum_ticks.get(map_id, {})
                remaining = grid.get((x, y), 0.0) - r
                if still_ticked:
                    grid[(x, y)] = remaining
                else:
                    grid.pop((x, y), None)
                if not grid:
                    self.sum_rewards.pop(map_id, None)
            for (x, y, map_id), counts in battle_outcomes.items():
                bmap = self.sum_battle_outcomes.get(map_id)
                cell = bmap.get((x, y)) if bmap is not None else None
                if cell is None:
                    continue
                for k, c in counts.items():
                    cell[k] = cell.get(k, 0) - c
                if not any(cell.values()):
                    bmap.pop((x, y), None)
                    if not bmap:
                        self.sum_battle_outcomes.pop(map_id, None)
            for (x, y, map_id), n in milestones.items():
                grid = self.sum_milestones.get(map_id)
                if grid is None:
                    continue
                # Same piggyback-on-ticks rule as reward: milestone hits are
                # rare and sparse, so drop the entry once its tile has no
                # ticks left rather than tracking staleness independently.
                still_ticked = (x, y) in self.sum_ticks.get(map_id, {})
                remaining = grid.get((x, y), 0) - n
                if still_ticked:
                    grid[(x, y)] = remaining
                else:
                    grid.pop((x, y), None)
                if not grid:
                    self.sum_milestones.pop(map_id, None)
            for (x, y, map_id), n in truncations.items():
                grid = self.sum_truncations.get(map_id)
                if grid is None:
                    continue
                # Same piggyback-on-ticks rule as milestones.
                still_ticked = (x, y) in self.sum_ticks.get(map_id, {})
                remaining = grid.get((x, y), 0) - n
                if still_ticked:
                    grid[(x, y)] = remaining
                else:
                    grid.pop((x, y), None)
                if not grid:
                    self.sum_truncations.pop(map_id, None)
            for (x, y, map_id), n in visit_counts.items():
                grid = self.sum_visit_counts.get(map_id)
                if grid is None:
                    continue
                # Same piggyback-on-ticks rule as milestones/truncations.
                still_ticked = (x, y) in self.sum_ticks.get(map_id, {})
                remaining = grid.get((x, y), 0) - n
                if still_ticked:
                    grid[(x, y)] = remaining
                else:
                    grid.pop((x, y), None)
                if not grid:
                    self.sum_visit_counts.pop(map_id, None)
            # transition_votes is intentionally NOT evicted here — map
            # connectivity is structural, not recent-behavior traffic.

    def maps(self) -> list[int]:
        """Map ids currently in the window, most-visited first."""
        return sorted(self.sum_ticks, key=lambda m: -sum(self.sum_ticks[m].values()))

    def avg_party(self) -> tuple[float, float] | None:
        """(avg party size, avg level) over the last _PARTY_HISTORY episodes
        that reported party data, or None if none have yet."""
        if not self._party_history:
            return None
        counts = [c for c, _ in self._party_history]
        levels = [lv for _, lv in self._party_history]
        return sum(counts) / len(counts), sum(levels) / len(levels)

    def _metric_sums(self, metric: str) -> dict[int, dict[tuple[int, int], float]]:
        if metric == "reward":
            return self.sum_rewards
        if metric == "milestones":
            return self.sum_milestones
        if metric == "truncations":
            return self.sum_truncations
        if metric == "visits":
            return self.sum_visit_counts
        return self.sum_ticks

    @staticmethod
    def _outcome_rate(
        counts: dict[str, int], num_key: str, den_key: str
    ) -> float | None:
        """num_key / (num_key + den_key), or None if under _MIN_BATTLE_SAMPLES.
        Shared by winrate (win vs loss) and fleerate (smart vs coward) —
        both are ratios over the same two-bucket slice of battle_outcome_counts."""
        total = counts.get(num_key, 0) + counts.get(den_key, 0)
        if total < _MIN_BATTLE_SAMPLES:
            return None
        return counts.get(num_key, 0) / total

    def average_grid(self, map_id: int, metric: str = "ticks"):
        """(grid, x0, y0): grid[y - y0, x - x0] = avg value/run for
        ``metric`` ("ticks", "reward", "milestones", or "truncations"), avg
        visits/run divided by VISIT_PENALTY_SOFT_THRESHOLD for "visits" (see
        _visit_metric_scale), a rate in [0, 1] for "winrate" (win /
        (win+loss)) or "fleerate" (smart / (smart+coward)), or
        episodes-since-last-visit for "recency" / episodes-since-last-dialog
        for "dialog_recency", NaN where unvisited (or, for the rate metrics,
        under-sampled; for dialog_recency, never dialog-triggered) in the
        current window. None if the map isn't in the window.

        The footprint (bounding box + which cells are "visited") always
        follows the ticks grid — reward/winrate/fleerate/milestones/
        truncations are attributed to a subset of the tiles
        visited_positions tracks, so ticks is the authoritative
        visited-set; a tile with exactly zero net reward is still
        "visited", not NaN.
        """
        grid_ticks = self.sum_ticks.get(map_id)
        n_runs = self.run_count.get(map_id, 0)
        if not grid_ticks or n_runs <= 0:
            return None
        xs = [p[0] for p in grid_ticks]
        ys = [p[1] for p in grid_ticks]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        out = np.full((y1 - y0 + 1, x1 - x0 + 1), np.nan, dtype=np.float32)
        rate_keys = _RATE_METRICS.get(metric)
        if rate_keys is not None:
            outcome_grid = self.sum_battle_outcomes.get(map_id, {})
            for (x, y), counts in outcome_grid.items():
                rate = self._outcome_rate(counts, *rate_keys)
                if rate is not None:
                    out[y - y0, x - x0] = rate
            return out, x0, y0
        if metric == "recency":
            seen = self.last_seen.get(map_id, {})
            for (x, y) in grid_ticks:
                last = seen.get((x, y))
                if last is not None:
                    out[y - y0, x - x0] = self.total_episodes - last
            return out, x0, y0
        if metric == "dialog_recency":
            seen = self.last_dialog_seen.get(map_id, {})
            for (x, y) in grid_ticks:
                last = seen.get((x, y))
                if last is not None:
                    out[y - y0, x - x0] = self.total_episodes - last
            return out, x0, y0
        source = self._metric_sums(metric).get(map_id, {})
        divisor = n_runs * _visit_metric_scale() if metric == "visits" else n_runs
        for (x, y), ticks in grid_ticks.items():
            val = ticks if metric == "ticks" else source.get((x, y), 0.0)
            out[y - y0, x - x0] = val / divisor
        return out, x0, y0

    def direction_grid(self, map_id: int):
        """(xs, ys, us, vs): quiver-ready arrays for cells with enough
        direction samples. (us, vs) is the net right/left, down/up count
        difference normalized to [-1, 1] per axis — arrow direction is the
        most common walking direction, length reflects how dominant it is."""
        dmap = self.direction_sums.get(map_id)
        if not dmap:
            return None
        return self._direction_arrays({map_id: (0, 0)})

    def _direction_arrays(self, offsets: dict[int, tuple[int, int]]):
        xs, ys, us, vs = [], [], [], []
        for map_id, (gx0, gy0) in offsets.items():
            dmap = self.direction_sums.get(map_id)
            if not dmap:
                continue
            for (x, y), counts in dmap.items():
                total = sum(counts.values())
                if total < _MIN_DIRECTION_SAMPLES:
                    continue
                u = (counts.get("right", 0) - counts.get("left", 0)) / total
                v = (counts.get("down", 0) - counts.get("up", 0)) / total
                if u == 0 and v == 0:
                    continue
                xs.append(x + gx0)
                ys.append(y + gy0)
                us.append(u)
                vs.append(v)
        if not xs:
            return None
        return np.array(xs), np.array(ys), np.array(us), np.array(vs)

    def _confirmed_transitions(self) -> dict[tuple[int, int], tuple[int, int]]:
        """(from_map, to_map) -> delta for pairs with enough consistent votes.

        Best case: both directions were walked organically and their deltas
        agree (delta_ab == -delta_ba) — cross-checked, so the base
        vote/majority bar applies. If the two directions were both measured
        but *disagree*, the pair is untrustworthy (a mismeasured or genuinely
        mixed crossing) and is dropped entirely.

        Many door/stairs warps only ever accumulate votes in one direction,
        though: a map first reached through a scripted/cutscene walk (e.g.
        the Oak's-Lab intercept) never contributes votes for the entry
        direction — count() already skips cutscene-locked steps — so only
        the *exit* direction, recorded later whenever the agent organically
        walks back out, ever accumulates. That's not a corrupted pair, just
        a permanently one-sided one — requiring both directions would leave
        it unconfirmed forever regardless of sample count. Accept it off the
        single direction instead, but on a much stricter bar
        (_MIN_TRANSITION_VOTES_SOLO / _MAJORITY_SOLO) since there's no
        independent second measurement to catch a mis-measured delta.
        """
        candidates: dict[tuple[int, int], tuple[tuple[int, int], int, int]] = {}
        for key, votes in self.transition_votes.items():
            total = sum(votes.values())
            if total < _MIN_TRANSITION_VOTES:
                continue
            best_delta, best_count = max(votes.items(), key=lambda kv: kv[1])
            if best_count / total < _MIN_TRANSITION_MAJORITY:
                continue
            candidates[key] = (best_delta, best_count, total)

        out: dict[tuple[int, int], tuple[int, int]] = {}
        for (a, b), (delta, best_count, total) in candidates.items():
            reverse = candidates.get((b, a))
            if reverse is not None:
                rev_delta = reverse[0]
                if rev_delta == (-delta[0], -delta[1]):
                    out[(a, b)] = delta
                continue
            if (
                total >= _MIN_TRANSITION_VOTES_SOLO
                and best_count / total >= _MIN_TRANSITION_MAJORITY_SOLO
            ):
                out[(a, b)] = delta
        return out

    def global_offsets(self, anchor: int) -> dict[int, tuple[int, int]]:
        """Every map's (x, y) world-cell offset relative to ``anchor``
        (itself at (0, 0)): pokemon.map_connections.overworld_offsets
        (exact, static, derived straight from reference/pokered's
        `connection` map-header data — see that module's docstring) for
        every outdoor map reachable that way, plus ``_empirical_offsets``
        for anything else (interiors etc., reachable only through
        non-geometric warps, which the static graph doesn't cover at all)
        that live episode data has stitched in.

        The static graph is available immediately — no episodes needed —
        and always wins on conflict: it's exact game data, not a live
        heuristic guess, so it's applied last and unconditionally
        overwrites whatever the empirical BFS came up with for the same
        map_id. Both dicts already share one coordinate frame by
        construction (both place ``anchor`` at (0, 0)), so overlaying them
        is a plain dict update, no re-basing needed.
        """
        from pokemon import map_connections

        offsets = self._empirical_offsets(anchor)
        offsets.update(map_connections.overworld_offsets(anchor))
        offsets.setdefault(anchor, (0, 0))
        return offsets

    def _empirical_offsets(self, anchor: int) -> dict[int, tuple[int, int]]:
        """BFS the empirical connection graph from ``anchor`` (offset (0,0)),
        inferred from observed live step deltas across map_id boundaries
        (see RollingHeatmapAggregator.add_episode's ``transitions`` arg).
        Maps with no discovered path to the anchor are simply absent.

        A plain BFS spanning tree trusts whichever path reaches a map first
        and never looks back — if that map is *also* reachable via a second
        confirmed edge implying a different offset (a door-warp delta that
        got "confirmed" as if it were geometric, most commonly), the
        disagreement would otherwise go unnoticed and the map gets stitched
        in at one arbitrary, possibly overlapping position. Detect that case
        and drop the map — and everything stitched in only through it —
        rather than render a silently-wrong placement.

        Superseded for outdoor maps by global_offsets' static
        pokemon.map_connections overlay — this remains the only source of
        placement for interiors/caves/etc. that only ever warp, which have
        no `connection` header data at all.
        """
        confirmed = self._confirmed_transitions()
        adj: dict[int, list[tuple[int, tuple[int, int]]]] = {}
        for (a, b), delta in confirmed.items():
            adj.setdefault(a, []).append((b, delta))
            adj.setdefault(b, []).append((a, (-delta[0], -delta[1])))

        offsets: dict[int, tuple[int, int]] = {anchor: (0, 0)}
        children: dict[int, list[int]] = {}
        conflicted: set[int] = set()
        frontier = deque([anchor])
        while frontier:
            cur = frontier.popleft()
            if cur in conflicted:
                # cur's own offset is already known-unreliable — don't use
                # its edges to cast doubt on otherwise-solid neighbors.
                continue
            gx, gy = offsets[cur]
            for nxt, delta in adj.get(cur, []):
                # world position at the crossing is invariant: offsets[nxt]
                # is defined by G[next] = G[cur] - delta_{cur->next}.
                implied = (gx - delta[0], gy - delta[1])
                if nxt in offsets:
                    if nxt == anchor:
                        # cur's placement, cross-checked via this edge, no
                        # longer lands back on the anchor's fixed (0, 0) —
                        # e.g. a door whose exit delta isn't the exact
                        # negation of its entry delta. cur (not the anchor)
                        # is the unreliable one here.
                        if implied != (0, 0):
                            conflicted.add(cur)
                    elif nxt != cur and offsets[nxt] != implied:
                        conflicted.add(nxt)
                    continue
                offsets[nxt] = implied
                children.setdefault(cur, []).append(nxt)
                frontier.append(nxt)

        if conflicted:
            drop = set()
            stack = list(conflicted)
            while stack:
                node = stack.pop()
                if node in drop:
                    continue
                drop.add(node)
                stack.extend(children.get(node, []))
            for node in drop:
                offsets.pop(node, None)
        return offsets

    def combined_view(self, anchor: int | None = None, metric: str = "ticks"):
        """Stitches every map reachable from ``anchor`` (default: the
        most-visited map in the current window) into one canvas, colored by
        ``metric`` ("ticks", "reward", "winrate", "fleerate", "recency",
        "dialog_recency", "milestones", "truncations", or "visits").

        Returns (grid, x0, y0, offsets, connected_maps, unconnected_maps) or
        None if there's nothing in the window yet.
        """
        maps_in_window = self.maps()
        if not maps_in_window:
            return None
        if anchor is None or anchor not in maps_in_window:
            anchor = maps_in_window[0]

        offsets = self.global_offsets(anchor)
        connected = [m for m in maps_in_window if m in offsets]
        if not connected:
            connected = [anchor]
            offsets = {anchor: (0, 0)}
        unconnected = [m for m in maps_in_window if m not in offsets]

        cells: list[tuple[int, int, float]] = []
        rate_keys = _RATE_METRICS.get(metric)
        for map_id in connected:
            gx0, gy0 = offsets[map_id]
            if rate_keys is not None:
                for (x, y), counts in self.sum_battle_outcomes.get(map_id, {}).items():
                    rate = self._outcome_rate(counts, *rate_keys)
                    if rate is not None:
                        cells.append((x + gx0, y + gy0, rate))
                continue
            if metric == "recency":
                seen = self.last_seen.get(map_id, {})
                for (x, y) in self.sum_ticks.get(map_id, {}):
                    last = seen.get((x, y))
                    if last is not None:
                        cells.append((x + gx0, y + gy0, self.total_episodes - last))
                continue
            if metric == "dialog_recency":
                seen = self.last_dialog_seen.get(map_id, {})
                for (x, y) in self.sum_ticks.get(map_id, {}):
                    last = seen.get((x, y))
                    if last is not None:
                        cells.append((x + gx0, y + gy0, self.total_episodes - last))
                continue
            n_runs = max(self.run_count.get(map_id, 1), 1)
            divisor = n_runs * _visit_metric_scale() if metric == "visits" else n_runs
            metric_grid = self._metric_sums(metric).get(map_id, {})
            for (x, y), ticks in self.sum_ticks.get(map_id, {}).items():
                val = ticks if metric == "ticks" else metric_grid.get((x, y), 0.0)
                cells.append((x + gx0, y + gy0, val / divisor))
        if not cells:
            return None

        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        grid = np.full((y1 - y0 + 1, x1 - x0 + 1), np.nan, dtype=np.float32)
        for gx, gy, avg in cells:
            prev = grid[gy - y0, gx - x0]
            grid[gy - y0, gx - x0] = avg if np.isnan(prev) else max(prev, avg)

        return grid, x0, y0, offsets, connected, unconnected


def _populate_unconnected_panels(
    side_axes: list,
    agg: "RollingHeatmapAggregator",
    unconnected: list[int],
    metric: str,
    name_of,
    allowed: frozenset[int] | None,
    cmap,
    vmin: float,
    vmax: float,
) -> None:
    """Render each map the combined-view stitcher couldn't place (no
    confirmed connection to the anchor — see global_offsets) as its own
    small thumbnail in ``side_axes``, sharing the main image's colormap/clim
    so colors stay comparable. Previously these maps were only named in the
    title, truncated past 6; now they stay visually inspectable even though
    their position relative to the stitched cluster is unknown.

    Always clears/hides every side axis first, so calling this with an
    empty ``unconnected`` (or from a context with no side panels, e.g. the
    single-map view) is the correct way to blank them out.

    Each thumbnail gets the real map pixel art (pokemon.map_render) as its
    background, same as the main view — imported lazily here (like this
    module's other pokemon.* imports) so importing PositionHeatmap itself
    stays cheap for the main training process.
    """
    from pokemon import map_render

    for sax in side_axes:
        sax.clear()
        sax.set_visible(False)
    max_panels = len(side_axes)
    if not unconnected or max_panels == 0:
        return
    overflow = len(unconnected) > max_panels
    thumb_ids = unconnected[: max_panels - 1] if overflow else unconnected[:max_panels]
    for sax, map_id in zip(side_axes, thumb_ids):
        sax.set_visible(True)
        sax.set_facecolor(_BG_FACECOLOR)
        sax.set_xticks([])
        sax.set_yticks([])
        off_goal = allowed is not None and map_id not in allowed
        title = name_of(map_id) + (" [OFF-GOAL]" if off_goal else "")
        result = agg.average_grid(map_id, metric=metric)
        if result is None:
            sax.text(
                0.5, 0.5, title, ha="center", va="center", fontsize=6,
                color="white", wrap=True, transform=sax.transAxes,
            )
            continue
        grid, x0, y0 = result
        bg = map_render.render_map_rgb(map_id)
        if bg is not None:
            h_cells, w_cells = bg.shape[0] / map_render.CELL_PX, bg.shape[1] / map_render.CELL_PX
            sax.imshow(
                bg, origin="upper", zorder=1,
                extent=(-0.5, w_cells - 0.5, h_cells - 0.5, -0.5),
            )
        sax.imshow(
            grid, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper",
            alpha=_OVERLAY_ALPHA, zorder=2,
            extent=(x0 - 0.5, x0 + grid.shape[1] - 0.5, y0 + grid.shape[0] - 0.5, y0 - 0.5),
        )
        sax.set_title(title, fontsize=7, color="#ff8080" if off_goal else "white")
    if overflow:
        remaining = unconnected[max_panels - 1:]
        last_ax = side_axes[max_panels - 1]
        last_ax.set_visible(True)
        last_ax.set_xticks([])
        last_ax.set_yticks([])
        names = ", ".join(name_of(m) for m in remaining[:8])
        more = " ..." if len(remaining) > 8 else ""
        last_ax.text(
            0.5, 0.5, f"+{len(remaining)} more:\n{names}{more}",
            ha="center", va="center", fontsize=6, color="#aaaaaa", wrap=True,
            transform=last_ax.transAxes,
        )
        last_ax.set_title("(not shown)", fontsize=7, color="#888888")


def _map_name_lookup() -> dict[int, str]:
    from pokemon import Data as _data_module

    names: dict[int, str] = {}
    for key, value in vars(_data_module).items():
        if key.startswith("MAP_") and isinstance(value, int):
            names.setdefault(value, key[len("MAP_"):].replace("_", " ").title())
    return names


_ALL_STAGES = "All"


# Goal saturation panel status dots -- see curriculum_config.goal_status.
# "locked" gets no dot at all (nothing to flag: pick_new_goal won't hand it
# out yet either way).
_GOAL_STATUS_DOT = {"mastered": "\U0001F7E2 ", "practicable": "\U0001F7E1 "}
# Second-tier sort key for the goal saturation panel (see
# _refresh_goal_saturation): among goals tied on hit count, rank by how
# "active" a possibility the goal currently is -- mastered (already
# reachable and cleared) and practicable (reachable, just not yet hit) sort
# above locked (pick_new_goal can't hand it out at all yet).
_GOAL_STATUS_RANK = {"mastered": 0, "practicable": 1, "locked": 2}


def _default_map_id() -> int | None:
    """Map to show before any episode data has arrived — Red's bedroom
    (REDS_HOUSE_2F), the map a fresh game (and so the very first episode)
    actually starts on (see tools/render_route.py's own DEFAULT_START).
    Lets the window open showing real map art immediately instead of a
    blank "waiting for runs" screen — see redraw_single/redraw_combined's
    early-return branches. None only if map_constants can't be imported at
    all (should not happen in practice)."""
    try:
        from pokemon.map_constants import MAPS

        return MAPS["REDS_HOUSE_2F"][0]
    except Exception:
        return None


def _goal_status(goal: str) -> str:
    try:
        from curriculum_config import goal_status

        return goal_status(goal)
    except Exception:
        return "locked"


def _goal_home_map(goal: str) -> int | None:
    try:
        from curriculum_config import _goal_map_id

        return _goal_map_id(goal)
    except Exception:
        return None


def _stage_order_lookup() -> list[str]:
    """Canonical curriculum stage order, used only to sort the dropdown —
    stages actually seen in the data are shown regardless of whether they're
    in this list."""
    try:
        from curriculum_config import STAGE_ORDER

        return list(STAGE_ORDER)
    except Exception:
        return []


def _allowed_maps_lookup() -> dict[str, frozenset[int] | None]:
    """stage label -> the small set of maps relevant to that stage's goal,
    for the off-goal-map visual cue in the combined/single views.

    There is no more hand-curated GOAL_ALLOWED_MAPS critical-path set to
    read (see Data.py's generic goal system) — this instead uses
    event_graph.py's static-analysis map attribution: the goal event's own
    home map plus its inferred parent/child events' home maps (the same
    small neighborhood Data.active_map_events draws on for
    reward_active_map_presence, not a hand-picked full path). A goal with no
    event_graph entry (e.g. a badge — wObtainedBadges has no map of its
    own) or an unrecognized stage just skips highlighting for that stage."""
    try:
        from curriculum_config import get_goal_for_stage

        from pokemon.event_graph import EVENT_GRAPH
    except Exception:
        return {}
    out: dict[str, frozenset[int] | None] = {}
    for stage in _stage_order_lookup():
        try:
            goal = get_goal_for_stage(stage)
            info = EVENT_GRAPH.get(goal)
            if info is None:
                out[stage] = None
                continue
            map_ids = {info["map_id"]} if info["map_id"] is not None else set()
            for name in (*info["parents"], *info["children"]):
                neighbor = EVENT_GRAPH.get(name)
                if neighbor and neighbor["map_id"] is not None:
                    map_ids.add(neighbor["map_id"])
            out[stage] = frozenset(map_ids) if map_ids else None
        except Exception:
            continue
    return out


def _run_window(q: "Queue", window_frames: int, title: str) -> None:
    import json
    import matplotlib
    from pathlib import Path

    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from pokemon import map_connections, map_render
    from pokemon.Data import VISIT_PENALTY_HARD_THRESHOLD, VISIT_PENALTY_SOFT_THRESHOLD

    default_map_id = _default_map_id()
    map_names = _map_name_lookup()
    stage_order = _stage_order_lookup()
    allowed_maps_by_stage = _allowed_maps_lookup()
    # "All" pools every run regardless of stage; stage_aggs holds one
    # independent rolling aggregator per curriculum stage seen so far.
    agg_all = RollingHeatmapAggregator(window_frames)
    stage_aggs: dict[str, RollingHeatmapAggregator] = {}
    # ticks: non-negative "time spent", sequential colormap from a dark floor.
    # reward: signed "reward earned", diverging colormap centered on zero so
    # farming/exploit hotspots (strongly positive) read differently from
    # penalty hotspots (strongly negative) instead of both just being "hot".
    # winrate: a fixed-range [0, 1] fraction, red-to-green so losing tiles
    # and winning tiles are immediately distinguishable regardless of sample
    # size (unlike ticks/reward, the color scale never rescales to the data).
    ticks_cmap = plt.get_cmap("inferno").copy()
    ticks_cmap.set_bad(alpha=0.0)
    reward_cmap = plt.get_cmap("coolwarm").copy()
    reward_cmap.set_bad(alpha=0.0)
    winrate_cmap = plt.get_cmap("RdYlGn").copy()
    winrate_cmap.set_bad(alpha=0.0)
    # fleerate shares winrate's fixed-range red-green scale — same "0-1
    # fraction, green is the good outcome" shape, just a different pair of
    # battle_outcome_counts buckets (smart/coward instead of win/loss).
    # recency: non-negative "episodes since last visit", sequential like
    # ticks but a distinct colormap so the two aren't visually confused.
    recency_cmap = plt.get_cmap("viridis").copy()
    recency_cmap.set_bad(alpha=0.0)
    # dialog_recency: same non-negative "episodes since" shape as recency,
    # but scoped to dialog-triggering tiles only — distinct colormap so it
    # doesn't read as the same metric at a glance.
    dialog_recency_cmap = plt.get_cmap("cividis").copy()
    dialog_recency_cmap.set_bad(alpha=0.0)
    # milestones: non-negative sparse counts (most tiles are 0), same
    # avg/run convention as reward — distinct colormap from ticks/recency.
    milestones_cmap = plt.get_cmap("plasma").copy()
    milestones_cmap.set_bad(alpha=0.0)
    # truncations: same non-negative sparse-count shape as milestones (most
    # tiles are 0, a hit means a run actually ended there) — distinct
    # colormap so it doesn't read as the same metric at a glance.
    truncations_cmap = plt.get_cmap("magma").copy()
    truncations_cmap.set_bad(alpha=0.0)
    # visits: avg position_visit_counts/run, normalized by
    # VISIT_PENALTY_SOFT_THRESHOLD (see _visit_metric_scale) so 1.0 reads as
    # "at the anti-loop soft-penalty threshold" and 2.0 as "at the hard
    # threshold" — fixed [0, 2] range like winrate/fleerate (doesn't rescale
    # to the data) rather than ticks' open-ended sequential scale, since the
    # threshold multiples are the meaningful reference points here, not
    # whatever the hottest tile in the window happens to be.
    visits_cmap = plt.get_cmap("YlOrRd").copy()
    visits_cmap.set_bad(alpha=0.0)
    _METRICS = (
        "ticks", "reward", "winrate", "fleerate", "recency", "dialog_recency",
        "milestones", "truncations", "visits",
    )

    fig = plt.figure(figsize=(11, 8))
    fig.canvas.manager.set_window_title(title)
    # Main plot spans the left 3/4; a stacked column of small axes on the
    # right holds combined-view thumbnails for maps that couldn't be
    # stitched in (see _populate_unconnected_panels). Hidden (not just
    # empty) whenever the single-map view is active or nothing's unconnected.
    gs = GridSpec(_MAX_SIDE_PANELS, 4, figure=fig, wspace=0.7, hspace=0.6)
    ax = fig.add_subplot(gs[:, :3])
    ax.set_facecolor(_BG_FACECOLOR)
    side_axes = [fig.add_subplot(gs[i, 3]) for i in range(_MAX_SIDE_PANELS)]
    for _sax in side_axes:
        _sax.set_visible(False)
    # Real map pixel art (pokemon.map_render) drawn first as a background,
    # the metric-color grid on top at _OVERLAY_ALPHA — see that constant's
    # comment. bg_im starts blank (no map picked yet) and is filled in by
    # _update_background on the first redraw.
    bg_im = ax.imshow(np.zeros((1, 1, 3), dtype=np.uint8), origin="upper", zorder=1)
    im = ax.imshow(
        np.zeros((1, 1)), cmap=ticks_cmap, origin="upper",
        alpha=_OVERLAY_ALPHA, zorder=2,
    )
    cbar = fig.colorbar(im, ax=ax, label="avg ticks / run")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    quiv = {"artist": None}
    labels: list = []
    value_texts: list = []

    state = {
        "map_id": None,
        "closed": False,
        "mode": "single",
        "stage": _ALL_STAGES,
        "metric": "ticks",
        "show_values": False,
        # True once the user scroll-zooms or drag-pans the view. While True,
        # redraws leave the view alone (so the manual zoom/pan sticks even as
        # newly-discovered tiles grow the grid off-screen). Reset to False on
        # any context switch (map/stage/mode) so the view snaps back to
        # fitting whatever's been discovered in the new context.
        "user_viewport": False,
    }
    # Drag-to-pan bookkeeping (left mouse button held over the axes).
    pan_state = {"active": False, "x0": None, "y0": None, "xlim0": None, "ylim0": None}

    def current_agg() -> RollingHeatmapAggregator:
        if state["stage"] == _ALL_STAGES:
            return agg_all
        return stage_aggs.get(state["stage"], agg_all)

    def _stage_sort_key(s: str):
        try:
            return (0, stage_order.index(s))
        except ValueError:
            return (1, s)

    def name_of(map_id: int) -> str:
        return map_names.get(map_id, f"map {map_id}")

    def on_close(_event) -> None:
        state["closed"] = True

    def on_key(event) -> None:
        if event.key == "c":
            state["mode"] = "combined" if state["mode"] == "single" else "single"
            state["user_viewport"] = False
            redraw()
            return
        if event.key == "r":
            idx = _METRICS.index(state["metric"])
            state["metric"] = _METRICS[(idx + 1) % len(_METRICS)]
            redraw()
            return
        if event.key == "v":
            state["show_values"] = not state["show_values"]
            redraw()
            return
        if event.key == "0":
            state["user_viewport"] = False
            redraw()
            return
        if state["mode"] != "single":
            return
        maps = current_agg().maps()
        if not maps:
            return
        if state["map_id"] not in maps:
            state["map_id"] = maps[0]
        idx = maps.index(state["map_id"])
        if event.key in ("right", "n"):
            state["map_id"] = maps[(idx + 1) % len(maps)]
            state["user_viewport"] = False
            redraw()
        elif event.key in ("left", "p"):
            state["map_id"] = maps[(idx - 1) % len(maps)]
            state["user_viewport"] = False
            redraw()

    def on_scroll(event) -> None:
        """Scroll-wheel zoom, centered on the cursor's data position."""
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        base_scale = 1.15
        if event.button == "up":
            scale = 1.0 / base_scale
        elif event.button == "down":
            scale = base_scale
        else:
            return
        xlim0, ylim0 = ax.get_xlim(), ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_w = (xlim0[1] - xlim0[0]) * scale
        new_h = (ylim0[1] - ylim0[0]) * scale
        relx = (xlim0[1] - xdata) / (xlim0[1] - xlim0[0])
        rely = (ylim0[1] - ydata) / (ylim0[1] - ylim0[0])
        ax.set_xlim(xdata - new_w * (1 - relx), xdata + new_w * relx)
        ax.set_ylim(ydata - new_h * (1 - rely), ydata + new_h * rely)
        state["user_viewport"] = True
        fig.canvas.draw_idle()

    def on_button_press(event) -> None:
        """Start a drag-to-pan gesture (left mouse button over the axes)."""
        if event.inaxes != ax or event.button != 1:
            return
        pan_state["active"] = True
        pan_state["x0"] = event.xdata
        pan_state["y0"] = event.ydata
        pan_state["xlim0"] = ax.get_xlim()
        pan_state["ylim0"] = ax.get_ylim()

    def on_motion(event) -> None:
        if not pan_state["active"] or event.xdata is None or event.ydata is None:
            return
        dx = event.xdata - pan_state["x0"]
        dy = event.ydata - pan_state["y0"]
        xlim0, ylim0 = pan_state["xlim0"], pan_state["ylim0"]
        ax.set_xlim(xlim0[0] - dx, xlim0[1] - dx)
        ax.set_ylim(ylim0[0] - dy, ylim0[1] - dy)
        state["user_viewport"] = True
        fig.canvas.draw_idle()

    def on_button_release(event) -> None:
        if event.button == 1:
            pan_state["active"] = False

    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_button_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_button_release)

    # Stage dropdown, embedded directly in the TkAgg window above the canvas.
    stage_combo = None
    goal_tree = None
    try:
        import tkinter as tk
        from tkinter import ttk

        manager = fig.canvas.manager
        root = getattr(manager, "window", None)
        canvas_widget = fig.canvas.get_tk_widget()
        if root is not None:
            control_frame = tk.Frame(root)
            control_frame.pack(side=tk.TOP, fill=tk.X, before=canvas_widget)
            tk.Label(control_frame, text="Stage:").pack(
                side=tk.LEFT, padx=(6, 2), pady=4
            )
            stage_var = tk.StringVar(value=_ALL_STAGES)
            stage_combo = ttk.Combobox(
                control_frame, textvariable=stage_var, state="readonly", width=28
            )
            stage_combo["values"] = [_ALL_STAGES]
            stage_combo.pack(side=tk.LEFT, padx=2, pady=4)

            def _on_stage_selected(_event=None) -> None:
                state["stage"] = stage_var.get()
                state["map_id"] = None
                state["user_viewport"] = False
                redraw()

            stage_combo.bind("<<ComboboxSelected>>", _on_stage_selected)

            # Goal saturation panel: cumulative per-goal hit counts across
            # the whole training run (see MilestoneCallback._goal_hit_counts
            # / GOAL_HIT_COUNTS_PATH in ppo/callbacks.py, which persists the
            # exact file this reads). Read straight off disk on a timer
            # rather than piped through the queue like the map data --
            # it's whole-run state, not a per-episode snapshot, and this
            # keeps the panel decoupled from HeatmapCallback entirely (it
            # still populates even if that feed is idle). Most-practiced
            # goals sort to the top -- zero-hit ones (greyed out) sink to
            # the bottom, so scrolling down surfaces what pick_new_goal's
            # weights are currently favoring least.
            sat_frame = tk.Frame(root)
            sat_frame.pack(side=tk.RIGHT, fill=tk.Y, before=canvas_widget)
            tk.Label(sat_frame, text="Goal saturation (most-practiced first)").pack(
                side=tk.TOP, pady=(4, 0)
            )
            tk.Label(
                sat_frame,
                text="\U0001F7E2 mastered   \U0001F7E1 practicable (0 hits ok)\n"
                "visited maps only, locked goals hidden",
                fg="#666666",
            ).pack(side=tk.TOP, pady=(0, 4))
            tree_frame = tk.Frame(sat_frame)
            tree_frame.pack(side=tk.TOP, fill=tk.Y, expand=True, padx=4, pady=4)
            goal_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
            goal_tree = ttk.Treeview(
                tree_frame,
                columns=("hits", "success"),
                show="tree headings",
                height=32,
                yscrollcommand=goal_scroll.set,
            )
            goal_tree.heading("#0", text="Goal")
            goal_tree.heading("hits", text="Hits")
            goal_tree.heading("success", text="Success %")
            goal_tree.column("#0", width=220, anchor=tk.W, stretch=False)
            goal_tree.column("hits", width=50, anchor=tk.E, stretch=False)
            goal_tree.column("success", width=70, anchor=tk.E, stretch=False)
            goal_tree.tag_configure("zero", foreground="#999999")
            goal_scroll.config(command=goal_tree.yview)
            goal_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            goal_tree.pack(side=tk.LEFT, fill=tk.Y, expand=True)
    except Exception:
        stage_combo = None
        goal_tree = None

    def _refresh_stage_dropdown() -> None:
        """Rebuild the dropdown's option list from stages seen so far."""
        if stage_combo is None:
            return
        values = [_ALL_STAGES] + sorted(stage_aggs.keys(), key=_stage_sort_key)
        if list(stage_combo["values"]) != values:
            stage_combo["values"] = values

    goal_hit_counts_path = Path("saves") / "goal_hit_counts.json"
    visited_maps_path = Path("saves") / "visited_maps.json"
    # Per-goal rolling success window (most recent up to MilestoneCallback.
    # window resume-attempts, as a list of 1/0) while that goal was the
    # active base/resume checkpoint -- see MilestoneCallback._goal_success_
    # windows / GOAL_SUCCESS_STATS_PATH in ppo/callbacks.py. Distinct from
    # goal_hit_counts_path's "hits" (any goal touched by an episode,
    # regardless of what was assigned, cumulative forever): this windowed
    # rate is what curriculum_config.pick_next_goal_by_success_rate actually
    # picks on. Optional -- older saves/ dirs or a run that hasn't hit the
    # periodic save yet won't have it, so its absence doesn't block the rest
    # of the panel, it just shows "n/a" success rates.
    goal_success_stats_path = Path("saves") / "goal_success_stats.json"
    _goal_panel_state = {"mtime": None}

    def _refresh_goal_saturation() -> None:
        """Reload saves/goal_hit_counts.json, saves/goal_success_stats.json
        and saves/visited_maps.json (see MilestoneCallback._visited_maps in
        ppo/callbacks.py, fed into pick_new_goal's own ``visited_maps`` arg)
        when any changes, and repaint the side panel.

        Only goals pick_new_goal could actually hand out right now are
        listed: status "locked" (parents not yet satisfied -- see
        curriculum_config.goal_status) is dropped entirely, and so is any
        "mastered"/"practicable" goal whose home map (event_graph.py's
        map_id -- see _goal_home_map) hasn't been visited by any episode
        this run. A goal with no resolvable home map (e.g. a badge with no
        EVENT_GRAPH anchor) can't be checked this way and is kept, same as
        goal_status treats missing EVENT_GRAPH info as trivially satisfied.

        Sorted most-practiced goal first (primary key: hit count,
        descending) with ties -- overwhelmingly the pile of zero-hit goals
        -- broken by _goal_status's own "possible activity" ordering
        (mastered, then practicable: see _GOAL_STATUS_RANK) so a
        not-yet-hit-but-currently-pickable goal surfaces above a mastered
        one, instead of both sinking to the same arbitrary STAGE_ORDER
        position."""
        if goal_tree is None:
            return
        try:
            mtime = goal_hit_counts_path.stat().st_mtime
            visited_mtime = visited_maps_path.stat().st_mtime
        except OSError:
            return
        try:
            stats_mtime = goal_success_stats_path.stat().st_mtime
        except OSError:
            stats_mtime = None
        combined_mtime = (mtime, visited_mtime, stats_mtime)
        if combined_mtime == _goal_panel_state["mtime"]:
            return
        try:
            with open(goal_hit_counts_path, encoding="utf-8") as f:
                counts = json.load(f)
            with open(visited_maps_path, encoding="utf-8") as f:
                visited_maps = {int(m) for m in json.load(f)}
        except (OSError, ValueError):
            return
        stats: dict = {}
        if stats_mtime is not None:
            try:
                with open(goal_success_stats_path, encoding="utf-8") as f:
                    stats = json.load(f)
            except (OSError, ValueError):
                stats = {}
        _goal_panel_state["mtime"] = combined_mtime
        names = stage_order or sorted(counts)
        statuses = {name: _goal_status(name) for name in names}
        home_maps = {name: _goal_home_map(name) for name in names}
        names = [
            name
            for name in names
            if statuses[name] != "locked"
            and (home_maps[name] is None or home_maps[name] in visited_maps)
        ]
        rows = sorted(
            ((name, int(counts.get(name, 0))) for name in names),
            key=lambda kv: (-kv[1], _GOAL_STATUS_RANK.get(statuses[kv[0]], 99)),
        )
        goal_tree.delete(*goal_tree.get_children())
        for name, hits in rows:
            label = f"{_GOAL_STATUS_DOT.get(statuses[name], '')}{name}"
            window = stats.get(name) or []
            attempts = len(window)
            successes = sum(window)
            success_pct = f"{successes / attempts:.0%}" if attempts else "n/a"
            goal_tree.insert(
                "", tk.END, text=label, values=(hits, success_pct),
                tags=("zero",) if hits == 0 else (),
            )

    def _clear_overlays() -> None:
        if quiv["artist"] is not None:
            quiv["artist"].remove()
            quiv["artist"] = None
        for t in labels:
            t.remove()
        labels.clear()
        for t in value_texts:
            t.remove()
        value_texts.clear()

    def _apply_metric_style(grid) -> None:
        """Colormap/scale/colorbar-label for the active metric. Reward is
        signed (farming a tile can look identical to time-spent in the
        ticks view but shows as strongly positive here; a real penalty spot
        shows negative) so it gets a diverging map centered on zero instead
        of ticks' 0-based sequential one."""
        finite = grid[~np.isnan(grid)]
        if state["metric"] == "reward":
            im.set_cmap(reward_cmap)
            vmax = float(np.abs(finite).max()) if finite.size else 0.0
            vmax = vmax if vmax > 0 else 1.0
            im.set_clim(-vmax, vmax)
            cbar.set_label("avg reward / run")
        elif state["metric"] == "winrate":
            im.set_cmap(winrate_cmap)
            im.set_clim(0.0, 1.0)
            cbar.set_label("battle win rate")
        elif state["metric"] == "fleerate":
            im.set_cmap(winrate_cmap)
            im.set_clim(0.0, 1.0)
            cbar.set_label("flee smart-rate (smart / smart+coward)")
        elif state["metric"] == "recency":
            im.set_cmap(recency_cmap)
            vmax = float(finite.max()) if finite.size else 0.0
            im.set_clim(0, vmax if vmax > 0 else 1.0)
            cbar.set_label("episodes since last visit")
        elif state["metric"] == "dialog_recency":
            im.set_cmap(dialog_recency_cmap)
            vmax = float(finite.max()) if finite.size else 0.0
            im.set_clim(0, vmax if vmax > 0 else 1.0)
            cbar.set_label("episodes since last dialog")
        elif state["metric"] == "milestones":
            im.set_cmap(milestones_cmap)
            vmax = float(finite.max()) if finite.size else 0.0
            im.set_clim(0, vmax if vmax > 0 else 1.0)
            cbar.set_label("avg milestone hits / run")
        elif state["metric"] == "truncations":
            im.set_cmap(truncations_cmap)
            vmax = float(finite.max()) if finite.size else 0.0
            im.set_clim(0, vmax if vmax > 0 else 1.0)
            cbar.set_label("avg truncations / run")
        elif state["metric"] == "visits":
            im.set_cmap(visits_cmap)
            # Fixed scale (see visits_cmap comment above) -- hard threshold
            # ratio, not the data's own max.
            im.set_clim(0.0, VISIT_PENALTY_HARD_THRESHOLD / VISIT_PENALTY_SOFT_THRESHOLD)
            cbar.set_label(f"avg visits / run ÷ {VISIT_PENALTY_SOFT_THRESHOLD}")
        else:
            im.set_cmap(ticks_cmap)
            vmax = float(finite.max()) if finite.size else 0.0
            im.set_clim(0, vmax if vmax > 0 else 1.0)
            cbar.set_label("avg ticks / run")

    def _apply_extent(x0: int, y0: int, grid, bg_extent: tuple | None = None) -> None:
        """Set the image extent to the grid's bounding box and, unless the
        user has manually zoomed/panned this context, snap the view to fit
        it. Without this the axes view stays wherever it last was (initially
        a 1x1 box), so a map that grows as more tiles get discovered — or a
        freshly switched-to map with a totally different footprint — never
        visibly resizes to match.

        ``bg_extent`` (from ``_update_background``) is usually bigger than
        the data grid's own bbox — the visited-tile subset vs. the map's
        full pixel-art footprint — so the auto-fit view is the *union* of
        the two: on first arriving at a mostly-unexplored map, this shows
        the whole map (background) immediately rather than cropping to just
        the handful of visited tiles.
        """
        extent = (
            x0 - 0.5, x0 + grid.shape[1] - 0.5, y0 + grid.shape[0] - 0.5, y0 - 0.5,
        )
        im.set_extent(extent)
        if state["user_viewport"]:
            return
        fit = extent
        if bg_extent is not None:
            fit = (
                min(extent[0], bg_extent[0]), max(extent[1], bg_extent[1]),
                max(extent[2], bg_extent[2]), min(extent[3], bg_extent[3]),
            )
        ax.set_xlim(fit[0], fit[1])
        ax.set_ylim(fit[2], fit[3])

    def _update_background(img_data: tuple | None) -> tuple | None:
        """img_data: (rgb_array, x0, y0) in cell coords (e.g. from
        map_render.render_map_rgb/stitched_rgb), or None if unavailable —
        blanks bg_im so the axes' dark ``_BG_FACECOLOR`` shows through
        instead (only a handful of unused placeholder maps hit this).
        Returns the extent bg_im was set to, for ``_apply_extent``'s
        auto-fit union, or None."""
        if img_data is None:
            bg_im.set_data(np.zeros((1, 1, 3), dtype=np.uint8))
            bg_im.set_extent((0, 0, 0, 0))
            return None
        img, bx0, by0 = img_data
        h_cells = img.shape[0] / map_render.CELL_PX
        w_cells = img.shape[1] / map_render.CELL_PX
        extent = (bx0 - 0.5, bx0 + w_cells - 0.5, by0 + h_cells - 0.5, by0 - 0.5)
        bg_im.set_data(img)
        bg_im.set_extent(extent)
        return extent

    def _show_placeholder(context_label: str) -> None:
        """No episode data for the current stage/mode yet — show the real
        map art for ``state["map_id"]`` (or, before anything's ever been
        picked, ``default_map_id`` — Red's bedroom, where the game and so
        the first episode actually starts) immediately, with the metric
        overlay left empty, instead of a blank screen. "Cała mapa" (the
        whole map) should be visible the moment the window opens, not only
        once the first run lands."""
        map_id = state["map_id"] if state["map_id"] is not None else default_map_id
        bg_img = map_render.render_map_rgb(map_id) if map_id is not None else None
        bg_extent = _update_background((bg_img, 0, 0) if bg_img is not None else None)
        im.set_data(np.full((1, 1), np.nan))
        im.set_extent((0, 0, 0, 0))
        if bg_extent is not None and not state["user_viewport"]:
            ax.set_xlim(bg_extent[0], bg_extent[1])
            ax.set_ylim(bg_extent[2], bg_extent[3])
        name = name_of(map_id) if map_id is not None else "?"
        ax.set_title(
            f"[{state['stage']}] {context_label}{name} (id={map_id}) "
            f"— waiting for runs..."
        )
        fig.canvas.draw_idle()

    def _show_placeholder_combined() -> None:
        """No episode data for the current stage yet — combined view still
        shows the *whole* statically-connected outdoor world immediately
        (pokemon.map_connections.overworld_offsets, built straight from
        reference/pokered's source map-header data), not just one map.
        Unlike ``_show_placeholder``'s single-map fallback, this needs no
        live data at all to be "fully built" — that's the whole point of
        using the game's own source geometry instead of the empirical,
        live-inferred BFS (RollingHeatmapAggregator._empirical_offsets)."""
        anchor = state["map_id"] if state["map_id"] is not None else default_map_id
        offsets = map_connections.overworld_offsets(anchor) if anchor is not None else {}
        bg_extent = _update_background(map_render.stitched_rgb(offsets) if offsets else None)
        im.set_data(np.full((1, 1), np.nan))
        im.set_extent((0, 0, 0, 0))
        if bg_extent is not None and not state["user_viewport"]:
            ax.set_xlim(bg_extent[0], bg_extent[1])
            ax.set_ylim(bg_extent[2], bg_extent[3])
        ax.set_title(
            f"[{state['stage']}] Combined view: {len(offsets)} maps "
            f"(static, from game source) — waiting for runs to color them in..."
        )
        fig.canvas.draw_idle()

    def _draw_quiver(dgrid) -> None:
        if dgrid is None:
            return
        xs, ys, us, vs = dgrid
        # Extent's y-axis is already inverted to match world "down" —
        # (u, v) = (dx, dy) plots correctly with no manual sign flip.
        quiv["artist"] = ax.quiver(
            xs, ys, us, vs,
            color="#4fd1ff", alpha=0.9, pivot="mid",
            angles="xy", scale_units="xy", scale=1.3, width=0.006,
        )

    def _draw_value_labels(grid, x0: int, y0: int) -> None:
        """Overlay the per-tile numeric value (avg ticks or avg reward, per
        the active metric) on top of each non-NaN cell. Skipped above
        _MAX_VALUE_LABELS cells — text artists are too slow at that scale to
        keep the live view responsive."""
        ax.set_xlabel("x")
        if not state["show_values"]:
            return
        ys, xs = np.where(~np.isnan(grid))
        if xs.size == 0:
            return
        if xs.size > _MAX_VALUE_LABELS:
            ax.set_xlabel(
                f"x  (values hidden: {xs.size} cells > {_MAX_VALUE_LABELS} cap)"
            )
            return
        if state["metric"] == "reward":
            fmt = "{:+.2f}"
        elif state["metric"] in ("winrate", "fleerate"):
            fmt = "{:.0%}"
        elif state["metric"] in ("milestones", "visits"):
            fmt = "{:.2f}"
        else:
            fmt = "{:.0f}"
        stroke = [pe.withStroke(linewidth=1.5, foreground="black")]
        for iy, ix in zip(ys, xs):
            value_texts.append(
                ax.text(
                    x0 + ix, y0 + iy, fmt.format(float(grid[iy, ix])),
                    color="white", fontsize=6, ha="center", va="center",
                    path_effects=stroke, zorder=5,
                )
            )

    def _hide_side_panels() -> None:
        for sax in side_axes:
            sax.clear()
            sax.set_visible(False)

    def _party_line(agg: RollingHeatmapAggregator) -> str:
        """'party: N pokemon, avg lvl X.X (last episode)  |  running avg over
        last _PARTY_HISTORY: N pokemon, lvl X.X' — blank until at least one
        episode has reported party data."""
        if agg.last_party_count is None or agg.last_party_avg_level is None:
            return ""
        running = agg.avg_party()
        running_s = ""
        if running is not None:
            avg_count, avg_level = running
            running_s = (
                f"  |  running avg (last {len(agg._party_history)}): "
                f"{avg_count:.1f} pokemon, lvl {avg_level:.1f}"
            )
        return (
            f"party: {agg.last_party_count} pokemon, "
            f"avg lvl {agg.last_party_avg_level:.1f} (last episode)"
            f"{running_s}\n"
        )

    def redraw_single() -> None:
        _hide_side_panels()
        agg = current_agg()
        maps = agg.maps()
        if not maps:
            _show_placeholder("")
            return
        if state["map_id"] not in maps:
            state["map_id"] = maps[0]
        result = agg.average_grid(state["map_id"], metric=state["metric"])
        if result is None:
            return
        grid, x0, y0 = result
        bg_img = map_render.render_map_rgb(state["map_id"])
        bg_extent = _update_background((bg_img, 0, 0) if bg_img is not None else None)
        im.set_data(grid)
        _apply_extent(x0, y0, grid, bg_extent)
        _apply_metric_style(grid)

        _clear_overlays()
        _draw_quiver(agg.direction_grid(state["map_id"]))
        _draw_value_labels(grid, x0, y0)

        name = name_of(state["map_id"])
        n_runs = agg.run_count.get(state["map_id"], 0)
        idx = maps.index(state["map_id"]) + 1
        allowed = allowed_maps_by_stage.get(state["stage"])
        off_goal_s = (
            "  [OFF-GOAL MAP]"
            if allowed is not None and state["map_id"] not in allowed
            else ""
        )
        ax.set_title(
            f"[{state['stage']}] {name} (id={state['map_id']}){off_goal_s}  "
            f"[{idx}/{len(maps)}]  "
            f"metric={state['metric']}\n"
            f"{n_runs} runs - {agg.total_frames:,} frames in window\n"
            f"{_party_line(agg)}"
            f"<-/-> switch map, c: combined view, r: cycle metric, "
            f"v: toggle value labels ({'on' if state['show_values'] else 'off'})\n"
            f"scroll: zoom, drag: pan, 0: reset view"
        )

    def redraw_combined() -> None:
        agg = current_agg()
        result = agg.combined_view(metric=state["metric"])
        if result is None:
            _hide_side_panels()
            _show_placeholder_combined()
            return
        grid, x0, y0, offsets, connected, unconnected = result
        bg_extent = _update_background(map_render.stitched_rgb(offsets))
        im.set_data(grid)
        _apply_extent(x0, y0, grid, bg_extent)
        _apply_metric_style(grid)

        _clear_overlays()
        _draw_quiver(agg._direction_arrays(offsets))
        _draw_value_labels(grid, x0, y0)
        allowed = allowed_maps_by_stage.get(state["stage"])
        vmin, vmax = im.get_clim()
        _populate_unconnected_panels(
            side_axes, agg, unconnected, state["metric"], name_of, allowed,
            im.get_cmap(), vmin, vmax,
        )
        off_goal_count = 0
        for map_id in connected:
            gx0, gy0 = offsets[map_id]
            cells = agg.sum_ticks.get(map_id)
            if not cells:
                continue
            cx = gx0 + sum(p[0] for p in cells) / len(cells)
            cy = gy0 + sum(p[1] for p in cells) / len(cells)
            # Off-goal = this map isn't in the current stage's
            # GOAL_ALLOWED_MAPS, i.e. the same "critical path" set
            # reward_off_goal_camp penalizes dwelling outside of — a purely
            # visual cue, no new per-tile data needed since it's a map-level
            # (not tile-level) classification.
            off_goal = allowed is not None and map_id not in allowed
            if off_goal:
                off_goal_count += 1
            labels.append(
                ax.text(
                    cx, cy, name_of(map_id) + (" [OFF-GOAL]" if off_goal else ""),
                    color="white", fontsize=8, ha="center", va="center",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        fc="#8b1a1a" if off_goal else "black",
                        alpha=0.75 if off_goal else 0.5, lw=0,
                    ),
                )
            )

        off_goal_s = f"  |  {off_goal_count} off-goal map(s)" if off_goal_count else ""
        unconnected_s = (
            f"  |  {len(unconnected)} not yet connected (see side panels)"
            if unconnected else ""
        )
        ax.set_title(
            f"[{state['stage']}] Combined view: {len(connected)} maps stitched  "
            f"metric={state['metric']}\n"
            f"{agg.total_frames:,} frames in window{unconnected_s}{off_goal_s}\n"
            f"{_party_line(agg)}"
            f"c: single-map view, r: cycle metric, "
            f"v: toggle value labels ({'on' if state['show_values'] else 'off'}) "
            f"(best-effort — connections are inferred, not authoritative)\n"
            f"scroll: zoom, drag: pan, 0: reset view"
        )

    def redraw() -> None:
        if state["mode"] == "combined":
            redraw_combined()
        else:
            redraw_single()
        fig.canvas.draw_idle()

    plt.ion()
    plt.show(block=False)
    # Draw the placeholder map immediately (see _show_placeholder) instead
    # of leaving the window blank until the first queue item arrives — the
    # main loop below only calls redraw() when got_any is True.
    redraw()

    while not state["closed"]:
        got_any = False
        try:
            while True:
                item = q.get_nowait()
                if item is STOP:
                    plt.close(fig)
                    return
                (
                    positions, directions, transitions, rewards,
                    battle_outcomes, milestones, dialogs, truncations, visit_counts,
                    steps, stage, party_count, party_avg_level,
                ) = item
                agg_all.add_episode(
                    positions, directions, transitions, rewards,
                    battle_outcomes, milestones, dialogs, truncations, visit_counts,
                    steps, party_count, party_avg_level,
                )
                stage_label = stage or "unknown"
                is_new_stage = stage_label not in stage_aggs
                stage_aggs.setdefault(
                    stage_label, RollingHeatmapAggregator(window_frames)
                ).add_episode(
                    positions, directions, transitions, rewards,
                    battle_outcomes, milestones, dialogs, truncations, visit_counts,
                    steps, party_count, party_avg_level,
                )
                if is_new_stage:
                    _refresh_stage_dropdown()
                got_any = True
        except _queue_mod.Empty:
            pass
        if got_any:
            redraw()
        _refresh_goal_saturation()
        plt.pause(0.5)


def start_heatmap_process(
    window_frames: int, title: str = "Pokemon Red AI - Position Heatmap"
) -> tuple[Process, "Queue"]:
    """Spawn the heatmap window in its own process. Feed it via push_episode()."""
    q: Queue = Queue(maxsize=256)
    proc = Process(target=_run_window, args=(q, window_frames, title), daemon=True)
    proc.start()
    return proc, q


def stop_heatmap_process(proc: Process, q: "Queue") -> None:
    try:
        q.put_nowait(STOP)
    except Exception:
        pass
    proc.join(timeout=2)
    if proc.is_alive():
        proc.terminate()


def push_episode(
    q: "Queue",
    positions: dict,
    directions: dict | None,
    transitions: dict | None,
    rewards: dict | None,
    battle_outcomes: dict | None,
    milestones: dict | None,
    dialogs: dict | None,
    truncations: dict | None,
    visit_counts: dict | None,
    steps: int,
    stage: str | None = None,
    party_count: int | None = None,
    party_avg_level: float | None = None,
) -> None:
    """Non-blocking: drop the update rather than stall the caller if the
    visualizer process is behind."""
    try:
        q.put_nowait(
            (
                positions, directions, transitions, rewards,
                battle_outcomes, milestones, dialogs, truncations, visit_counts,
                steps, stage, party_count, party_avg_level,
            )
        )
    except _queue_mod.Full:
        pass
    except Exception:
        pass


def _slug(label: str) -> str:
    """Filesystem-safe stem for a stage/metric label."""
    out = "".join(c if c.isalnum() else "_" for c in str(label))
    return out.strip("_") or "x"


def save_heatmap_images(
    aggregators: dict[str, "RollingHeatmapAggregator"],
    out_dir,
    metrics: tuple[str, ...] = ("ticks",),
    top_maps: int = 6,
) -> list:
    """Render static PNGs for a batch of pooled runs — one combined-view
    image plus up to ``top_maps`` single-map images (most-visited first),
    per metric, per aggregator in ``aggregators`` (e.g. {"all": agg_all,
    **stage_aggs}).

    Unlike ``_run_window``, this has no event loop or live queue: it's meant
    for the --batch eval mode, which can run far too many episodes (e.g.
    300k) to drive an interactive matplotlib window. Uses the non-interactive
    Agg backend, so it works headless (no display needed).

    Returns the list of written file paths.
    """
    from pathlib import Path

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    from pokemon import map_render
    from pokemon.Data import VISIT_PENALTY_HARD_THRESHOLD, VISIT_PENALTY_SOFT_THRESHOLD

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    map_names = _map_name_lookup()

    _rate_cmap = plt.get_cmap("RdYlGn").copy()
    cmap_of = {
        "ticks": plt.get_cmap("inferno").copy(),
        "reward": plt.get_cmap("coolwarm").copy(),
        "winrate": _rate_cmap,
        "fleerate": _rate_cmap,
        "recency": plt.get_cmap("viridis").copy(),
        "dialog_recency": plt.get_cmap("cividis").copy(),
        "milestones": plt.get_cmap("plasma").copy(),
        "truncations": plt.get_cmap("magma").copy(),
        "visits": plt.get_cmap("YlOrRd").copy(),
    }
    for cmap in cmap_of.values():
        cmap.set_bad(alpha=0.0)
    label_of = {
        "ticks": "avg ticks / run",
        "reward": "avg reward / run",
        "winrate": "battle win rate",
        "fleerate": "flee smart-rate (smart / smart+coward)",
        "recency": "episodes since last visit",
        "dialog_recency": "episodes since last dialog",
        "milestones": "avg milestone hits / run",
        "truncations": "avg truncations / run",
        "visits": f"avg visits / run ÷ {VISIT_PENALTY_SOFT_THRESHOLD}",
    }

    def _clim(grid, metric: str) -> tuple[float, float]:
        finite = grid[~np.isnan(grid)]
        if metric == "reward":
            vmax = float(np.abs(finite).max()) if finite.size else 0.0
            vmax = vmax if vmax > 0 else 1.0
            return -vmax, vmax
        if metric in ("winrate", "fleerate"):
            return 0.0, 1.0
        if metric == "visits":
            # Fixed scale (see the live window's visits_cmap comment) --
            # the threshold ratio, not the data's own max.
            return 0.0, VISIT_PENALTY_HARD_THRESHOLD / VISIT_PENALTY_SOFT_THRESHOLD
        vmax = float(finite.max()) if finite.size else 0.0
        return 0.0, vmax if vmax > 0 else 1.0

    def _value_fmt(metric: str) -> str:
        if metric == "reward":
            return "{:+.2f}"
        if metric in ("winrate", "fleerate"):
            return "{:.0%}"
        if metric in ("milestones", "truncations", "visits"):
            return "{:.2f}"
        return "{:.0f}"

    def _render(
        grid, x0: int, y0: int, title: str, metric: str, path: "Path", dgrid,
        unconnected: list[int] | None = None,
        agg_for_panels: "RollingHeatmapAggregator | None" = None,
        background: tuple | None = None,
    ) -> None:
        # unconnected/agg_for_panels are only passed for the combined-view
        # image, so it can grow a side column of per-map thumbnails for maps
        # the stitcher couldn't place — same purpose as the live window's
        # side panels (_populate_unconnected_panels), just rendered once
        # into the static PNG instead of redrawn interactively.
        if unconnected:
            fig = plt.figure(figsize=(11, 8))
            gs = GridSpec(_MAX_SIDE_PANELS, 4, figure=fig, wspace=0.7, hspace=0.6)
            ax = fig.add_subplot(gs[:, :3])
            side_axes = [fig.add_subplot(gs[i, 3]) for i in range(_MAX_SIDE_PANELS)]
        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            side_axes = []
        ax.set_facecolor(_BG_FACECOLOR)
        # Real map pixel art (pokemon.map_render), drawn first so the
        # metric-color grid layers on top at _OVERLAY_ALPHA. background's own
        # (rgb, bx0, by0) usually covers more than grid's visited-tile bbox
        # (the map's full footprint vs. just what got visited) — but
        # AxesImage.set_extent (below) forcibly sets the axes view to its
        # own extent rather than unioning with what's already there, so the
        # view has to be explicitly widened afterward or it crops to the
        # (smaller) data grid and hides the rest of the background.
        bg_extent = None
        if background is not None:
            bg_img, bx0, by0 = background
            h_cells = bg_img.shape[0] / map_render.CELL_PX
            w_cells = bg_img.shape[1] / map_render.CELL_PX
            bg_extent = (bx0 - 0.5, bx0 + w_cells - 0.5, by0 + h_cells - 0.5, by0 - 0.5)
            ax.imshow(bg_img, origin="upper", zorder=1, extent=bg_extent)
        im = ax.imshow(
            grid, cmap=cmap_of[metric], origin="upper",
            alpha=_OVERLAY_ALPHA, zorder=2,
        )
        data_extent = (x0 - 0.5, x0 + grid.shape[1] - 0.5, y0 + grid.shape[0] - 0.5, y0 - 0.5)
        im.set_extent(data_extent)
        if bg_extent is not None:
            ax.set_xlim(min(data_extent[0], bg_extent[0]), max(data_extent[1], bg_extent[1]))
            ax.set_ylim(max(data_extent[2], bg_extent[2]), min(data_extent[3], bg_extent[3]))
        im.set_clim(*_clim(grid, metric))
        fig.colorbar(im, ax=ax, label=label_of[metric])
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(title, fontsize=10)
        if dgrid is not None:
            xs, ys, us, vs = dgrid
            ax.quiver(
                xs, ys, us, vs,
                color="#4fd1ff", alpha=0.9, pivot="mid",
                angles="xy", scale_units="xy", scale=1.3, width=0.006,
            )
        ys_idx, xs_idx = np.where(~np.isnan(grid))
        if 0 < xs_idx.size <= _MAX_VALUE_LABELS:
            fmt = _value_fmt(metric)
            stroke = [pe.withStroke(linewidth=1.5, foreground="black")]
            for iy, ix in zip(ys_idx, xs_idx):
                ax.text(
                    x0 + ix, y0 + iy, fmt.format(float(grid[iy, ix])),
                    color="white", fontsize=6, ha="center", va="center",
                    path_effects=stroke, zorder=5,
                )
        if side_axes:
            vmin, vmax = im.get_clim()
            _populate_unconnected_panels(
                side_axes, agg_for_panels, unconnected, metric,
                lambda m: map_names.get(m, f"map {m}"), None,
                cmap_of[metric], vmin, vmax,
            )
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    written: list = []
    for label, agg in aggregators.items():
        maps = agg.maps()
        if not maps:
            continue
        slug_label = _slug(label)
        for metric in metrics:
            combined = agg.combined_view(metric=metric)
            if combined is not None:
                grid, x0, y0, offsets, connected, unconnected = combined
                path = out_dir / f"{slug_label}_combined_{metric}.png"
                title = (
                    f"[{label}] combined: {len(connected)} maps stitched "
                    f"({len(unconnected)} unconnected)  metric={metric}\n"
                    f"{agg.total_episodes:,} runs pooled, "
                    f"{agg.total_frames:,} frames"
                )
                _render(
                    grid, x0, y0, title, metric, path,
                    dgrid=agg._direction_arrays(offsets),
                    unconnected=unconnected, agg_for_panels=agg,
                    background=map_render.stitched_rgb(offsets),
                )
                written.append(path)

            for map_id in maps[:top_maps]:
                result = agg.average_grid(map_id, metric=metric)
                if result is None:
                    continue
                grid, x0, y0 = result
                name = map_names.get(map_id, f"map {map_id}")
                path = out_dir / f"{slug_label}_map{map_id}_{_slug(name)}_{metric}.png"
                title = (
                    f"[{label}] {name} (id={map_id})  metric={metric}\n"
                    f"{agg.run_count.get(map_id, 0):,} runs"
                )
                bg_img = map_render.render_map_rgb(map_id)
                _render(
                    grid, x0, y0, title, metric, path,
                    dgrid=agg.direction_grid(map_id),
                    background=(bg_img, 0, 0) if bg_img is not None else None,
                )
                written.append(path)
    return written
