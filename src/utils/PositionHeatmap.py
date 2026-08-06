"""Live exploration heatmap: rolling window of visited (map, x, y) ticks,
averaged per run, rendered in its own process so it never blocks
training/eval. Opt-in via --heatmap on train_ppo.py / run_eval_ppo.py.

Data flow: PokemonRedEnv snapshots Data.visited_positions / direction_counts
/ map_transitions / reward_sums at every "run" boundary (episode end, or a
mid-episode curriculum-leg clear) into info["heatmap_positions"/
"heatmap_directions"/"heatmap_transitions"/"heatmap_rewards"/
"heatmap_steps"]. A callback (training) or the eval loop (eval) forwards
those snapshots through a multiprocessing.Queue to _run_window, which runs
in a separate process and owns the matplotlib window.

Two view modes (toggle with 'c'):
  - single map: one map_id's grid + direction arrows, cycled with left/right.
  - combined: every currently-connected map auto-stitched into one canvas.
    Connectivity is inferred empirically, not from any hardcoded map data —
    every step that crosses a map_id boundary backs out the coordinate
    offset between the two maps' local (x, y) grids from the arrow-key
    direction taken. This is a best-effort heuristic: a door warp into an
    unrelated interior can *look* like a consistent edge connection (same
    fixed offset every time) and get stitched in at a nonsense position —
    there's no memory-flag here distinguishing "walked across an edge" from
    "used a door", so treat the combined view as approximate.

Two color metrics (toggle with 'r'), independent of the view mode:
  - ticks: avg ticks/run per tile (time spent) — the original view.
  - reward: avg reward/run per tile — where reward is actually earned, not
    just where time is spent. A battle-won/lost payout is attributed to the
    grass/trainer tile that started the fight even though the fight itself
    has no world position, so farming loops (grind a tile for repeat battle
    reward, spam a menu for event-flag reward, etc.) show up as a hot tile
    here even when the ticks view looks unremarkable.

Per-tile numeric value labels (toggle with 'v'), independent of view mode
and metric: prints the exact avg-ticks or avg-reward number on top of each
cell, for reading precise values instead of eyeballing color. Auto-hidden
above _MAX_VALUE_LABELS cells since that many text artists stalls redraws.
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


class RollingHeatmapAggregator:
    """Average ticks-spent-per-run, per (map_id, x, y), over the last
    ``window_frames`` steps pooled across every run that fed it. Also tracks
    the dominant walking direction per tile, and (permanently — map
    connectivity doesn't age out like traffic does) empirical map-to-map
    coordinate offsets for the combined view.

    One ``add_episode`` call = one "run": a full episode, or one curriculum
    leg when auto-curriculum clears visited_positions mid-episode. That's
    the unit "average per run" is taken over.
    """

    def __init__(self, window_frames: int):
        self.window_frames = int(window_frames)
        self._runs: deque[
            tuple[
                int,
                dict[tuple[int, int, int], int],
                dict[tuple[int, int, int], dict[str, int]],
                dict[tuple[int, int, int], float],
            ]
        ] = deque()
        self.total_frames = 0
        # map_id -> {(x, y): summed ticks across runs in the window}
        self.sum_ticks: dict[int, dict[tuple[int, int], int]] = {}
        # map_id -> {(x, y): summed reward across runs in the window}
        self.sum_rewards: dict[int, dict[tuple[int, int], float]] = {}
        # map_id -> number of runs in the window that touched this map
        self.run_count: dict[int, int] = {}
        # map_id -> {(x, y): {"up"/"down"/"left"/"right": step count}}
        self.direction_sums: dict[int, dict[tuple[int, int], dict[str, int]]] = {}
        # (from_map, to_map) -> {(dx, dy): vote count} — permanent, never evicted.
        self.transition_votes: dict[tuple[int, int], dict[tuple[int, int], int]] = {}

    def add_episode(
        self,
        positions: dict[tuple[int, int, int], int],
        directions: dict[tuple[int, int, int], dict[str, int]] | None,
        transitions: dict[tuple[int, int], dict[tuple[int, int], int]] | None,
        rewards: dict[tuple[int, int, int], float] | None,
        steps: int,
    ) -> None:
        if not positions or steps <= 0:
            return
        directions = directions or {}
        rewards = rewards or {}
        self._runs.append((steps, positions, directions, rewards))
        self.total_frames += steps
        touched_maps = set()
        for (x, y, map_id), ticks in positions.items():
            touched_maps.add(map_id)
            grid = self.sum_ticks.setdefault(map_id, {})
            grid[(x, y)] = grid.get((x, y), 0) + ticks
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
        for key, votes in (transitions or {}).items():
            dest = self.transition_votes.setdefault(key, {})
            for delta, c in votes.items():
                dest[delta] = dest.get(delta, 0) + c
        self._evict()

    def _evict(self) -> None:
        while self._runs and self.total_frames > self.window_frames:
            steps, positions, directions, rewards = self._runs.popleft()
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
            # transition_votes is intentionally NOT evicted here — map
            # connectivity is structural, not recent-behavior traffic.

    def maps(self) -> list[int]:
        """Map ids currently in the window, most-visited first."""
        return sorted(self.sum_ticks, key=lambda m: -sum(self.sum_ticks[m].values()))

    def _metric_sums(self, metric: str) -> dict[int, dict[tuple[int, int], float]]:
        return self.sum_rewards if metric == "reward" else self.sum_ticks

    def average_grid(self, map_id: int, metric: str = "ticks"):
        """(grid, x0, y0): grid[y - y0, x - x0] = avg value/run for
        ``metric`` ("ticks" or "reward"), NaN where unvisited in the current
        window. None if the map isn't in the window.

        The footprint (bounding box + which cells are "visited") always
        follows the ticks grid — reward is attributed to the same tiles
        visited_positions tracks, so ticks is the authoritative visited-set;
        a tile with exactly zero net reward is still "visited", not NaN.
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
        source = self._metric_sums(metric).get(map_id, {})
        for (x, y), ticks in grid_ticks.items():
            val = ticks if metric == "ticks" else source.get((x, y), 0.0)
            out[y - y0, x - x0] = val / n_runs
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
        """BFS the empirical connection graph from ``anchor`` (offset (0,0)).
        Maps with no discovered path to the anchor are simply absent.

        A plain BFS spanning tree trusts whichever path reaches a map first
        and never looks back — if that map is *also* reachable via a second
        confirmed edge implying a different offset (a door-warp delta that
        got "confirmed" as if it were geometric, most commonly), the
        disagreement would otherwise go unnoticed and the map gets stitched
        in at one arbitrary, possibly overlapping position. Detect that case
        and drop the map — and everything stitched in only through it —
        rather than render a silently-wrong placement.
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
        ``metric`` ("ticks" or "reward").

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
        for map_id in connected:
            gx0, gy0 = offsets[map_id]
            n_runs = max(self.run_count.get(map_id, 1), 1)
            reward_grid = self.sum_rewards.get(map_id, {})
            for (x, y), ticks in self.sum_ticks.get(map_id, {}).items():
                val = ticks if metric == "ticks" else reward_grid.get((x, y), 0.0)
                cells.append((x + gx0, y + gy0, val / n_runs))
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


def _map_name_lookup() -> dict[int, str]:
    from pokemon import Data as _data_module

    names: dict[int, str] = {}
    for key, value in vars(_data_module).items():
        if key.startswith("MAP_") and isinstance(value, int):
            names.setdefault(value, key[len("MAP_"):].replace("_", " ").title())
    return names


_ALL_STAGES = "All"


def _stage_order_lookup() -> list[str]:
    """Canonical curriculum stage order, used only to sort the dropdown —
    stages actually seen in the data are shown regardless of whether they're
    in this list."""
    try:
        from curriculum_config import STAGE_ORDER

        return list(STAGE_ORDER)
    except Exception:
        return []


def _run_window(q: "Queue", window_frames: int, title: str) -> None:
    import matplotlib

    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    map_names = _map_name_lookup()
    stage_order = _stage_order_lookup()
    # "All" pools every run regardless of stage; stage_aggs holds one
    # independent rolling aggregator per curriculum stage seen so far.
    agg_all = RollingHeatmapAggregator(window_frames)
    stage_aggs: dict[str, RollingHeatmapAggregator] = {}
    # ticks: non-negative "time spent", sequential colormap from a dark floor.
    # reward: signed "reward earned", diverging colormap centered on zero so
    # farming/exploit hotspots (strongly positive) read differently from
    # penalty hotspots (strongly negative) instead of both just being "hot".
    ticks_cmap = plt.get_cmap("inferno").copy()
    ticks_cmap.set_bad(color="#111111")
    reward_cmap = plt.get_cmap("coolwarm").copy()
    reward_cmap.set_bad(color="#111111")

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.canvas.manager.set_window_title(title)
    im = ax.imshow(np.zeros((1, 1)), cmap=ticks_cmap, origin="upper")
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
    }

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
            redraw()
            return
        if event.key == "r":
            state["metric"] = "reward" if state["metric"] == "ticks" else "ticks"
            redraw()
            return
        if event.key == "v":
            state["show_values"] = not state["show_values"]
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
            redraw()
        elif event.key in ("left", "p"):
            state["map_id"] = maps[(idx - 1) % len(maps)]
            redraw()

    fig.canvas.mpl_connect("close_event", on_close)
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Stage dropdown, embedded directly in the TkAgg window above the canvas.
    stage_combo = None
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
                redraw()

            stage_combo.bind("<<ComboboxSelected>>", _on_stage_selected)
    except Exception:
        stage_combo = None

    def _refresh_stage_dropdown() -> None:
        """Rebuild the dropdown's option list from stages seen so far."""
        if stage_combo is None:
            return
        values = [_ALL_STAGES] + sorted(stage_aggs.keys(), key=_stage_sort_key)
        if list(stage_combo["values"]) != values:
            stage_combo["values"] = values

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
            vmax = float(np.abs(finite).max()) if finite.size else 1.0
            vmax = max(vmax, 1.0)
            im.set_clim(-vmax, vmax)
            cbar.set_label("avg reward / run")
        else:
            im.set_cmap(ticks_cmap)
            vmax = float(finite.max()) if finite.size else 1.0
            im.set_clim(0, max(vmax, 1.0))
            cbar.set_label("avg ticks / run")

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
        fmt = "{:+.2f}" if state["metric"] == "reward" else "{:.0f}"
        stroke = [pe.withStroke(linewidth=1.5, foreground="black")]
        for iy, ix in zip(ys, xs):
            value_texts.append(
                ax.text(
                    x0 + ix, y0 + iy, fmt.format(float(grid[iy, ix])),
                    color="white", fontsize=6, ha="center", va="center",
                    path_effects=stroke, zorder=5,
                )
            )

    def redraw_single() -> None:
        agg = current_agg()
        maps = agg.maps()
        if not maps:
            ax.set_title(f"Heatmap [{state['stage']}] - waiting for runs...")
            fig.canvas.draw_idle()
            return
        if state["map_id"] not in maps:
            state["map_id"] = maps[0]
        result = agg.average_grid(state["map_id"], metric=state["metric"])
        if result is None:
            return
        grid, x0, y0 = result
        im.set_data(grid)
        im.set_extent(
            (x0 - 0.5, x0 + grid.shape[1] - 0.5, y0 + grid.shape[0] - 0.5, y0 - 0.5)
        )
        _apply_metric_style(grid)

        _clear_overlays()
        _draw_quiver(agg.direction_grid(state["map_id"]))
        _draw_value_labels(grid, x0, y0)

        name = name_of(state["map_id"])
        n_runs = agg.run_count.get(state["map_id"], 0)
        idx = maps.index(state["map_id"]) + 1
        ax.set_title(
            f"[{state['stage']}] {name} (id={state['map_id']})  [{idx}/{len(maps)}]  "
            f"metric={state['metric']}\n"
            f"{n_runs} runs - {agg.total_frames:,} frames in window\n"
            f"<-/-> switch map, c: combined view, r: toggle ticks/reward, "
            f"v: toggle value labels ({'on' if state['show_values'] else 'off'})"
        )

    def redraw_combined() -> None:
        agg = current_agg()
        result = agg.combined_view(metric=state["metric"])
        if result is None:
            ax.set_title(f"Heatmap [{state['stage']}] - waiting for runs...")
            fig.canvas.draw_idle()
            return
        grid, x0, y0, offsets, connected, unconnected = result
        im.set_data(grid)
        im.set_extent(
            (x0 - 0.5, x0 + grid.shape[1] - 0.5, y0 + grid.shape[0] - 0.5, y0 - 0.5)
        )
        _apply_metric_style(grid)

        _clear_overlays()
        _draw_quiver(agg._direction_arrays(offsets))
        _draw_value_labels(grid, x0, y0)
        for map_id in connected:
            gx0, gy0 = offsets[map_id]
            cells = agg.sum_ticks.get(map_id)
            if not cells:
                continue
            cx = gx0 + sum(p[0] for p in cells) / len(cells)
            cy = gy0 + sum(p[1] for p in cells) / len(cells)
            labels.append(
                ax.text(
                    cx, cy, name_of(map_id),
                    color="white", fontsize=8, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5, lw=0),
                )
            )

        unconnected_s = (
            f"  |  {len(unconnected)} not yet connected: "
            f"{', '.join(name_of(m) for m in unconnected[:6])}"
            f"{' ...' if len(unconnected) > 6 else ''}"
            if unconnected else ""
        )
        ax.set_title(
            f"[{state['stage']}] Combined view: {len(connected)} maps stitched  "
            f"metric={state['metric']}\n"
            f"{agg.total_frames:,} frames in window{unconnected_s}\n"
            f"c: single-map view, r: toggle ticks/reward, "
            f"v: toggle value labels ({'on' if state['show_values'] else 'off'}) "
            f"(best-effort — connections are inferred, not authoritative)"
        )

    def redraw() -> None:
        if state["mode"] == "combined":
            redraw_combined()
        else:
            redraw_single()
        fig.canvas.draw_idle()

    plt.ion()
    plt.show(block=False)

    while not state["closed"]:
        got_any = False
        try:
            while True:
                item = q.get_nowait()
                if item is STOP:
                    plt.close(fig)
                    return
                positions, directions, transitions, rewards, steps, stage = item
                agg_all.add_episode(positions, directions, transitions, rewards, steps)
                stage_label = stage or "unknown"
                is_new_stage = stage_label not in stage_aggs
                stage_aggs.setdefault(
                    stage_label, RollingHeatmapAggregator(window_frames)
                ).add_episode(positions, directions, transitions, rewards, steps)
                if is_new_stage:
                    _refresh_stage_dropdown()
                got_any = True
        except _queue_mod.Empty:
            pass
        if got_any:
            redraw()
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
    steps: int,
    stage: str | None = None,
) -> None:
    """Non-blocking: drop the update rather than stall the caller if the
    visualizer process is behind."""
    try:
        q.put_nowait((positions, directions, transitions, rewards, steps, stage))
    except _queue_mod.Full:
        pass
    except Exception:
        pass
