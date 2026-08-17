#!/usr/bin/env python
"""Offline viewer: every coordinate-gated event entry point
(pokemon.map_scripts.COORD_TRIGGERS -- state coordinate guards AND NPC/sign/
item/trainer interactions, see tools/gen_map_scripts.py) pinned onto the
map's real pixel art (pokemon.map_render), per map. Click a pin to see the
dispatch-table state it belongs to: its guard chain, its real asm body, and
its immediate neighbors (which states can reach it / where it goes next) --
a first cut at a "reward compass" that's grounded in real entry points
instead of a single BFS direction vector (see pokemon.Data.compass_progress).

This is a companion to tools/view_script_graph.py (that one shows a whole
map's dispatch-state DAG in the abstract; this one shows WHERE on the map
each event-reachable point actually is). Same offline HTML convention: one
self-contained file, no server, no CDN.

The viewer also lets you manually check off events you've already done
(there's no live game connection -- this is a static analysis tool, not an
overlay) and, per pokemon.event_graph's parent/child edges, highlights which
un-done events are "next-doable" right now (every direct parent already
checked off) -- the same rule pokemon.Data._goal_parents_satisfied applies
against live game state, just driven by hand here. A "show only next-doable"
toggle filters the map down to that frontier. Progress persists in the
browser's localStorage.

Usage:
  python tools/view_map_pins.py
  python tools/view_map_pins.py --out my_view.html
  python tools/view_map_pins.py --map PALLET_TOWN
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
from gen_script_graph_data import build_data as build_script_data  # noqa: E402

# gen_script_graph_data -> gen_map_scripts -> pokered_asm puts src/ on
# sys.path as a side effect of the import above, so pokemon.* is importable
# from here on.
from pokemon import map_render as mr  # noqa: E402
from pokemon import map_scripts as ms  # noqa: E402
from pokemon.map_constants import MAPS_BY_ID  # noqa: E402


def prettify_map_name(const_name: str) -> str:
    return " ".join(w.capitalize() for w in const_name.split("_"))


def build_data() -> dict[str, dict]:
    """map_id (str) -> {map_id, const_name, name, w, h, cell_px, png (base64),
    pins: [{event, kind: "point"|"line", x, y, axis, value, state_idx,
    handler, guards, sets, code, next_state, preds}]}, for every map with at
    least one COORD_TRIGGERS entry AND renderable pixel art. Most entries
    have both axes known ("point"); a handful (e.g. Pallet Town's "Oak
    appears" cutscene, gated only on wYCoord == 1, any x -- the player can
    trigger it by entering from Route 1 at any column) only constrain one
    axis -- those become a "line" pin spanning the whole row/column instead
    of being silently dropped."""
    script_data = build_script_data()

    pins_by_map: dict[int, list[dict]] = {}
    for event, locs in ms.COORD_TRIGGERS.items():
        for map_id, x, y in locs:
            if x is not None and y is not None:
                pin = {"event": event, "kind": "point", "x": x, "y": y}
            elif x is not None:
                pin = {"event": event, "kind": "line", "axis": "x", "value": x}
            elif y is not None:
                pin = {"event": event, "kind": "line", "axis": "y", "value": y}
            else:
                continue  # both axes unconstrained -- not a placeable trigger
            pins_by_map.setdefault(map_id, []).append(pin)

    out: dict[str, dict] = {}
    for map_id, pins in sorted(pins_by_map.items()):
        img = mr.render_map_rgb(map_id)
        if img is None:
            continue

        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        map_states = script_data.get(str(map_id), {}).get("states", {})
        preds: dict[str, list[str]] = {}
        for idx, st in map_states.items():
            ns = st.get("next_state")
            if ns is not None:
                preds.setdefault(str(ns), []).append(idx)

        enriched: list[dict] = []
        for p in pins:
            match_idx = next(
                (idx for idx, st in map_states.items() if p["event"] in (st.get("sets") or [])),
                None,
            )
            entry = dict(p)
            if match_idx is not None:
                st = map_states[match_idx]
                entry.update(
                    state_idx=match_idx,
                    handler=st["handler"],
                    guards=st["guards"],
                    sets=st["sets"],
                    code=st["code"],
                    next_state=st["next_state"],
                    preds=preds.get(match_idx, []),
                )
            enriched.append(entry)

        entry_name = MAPS_BY_ID[map_id][0]
        out[str(map_id)] = {
            "map_id": map_id,
            "const_name": entry_name,
            "name": prettify_map_name(entry_name),
            "w": int(img.shape[1]),
            "h": int(img.shape[0]),
            "cell_px": mr.CELL_PX,
            "png": png_b64,
            "pins": enriched,
            "pin_count": len(enriched),
        }
    return out


def build_event_parents(maps: dict[str, dict]) -> dict[str, list[str]]:
    """event name -> its direct pokemon.event_graph parents, for every event
    that appears as a pin anywhere, mirroring Data._goal_parents_satisfied's
    own (non-recursive -- only direct parents matter) semantics: an event is
    "doable next" once every one of these is marked done in the viewer, the
    same rule the live compass uses against real game flags. Only direct
    parents are included (not transitively); the client still lets a parent
    that isn't itself a pin be marked done via its guard-chip in the detail
    panel."""
    from pokemon import event_graph as eg

    out: dict[str, list[str]] = {}
    for m in maps.values():
        for pin in m["pins"]:
            event = pin["event"]
            if event in out:
                continue
            info = eg.EVENT_GRAPH.get(event)
            out[event] = list(info["parents"]) if info else []
    return out


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Map Pins Explorer</title>
<style>
:root {
  --bg: #14171b;
  --surface: #1b1f26;
  --surface-2: #242a33;
  --surface-3: #2c333e;
  --line: #333c49;
  --line-soft: #262d38;
  --text: #d7dce3;
  --text-dim: #8590a3;
  --text-faint: #576078;
  --accent: #9bbc0f;
  --accent-strong: #c5e05a;
  --accent-dim: #3c4a16;
  --accent-ink: #0f1400;
  --guard-event: #e8a33d;
  --guard-event-dim: #4a3517;
  --guard-coord: #5fb3d9;
  --guard-coord-dim: #17323d;
  --set-chip: #7fd88f;
  --set-chip-dim: #1c3320;
  --pin: #ff5f6d;
  --pin-ink: #2a0002;
  --pin-ring: rgba(255,95,109,0.35);
  --shadow: 0 8px 24px rgba(0,0,0,0.45);
  --mono: ui-monospace, "JetBrains Mono", "Cascadia Code", "SF Mono", Consolas, "Roboto Mono", "Liberation Mono", monospace;
}
:root[data-theme="light"] {
  --bg: #eef0e4;
  --surface: #ffffff;
  --surface-2: #f4f6ec;
  --surface-3: #e9ecdd;
  --line: #cdd3bd;
  --line-soft: #dde1cf;
  --text: #1d2417;
  --text-dim: #566047;
  --text-faint: #8a927a;
  --accent: #4d6b0a;
  --accent-strong: #3a5006;
  --accent-dim: #dbe8b8;
  --accent-ink: #eefccb;
  --guard-event: #9a6111;
  --guard-event-dim: #f5e3c4;
  --guard-coord: #1c6f8e;
  --guard-coord-dim: #d3ecf5;
  --set-chip: #2f7a44;
  --set-chip-dim: #d9f0dd;
  --pin: #c9313c;
  --pin-ink: #fff0ef;
  --pin-ring: rgba(201,49,60,0.25);
  --shadow: 0 8px 24px rgba(60,66,40,0.12);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14171b; --surface: #1b1f26; --surface-2: #242a33; --surface-3: #2c333e;
    --line: #333c49; --line-soft: #262d38; --text: #d7dce3; --text-dim: #8590a3; --text-faint: #576078;
    --accent: #9bbc0f; --accent-strong: #c5e05a; --accent-dim: #3c4a16; --accent-ink: #0f1400;
    --guard-event: #e8a33d; --guard-event-dim: #4a3517; --guard-coord: #5fb3d9; --guard-coord-dim: #17323d;
    --set-chip: #7fd88f; --set-chip-dim: #1c3320;
    --pin: #ff5f6d; --pin-ink: #2a0002; --pin-ring: rgba(255,95,109,0.35);
    --shadow: 0 8px 24px rgba(0,0,0,0.45);
  }
}
:root[data-theme="dark"] {
  --bg: #14171b; --surface: #1b1f26; --surface-2: #242a33; --surface-3: #2c333e;
  --line: #333c49; --line-soft: #262d38; --text: #d7dce3; --text-dim: #8590a3; --text-faint: #576078;
  --accent: #9bbc0f; --accent-strong: #c5e05a; --accent-dim: #3c4a16; --accent-ink: #0f1400;
  --guard-event: #e8a33d; --guard-event-dim: #4a3517; --guard-coord: #5fb3d9; --guard-coord-dim: #17323d;
  --set-chip: #7fd88f; --set-chip-dim: #1c3320;
  --pin: #ff5f6d; --pin-ink: #2a0002; --pin-ring: rgba(255,95,109,0.35);
  --shadow: 0 8px 24px rgba(0,0,0,0.45);
}

* { box-sizing: border-box; }
html, body { height: 100%; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: var(--mono);
  font-size: 13px;
  overflow: hidden;
}

.app { display: grid; grid-template-columns: 260px 1fr; grid-template-rows: 48px 1fr; height: 100vh; }
.app.with-detail { grid-template-columns: 260px 1fr 380px; }

.topbar {
  grid-column: 1 / -1;
  display: flex; align-items: center; gap: 14px;
  padding: 0 16px;
  background: var(--surface);
  border-bottom: 1px solid var(--line);
}
.brand { display: flex; align-items: baseline; gap: 8px; }
.brand .mark { color: var(--accent); font-weight: 700; letter-spacing: 0.04em; }
.brand .sub { color: var(--text-faint); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase; }
.topbar .stats { margin-left: auto; display: flex; gap: 18px; color: var(--text-dim); font-size: 11.5px; }
.topbar .stats b { color: var(--text); font-variant-numeric: tabular-nums; }
.theme-toggle {
  background: var(--surface-2); border: 1px solid var(--line); color: var(--text-dim);
  border-radius: 4px; padding: 5px 9px; font-family: var(--mono); font-size: 11px; cursor: pointer;
}
.theme-toggle:hover { color: var(--text); border-color: var(--text-faint); }

.sidebar {
  background: var(--surface);
  border-right: 1px solid var(--line);
  display: flex; flex-direction: column;
  min-height: 0;
}
.search-wrap { padding: 10px; border-bottom: 1px solid var(--line-soft); }
.search-wrap input {
  width: 100%; background: var(--surface-2); border: 1px solid var(--line); color: var(--text);
  padding: 7px 9px; border-radius: 4px; font-family: var(--mono); font-size: 12.5px;
}
.search-wrap input::placeholder { color: var(--text-faint); }
.progress-wrap { padding: 8px 10px; border-bottom: 1px solid var(--line-soft); }
.chk-row { display: flex; align-items: center; gap: 6px; font-size: 11.5px; color: var(--text-dim); cursor: pointer; user-select: none; }
.chk-row input { accent-color: var(--accent); cursor: pointer; }
.progress-stats { margin-top: 6px; font-size: 10.5px; color: var(--text-faint); }
.progress-stats #doneCount { color: var(--set-chip); font-weight: 700; }
.progress-stats button {
  background: none; border: none; color: var(--text-faint); text-decoration: underline;
  cursor: pointer; font-family: var(--mono); font-size: 10.5px; padding: 0;
}
.progress-stats button:hover { color: var(--text); }
.map-list { overflow-y: auto; flex: 1; padding: 6px; }
.map-item {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 9px; border-radius: 4px; cursor: pointer; color: var(--text-dim);
  border: 1px solid transparent;
}
.map-item:hover { background: var(--surface-2); color: var(--text); }
.map-item.active { background: var(--accent-dim); color: var(--accent-strong); border-color: var(--accent-dim); }
.map-item .n { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12.5px; }
.map-item .c {
  font-size: 10.5px; color: var(--text-faint); background: var(--surface-3);
  border-radius: 999px; padding: 1px 7px; font-variant-numeric: tabular-nums;
}
.map-item.active .c { background: var(--accent-dim); color: var(--accent); }
.map-item .nx {
  font-size: 10.5px; color: var(--guard-event); background: var(--guard-event-dim);
  border-radius: 999px; padding: 1px 7px; font-variant-numeric: tabular-nums;
}
.empty-hint { color: var(--text-faint); font-size: 11.5px; padding: 14px 10px; text-align: center; }

.canvas-wrap { position: relative; overflow: hidden; background: var(--bg); }
.canvas-wrap.grabbing { cursor: grabbing; }
.canvas-wrap:not(.grabbing) { cursor: grab; }
.bgdots {
  position: absolute; inset: 0;
  background-image: radial-gradient(var(--line-soft) 1px, transparent 1px);
  background-size: 22px 22px;
  pointer-events: none;
}
.viewport {
  position: absolute; top: 0; left: 0;
  transform-origin: 0 0;
}
.viewport img.mapart {
  display: block;
  image-rendering: pixelated;
  image-rendering: crisp-edges;
}
.header-strip {
  position: absolute; top: 10px; left: 14px; right: 14px; z-index: 3;
  display: flex; align-items: baseline; gap: 10px; pointer-events: none;
}
.header-strip .fn {
  color: var(--text); font-weight: 700; font-size: 14px; letter-spacing: 0.01em;
  background: var(--surface); padding: 4px 10px; border: 1px solid var(--line); border-radius: 4px;
  pointer-events: auto; box-shadow: var(--shadow);
}
.header-strip .meta { color: var(--text-dim); font-size: 11px; }
.zoom-controls {
  position: absolute; bottom: 14px; right: 14px; z-index: 3;
  display: flex; flex-direction: column; gap: 4px;
}
.zoom-controls button {
  width: 28px; height: 28px; background: var(--surface); border: 1px solid var(--line); color: var(--text-dim);
  border-radius: 4px; cursor: pointer; font-family: var(--mono); font-size: 14px; box-shadow: var(--shadow);
}
.zoom-controls button:hover { color: var(--text); border-color: var(--text-faint); }
.legend {
  position: absolute; bottom: 14px; left: 14px; z-index: 3;
  background: var(--surface); border: 1px solid var(--line); border-radius: 5px;
  padding: 8px 10px; font-size: 10.5px; color: var(--text-dim); box-shadow: var(--shadow);
  display: flex; flex-direction: column; gap: 5px;
}
.legend .row { display: flex; align-items: center; gap: 6px; }
.legend .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.legend .sw { width: 14px; border-top: 2px dashed var(--pin); display: inline-block; }

.pin {
  position: absolute;
  width: 14px; height: 14px;
  margin-left: -7px; margin-top: -7px;
  border-radius: 50%;
  background: var(--pin);
  border: 2px solid var(--pin-ink);
  box-shadow: 0 0 0 4px var(--pin-ring);
  cursor: pointer;
  z-index: 2;
}
.pin:hover { transform: scale(1.35); }
.pin.active { transform: scale(1.5); box-shadow: 0 0 0 6px var(--pin-ring); }
.pin.next { background: var(--guard-event); box-shadow: 0 0 0 4px var(--guard-event-dim); animation: pin-pulse 1.6s ease-in-out infinite; }
.pin.done { background: var(--set-chip); box-shadow: 0 0 0 4px var(--set-chip-dim); opacity: 0.85; }
.pin.locked { opacity: 0.45; }
.pin.dimmed { display: none; }
@keyframes pin-pulse {
  0%, 100% { box-shadow: 0 0 0 4px var(--guard-event-dim); }
  50% { box-shadow: 0 0 0 8px var(--guard-event-dim); }
}
.pin-line {
  position: absolute;
  background: var(--pin-ring);
  border-top: 2px dashed var(--pin);
  border-bottom: 2px dashed var(--pin);
  cursor: pointer;
  z-index: 1;
}
.pin-line.vertical { border-top: none; border-bottom: none; border-left: 2px dashed var(--pin); border-right: 2px dashed var(--pin); }
.pin-line:hover, .pin-line.active { background: var(--pin); opacity: 0.55; }
.pin-line.next { border-color: var(--guard-event); background: var(--guard-event-dim); }
.pin-line.done { border-color: var(--set-chip); background: var(--set-chip-dim); opacity: 0.6; }
.pin-line.locked { opacity: 0.4; }
.pin-line.dimmed { display: none; }
.pin-tip {
  position: absolute;
  transform: translate(-50%, -130%);
  background: var(--surface); border: 1px solid var(--line); border-radius: 4px;
  padding: 3px 7px; font-size: 10.5px; color: var(--text); white-space: nowrap;
  pointer-events: none; box-shadow: var(--shadow); z-index: 4;
}

.detail {
  background: var(--surface);
  border-left: 1px solid var(--line);
  overflow-y: auto;
  padding: 14px;
  display: none;
}
.app.with-detail .detail { display: block; }
.detail h2 { margin: 0 0 4px 0; font-size: 13px; color: var(--accent-strong); }
.detail .sub { color: var(--text-faint); font-size: 10.5px; margin-bottom: 12px; }
.detail .close {
  float: right; background: none; border: none; color: var(--text-faint);
  cursor: pointer; font-size: 15px; line-height: 1;
}
.detail .close:hover { color: var(--text); }
.guards { display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0; }
.guard-chip {
  font-size: 10px; padding: 2px 6px; border-radius: 3px; border: 1px solid transparent; white-space: nowrap;
}
.guard-chip.event-req { color: var(--guard-event); background: var(--guard-event-dim); cursor: pointer; }
.guard-chip.event-forbid { color: var(--guard-event); background: var(--guard-event-dim); opacity: 0.65; text-decoration: line-through; cursor: pointer; }
.guard-chip.event-req:hover, .guard-chip.event-forbid:hover { outline: 1px solid var(--guard-event); }
.guard-chip.chip-done { color: var(--set-chip); background: var(--set-chip-dim); text-decoration: none; opacity: 1; }
.guard-chip.coord { color: var(--guard-coord); background: var(--guard-coord-dim); }
.set-chip {
  font-size: 10px; padding: 2px 7px; border-radius: 999px; margin: 2px 4px 2px 0; display: inline-block;
  color: var(--set-chip); background: var(--set-chip-dim); cursor: pointer; border: 1px solid transparent;
}
.set-chip.unmarked { color: var(--text-faint); background: var(--surface-2); border-color: var(--line); }
.set-chip:hover { border-color: var(--set-chip); }
.rel-chip {
  font-size: 10px; padding: 2px 7px; border-radius: 4px; margin: 2px 4px 2px 0; display: inline-block;
  color: var(--text-dim); background: var(--surface-2); border: 1px solid var(--line); cursor: pointer;
}
.rel-chip:hover { color: var(--text); border-color: var(--text-faint); }
.detail .code {
  margin: 8px 0; padding: 8px 9px; background: var(--bg); border: 1px solid var(--line-soft);
  border-radius: 4px; font-size: 11px; line-height: 1.55; max-height: 320px; overflow: auto; white-space: pre;
}
.detail .code::-webkit-scrollbar { width: 8px; }
.detail .code::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }
.detail .section-label { font-size: 10.5px; color: var(--text-faint); text-transform: uppercase; letter-spacing: 0.06em; margin: 12px 0 4px; }
.detail .none { color: var(--text-faint); font-size: 11px; font-style: italic; }
.tok-label { color: var(--accent-strong); }
.tok-mnem { color: var(--guard-coord); }
.tok-macro { color: #c792ea; }
.tok-const { color: var(--set-chip); }
.tok-num { color: var(--guard-event); }
.tok-comment { color: var(--text-faint); font-style: italic; }
:root[data-theme="light"] .tok-macro { color: #9d5bc9; }

.welcome {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  flex-direction: column; gap: 10px; color: var(--text-faint); text-align: center; padding: 20px;
}
.welcome .big { font-size: 15px; color: var(--text-dim); }
.welcome .small { font-size: 11.5px; max-width: 420px; }
</style>
</head>
<body>

<div class="app" id="app">
  <div class="topbar">
    <div class="brand">
      <span class="mark">&#9646;&#9646; MAP PINS</span>
      <span class="sub">event entry points &middot; projected on real map art &middot; offline</span>
    </div>
    <div class="stats">
      <span>maps <b id="statMaps">0</b></span>
      <span>pins <b id="statPins">0</b></span>
    </div>
    <button class="theme-toggle" id="themeToggle" type="button" title="Toggle theme">&#9680; theme</button>
  </div>

  <div class="sidebar">
    <div class="search-wrap">
      <input id="search" type="text" placeholder="Filter maps... (e.g. oak, gym, route)" autocomplete="off" />
    </div>
    <div class="progress-wrap">
      <label class="chk-row"><input type="checkbox" id="filterNext" /> Show only next-doable</label>
      <div class="progress-stats"><span id="doneCount">0</span> event(s) marked done &middot; <button id="clearDone" type="button">clear</button></div>
    </div>
    <div class="map-list" id="mapList" role="listbox" aria-label="Maps with event pins"></div>
  </div>

  <div class="canvas-wrap" id="canvasWrap">
    <div class="bgdots"></div>
    <div class="header-strip" id="headerStrip" style="display:none">
      <span class="fn" id="fnName"></span>
      <span class="meta" id="fnMeta"></span>
    </div>
    <div class="viewport" id="viewport"></div>
    <div class="welcome" id="welcome">
      <div class="big">Select a map on the left</div>
      <div class="small">Every pin is a coordinate the game checks against a real <code>EVENT_*</code> flag getting set (pokemon.map_scripts.COORD_TRIGGERS) -- a state's own coordinate guard, or an NPC/sign/item/trainer interaction. Click a pin to see its guard chain and real asm body, and tick "Mark this event done" (or click any event chip) to track progress -- amber pins are next-doable once their event_graph parents are all marked done, green is done. "Show only next-doable" hides the rest. Drag to pan, scroll to zoom.</div>
    </div>
    <div class="zoom-controls">
      <button id="zoomIn" type="button" title="Zoom in">+</button>
      <button id="zoomReset" type="button" title="Reset view">&#9678;</button>
      <button id="zoomOut" type="button" title="Zoom out">&minus;</button>
    </div>
    <div class="legend" id="legend" style="display:none">
      <div class="row"><span class="dot" style="background:var(--guard-event)"></span> next-doable (parents all marked done)</div>
      <div class="row"><span class="dot" style="background:var(--set-chip)"></span> marked done</div>
      <div class="row"><span class="dot" style="background:var(--pin)"></span> locked / unknown</div>
      <div class="row"><span class="sw"></span> whole row/col trigger</div>
    </div>
  </div>

  <div class="detail" id="detail"></div>
</div>

<script id="graph-data" type="application/json">__DATA_JSON__</script>
<script>
(function () {
  "use strict";
  var PAYLOAD = JSON.parse(document.getElementById("graph-data").textContent);
  var DATA = PAYLOAD.maps;
  var EVENT_PARENTS = PAYLOAD.event_parents;
  var root = document.documentElement;
  var INITIAL_MAP = __INITIAL_MAP__;

  // ---------- "done" event tracking (manual checklist -- there's no live
  // game connection here, so progress is whatever the user marks) ----------
  var doneEvents = (function () {
    try {
      var raw = localStorage.getItem("mp-done-events");
      return raw ? new Set(JSON.parse(raw)) : new Set();
    } catch (e) { return new Set(); }
  })();
  var filterNext = false;
  var doneChangeListeners = [];

  function saveDone() {
    try { localStorage.setItem("mp-done-events", JSON.stringify(Array.from(doneEvents))); } catch (e) {}
  }
  function isDone(event) { return doneEvents.has(event); }
  // Mirrors Data._goal_parents_satisfied: unlocked once every direct
  // event_graph parent is satisfied -- here "satisfied" means "marked done
  // in this checklist", the manual stand-in for live game state.
  function isNext(event) {
    if (isDone(event)) return false;
    var parents = EVENT_PARENTS[event] || [];
    return parents.every(function (p) { return doneEvents.has(p); });
  }
  function pinState(event) {
    if (isDone(event)) return "done";
    if (isNext(event)) return "next";
    return "locked";
  }
  function toggleDone(event) {
    if (doneEvents.has(event)) doneEvents.delete(event); else doneEvents.add(event);
    saveDone();
    doneChangeListeners.forEach(function (fn) { fn(); });
  }
  function onDoneChange(fn) { doneChangeListeners.push(fn); }

  function applyStoredTheme() {
    try {
      var t = localStorage.getItem("mp-theme");
      if (t === "dark" || t === "light") root.setAttribute("data-theme", t);
    } catch (e) {}
  }
  applyStoredTheme();
  document.getElementById("themeToggle").addEventListener("click", function () {
    var cur = root.getAttribute("data-theme");
    var mql = window.matchMedia("(prefers-color-scheme: dark)").matches;
    var next;
    if (!cur) next = mql ? "light" : "dark";
    else next = cur === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("mp-theme", next); } catch (e) {}
  });

  var MACROS = /^(CheckEvent|CheckEventReuseA|CheckEventHL|CheckBothEventsSet|CheckEitherEventSet|SetEvent|SetEvents|SetEventReuseHL|ResetEvent|trainer|predef|predef_jump|farcall|def_script_pointers|dw_const|text_far|text_asm|text_end)\b/;
  var MNEM = /^(ld|ldh|jp|jr|call|ret|reti|cp|and|or|xor|inc|dec|push|pop|bit|res|set|rst|add|sub|adc|sbc|swap|rlca|rrca|rla|rra|nop|halt|di|ei|db|dw)\b/i;

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function highlightLine(line) {
    var commentIdx = -1, inStr = false;
    for (var i = 0; i < line.length; i++) {
      var c = line[i];
      if (c === '"') inStr = !inStr;
      if (c === ";" && !inStr) { commentIdx = i; break; }
    }
    var code = commentIdx >= 0 ? line.slice(0, commentIdx) : line;
    var comment = commentIdx >= 0 ? line.slice(commentIdx) : "";
    var trimmed = code.replace(/^\t/, "");
    var leadTab = code.length !== trimmed.length ? "\t" : "";
    var html;
    var labelMatch = trimmed.match(/^([.\w]+)(::?)(\s*)$/);
    if (labelMatch) {
      html = leadTab + '<span class="tok-label">' + escapeHtml(labelMatch[1] + labelMatch[2]) + "</span>" + labelMatch[3];
    } else if (leadTab && MACROS.test(trimmed)) {
      var m = trimmed.match(MACROS);
      html = leadTab + '<span class="tok-macro">' + escapeHtml(m[0]) + "</span>" + escapeHtml(trimmed.slice(m[0].length));
    } else if (leadTab && MNEM.test(trimmed)) {
      var mm = trimmed.match(MNEM);
      var rest = trimmed.slice(mm[0].length);
      rest = escapeHtml(rest)
        .replace(/\b(EVENT_[A-Z0-9_]+|SCRIPT_[A-Z0-9_]+|SPRITE_[A-Z0-9_]+|TEXT_[A-Z0-9_]+|PLAYER_DIR_[A-Z0-9_]+|SFX_[A-Z0-9_]+|MUSIC_[A-Z0-9_]+|PAD_[A-Z0-9_]+|w[A-Z]\w*|h[A-Z]\w*)\b/g, '<span class="tok-const">$1</span>')
        .replace(/(?<![\w#])(-?\d+)\b/g, '<span class="tok-num">$1</span>');
      html = leadTab + '<span class="tok-mnem">' + escapeHtml(mm[0]) + "</span>" + rest;
    } else {
      html = escapeHtml(trimmed) ? leadTab + escapeHtml(trimmed) : "";
    }
    if (comment) html += '<span class="tok-comment">' + escapeHtml(comment) + "</span>";
    return html || "&nbsp;";
  }
  function highlightCode(code) {
    return code.split("\n").map(highlightLine).join("\n");
  }

  var mapEntries = Object.keys(DATA).map(function (id) { return DATA[id]; });
  mapEntries.sort(function (a, b) { return b.pin_count - a.pin_count || a.name.localeCompare(b.name); });

  document.getElementById("statMaps").textContent = mapEntries.length;
  document.getElementById("statPins").textContent = mapEntries.reduce(function (s, m) { return s + m.pin_count; }, 0);

  var mapListEl = document.getElementById("mapList");
  var appEl = document.getElementById("app");
  var activeId = null;
  var activePinIdx = null;

  function nextCount(m) {
    return m.pins.reduce(function (n, p) { return n + (isNext(p.event) ? 1 : 0); }, 0);
  }

  function renderList(filter) {
    mapListEl.innerHTML = "";
    var f = (filter || "").trim().toLowerCase();
    var shown = 0;
    mapEntries.forEach(function (m) {
      if (f && m.name.toLowerCase().indexOf(f) === -1 && m.const_name.toLowerCase().indexOf(f) === -1) return;
      shown++;
      var el = document.createElement("div");
      el.className = "map-item" + (String(m.map_id) === activeId ? " active" : "");
      el.setAttribute("role", "option");
      el.tabIndex = 0;
      var nx = nextCount(m);
      el.innerHTML = '<span class="n">' + escapeHtml(m.name) + '</span>' +
        (nx ? '<span class="nx">' + nx + " next</span>" : "") +
        '<span class="c">' + m.pin_count + "</span>";
      el.addEventListener("click", function () { selectMap(String(m.map_id)); });
      el.addEventListener("keydown", function (e) { if (e.key === "Enter") selectMap(String(m.map_id)); });
      mapListEl.appendChild(el);
    });
    if (!shown) {
      var hint = document.createElement("div");
      hint.className = "empty-hint";
      hint.textContent = 'Brak wynikow dla "' + filter + '"';
      mapListEl.appendChild(hint);
    }
  }
  renderList("");
  document.getElementById("search").addEventListener("input", function (e) { renderList(e.target.value); });

  function updateProgressUi() {
    document.getElementById("doneCount").textContent = doneEvents.size;
    renderList(document.getElementById("search").value);
  }
  onDoneChange(updateProgressUi);
  updateProgressUi();

  document.getElementById("filterNext").addEventListener("change", function (e) {
    filterNext = e.target.checked;
    if (activeId) selectMap(activeId, true);
  });
  document.getElementById("clearDone").addEventListener("click", function () {
    if (!doneEvents.size) return;
    if (!confirm("Clear all " + doneEvents.size + " marked-done event(s)?")) return;
    doneEvents.clear();
    saveDone();
    doneChangeListeners.forEach(function (fn) { fn(); });
    if (activeId) selectMap(activeId, true);
  });

  var viewport = document.getElementById("viewport");
  var welcome = document.getElementById("welcome");
  var headerStrip = document.getElementById("headerStrip");
  var legend = document.getElementById("legend");
  var scale = 1, panX = 40, panY = 40;

  function applyTransform() {
    viewport.style.transform = "translate(" + panX + "px," + panY + "px) scale(" + scale + ")";
  }

  function guardChip(g) {
    if (g[0] === "event") {
      var cls = g[2] ? "event-req" : "event-forbid";
      var done = isDone(g[1]);
      var label = escapeHtml((g[2] ? "" : "NOT ") + g[1].replace(/^EVENT_/, ""));
      return '<span class="guard-chip ' + cls + (done ? " chip-done" : "") + '" data-toggle-event="' +
        escapeHtml(g[1]) + '" title="click to mark ' + escapeHtml(g[1]) + ' as done/not done" >' +
        (done ? "&#10003; " : "") + label + "</span>";
    }
    return '<span class="guard-chip coord" title="w' + (g[1] === "x" ? "X" : "Y") + 'Coord must equal ' + g[2] + '">' + g[1] + "==" + g[2] + "</span>";
  }

  function closeDetail() {
    appEl.classList.remove("with-detail");
    activePinIdx = null;
    document.querySelectorAll(".pin.active, .pin-line.active").forEach(function (p) { p.classList.remove("active"); });
  }

  function showDetail(m, pin, pinIdx, pinEl) {
    activePinIdx = pinIdx;
    document.querySelectorAll(".pin.active, .pin-line.active").forEach(function (p) { p.classList.remove("active"); });
    if (pinEl) pinEl.classList.add("active");
    appEl.classList.add("with-detail");

    var det = document.getElementById("detail");
    var html = '<button class="close" id="detailClose" title="Close">&times;</button>';
    var whereStr = pin.kind === "line"
      ? (pin.axis === "y" ? "any x, y=" + pin.value : "x=" + pin.value + ", any y")
      : "(" + pin.x + "," + pin.y + ")";
    html += "<h2>" + escapeHtml(pin.event.replace(/^EVENT_/, "")) + "</h2>";
    html += '<div class="sub">' + m.name + " &middot; " + whereStr + "</div>";
    html += '<label class="chk-row" style="margin-bottom:10px"><input type="checkbox" id="markDone"' +
      (isDone(pin.event) ? " checked" : "") + ' /> Mark this event done</label>';
    var parents = EVENT_PARENTS[pin.event] || [];
    html += '<div class="section-label">Requires (event_graph parents)</div>';
    html += parents.length
      ? '<div>' + parents.map(function (p) { return guardChip(["event", p, true]); }).join("") + "</div>"
      : '<div class="none">none -- always doable</div>';

    if (pin.state_idx === undefined) {
      html += '<div class="none">Not tied to a def_script_pointers state directly -- reached via an NPC/sign/item/trainer interaction script (pokemon.map_scripts build_object_triggers / build_bg_triggers). No state-table asm body to show here.</div>';
    } else {
      html += '<div class="section-label">Guard chain</div>';
      html += (pin.guards && pin.guards.length)
        ? '<div class="guards">' + pin.guards.map(guardChip).join("") + "</div>"
        : '<div class="none">none</div>';

      html += '<div class="section-label">Sets (click to mark done)</div>';
      html += '<div>' + (pin.sets || []).map(function (s) {
        var d = isDone(s);
        return '<span class="set-chip' + (d ? "" : " unmarked") + '" data-toggle-event="' + escapeHtml(s) + '" title="click to mark ' + escapeHtml(s) + ' as done/not done">' +
          (d ? "&#10003; " : "") + escapeHtml(s.replace(/^EVENT_/, "")) + "</span>";
      }).join("") + "</div>";

      html += '<div class="section-label">Reached from (state #)</div>';
      html += (pin.preds && pin.preds.length)
        ? '<div>' + pin.preds.map(function (p) { return '<span class="rel-chip" data-state="' + p + '">#' + p + "</span>"; }).join("") + "</div>"
        : '<div class="none">none (entry state)</div>';

      html += '<div class="section-label">Next state</div>';
      html += (pin.next_state !== null && pin.next_state !== undefined)
        ? '<div><span class="rel-chip" data-state="' + pin.next_state + '">#' + pin.next_state + "</span></div>"
        : '<div class="none">none</div>';

      html += '<div class="section-label">' + pin.handler + ' (state #' + pin.state_idx + ')</div>';
      html += '<div class="code">' + highlightCode(pin.code) + "</div>";
    }

    det.innerHTML = html;
    document.getElementById("detailClose").addEventListener("click", closeDetail);
    document.getElementById("markDone").addEventListener("change", function () {
      toggleDone(pin.event);
      showDetail(m, pin, pinIdx, pinEl);
      refreshPinVisuals(m);
    });
    det.querySelectorAll("[data-toggle-event]").forEach(function (chip) {
      chip.addEventListener("click", function () {
        toggleDone(chip.dataset.toggleEvent);
        showDetail(m, pin, pinIdx, pinEl);
        refreshPinVisuals(m);
      });
    });
    det.querySelectorAll(".rel-chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        var targetIdx = chip.dataset.state;
        var targetPinIdx = m.pins.findIndex(function (p) { return String(p.state_idx) === String(targetIdx); });
        if (targetPinIdx === -1) return;
        var targetEl = viewport.querySelector('[data-idx="' + targetPinIdx + '"]');
        showDetail(m, m.pins[targetPinIdx], targetPinIdx, targetEl);
        if (targetEl) targetEl.scrollIntoView({ behavior: "smooth", block: "center", inline: "center" });
      });
    });
  }

  function pinClassSuffix(pin) {
    var state = pinState(pin.event);
    var cls = " " + state;
    if (filterNext && state !== "next") cls += " dimmed";
    return cls;
  }

  function refreshPinVisuals(m) {
    viewport.querySelectorAll(".pin, .pin-line").forEach(function (el) {
      var pin = m.pins[Number(el.dataset.idx)];
      if (!pin) return;
      var base = el.classList.contains("vertical")
        ? "pin-line vertical"
        : (pin.kind === "line" ? "pin-line" : "pin");
      var wasActive = el.classList.contains("active");
      el.className = base + pinClassSuffix(pin) + (wasActive ? " active" : "");
    });
    updateProgressUi();
  }

  function selectMap(id, keepView) {
    activeId = id;
    closeDetail();
    var m = DATA[id];
    welcome.style.display = "none";
    headerStrip.style.display = "flex";
    legend.style.display = "flex";
    document.getElementById("fnName").textContent = m.name;
    document.getElementById("fnMeta").textContent = m.pin_count + " event pin(s) · " + m.w + "×" + m.h + "px";

    viewport.innerHTML = "";
    var img = document.createElement("img");
    img.className = "mapart";
    img.width = m.w;
    img.height = m.h;
    img.src = "data:image/png;base64," + m.png;
    viewport.appendChild(img);

    m.pins.forEach(function (pin, idx) {
      var el = document.createElement("div");
      var label = pin.event.replace(/^EVENT_/, "");
      if (pin.kind === "line") {
        el.className = "pin-line" + (pin.axis === "x" ? " vertical" : "") + pinClassSuffix(pin);
        el.dataset.idx = idx;
        if (pin.axis === "y") {
          el.style.left = "0px";
          el.style.top = (pin.value * m.cell_px) + "px";
          el.style.width = m.w + "px";
          el.style.height = m.cell_px + "px";
          el.title = label + " (any x, y=" + pin.value + ")";
        } else {
          el.style.top = "0px";
          el.style.left = (pin.value * m.cell_px) + "px";
          el.style.height = m.h + "px";
          el.style.width = m.cell_px + "px";
          el.title = label + " (x=" + pin.value + ", any y)";
        }
      } else {
        el.className = "pin" + pinClassSuffix(pin);
        el.dataset.idx = idx;
        el.style.left = (pin.x * m.cell_px + m.cell_px / 2) + "px";
        el.style.top = (pin.y * m.cell_px + m.cell_px / 2) + "px";
        el.title = label + " (" + pin.x + "," + pin.y + ")";
      }
      el.addEventListener("click", function (e) {
        e.stopPropagation();
        showDetail(m, pin, idx, el);
      });
      viewport.appendChild(el);
    });

    updateProgressUi();
    if (!keepView) resetView(m);
  }

  function resetView(m) {
    var wrap = document.getElementById("canvasWrap").getBoundingClientRect();
    var availW = wrap.width - 80, availH = wrap.height - 80;
    scale = Math.max(1, Math.min(availW / m.w, availH / m.h, 6));
    panX = Math.max(20, (wrap.width - m.w * scale) / 2);
    panY = Math.max(56, (wrap.height - m.h * scale) / 2);
    applyTransform();
  }

  var wrapEl = document.getElementById("canvasWrap");
  var dragging = false, lastX = 0, lastY = 0;
  wrapEl.addEventListener("pointerdown", function (e) {
    if (e.target.closest(".pin") || e.target.closest(".pin-line")) return;
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    wrapEl.classList.add("grabbing");
    wrapEl.setPointerCapture(e.pointerId);
  });
  wrapEl.addEventListener("pointermove", function (e) {
    if (!dragging) return;
    panX += e.clientX - lastX; panY += e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    applyTransform();
  });
  ["pointerup", "pointercancel"].forEach(function (ev) {
    wrapEl.addEventListener(ev, function () { dragging = false; wrapEl.classList.remove("grabbing"); });
  });
  wrapEl.addEventListener("wheel", function (e) {
    e.preventDefault();
    var wrap = wrapEl.getBoundingClientRect();
    var mx = e.clientX - wrap.left, my = e.clientY - wrap.top;
    var prevScale = scale;
    var delta = e.deltaY > 0 ? 0.9 : 1.1;
    scale = Math.min(10, Math.max(0.5, scale * delta));
    panX = mx - ((mx - panX) / prevScale) * scale;
    panY = my - ((my - panY) / prevScale) * scale;
    applyTransform();
  }, { passive: false });

  document.getElementById("zoomIn").addEventListener("click", function () { scale = Math.min(10, scale * 1.15); applyTransform(); });
  document.getElementById("zoomOut").addEventListener("click", function () { scale = Math.max(0.5, scale / 1.15); applyTransform(); });
  document.getElementById("zoomReset").addEventListener("click", function () {
    if (activeId) resetView(DATA[activeId]);
  });

  var initial = (INITIAL_MAP && DATA[INITIAL_MAP]) ? DATA[INITIAL_MAP] : mapEntries[0];
  if (initial) selectMap(String(initial.map_id));
})();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out", default="map_pins.html",
        help="docelowy plik HTML (domyslnie map_pins.html) -- dziala offline",
    )
    parser.add_argument(
        "--map",
        help="nazwa mapy do pokazania od razu po otwarciu (np. PALLET_TOWN) -- "
        "domyslnie mapa z najwieksza liczba pinow",
    )
    args = parser.parse_args()

    data = build_data()
    if not data:
        raise SystemExit(
            "brak map z pinami -- upewnij sie, ze src/pokemon/map_scripts.py "
            "jest wygenerowany (python tools/gen_map_scripts.py) i ze "
            "reference/pokered jest obecne"
        )

    initial_map_id = None
    if args.map:
        target = args.map.strip().upper()
        for map_id_str, entry in data.items():
            if entry["const_name"] == target:
                initial_map_id = map_id_str
                break
        if initial_map_id is None:
            print(f"uwaga: nie znaleziono mapy '{args.map}', uzywam domyslnej")

    payload = {"maps": data, "event_parents": build_event_parents(data)}
    html = HTML_TEMPLATE.replace(
        "__DATA_JSON__", json.dumps(payload, separators=(",", ":"))
    ).replace("__INITIAL_MAP__", json.dumps(initial_map_id))

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    total_pins = sum(m["pin_count"] for m in data.values())
    print(f"zapisano {out_path.resolve()} ({len(data)} map, {total_pins} pinow)")
    print("otworz w przegladarce -- narzedzie jest w pelni offline (bez serwera, bez CDN)")


if __name__ == "__main__":
    main()
