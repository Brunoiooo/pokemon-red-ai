#!/usr/bin/env python
"""Offline viewer: a static snapshot of pokemon.event_compass's live
picture for one save state -- every currently-unset, currently-satisfiable
event trigger (pokemon.event_triggers.EVENT_TRIGGERS, filtered by that
save's own wEventFlags) pinned onto its map's real pixel art
(pokemon.map_render), with the single tile
pokemon.event_compass.nearest_unlockable_event actually picks from the
player's saved position highlighted.

Companion to tools/view_map_pins.py (that one shows the static per-map
COORD_TRIGGERS graph in the abstract; this one shows the *live*, filtered
"what's actually next" picture for one specific save, across every map that
has a candidate -- not just the player's current one). No live game
connection -- point it at a .state file (Emulator/PyBoy's save_state
format, e.g. saves/start/checkpoint.state) and it renders that one moment.

Usage:
  python tools/view_event_compass.py
  python tools/view_event_compass.py --state saves/EVENT_GOT_STARTER/checkpoint.state
  python tools/view_event_compass.py --rom rom.gb --out compass.html
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
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokemon import event_compass as ec  # noqa: E402
from pokemon import event_triggers as et  # noqa: E402
from pokemon import map_render as mr  # noqa: E402
from pokemon import ram_constants as rc  # noqa: E402
from pokemon.map_constants import MAPS_BY_ID  # noqa: E402


def prettify_map_name(const_name: str) -> str:
    return " ".join(w.capitalize() for w in const_name.split("_"))


def load_memory(rom: str, state: str):
    """Loads a saved PyBoy state headlessly (window='null') and returns its
    memory view plus the player's (x, y, map_id) -- same RAM addresses
    Data.position_x/position_y/get_position read (pokemon.ram_constants.RAM)."""
    from pyboy import PyBoy

    pyboy = PyBoy(rom, window="null")
    with open(state, "rb") as f:
        pyboy.load_state(f)
    memory = pyboy.memory
    pos = (memory[rc.RAM.wXCoord], memory[rc.RAM.wYCoord], memory[rc.RAM.wCurMap])
    return pyboy, memory, pos


def build_data(memory, pos) -> tuple[dict[str, dict], dict | None, dict[str, bool]]:
    """map_id (str) -> {..., pins: [{event, kind, x, y, requires, ...}]} for
    every map with at least one not-yet-set trigger tile, the
    nearest_unlockable_event result (or None), and live_flags: the real
    wEventFlags bit (at this save) for every event name the client-side
    "what if I mark this done" simulation might need -- every
    EVENT_TRIGGERS key plus every name referenced inside any site's own
    `requires` (some guard events have no trigger site of their own, e.g.
    badges/HM gates event_graph already excludes, but their live value is
    still needed to color other pins correctly).

    Unlike the first version of this tool, sites are NOT pre-filtered by
    ec.requires_satisfied here -- every not-yet-set event's every site is
    included, tagged with its own `requires`, so the browser can recompute
    "available" (dostepne) vs "blocked" (zakazane) live as the user toggles
    hypothetical selections, instead of only ever seeing the fixed snapshot
    this Python pass happened to compute requires against."""
    nearest = ec.nearest_unlockable_event(pos, memory, max_hops=100000)
    nearest_event = nearest[3] if nearest else None

    referenced_names = set(et.EVENT_TRIGGERS.keys())
    for sites in et.EVENT_TRIGGERS.values():
        for site in sites:
            referenced_names.update(name for _kind, name, _req in site["requires"])
    live_flags = {name: ec.event_flag_set(name, memory) for name in referenced_names}

    pins_by_map: dict[int, list[dict]] = {}
    for event_name, sites in et.EVENT_TRIGGERS.items():
        if live_flags.get(event_name, False):
            continue  # already set live -- nothing left to route to
        for site in sites:
            for tx, ty, tmap in ec.tiles_for_site(site):
                pins_by_map.setdefault(tmap, []).append(
                    {
                        "event": event_name,
                        "x": tx,
                        "y": ty,
                        "kind": site["kind"],
                        "requires": list(site["requires"]),
                        "is_nearest": event_name == nearest_event,
                    }
                )

    out: dict[str, dict] = {}
    for map_id, pins in sorted(pins_by_map.items()):
        img = mr.render_map_rgb(map_id)
        if img is None:
            continue
        from PIL import Image

        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        entry_name = MAPS_BY_ID[map_id][0]
        out[str(map_id)] = {
            "map_id": map_id,
            "const_name": entry_name,
            "name": prettify_map_name(entry_name),
            "w": int(img.shape[1]),
            "h": int(img.shape[0]),
            "cell_px": mr.CELL_PX,
            "png": png_b64,
            "pins": pins,
            "pin_count": len(pins),
            "is_player_here": map_id == pos[2],
        }
    nearest_payload = (
        {"event": nearest_event, "dx": nearest[0], "dy": nearest[1], "dist": nearest[2]}
        if nearest
        else None
    )
    return out, nearest_payload, live_flags


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Event Compass Snapshot</title>
<style>
:root {
  --bg: #14171b; --surface: #1b1f26; --surface-2: #242a33; --surface-3: #2c333e;
  --line: #333c49; --line-soft: #262d38; --text: #d7dce3; --text-dim: #8590a3; --text-faint: #576078;
  --accent: #9bbc0f; --accent-strong: #c5e05a; --accent-dim: #3c4a16;
  --guard-event: #e8a33d; --guard-event-dim: #4a3517;
  --pin: #ff5f6d; --pin-ink: #2a0002; --pin-ring: rgba(255,95,109,0.35);
  --nearest: #7fd88f; --nearest-ink: #06210c; --nearest-ring: rgba(127,216,143,0.4);
  --player: #5fb3d9;
  --select: #f2c94c; --select-ink: #3a2c00;
  --avail: #5fb3d9; --avail-ink: #06202a; --avail-ring: rgba(95,179,217,0.4);
  --blocked: #6b7688; --blocked-ink: #14171b; --blocked-ring: rgba(107,118,136,0.3);
  --shadow: 0 8px 24px rgba(0,0,0,0.45);
  --mono: ui-monospace, "JetBrains Mono", "Cascadia Code", "SF Mono", Consolas, monospace;
}
@media (prefers-color-scheme: light) {
  :root {
    --bg: #eef0e4; --surface: #ffffff; --surface-2: #f4f6ec; --surface-3: #e9ecdd;
    --line: #cdd3bd; --line-soft: #dde1cf; --text: #1d2417; --text-dim: #566047; --text-faint: #8a927a;
    --accent: #4d6b0a; --accent-strong: #3a5006; --accent-dim: #dbe8b8;
    --guard-event: #9a6111; --guard-event-dim: #f5e3c4;
    --pin: #c9313c; --pin-ink: #fff0ef; --pin-ring: rgba(201,49,60,0.25);
    --nearest: #2f7a44; --nearest-ink: #eafcee; --nearest-ring: rgba(47,122,68,0.3);
    --player: #1c6f8e;
    --select: #a5760a; --select-ink: #fff4dc;
    --avail: #1c6f8e; --avail-ink: #eaf6fb; --avail-ring: rgba(28,111,142,0.25);
    --blocked: #9aa3b2; --blocked-ink: #eef0e4; --blocked-ring: rgba(154,163,178,0.25);
    --shadow: 0 8px 24px rgba(60,66,40,0.12);
  }
}
* { box-sizing: border-box; }
html, body { height: 100%; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: var(--mono); font-size: 13px; overflow: hidden; }
.app { display: grid; grid-template-columns: 260px 1fr; grid-template-rows: 56px 1fr; height: 100vh; }
.topbar { grid-column: 1 / -1; display: flex; align-items: center; gap: 14px; padding: 0 16px; background: var(--surface); border-bottom: 1px solid var(--line); }
.brand .mark { color: var(--accent); font-weight: 700; letter-spacing: 0.04em; }
.brand .sub { color: var(--text-faint); font-size: 11px; }
.topbar .nearest { margin-left: auto; font-size: 12px; color: var(--nearest); }
.topbar .sel-controls { display: flex; align-items: center; gap: 8px; font-size: 12px; color: var(--select); }
.topbar .sel-controls b { font-variant-numeric: tabular-nums; }
.topbar .sel-controls button { background: none; border: none; color: var(--text-faint); text-decoration: underline; cursor: pointer; font-family: var(--mono); font-size: 11px; padding: 0; }
.topbar .sel-controls button:hover { color: var(--text); }
.sidebar { background: var(--surface); border-right: 1px solid var(--line); overflow-y: auto; padding: 6px; }
.map-item { display: flex; align-items: center; gap: 8px; padding: 7px 9px; border-radius: 4px; cursor: pointer; color: var(--text-dim); border: 1px solid transparent; }
.map-item:hover { background: var(--surface-2); color: var(--text); }
.map-item.active { background: var(--accent-dim); color: var(--accent-strong); border-color: var(--accent-dim); }
.map-item .n { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 12.5px; }
.map-item .c { font-size: 10.5px; color: var(--text-faint); background: var(--surface-3); border-radius: 999px; padding: 1px 7px; }
.map-item .sel { font-size: 10.5px; color: var(--select-ink); background: var(--select); border-radius: 999px; padding: 1px 7px; font-weight: 700; }
.map-item .p { color: var(--player); font-size: 13px; }
.canvas-wrap { position: relative; overflow: hidden; background: var(--bg); }
.viewport { position: absolute; top: 0; left: 0; transform-origin: 0 0; }
.viewport img.mapart { display: block; image-rendering: pixelated; image-rendering: crisp-edges; }
.pin { position: absolute; width: 12px; height: 12px; margin-left: -6px; margin-top: -6px; border-radius: 50%; background: var(--blocked); border: 2px solid var(--blocked-ink); box-shadow: 0 0 0 3px var(--blocked-ring); cursor: pointer; z-index: 1; opacity: 0.55; }
.pin:hover { transform: scale(1.3); }
.pin.available { background: var(--avail); border-color: var(--avail-ink); box-shadow: 0 0 0 3px var(--avail-ring); opacity: 1; z-index: 2; }
.pin.nearest { background: var(--nearest); border-color: var(--nearest-ink); box-shadow: 0 0 0 5px var(--nearest-ring); width: 16px; height: 16px; margin-left: -8px; margin-top: -8px; opacity: 1; z-index: 3; }
.pin.selected { outline: 3px solid var(--select); outline-offset: 2px; opacity: 1; }
.pin-tip { position: absolute; transform: translate(-50%, -130%); background: var(--surface); border: 1px solid var(--line); border-radius: 4px; padding: 3px 7px; font-size: 10px; color: var(--text); white-space: nowrap; pointer-events: none; z-index: 4; box-shadow: var(--shadow); }
.legend { position: absolute; bottom: 14px; left: 14px; z-index: 3; background: var(--surface); border: 1px solid var(--line); border-radius: 5px; padding: 8px 10px; font-size: 10.5px; color: var(--text-dim); box-shadow: var(--shadow); display: flex; flex-direction: column; gap: 5px; }
.legend .row { display: flex; align-items: center; gap: 6px; }
.legend .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }
.legend .ring { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; outline: 2px solid var(--select); outline-offset: 1px; background: var(--blocked); }
.zoom-controls { position: absolute; bottom: 14px; right: 14px; z-index: 3; display: flex; flex-direction: column; gap: 4px; }
.zoom-controls button { width: 28px; height: 28px; background: var(--surface); border: 1px solid var(--line); color: var(--text-dim); border-radius: 4px; cursor: pointer; font-family: var(--mono); font-size: 14px; box-shadow: var(--shadow); }
.welcome { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; color: var(--text-faint); }
</style>
</head>
<body>
<div class="app" id="app">
  <div class="topbar">
    <div class="brand"><span class="mark">&#9646;&#9646; EVENT COMPASS</span> <span class="sub">live snapshot &middot; offline</span></div>
    <div class="sel-controls"><b id="selCount">0</b> zaznaczonych <button id="selClear" type="button">wyczysc</button></div>
    <div class="nearest" id="nearestLabel"></div>
  </div>
  <div class="sidebar" id="mapList"></div>
  <div class="canvas-wrap" id="canvasWrap">
    <div class="viewport" id="viewport"></div>
    <div class="welcome" id="welcome">Select a map on the left</div>
    <div class="zoom-controls">
      <button id="zoomIn" type="button">+</button>
      <button id="zoomReset" type="button">&#9678;</button>
      <button id="zoomOut" type="button">&minus;</button>
    </div>
    <div class="legend">
      <div class="row"><span class="dot" style="background:var(--nearest)"></span> najblizszy (prawdziwy stan zapisu)</div>
      <div class="row"><span class="dot" style="background:var(--avail)"></span> dostepne teraz (wymagania spelnione)</div>
      <div class="row"><span class="dot" style="background:var(--blocked)"></span> zakazane / zablokowane (wymagania niespelnione)</div>
      <div class="row"><span class="ring"></span> zaznaczone przez Ciebie (traktowane jako "zrobione" w symulacji)</div>
    </div>
  </div>
</div>
<script id="payload" type="application/json">__DATA_JSON__</script>
<script>
(function () {
  "use strict";
  var PAYLOAD = JSON.parse(document.getElementById("payload").textContent);
  var DATA = PAYLOAD.maps;
  var NEAREST = PAYLOAD.nearest;
  var LIVE_FLAGS = PAYLOAD.live_flags || {};
  var mapEntries = Object.keys(DATA).map(function (id) { return DATA[id]; });
  mapEntries.sort(function (a, b) { return b.pin_count - a.pin_count || a.name.localeCompare(b.name); });

  document.getElementById("nearestLabel").textContent = NEAREST
    ? "nearest: " + NEAREST.event.replace(/^EVENT_/, "") + " (" + NEAREST.dist + " tiles)"
    : "nearest: none reachable";

  // ---------- "should happen" event marking -- click a pin to toggle it,
  // by event name (an event can have several trigger sites/maps; marking
  // one marks all of them, since they're the same underlying event). No
  // live game connection here, so this is a manual checklist the player
  // drives, persisted in the browser's localStorage across reloads --
  // same convention tools/view_map_pins.py's own doneEvents uses.
  var selectedEvents = (function () {
    try {
      var raw = localStorage.getItem("ec-selected-events");
      return raw ? new Set(JSON.parse(raw)) : new Set();
    } catch (e) { return new Set(); }
  })();
  function saveSelected() {
    try { localStorage.setItem("ec-selected-events", JSON.stringify(Array.from(selectedEvents))); } catch (e) {}
  }
  function toggleSelected(event) {
    if (selectedEvents.has(event)) selectedEvents.delete(event); else selectedEvents.add(event);
    saveSelected();
    document.getElementById("selCount").textContent = selectedEvents.size;
    renderList();
    if (activeId) refreshPinVisuals(DATA[activeId]);
  }
  document.getElementById("selCount").textContent = selectedEvents.size;
  document.getElementById("selClear").addEventListener("click", function () {
    if (!selectedEvents.size) return;
    if (!confirm("Wyczyscic wszystkie " + selectedEvents.size + " zaznaczonych event(ow)?")) return;
    selectedEvents.clear();
    saveSelected();
    document.getElementById("selCount").textContent = 0;
    renderList();
    if (activeId) refreshPinVisuals(DATA[activeId]);
  });

  var listEl = document.getElementById("mapList");
  var activeId = null;

  function selectedCount(m) {
    var seen = {}, n = 0;
    m.pins.forEach(function (p) {
      if (selectedEvents.has(p.event) && !seen[p.event]) { seen[p.event] = true; n++; }
    });
    return n;
  }

  function renderList() {
    listEl.innerHTML = "";
    mapEntries.forEach(function (m) {
      var el = document.createElement("div");
      el.className = "map-item" + (String(m.map_id) === activeId ? " active" : "");
      var sel = selectedCount(m);
      el.innerHTML = (m.is_player_here ? '<span class="p">&#9679;</span>' : '<span style="width:10px;display:inline-block"></span>') +
        '<span class="n">' + m.name + '</span>' +
        (sel ? '<span class="sel">' + sel + '</span>' : '') +
        '<span class="c">' + m.pin_count + '</span>';
      el.addEventListener("click", function () { selectMap(String(m.map_id)); });
      listEl.appendChild(el);
    });
  }
  renderList();

  var viewport = document.getElementById("viewport");
  var welcome = document.getElementById("welcome");
  var scale = 1, panX = 40, panY = 40;

  function applyTransform() {
    viewport.style.transform = "translate(" + panX + "px," + panY + "px) scale(" + scale + ")";
  }

  // ---------- live "dostepne" (available) / "zakazane" (blocked)
  // simulation -- a pin's own requires (the same ("event", name,
  // required_bool) guards pokemon.event_triggers computed statically) are
  // re-evaluated here against a hypothetical state: any event the user has
  // selected is treated as done (True), everything else falls back to its
  // real live value from this save (LIVE_FLAGS). Selecting an event that a
  // NOT-required guard names immediately flips those pins to "blocked" --
  // e.g. Pallet Town's EVENT_OAK_APPEARED_IN_PALLET requires
  // EVENT_FOLLOWED_OAK_INTO_LAB to stay False, so selecting the latter
  // blocks the former live, the same one-way idiom event_triggers'
  // module docstring documents.
  function effectiveFlag(name) {
    if (selectedEvents.has(name)) return true;
    return !!LIVE_FLAGS[name];
  }
  function pinSatisfied(pin) {
    return pin.requires.every(function (r) { return effectiveFlag(r[1]) === r[2]; });
  }

  function pinTitle(pin) {
    var label = pin.event.replace(/^EVENT_/, "") + " (" + pin.x + "," + pin.y + ")" +
      (pin.requires.length ? " requires " + pin.requires.map(function (r) { return (r[2] ? "" : "NOT ") + r[1].replace(/^EVENT_/, ""); }).join(", ") : "");
    var status = selectedEvents.has(pin.event)
      ? " -- zaznaczone (klik, aby odznaczyc)"
      : (pinSatisfied(pin) ? " -- dostepne teraz (klik, aby zaznaczyc)" : " -- zakazane/zablokowane (klik, aby mimo to zaznaczyc)");
    return label + status;
  }

  function refreshPinVisuals(m) {
    viewport.querySelectorAll(".pin").forEach(function (el) {
      var pin = m.pins[Number(el.dataset.idx)];
      if (!pin) return;
      el.classList.toggle("selected", selectedEvents.has(pin.event));
      el.classList.toggle("available", pinSatisfied(pin));
      el.title = pinTitle(pin);
    });
  }

  function selectMap(id) {
    activeId = id;
    renderList();
    var m = DATA[id];
    welcome.style.display = "none";
    viewport.innerHTML = "";
    var img = document.createElement("img");
    img.className = "mapart";
    img.width = m.w; img.height = m.h;
    img.src = "data:image/png;base64," + m.png;
    viewport.appendChild(img);

    m.pins.forEach(function (pin, idx) {
      var el = document.createElement("div");
      el.className = "pin" + (pinSatisfied(pin) ? " available" : "") + (pin.is_nearest ? " nearest" : "") + (selectedEvents.has(pin.event) ? " selected" : "");
      el.dataset.idx = idx;
      el.style.left = (pin.x * m.cell_px + m.cell_px / 2) + "px";
      el.style.top = (pin.y * m.cell_px + m.cell_px / 2) + "px";
      el.title = pinTitle(pin);
      el.addEventListener("click", function (e) {
        e.stopPropagation();
        toggleSelected(pin.event);
      });
      viewport.appendChild(el);
    });

    var wrap = document.getElementById("canvasWrap").getBoundingClientRect();
    scale = Math.max(1, Math.min((wrap.width - 80) / m.w, (wrap.height - 80) / m.h, 6));
    panX = Math.max(20, (wrap.width - m.w * scale) / 2);
    panY = Math.max(20, (wrap.height - m.h * scale) / 2);
    applyTransform();
  }

  var wrapEl = document.getElementById("canvasWrap");
  var dragging = false, lastX = 0, lastY = 0;
  wrapEl.addEventListener("pointerdown", function (e) {
    if (e.target.closest(".pin")) return;  // let the pin's own click handler fire (see toggleSelected) instead of capturing it into a drag
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    wrapEl.setPointerCapture(e.pointerId);
  });
  wrapEl.addEventListener("pointermove", function (e) { if (!dragging) return; panX += e.clientX - lastX; panY += e.clientY - lastY; lastX = e.clientX; lastY = e.clientY; applyTransform(); });
  ["pointerup", "pointercancel"].forEach(function (ev) { wrapEl.addEventListener(ev, function () { dragging = false; }); });
  wrapEl.addEventListener("wheel", function (e) {
    e.preventDefault();
    var wrap = wrapEl.getBoundingClientRect();
    var mx = e.clientX - wrap.left, my = e.clientY - wrap.top, prevScale = scale;
    scale = Math.min(10, Math.max(0.5, scale * (e.deltaY > 0 ? 0.9 : 1.1)));
    panX = mx - ((mx - panX) / prevScale) * scale;
    panY = my - ((my - panY) / prevScale) * scale;
    applyTransform();
  }, { passive: false });
  document.getElementById("zoomIn").addEventListener("click", function () { scale = Math.min(10, scale * 1.15); applyTransform(); });
  document.getElementById("zoomOut").addEventListener("click", function () { scale = Math.max(0.5, scale / 1.15); applyTransform(); });
  document.getElementById("zoomReset").addEventListener("click", function () { if (mapEntries[0]) selectMap(String(mapEntries[0].map_id)); });

  if (mapEntries[0]) selectMap(String(mapEntries[0].map_id));
})();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rom", default=str(REPO_ROOT / "rom.gb"))
    parser.add_argument("--state", default=str(REPO_ROOT / "saves" / "start" / "checkpoint.state"))
    parser.add_argument("--out", default="event_compass.html")
    args = parser.parse_args()

    pyboy, memory, pos = load_memory(args.rom, args.state)
    try:
        data, nearest, live_flags = build_data(memory, pos)
    finally:
        pyboy.stop(False)

    if not data:
        raise SystemExit("no currently-unlockable events found for this save state")

    payload = {"maps": data, "nearest": nearest, "live_flags": live_flags}
    html = HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, separators=(",", ":")))
    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    total_pins = sum(m["pin_count"] for m in data.values())
    print(f"player at {pos}, nearest: {nearest}")
    print(f"wrote {out_path.resolve()} ({len(data)} maps, {total_pins} pins)")


if __name__ == "__main__":
    main()
