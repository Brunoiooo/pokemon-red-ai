#!/usr/bin/env python
"""Offline IDA-Pro-style block-diagram viewer for pret/pokered's per-map
dispatch-table scripts (see tools/gen_map_scripts.py for the underlying
parser/data model this renders).

Emits ONE self-contained local HTML file -- no server, no CDN, no
internet required after generation, matching this project's existing
tools/view_event_graph.py --full --out foo.html convention (that one uses
a vendored Cytoscape.js+dagre; this one hand-rolls a simpler layered-DAG
layout instead, since each node here needs to hold real, syntax-tinted
asm text, not just a label -- Cytoscape's node model doesn't fit that
without extra extensions).

Each map's def_script_pointers table becomes one graph: a box per dispatch
state, showing its real asm body, its entry-guard chain (event/coordinate,
correctly signed -- see gen_map_scripts.parse_entry_guards), and which
events it SetEvents. Solid arrows are forward next_state transitions;
dashed red arrows are back-edges (a state whose next_state points at an
already-visited state on the same walk -- a real loop, e.g. Oak's Lab's
"wait for the player to press A" state pair).

Usage:
  python tools/view_script_graph.py
  python tools/view_script_graph.py --out my_view.html
  python tools/view_script_graph.py --map OAKS_LAB
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))
from gen_script_graph_data import build_data  # noqa: E402

DEFAULT_OUT = "script_graph.html"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="pl">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Script Graph Explorer</title>
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
  --edge-fwd: #55606f;
  --edge-back: #c2554d;
  --set-chip: #7fd88f;
  --set-chip-dim: #1c3320;
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
  --edge-fwd: #a7ae98;
  --edge-back: #a83d34;
  --set-chip: #2f7a44;
  --set-chip-dim: #d9f0dd;
  --shadow: 0 8px 24px rgba(60,66,40,0.12);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #14171b; --surface: #1b1f26; --surface-2: #242a33; --surface-3: #2c333e;
    --line: #333c49; --line-soft: #262d38; --text: #d7dce3; --text-dim: #8590a3; --text-faint: #576078;
    --accent: #9bbc0f; --accent-strong: #c5e05a; --accent-dim: #3c4a16; --accent-ink: #0f1400;
    --guard-event: #e8a33d; --guard-event-dim: #4a3517; --guard-coord: #5fb3d9; --guard-coord-dim: #17323d;
    --edge-fwd: #55606f; --edge-back: #c2554d; --set-chip: #7fd88f; --set-chip-dim: #1c3320;
    --shadow: 0 8px 24px rgba(0,0,0,0.45);
  }
}
:root[data-theme="dark"] {
  --bg: #14171b; --surface: #1b1f26; --surface-2: #242a33; --surface-3: #2c333e;
  --line: #333c49; --line-soft: #262d38; --text: #d7dce3; --text-dim: #8590a3; --text-faint: #576078;
  --accent: #9bbc0f; --accent-strong: #c5e05a; --accent-dim: #3c4a16; --accent-ink: #0f1400;
  --guard-event: #e8a33d; --guard-event-dim: #4a3517; --guard-coord: #5fb3d9; --guard-coord-dim: #17323d;
  --edge-fwd: #55606f; --edge-back: #c2554d; --set-chip: #7fd88f; --set-chip-dim: #1c3320;
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

.app { display: grid; grid-template-columns: 300px 1fr; grid-template-rows: 48px 1fr; height: 100vh; }

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
.theme-toggle:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }

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
.search-wrap input:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
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
.map-item:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
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
.zoom-controls button:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
.legend {
  position: absolute; bottom: 14px; left: 14px; z-index: 3;
  background: var(--surface); border: 1px solid var(--line); border-radius: 5px;
  padding: 8px 10px; font-size: 10.5px; color: var(--text-dim); box-shadow: var(--shadow);
  display: flex; flex-direction: column; gap: 5px;
}
.legend .row { display: flex; align-items: center; gap: 6px; }
.legend .sw { width: 14px; height: 2px; border-radius: 1px; }
.legend .chip { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }

svg.edges { position: absolute; top: 0; left: 0; overflow: visible; }
.edge-path { fill: none; stroke: var(--edge-fwd); stroke-width: 1.6; }
.edge-path.back { stroke: var(--edge-back); stroke-width: 1.5; stroke-dasharray: 4 3; }

.node {
  position: absolute;
  width: 380px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 6px;
  box-shadow: var(--shadow);
  overflow: hidden;
}
.node .nhead {
  display: flex; align-items: center; gap: 8px;
  padding: 7px 10px;
  background: var(--surface-2);
  border-bottom: 1px solid var(--line);
}
.node .idx {
  font-weight: 700; color: var(--accent-ink); background: var(--accent);
  border-radius: 3px; padding: 1px 6px; font-size: 11px; font-variant-numeric: tabular-nums;
}
.node .handler { color: var(--text); font-size: 12px; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.node .guards {
  display: flex; flex-wrap: wrap; gap: 4px;
  padding: 7px 10px 0 10px;
}
.guard-chip {
  font-size: 10px; padding: 2px 6px; border-radius: 3px; border: 1px solid transparent;
  white-space: nowrap;
}
.guard-chip.event-req { color: var(--guard-event); background: var(--guard-event-dim); }
.guard-chip.event-forbid { color: var(--guard-event); background: var(--guard-event-dim); opacity: 0.65; text-decoration: line-through; }
.guard-chip.coord { color: var(--guard-coord); background: var(--guard-coord-dim); }
.node .code {
  margin: 8px 10px;
  padding: 8px 9px;
  background: var(--bg);
  border: 1px solid var(--line-soft);
  border-radius: 4px;
  font-size: 11px; line-height: 1.55;
  max-height: 220px; overflow: auto;
  white-space: pre;
}
.node .code::-webkit-scrollbar { width: 8px; height: 8px; }
.node .code::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }
.tok-label { color: var(--accent-strong); }
.tok-mnem { color: var(--guard-coord); }
.tok-macro { color: #c792ea; }
.tok-const { color: var(--set-chip); }
.tok-num { color: var(--guard-event); }
.tok-comment { color: var(--text-faint); font-style: italic; }
:root[data-theme="light"] .tok-macro { color: #9d5bc9; }

.node .nfoot {
  display: flex; flex-wrap: wrap; gap: 5px;
  padding: 0 10px 9px 10px;
}
.set-chip {
  font-size: 10px; padding: 2px 7px; border-radius: 999px;
  color: var(--set-chip); background: var(--set-chip-dim);
}
.node.noop { opacity: 0.72; }
.node.noop .code { display: none; }

.welcome {
  position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  flex-direction: column; gap: 10px; color: var(--text-faint); text-align: center; padding: 20px;
}
.welcome .big { font-size: 15px; color: var(--text-dim); }
.welcome .small { font-size: 11.5px; max-width: 420px; }
kbd {
  font-family: var(--mono); background: var(--surface-2); border: 1px solid var(--line);
  border-radius: 3px; padding: 1px 5px; font-size: 10.5px; color: var(--text-dim);
}
</style>
</head>
<body>

<div class="app">
  <div class="topbar">
    <div class="brand">
      <span class="mark">&#9646;&#9646; SCRIPT GRAPH</span>
      <span class="sub">pret/pokered &middot; def_script_pointers explorer &middot; offline</span>
    </div>
    <div class="stats">
      <span>maps <b id="statMaps">0</b></span>
      <span>states <b id="statStates">0</b></span>
    </div>
    <button class="theme-toggle" id="themeToggle" type="button" title="Toggle theme">&#9680; theme</button>
  </div>

  <div class="sidebar">
    <div class="search-wrap">
      <input id="search" type="text" placeholder="Filter maps... (e.g. oak, gym, route)" autocomplete="off" />
    </div>
    <div class="map-list" id="mapList" role="listbox" aria-label="Maps with a script table"></div>
  </div>

  <div class="canvas-wrap" id="canvasWrap">
    <div class="bgdots"></div>
    <div class="header-strip" id="headerStrip" style="display:none">
      <span class="fn" id="fnName"></span>
      <span class="meta" id="fnMeta"></span>
    </div>
    <div class="viewport" id="viewport">
      <svg class="edges" id="edgesSvg">
        <defs>
          <marker id="arrow-fwd" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L8,4 L0,8 Z" fill="var(--edge-fwd)"></path>
          </marker>
          <marker id="arrow-back" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L8,4 L0,8 Z" fill="var(--edge-back)"></path>
          </marker>
        </defs>
      </svg>
      <div id="nodesLayer"></div>
    </div>
    <div class="welcome" id="welcome">
      <div class="big">Select a map on the left</div>
      <div class="small">Each block is one dispatch state (a handler in that map's <code>def_script_pointers</code> table) with its real asm body. Solid arrows are forward <code>next_state</code> transitions; dashed red arrows loop back to an earlier state. Drag to pan, scroll to zoom, or use <kbd>+</kbd> / <kbd>&minus;</kbd>.</div>
    </div>
    <div class="zoom-controls">
      <button id="zoomIn" type="button" title="Zoom in">+</button>
      <button id="zoomReset" type="button" title="Reset view">&#9678;</button>
      <button id="zoomOut" type="button" title="Zoom out">&minus;</button>
    </div>
    <div class="legend" id="legend" style="display:none">
      <div class="row"><span class="sw" style="background:var(--edge-fwd)"></span> next_state</div>
      <div class="row"><span class="sw" style="background:var(--edge-back)"></span> back edge (loop)</div>
      <div class="row"><span class="chip" style="background:var(--guard-event)"></span> event guard</div>
      <div class="row"><span class="chip" style="background:var(--guard-coord)"></span> coordinate guard</div>
      <div class="row"><span class="chip" style="background:var(--set-chip)"></span> SetEvent</div>
    </div>
  </div>
</div>

<script id="graph-data" type="application/json">__DATA_JSON__</script>
<script>
(function () {
  "use strict";
  var DATA = JSON.parse(document.getElementById("graph-data").textContent);
  var root = document.documentElement;
  var INITIAL_MAP = __INITIAL_MAP__;

  // ---------- theme ----------
  function applyStoredTheme() {
    try {
      var t = localStorage.getItem("sg-theme");
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
    try { localStorage.setItem("sg-theme", next); } catch (e) {}
  });

  // ---------- tiny asm syntax highlighter ----------
  var MACROS = /^(CheckEvent|CheckEventReuseA|CheckEventHL|CheckBothEventsSet|CheckEitherEventSet|SetEvent|SetEvents|SetEventReuseHL|ResetEvent|trainer|predef|predef_jump|farcall|def_script_pointers|dw_const|text_far|text_asm|text_end)\b/;
  var MNEM = /^(ld|ldh|jp|jr|call|ret|reti|cp|and|or|xor|inc|dec|push|pop|bit|res|set|rst|add|sub|adc|sbc|swap|rlca|rrca|rla|rra|nop|halt|di|ei|db|dw)\b/i;

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  function highlightLine(line) {
    var commentIdx = -1;
    var inStr = false;
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

  // ---------- sidebar ----------
  var mapEntries = Object.keys(DATA).map(function (id) { return DATA[id]; });
  mapEntries.sort(function (a, b) { return a.name.localeCompare(b.name); });

  document.getElementById("statMaps").textContent = mapEntries.length;
  document.getElementById("statStates").textContent = mapEntries.reduce(function (s, m) { return s + m.state_count; }, 0);

  var mapListEl = document.getElementById("mapList");
  var activeId = null;

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
      el.innerHTML = '<span class="n">' + escapeHtml(m.name) + '</span><span class="c">' + m.state_count + "</span>";
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

  // ---------- layout engine (layered DAG, longest-path ranking) ----------
  function computeLayout(mapData) {
    var states = mapData.states;
    var ids = Object.keys(states).map(Number).sort(function (a, b) { return a - b; });
    var indeg = {};
    ids.forEach(function (i) { indeg[i] = 0; });
    ids.forEach(function (i) {
      var ns = states[i].next_state;
      if (ns !== null && ns !== undefined && states[ns]) indeg[ns]++;
    });

    var rank = {};
    var backEdges = {};
    var visiting = {};
    var visited = {};

    function dfs(i, depth) {
      if (visited[i]) { rank[i] = Math.max(rank[i], depth); return; }
      if (visiting[i]) { backEdges[i + ">" + states[i].next_state] = true; return; }
      visiting[i] = true;
      rank[i] = Math.max(rank[i] || 0, depth);
      var ns = states[i].next_state;
      if (ns !== null && ns !== undefined && states[ns]) {
        if (visiting[ns]) {
          backEdges[i + ">" + ns] = true;
        } else {
          dfs(ns, depth + 1);
        }
      }
      visiting[i] = false;
      visited[i] = true;
    }

    var roots = ids.filter(function (i) { return indeg[i] === 0; });
    if (!roots.length) roots = [ids[0]];
    roots.forEach(function (r) { dfs(r, 0); });
    ids.forEach(function (i) { if (rank[i] === undefined) dfs(i, 0); });

    var byRank = {};
    ids.forEach(function (i) {
      var r = rank[i] || 0;
      (byRank[r] = byRank[r] || []).push(i);
    });
    var maxRank = 0;
    Object.keys(byRank).forEach(function (r) { maxRank = Math.max(maxRank, Number(r)); });

    var NODE_W = 380, COL_GAP = 110, ROW_GAP = 26;
    var pos = {};
    for (var r = 0; r <= maxRank; r++) {
      var col = (byRank[r] || []).slice().sort(function (a, b) { return a - b; });
      col.forEach(function (i, idx) {
        pos[i] = { x: r * (NODE_W + COL_GAP) + 30, y: idx, col: r };
      });
    }
    return { ids: ids, rank: rank, pos: pos, backEdges: backEdges, nodeWidth: NODE_W, rowGap: ROW_GAP };
  }

  // ---------- render ----------
  var viewport = document.getElementById("viewport");
  var nodesLayer = document.getElementById("nodesLayer");
  var edgesSvg = document.getElementById("edgesSvg");
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
      var label = (g[2] ? "" : "NOT ") + g[1].replace(/^EVENT_/, "");
      return '<span class="guard-chip ' + cls + '" title="' + escapeHtml(g[1]) + " must be " + g[2] + '">' + escapeHtml(label) + "</span>";
    }
    return '<span class="guard-chip coord" title="w' + (g[1] === "x" ? "X" : "Y") + 'Coord must equal ' + g[2] + '">' + g[1] + "==" + g[2] + "</span>";
  }

  function selectMap(id) {
    activeId = id;
    renderList(document.getElementById("search").value);
    var m = DATA[id];
    welcome.style.display = "none";
    headerStrip.style.display = "flex";
    legend.style.display = "flex";
    document.getElementById("fnName").textContent = m.name;
    document.getElementById("fnMeta").textContent =
      m.file + (m.cur_script_ram ? "  ·  " + m.cur_script_ram : "  ·  shared wCurMapScript") + "  ·  " + m.state_count + " states";

    var layout = computeLayout(m);
    nodesLayer.innerHTML = "";
    var nodeEls = {};

    layout.ids.forEach(function (i) {
      var st = m.states[i];
      var isNoop = (!st.sets || !st.sets.length) && (!st.guards || !st.guards.length) &&
        /noop/i.test(st.handler);
      var el = document.createElement("div");
      el.className = "node" + (isNoop ? " noop" : "");
      el.dataset.idx = i;
      var guardsHtml = (st.guards || []).map(guardChip).join("");
      var setsHtml = (st.sets || []).map(function (s) {
        return '<span class="set-chip">SetEvent ' + escapeHtml(s.replace(/^EVENT_/, "")) + "</span>";
      }).join("");
      el.innerHTML =
        '<div class="nhead"><span class="idx">' + i + '</span><span class="handler">' + escapeHtml(st.handler) + '</span></div>' +
        (guardsHtml ? '<div class="guards">' + guardsHtml + "</div>" : "") +
        '<div class="code">' + highlightCode(st.code) + "</div>" +
        (setsHtml ? '<div class="nfoot">' + setsHtml + "</div>" : "");
      nodesLayer.appendChild(el);
      nodeEls[i] = el;
      el.style.left = layout.pos[i].x + "px";
    });

    // measure heights, then assign y within each column stacked with gaps
    var colTop = {};
    layout.ids.slice().sort(function (a, b) { return layout.pos[a].col - layout.pos[b].col || layout.pos[a].y - layout.pos[b].y; })
      .forEach(function (i) {
        var col = layout.pos[i].col;
        var top = colTop[col] || 0;
        nodeEls[i].style.top = top + "px";
        colTop[col] = top + nodeEls[i].getBoundingClientRect().height / scale + layout.rowGap;
      });

    // edges
    while (edgesSvg.lastChild && edgesSvg.lastChild.tagName !== "defs") edgesSvg.removeChild(edgesSvg.lastChild);
    var maxX = 0, maxY = 0;
    layout.ids.forEach(function (i) {
      var st = m.states[i];
      var ns = st.next_state;
      if (ns === null || ns === undefined || !m.states[ns]) return;
      var isBack = !!layout.backEdges[i + ">" + ns];
      var a = nodeEls[i].getBoundingClientRect();
      var b = nodeEls[ns].getBoundingClientRect();
      var vp = viewport.getBoundingClientRect();
      var ax = (a.left - vp.left) / scale, ay = (a.top - vp.top) / scale, aw = a.width / scale, ah = a.height / scale;
      var bx = (b.left - vp.left) / scale, by = (b.top - vp.top) / scale, bw = b.width / scale, bh = b.height / scale;
      var path = document.createElementNS("http://www.w3.org/2000/svg", "path");
      var d;
      if (!isBack) {
        var x1 = ax + aw, y1 = ay + ah / 2;
        var x2 = bx, y2 = by + bh / 2;
        var mx = (x1 + x2) / 2;
        d = "M" + x1 + "," + y1 + " C" + mx + "," + y1 + " " + mx + "," + y2 + " " + x2 + "," + y2;
      } else {
        var sx = ax + aw / 2, sy = ay + ah;
        var tx = bx + bw / 2, ty = by;
        var loopOut = 60;
        d = "M" + sx + "," + sy + " C" + sx + "," + (sy + loopOut) + " " + tx + "," + (ty - loopOut) + " " + tx + "," + ty;
      }
      path.setAttribute("d", d);
      path.setAttribute("class", "edge-path" + (isBack ? " back" : ""));
      path.setAttribute("marker-end", isBack ? "url(#arrow-back)" : "url(#arrow-fwd)");
      edgesSvg.appendChild(path);
      maxX = Math.max(maxX, ax + aw, bx + bw);
      maxY = Math.max(maxY, ay + ah, by + bh);
    });
    layout.ids.forEach(function (i) {
      var r = nodeEls[i].getBoundingClientRect();
      var vp = viewport.getBoundingClientRect();
      maxX = Math.max(maxX, (r.left - vp.left) / scale + r.width / scale);
      maxY = Math.max(maxY, (r.top - vp.top) / scale + r.height / scale);
    });
    edgesSvg.setAttribute("width", maxX + 40);
    edgesSvg.setAttribute("height", maxY + 40);

    resetView(layout);
  }

  function resetView(layout) {
    var wrap = document.getElementById("canvasWrap").getBoundingClientRect();
    scale = 1;
    var cols = 1;
    Object.keys(layout.pos).forEach(function (k) { cols = Math.max(cols, layout.pos[k].col + 1); });
    var totalW = cols * (layout.nodeWidth + 110);
    if (totalW > wrap.width - 80) scale = Math.max(0.35, (wrap.width - 80) / totalW);
    panX = 40; panY = 56;
    applyTransform();
  }

  // ---------- pan / zoom ----------
  var wrapEl = document.getElementById("canvasWrap");
  var dragging = false, lastX = 0, lastY = 0;
  wrapEl.addEventListener("pointerdown", function (e) {
    if (e.target.closest(".node") && e.target.closest(".code")) return;
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
    scale = Math.min(2.2, Math.max(0.2, scale * delta));
    panX = mx - ((mx - panX) / prevScale) * scale;
    panY = my - ((my - panY) / prevScale) * scale;
    applyTransform();
  }, { passive: false });

  document.getElementById("zoomIn").addEventListener("click", function () { scale = Math.min(2.2, scale * 1.15); applyTransform(); });
  document.getElementById("zoomOut").addEventListener("click", function () { scale = Math.max(0.2, scale / 1.15); applyTransform(); });
  document.getElementById("zoomReset").addEventListener("click", function () {
    if (activeId) selectMap(activeId);
  });

  var initial = (INITIAL_MAP && DATA[INITIAL_MAP])
    ? DATA[INITIAL_MAP]
    : (mapEntries.find(function (m) { return m.const_name === "OAKS_LAB"; }) || mapEntries[0]);
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
        "--out", default=DEFAULT_OUT,
        help=f"docelowy plik HTML (domyslnie {DEFAULT_OUT}) -- samodzielny, dziala offline",
    )
    parser.add_argument(
        "--map",
        help="nazwa mapy do pokazania od razu po otwarciu (np. OAKS_LAB) -- domyslnie OAKS_LAB albo pierwsza z listy",
    )
    args = parser.parse_args()

    data = build_data()
    initial_map_id = None
    if args.map:
        target = args.map.strip().upper()
        for map_id_str, entry in data.items():
            if entry["const_name"] == target:
                initial_map_id = map_id_str
                break
        if initial_map_id is None:
            print(f"uwaga: nie znaleziono mapy '{args.map}', uzywam domyslnej")

    html = HTML_TEMPLATE.replace(
        "__DATA_JSON__", json.dumps(data, separators=(",", ":"))
    ).replace("__INITIAL_MAP__", json.dumps(initial_map_id))

    out_path = Path(args.out)
    out_path.write_text(html, encoding="utf-8")
    print(f"zapisano {out_path.resolve()} ({len(data)} map, {sum(m['state_count'] for m in data.values())} stanow)")
    print("otworz w przegladarce -- narzedzie jest w pelni offline (bez serwera, bez CDN)")


if __name__ == "__main__":
    main()
