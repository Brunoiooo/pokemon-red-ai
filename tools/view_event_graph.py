#!/usr/bin/env python
"""CLI viewer for the pret/pokered event dependency graph.

Renders a layered parent -> (map, event) -> child dependency tree with
matplotlib -- no networkx/graphviz dependency, matching this project's
existing plotting stack (see src/utils/PositionHeatmap.py). Data comes from
src/pokemon/event_graph.py (see tools/gen_event_graph.py for how it's built
and why edges are a static-analysis heuristic, not proven causality).

Usage:
  python tools/view_event_graph.py --list-maps
  python tools/view_event_graph.py --map PEWTER_GYM
  python tools/view_event_graph.py --map-id 54
  python tools/view_event_graph.py --event EVENT_BEAT_BROCK
  python tools/view_event_graph.py --dead
  python tools/view_event_graph.py --map PEWTER_GYM --out pewter_gym.png
  python tools/view_event_graph.py --full --out event_graph.html   # interactive SVG: pan/zoom/search/click-for-detail
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from pokemon.event_graph import DEAD_EVENTS, EVENT_GRAPH  # noqa: E402
from pokemon.map_constants import MAPS, MAPS_BY_ID  # noqa: E402

WEVENTFLAGS_BASE = 0xD747

ROOT_COLOR = "#2f7d5f"
FOREIGN_COLOR = "#7c6f5f"
NORMAL_COLOR = "#8a8378"
EDGE_COLOR = "0.55"


def _short(name: str, limit: int = 24) -> str:
    s = name[len("EVENT_"):] if name.startswith("EVENT_") else name
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _map_name(map_id: int | None) -> str:
    if map_id is None:
        return "(brak)"
    entry = MAPS_BY_ID.get(map_id)
    return entry[0] if entry else f"MAP_{map_id}"


def build_map_layers(map_id: int) -> tuple[list[str], list[list[str]], set[str]]:
    """Layered-BFS layout for one map's local events, mirroring the HTML
    viewer's algorithm: level 0 = local roots (no in-map parent), each
    subsequent level = 1 + max(parent levels). Returns
    (local_names_sorted_by_bit_index, layers, foreign_neighbor_names)."""
    local = sorted(
        (n for n, d in EVENT_GRAPH.items() if d["map_id"] == map_id),
        key=lambda n: EVENT_GRAPH[n]["index"],
    )
    local_set = set(local)

    level: dict[str, int | None] = {
        n: (0 if not any(p in local_set for p in EVENT_GRAPH[n]["parents"]) else None)
        for n in local
    }
    for _ in range(50):
        changed = False
        for n in local:
            if level[n] is not None:
                continue
            parent_levels = [level[p] for p in EVENT_GRAPH[n]["parents"] if p in local_set]
            if parent_levels and all(l is not None for l in parent_levels):
                level[n] = max(parent_levels) + 1
                changed = True
        if not changed:
            break
    for n in local:
        if level[n] is None:
            level[n] = 0  # cyclic guard reference never resolved -> park at root row

    max_level = max(level.values(), default=0)
    layers: list[list[str]] = [[] for _ in range(max_level + 1)]
    for n in local:
        layers[level[n]].append(n)

    foreign: set[str] = set()
    for n in local:
        foreign.update(p for p in EVENT_GRAPH[n]["parents"] if p not in local_set)
        foreign.update(c for c in EVENT_GRAPH[n]["children"] if c not in local_set)

    return local, layers, foreign


def draw_tree(ax, local: list[str], layers: list[list[str]], foreign: set[str], title: str) -> None:
    import matplotlib.patches as mpatches

    local_set = set(local)
    col_w, row_h = 2.3, 1.2
    pos: dict[str, tuple[float, float]] = {}

    for row_idx, row in enumerate(layers):
        for i, n in enumerate(row):
            pos[n] = ((i + 0.5 - len(row) / 2) * col_w, -row_idx * row_h)

    foreign_sorted = sorted(foreign)
    foreign_row_y = -len(layers) * row_h
    for i, n in enumerate(foreign_sorted):
        pos[n] = ((i + 0.5 - len(foreign_sorted) / 2) * col_w, foreign_row_y)

    for n in local:
        for c in EVENT_GRAPH[n]["children"]:
            if n not in pos or c not in pos:
                continue
            x0, y0 = pos[n]
            x1, y1 = pos[c]
            ax.annotate(
                "", xy=(x1, y1 + 0.16), xytext=(x0, y0 - 0.16),
                arrowprops=dict(arrowstyle="-|>", color=EDGE_COLOR, shrinkA=0, shrinkB=0, lw=1.1),
            )

    for n, (x, y) in pos.items():
        is_foreign = n not in local_set
        is_root = (not is_foreign) and not EVENT_GRAPH[n]["parents"]
        edgecolor = FOREIGN_COLOR if is_foreign else (ROOT_COLOR if is_root else NORMAL_COLOR)
        lw = 1.8 if is_root else 1.2
        w = max(1.4, len(_short(n)) * 0.095)
        box = mpatches.FancyBboxPatch(
            (x - w / 2, y - 0.16), w, 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=lw, linestyle="--" if is_foreign else "-",
            edgecolor=edgecolor, facecolor="#f3f0ea" if is_foreign else "#fffdf9",
        )
        ax.add_patch(box)
        ax.text(x, y, _short(n), ha="center", va="center", fontsize=7.5, family="monospace")
        if is_foreign:
            ax.text(x, y - 0.27, _map_name(EVENT_GRAPH[n]["map_id"]), ha="center", va="center",
                     fontsize=6, color="0.5")

    ax.set_title(title, fontsize=11, family="monospace")
    xs = [p[0] for p in pos.values()] or [0.0]
    ax.set_xlim(min(xs) - 1.5, max(xs) + 1.5)
    ax.set_ylim(foreign_row_y - 0.8, 0.8)
    ax.axis("off")


MAP_ROOT_COLOR = "#3b5b8c"
TREE_COL_W = 2.3
TREE_ROW_H = 1.2
TREE_PAD = 1.4
FOREST_MAX_ROW_WIDTH = 55.0  # wrap to a new band past this much x -- keeps the
                              # canvas from becoming one endless horizontal
                              # strip, which forces DPI so low (to stay under
                              # the renderer's pixel-per-axis ceiling) that
                              # neither a saved PNG nor the interactive
                              # matplotlib zoom can resolve the text.
FOREST_BAND_GAP = 2.4  # extra vertical gap between wrapped bands


def build_forest_layout(map_ids: list[int], max_row_width: float = FOREST_MAX_ROW_WIDTH):
    """Lay every map's local tree out side by side, wrapping to a new band
    once a row gets wider than max_row_width: map name = a synthetic root
    node, its true in-map roots (layer 0) hang directly beneath it, and each
    map's subtree keeps its own build_map_layers() shape ("choinka").
    Returns (trees, rows) where trees is
    [(map_id, name, local, layers, x_offset, tree_width, row), ...] and rows
    is [(row_content_width, row_max_depth), ...], one entry per band."""
    cursor = 0.0
    row = 0
    row_max_depth = 0
    trees = []
    rows: list[tuple[float, int]] = []
    for mid in map_ids:
        local, layers, _foreign = build_map_layers(mid)
        if not local:
            continue
        name = _map_name(mid)
        events_width = max((len(r) for r in layers), default=1) * TREE_COL_W
        root_label_width = max(1.6, len(name) * 0.11)
        width = max(events_width, root_label_width, TREE_COL_W)
        if cursor > 0.0 and cursor + width > max_row_width:
            rows.append((cursor - TREE_PAD, row_max_depth))
            row += 1
            cursor = 0.0
            row_max_depth = 0
        trees.append((mid, name, local, layers, cursor, width, row))
        cursor += width + TREE_PAD
        row_max_depth = max(row_max_depth, len(layers))
    if cursor > 0.0:
        rows.append((cursor - TREE_PAD, row_max_depth))
    return trees, rows


def forest_row_tops(rows: list[tuple[float, int]]) -> list[float]:
    """y-coordinate of each band's map-root row, stacked top to bottom."""
    tops = []
    y = 0.0
    for _width, depth in rows:
        tops.append(y)
        y -= TREE_ROW_H * (depth + 1) + FOREST_BAND_GAP
    return tops


def draw_forest(ax, trees: list[tuple], rows: list[tuple[float, int]]) -> None:
    import matplotlib.patches as mpatches

    row_tops = forest_row_tops(rows)

    for _mid, mname, local, layers, x_offset, width, row in trees:
        root_y = row_tops[row]
        pos: dict[str, tuple[float, float]] = {}
        for row_idx, row_events in enumerate(layers):
            for i, n in enumerate(row_events):
                x = x_offset + width / 2 + (i + 0.5 - len(row_events) / 2) * TREE_COL_W
                pos[n] = (x, root_y - (row_idx + 1) * TREE_ROW_H)

        # edges within this map's own subtree (cross-map children are
        # skipped here -- use --event on the child to see that link).
        for n in local:
            for c in EVENT_GRAPH[n]["children"]:
                if n not in pos or c not in pos:
                    continue
                x0, y0 = pos[n]
                x1, y1 = pos[c]
                ax.annotate(
                    "", xy=(x1, y1 + 0.14), xytext=(x0, y0 - 0.14),
                    arrowprops=dict(arrowstyle="-|>", color=EDGE_COLOR, shrinkA=0, shrinkB=0, lw=1.0),
                )

        # map-root node + edges down to that map's true (parentless) roots.
        root_x = x_offset + width / 2
        for n in (layers[0] if layers else []):
            x1, y1 = pos[n]
            ax.annotate(
                "", xy=(x1, y1 + 0.14), xytext=(root_x, root_y - 0.14),
                arrowprops=dict(arrowstyle="-|>", color=MAP_ROOT_COLOR, shrinkA=0, shrinkB=0, lw=1.0),
            )
        rw = max(1.6, len(mname) * 0.11)
        ax.add_patch(mpatches.FancyBboxPatch(
            (root_x - rw / 2, root_y - 0.16), rw, 0.32,
            boxstyle="round,pad=0.02,rounding_size=0.06", linewidth=2.0,
            edgecolor=MAP_ROOT_COLOR, facecolor="#e7edf6",
        ))
        ax.text(root_x, root_y, mname, ha="center", va="center", fontsize=7, family="monospace", weight="bold")

        for n, (x, y) in pos.items():
            is_root = not EVENT_GRAPH[n]["parents"]
            edgecolor = ROOT_COLOR if is_root else NORMAL_COLOR
            lw = 1.5 if is_root else 1.0
            w = max(1.2, len(_short(n, limit=18)) * 0.09)
            ax.add_patch(mpatches.FancyBboxPatch(
                (x - w / 2, y - 0.14), w, 0.28,
                boxstyle="round,pad=0.02,rounding_size=0.05", linewidth=lw,
                edgecolor=edgecolor, facecolor="#fffdf9",
            ))
            ax.text(x, y, _short(n, limit=18), ha="center", va="center", fontsize=6.2, family="monospace")


def _event_detail(name: str) -> dict:
    """Everything cmd_event() prints for one event, for the HTML detail panel."""
    d = EVENT_GRAPH[name]
    addr = WEVENTFLAGS_BASE + d["index"] // 8
    bit = d["index"] % 8
    return {
        "name": name,
        "bit": f"{d['index']}  (0x{addr:04X} bit {bit})",
        "map": _map_name(d["map_id"]),
        "parents": d["parents"],
        "children": d["children"],
        "set_in": d["set_in"],
        "checked_in": d["checked_in"],
        "reset_in": d["reset_in"],
    }


VENDOR_DIR = Path(__file__).resolve().parent / "vendor"


def build_cytoscape_elements(map_ids: list[int]) -> dict:
    """Node/edge data for the HTML viewer, with NO position math at all --
    layout (and guaranteeing zero overlap) is handled entirely by the
    battle-tested dagre layered-graph algorithm via Cytoscape.js in the
    browser (see HTML_TEMPLATE), not by hand-rolled coordinates here."""
    nodes = []
    edges = []
    for mid in map_ids:
        local, layers, _foreign = build_map_layers(mid)
        if not local:
            continue
        name = _map_name(mid)
        map_node_id = f"__map_{mid}"
        local_set = set(local)
        nodes.append({"data": {
            "id": map_node_id, "label": name, "kind": "maproot",
            "detail": {"name": name, "map": name, "events": len(local)},
        }})
        for n in local:
            is_root = not EVENT_GRAPH[n]["parents"]
            nodes.append({"data": {
                "id": n, "label": _short(n, limit=48),
                "kind": "root" if is_root else "normal",
                "detail": _event_detail(n),
            }})
            for c in EVENT_GRAPH[n]["children"]:
                if c in local_set:
                    edges.append({"data": {"source": n, "target": c, "kind": "edge"}})
        if layers:
            for n in layers[0]:
                edges.append({"data": {"source": map_node_id, "target": n, "kind": "maproot"}})

    return {"nodes": nodes, "edges": edges}


HTML_TEMPLATE = r"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>@@TITLE@@</title>
<style>
  :root {
    --root: @@ROOT_COLOR@@; --foreign: @@FOREIGN_COLOR@@; --normal: @@NORMAL_COLOR@@;
    --maproot: @@MAPROOT_COLOR@@; --edge: @@EDGE_COLOR@@; --bg: #fbfaf7; --panel-bg: #ffffff;
    --text: #2a2a26; --muted: #7c7568; --accent: #c0392b;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; font-family: "Consolas", "SFMono-Regular", Menlo, monospace; color: var(--text); background: var(--bg); }
  #toolbar {
    position: fixed; top: 0; left: 0; right: 0; height: 48px; z-index: 10;
    display: flex; align-items: center; gap: 14px; padding: 0 14px;
    background: var(--panel-bg); border-bottom: 1px solid #ddd6c9; font-size: 13px;
  }
  #toolbar h1 { font-size: 13px; font-weight: 700; margin: 0; white-space: nowrap; }
  #toolbar .stats { color: var(--muted); white-space: nowrap; }
  #search { flex: 0 0 260px; padding: 5px 8px; border: 1px solid #ccc3ae; border-radius: 4px; font-family: inherit; font-size: 13px; }
  #match-count { color: var(--muted); min-width: 70px; }
  button { font-family: inherit; font-size: 12px; padding: 5px 10px; border: 1px solid #ccc3ae; border-radius: 4px; background: #fff; cursor: pointer; }
  button:hover { background: #f1ede2; }
  .legend { margin-left: auto; display: flex; gap: 12px; color: var(--muted); font-size: 12px; }
  .legend span { display: inline-flex; align-items: center; gap: 5px; }
  .swatch { width: 11px; height: 11px; border-radius: 3px; border: 1.6px solid; display: inline-block; }
  #cy { position: absolute; top: 48px; left: 0; right: 0; bottom: 0; }
  #panel {
    position: fixed; top: 48px; right: -360px; width: 340px; bottom: 0; padding: 16px;
    background: var(--panel-bg); border-left: 1px solid #ddd6c9; overflow-y: auto;
    transition: right 0.15s ease; font-size: 12.5px; z-index: 9;
  }
  #panel.open { right: 0; }
  #panel h2 { font-size: 14px; margin: 0 0 10px; word-break: break-all; }
  #panel .field { margin-bottom: 10px; }
  #panel .field .k { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; margin-bottom: 3px; }
  #panel .field .v { line-height: 1.5; word-break: break-all; }
  #panel .field .v ul { margin: 0; padding-left: 16px; }
  #panel .field .v a.jump { color: #2f5f9e; cursor: pointer; text-decoration: underline; }
  #panel .close { position: absolute; top: 12px; right: 12px; }
  #hint { position: fixed; bottom: 10px; left: 14px; color: var(--muted); font-size: 11px; z-index: 9; }
</style>
</head>
<body>
  <div id="toolbar">
    <h1>@@TITLE@@</h1>
    <span class="stats">@@STATS@@</span>
    <input id="search" type="text" placeholder="szukaj eventu lub mapy... (Enter = przejdz)">
    <span id="match-count"></span>
    <button id="reset-btn">fit do widoku</button>
    <span class="legend">
      <span><i class="swatch" style="border-color:var(--maproot);background:#e7edf6"></i>mapa</span>
      <span><i class="swatch" style="border-color:var(--root);background:#fffdf9"></i>root event</span>
      <span><i class="swatch" style="border-color:var(--normal);background:#fffdf9"></i>event</span>
    </span>
  </div>
  <div id="cy"></div>
  <div id="panel"><button class="close" id="panel-close">x</button><div id="panel-body"></div></div>
  <div id="hint">scroll = zoom, przeciagnij = pan, klik na wezel = szczegoly (layout: dagre)</div>
<script>@@CYTOSCAPE_JS@@</script>
<script>@@CYTOSCAPE_DAGRE_JS@@</script>
<script>
const DATA = @@DATA_JSON@@;

function esc(s) { return String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

// All layout (positioning, sizing to fit each label, avoiding overlap) is
// delegated to the dagre layered-graph algorithm via the cytoscape-dagre
// extension -- see tools/vendor/ for the vendored libraries.
const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: { nodes: DATA.nodes, edges: DATA.edges },
  wheelSensitivity: 0.25,
  minZoom: 0.02,
  maxZoom: 8,
  // Cytoscape's stylesheet engine parses these values itself (it isn't
  // real browser CSS), so colors must be literal -- var(--x) is not
  // understood here the way it is in the <style> block above.
  style: [
    { selector: 'node', style: {
        'label': 'data(label)', shape: 'round-rectangle',
        'width': 'label', 'height': 'label', 'padding': '7px',
        'font-family': 'Consolas, "SFMono-Regular", Menlo, monospace', 'font-size': 10,
        'text-valign': 'center', 'text-halign': 'center',
        'background-color': '#fffdf9', 'border-width': 1.2, 'border-color': '@@NORMAL_COLOR@@',
    } },
    { selector: 'node.root', style: { 'border-color': '@@ROOT_COLOR@@', 'border-width': 1.6 } },
    { selector: 'node.maproot', style: {
        'background-color': '#e7edf6', 'border-color': '@@MAPROOT_COLOR@@', 'border-width': 2,
        'font-weight': 'bold', 'font-size': 11,
    } },
    { selector: 'node.dim', style: { 'opacity': 0.15 } },
    { selector: 'node.match', style: { 'border-color': '@@ACCENT_COLOR@@', 'border-width': 2.6 } },
    { selector: 'node.selected', style: { 'border-color': '@@ACCENT_COLOR@@', 'border-width': 2.8, 'background-color': '#fff3ec' } },
    { selector: 'edge', style: {
        'width': 1, 'curve-style': 'bezier', 'target-arrow-shape': 'triangle', 'arrow-scale': 0.8,
        'line-color': '@@EDGE_COLOR@@', 'target-arrow-color': '@@EDGE_COLOR@@',
    } },
    { selector: 'edge.maproot', style: { 'line-color': '@@MAPROOT_COLOR@@', 'target-arrow-color': '@@MAPROOT_COLOR@@' } },
    { selector: 'edge.dim', style: { 'opacity': 0.06 } },
    { selector: 'edge.lit', style: { 'width': 2.2, 'opacity': 1, 'line-color': '@@ACCENT_COLOR@@', 'target-arrow-color': '@@ACCENT_COLOR@@' } },
  ],
  layout: { name: 'dagre', rankDir: 'TB', nodeSep: 18, rankSep: 55, edgeSep: 8 },
});
for (const n of cy.nodes()) n.addClass(n.data('kind'));
for (const e of cy.edges()) if (e.data('kind') === 'maproot') e.addClass('maproot');

// --- click-to-inspect detail panel (parents/children/set_in/... like a class diagram's member list) ---
const panel = document.getElementById('panel');
const panelBody = document.getElementById('panel-body');

function fieldList(label, items) {
  if (!items || !items.length) return `<div class="field"><div class="k">${label}</div><div class="v">-</div></div>`;
  const lis = items.map(x => {
    const isEvent = cy.getElementById(x).length > 0;
    return isEvent ? `<li><a class="jump" data-id="${esc(x)}">${esc(x)}</a></li>` : `<li>${esc(x)}</li>`;
  }).join('');
  return `<div class="field"><div class="k">${label}</div><div class="v"><ul>${lis}</ul></div></div>`;
}

function selectNode(node) {
  cy.elements().removeClass('selected dim lit');
  const neigh = node.closedNeighborhood();
  if (neigh.length > 1) {
    cy.elements().difference(neigh).addClass('dim');
    neigh.edges().addClass('lit');
  }
  node.addClass('selected');

  const d = node.data('detail');
  const kind = node.data('kind');
  let html = `<h2>${esc(d.name)}</h2>`;
  if (kind === 'maproot') {
    html += `<div class="field"><div class="k">mapa</div><div class="v">${esc(d.map)}</div></div>`;
    html += `<div class="field"><div class="k">eventy</div><div class="v">${d.events}</div></div>`;
  } else {
    html += `<div class="field"><div class="k">bit</div><div class="v">${esc(d.bit)}</div></div>`;
    html += `<div class="field"><div class="k">mapa</div><div class="v">${esc(d.map)}</div></div>`;
    html += fieldList('rodzice', d.parents);
    html += fieldList('dzieci', d.children);
    html += fieldList('SetEvent w', d.set_in);
    html += fieldList('CheckEvent w', d.checked_in);
    html += fieldList('ResetEvent w', d.reset_in);
  }
  panelBody.innerHTML = html;
  panel.classList.add('open');
  panelBody.querySelectorAll('a.jump').forEach(a => a.addEventListener('click', () => jumpTo(a.dataset.id)));
}

function jumpTo(id) {
  const node = cy.getElementById(id);
  if (!node.length) return;
  selectNode(node);
  cy.animate({ center: { eles: node }, zoom: Math.max(cy.zoom(), 2.2) }, { duration: 300 });
}

function closePanel() {
  panel.classList.remove('open');
  cy.elements().removeClass('selected dim lit');
}
document.getElementById('panel-close').addEventListener('click', closePanel);
cy.on('tap', 'node', (evt) => selectNode(evt.target));
cy.on('tap', (evt) => { if (evt.target === cy) closePanel(); });

// --- search ---
const searchBox = document.getElementById('search');
const matchCountEl = document.getElementById('match-count');
let matches = cy.collection();
function runSearch() {
  const q = searchBox.value.trim().toLowerCase();
  matches.removeClass('match');
  if (!q) { matches = cy.collection(); matchCountEl.textContent = ''; return; }
  matches = cy.nodes().filter(n => n.id().toLowerCase().includes(q) || n.data('label').toLowerCase().includes(q));
  matches.addClass('match');
  matchCountEl.textContent = matches.length ? `${matches.length} wynik(ow)` : 'brak wynikow';
}
searchBox.addEventListener('input', runSearch);
searchBox.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && matches.length) jumpTo(matches[0].id());
});

document.getElementById('reset-btn').addEventListener('click', () => { closePanel(); cy.fit(undefined, 40); });
</script>
</body>
</html>
"""


def render_full_html(nodes_edges: dict, title: str, stats: str) -> str:
    html = HTML_TEMPLATE
    for token, value in [
        ("@@TITLE@@", title),
        ("@@STATS@@", stats),
        ("@@DATA_JSON@@", json.dumps(nodes_edges, separators=(",", ":"))),
        ("@@ROOT_COLOR@@", ROOT_COLOR),
        ("@@FOREIGN_COLOR@@", FOREIGN_COLOR),
        ("@@NORMAL_COLOR@@", NORMAL_COLOR),
        ("@@MAPROOT_COLOR@@", MAP_ROOT_COLOR),
        ("@@EDGE_COLOR@@", "#8c8577"),
        ("@@ACCENT_COLOR@@", "#c0392b"),
        ("@@CYTOSCAPE_JS@@", (VENDOR_DIR / "cytoscape.min.js").read_text(encoding="utf-8")),
        ("@@CYTOSCAPE_DAGRE_JS@@", (VENDOR_DIR / "cytoscape-dagre.min.js").read_text(encoding="utf-8")),
    ]:
        html = html.replace(token, value)
    return html


FOREST_INCH_PER_UNIT = 2.4 / TREE_COL_W  # same box-to-canvas ratio _figure_for() uses -- keeps text inside its box
FOREST_MAX_PIXELS = 60_000  # stay under matplotlib/Agg's ~65536px-per-axis ceiling (hit by *both*
                             # savefig and the interactive window, since both render through Agg)


def cmd_full(args: argparse.Namespace) -> None:
    map_ids = sorted(
        {d["map_id"] for d in EVENT_GRAPH.values() if d["map_id"] is not None},
        key=_map_name,
    )
    if not map_ids:
        sys.exit("brak danych do narysowania")

    if args.out and args.out.lower().endswith((".html", ".htm")):
        data = build_cytoscape_elements(map_ids)
        n_maps = sum(1 for n in data["nodes"] if n["data"]["id"].startswith("__map_"))
        n_events = len(data["nodes"]) - n_maps
        html = render_full_html(
            data, "Graf zaleznosci eventow -- pret/pokered",
            f"{n_maps} map, {n_events} eventow",
        )
        Path(args.out).write_text(html, encoding="utf-8")
        print(f"zapisano {args.out} -- otworz w przegladarce (scroll = zoom, przeciagniecie = pan)")
        return

    trees, rows = build_forest_layout(map_ids)
    if not trees:
        sys.exit("brak danych do narysowania")
    n_events = sum(len(t[2]) for t in trees)

    import matplotlib.pyplot as plt

    row_tops = forest_row_tops(rows)
    total_width = max((w for w, _depth in rows), default=0.0)
    last_top = row_tops[-1] if row_tops else 0.0
    last_depth = rows[-1][1] if rows else 0
    total_height = -last_top + TREE_ROW_H * (last_depth + 1)

    # Figure size must grow with content, but wrapping into bands (done in
    # build_forest_layout) keeps it from becoming one endless strip -- a
    # canvas thousands of inches wide forces DPI so low (to dodge the
    # renderer's pixel-per-axis ceiling) that neither a saved PNG nor the
    # interactive matplotlib zoom can resolve the text.
    fig_w = max(18.0, total_width * FOREST_INCH_PER_UNIT)
    fig_h = max(10.0, total_height * FOREST_INCH_PER_UNIT)
    dpi = max(30, min(150, int(FOREST_MAX_PIXELS / max(fig_w, fig_h))))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_forest(ax, trees, rows)
    ax.set_title(
        f"Wszystkie mapy jako korzenie -- {len(trees)} map, {n_events} eventow "
        f"(uzyj lupy/pan w oknie matplotlib zeby przyblizyc)",
        fontsize=10, family="monospace",
    )
    ax.set_xlim(-1.5, total_width + 0.5)
    ax.set_ylim(last_top - TREE_ROW_H * (last_depth + 1) - 0.6, 0.6)
    ax.axis("off")
    fig.tight_layout()
    _emit(fig, args.out, args.show, dpi=dpi)


def _figure_for(layers: list[list[str]]):
    import matplotlib.pyplot as plt

    width = max(6.0, 2.4 * max((len(r) for r in layers), default=1))
    height = max(4.0, 1.3 * (len(layers) + 1))
    return plt.subplots(figsize=(width, height))


def _emit(fig, out: str | None, show: bool, dpi: int = 150) -> None:
    import matplotlib.pyplot as plt

    if out:
        fig.savefig(out, dpi=dpi)
        print(f"zapisano {out}")
    if show or not out:
        plt.show()


def cmd_list_maps() -> None:
    counts = Counter(d["map_id"] for d in EVENT_GRAPH.values() if d["map_id"] is not None)
    for mid, count in sorted(counts.items(), key=lambda kv: _map_name(kv[0])):
        print(f"{mid:4d}  {_map_name(mid):<32s} {count:3d} event(y)")


def cmd_dead() -> None:
    for n in sorted(DEAD_EVENTS):
        print(n)


def resolve_map_id(args: argparse.Namespace) -> int:
    if args.map_id is not None:
        return args.map_id
    name = args.map.upper()
    if name not in MAPS:
        sys.exit(f"nieznana mapa: {args.map!r} (zobacz --list-maps)")
    return MAPS[name][0]


def cmd_map(args: argparse.Namespace) -> None:
    map_id = resolve_map_id(args)
    mname = _map_name(map_id)
    local, layers, foreign = build_map_layers(map_id)
    if not local:
        sys.exit(f"mapa {mname} (id={map_id}) nie ma zadnych przypisanych eventow")

    fig, ax = _figure_for(layers)
    draw_tree(ax, local, layers, foreign, f"{mname}  (map_id={map_id}, {len(local)} event(y))")
    fig.tight_layout()
    _emit(fig, args.out, args.show)


def cmd_event(args: argparse.Namespace) -> None:
    name = args.event
    if name not in EVENT_GRAPH:
        sys.exit(f"nieznany event: {name!r}")
    d = EVENT_GRAPH[name]
    addr = WEVENTFLAGS_BASE + d["index"] // 8
    bit = d["index"] % 8

    print(name)
    print(f"  bit index    {d['index']}  ->  0x{addr:04X} bit {bit}")
    print(f"  mapa         {_map_name(d['map_id'])} (id={d['map_id']})")
    print(f"  rodzice      {', '.join(d['parents']) or '-'}")
    print(f"  dzieci       {', '.join(d['children']) or '-'}")
    print(f"  SetEvent w   {', '.join(d['set_in']) or '-'}")
    print(f"  CheckEvent w {', '.join(d['checked_in']) or '-'}")
    print(f"  ResetEvent w {', '.join(d['reset_in']) or '-'}")

    if d["map_id"] is None:
        return
    local, layers, foreign = build_map_layers(d["map_id"])
    fig, ax = _figure_for(layers)
    draw_tree(ax, local, layers, foreign, f"{_map_name(d['map_id'])} — kontekst dla {name}")
    fig.tight_layout()
    _emit(fig, args.out, args.show)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-maps", action="store_true", help="wypisz wszystkie mapy z liczba eventow")
    parser.add_argument("--dead", action="store_true", help="wypisz martwe eventy (nigdy nie ustawiane/sprawdzane/kasowane)")
    parser.add_argument("--map", help="narysuj drzewo eventow danej mapy (np. PEWTER_GYM)")
    parser.add_argument("--map-id", type=int, help="jak --map, ale po numerycznym id")
    parser.add_argument("--event", help="pokaz szczegoly + kontekstowe drzewo mapy danego eventu")
    parser.add_argument(
        "--full", action="store_true",
        help="narysuj caly las: kazda mapa jako korzen, jej eventy jako choinka pod nia "
             "(przyblizanie/przesuwanie lupa+dloni z paska narzedzi matplotlib)",
    )
    parser.add_argument(
        "--out",
        help="zapisz do pliku zamiast/obok pokazania okna -- PNG normalnie, "
             "a dla --full rozszerzenie .html daje interaktywny wykres SVG "
             "(pan/zoom/szukaj/klik-po-szczegoly) zamiast statycznego PNG",
    )
    parser.add_argument("--show", action="store_true", help="wymus pokazanie okna nawet z --out")
    args = parser.parse_args()

    if args.list_maps:
        cmd_list_maps()
    elif args.dead:
        cmd_dead()
    elif args.full:
        cmd_full(args)
    elif args.event:
        cmd_event(args)
    elif args.map or args.map_id is not None:
        cmd_map(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
