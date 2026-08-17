#!/usr/bin/env python
"""Extract per-map script-graph data (parsed structure + real asm source
text) for tools/view_script_graph.py's offline graph viewer.

Usage: python tools/gen_script_graph_data.py > out.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))
import gen_map_scripts as gms  # noqa: E402
import pokered_asm as pa  # noqa: E402


def prettify_map_name(const_name: str) -> str:
    return " ".join(w.capitalize() for w in const_name.split("_"))


def build_data() -> dict[str, dict]:
    """map_id (str) -> {map_id, const_name, name, file, cur_script_ram,
    state_count, states: {idx (str): {handler, const, guards, sets,
    next_state, code, line_start, line_end}}}, for every scripts/*.asm file
    with a def_script_pointers table."""
    from pokemon.map_constants import MAPS_BY_ID

    out: dict[str, dict] = {}
    for path in sorted(pa.SCRIPTS_DIR.glob("*.asm")):
        map_id = pa.resolve_map_id(path)
        if map_id is None:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        tables = gms.find_state_tables(text)
        if not tables:
            continue
        table = tables[0]
        label_ranges = pa.find_label_ranges(lines)
        script_const_to_index = {const: idx for idx, (_h, const) in enumerate(table)}

        states: dict[str, dict] = {}
        for idx, (handler, const) in enumerate(table):
            rng = label_ranges.get(handler)
            if rng is None:
                continue
            body_start = rng[0] + 1
            guards, after_guards = gms.parse_entry_guards(lines, body_start, rng[1])
            sets = sorted(gms.events_set_in_range(lines, after_guards, rng[1]))
            next_state = gms.parse_transition(
                lines, after_guards, rng[1], script_const_to_index
            )
            code = "\n".join(lines[rng[0] : rng[1]])
            states[str(idx)] = {
                "handler": handler,
                "const": const,
                "guards": guards,
                "sets": sets,
                "next_state": next_state,
                "code": code,
                "line_start": rng[0] + 1,
                "line_end": rng[1],
            }

        const_name = MAPS_BY_ID.get(map_id, (f"MAP_{map_id}",))[0]
        out[str(map_id)] = {
            "map_id": map_id,
            "const_name": const_name,
            "name": prettify_map_name(const_name),
            "file": path.name,
            "cur_script_ram": gms.find_cur_script_var(lines),
            "state_count": len(states),
            "states": states,
        }
    return out


def main() -> None:
    json.dump(build_data(), sys.stdout, indent=None, separators=(",", ":"))


if __name__ == "__main__":
    main()
