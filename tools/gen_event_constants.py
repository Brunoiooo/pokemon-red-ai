#!/usr/bin/env python
"""Regenerate src/pokemon/event_constants.py from the pret/pokered disassembly.

constants/event_constants.asm declares every wEventFlags bit *by name* using
rgbds' const_def/const/const_skip/const_next -- a compile-time enum, no
literal bit numbers in the source. This script re-implements that tiny
assembler subset in Python (no rgbds/rgblink needed here, unlike
gen_ram_constants.py -- these are pure integer constants, not linked
addresses) and resolves every EVENT_* name to:
  - its global bit index (0..NUM_EVENTS-1)
  - the absolute WRAM byte address (wEventFlags base, from ram_constants.py,
    + index // 8)
  - the bit position within that byte (index % 8)

Usage: python tools/gen_event_constants.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENT_CONSTANTS_ASM = REPO_ROOT / "reference" / "pokered" / "constants" / "event_constants.asm"
OUTPUT_PATH = REPO_ROOT / "src" / "pokemon" / "event_constants.py"

sys.path.insert(0, str(REPO_ROOT / "src"))

CONST_DEF_RE = re.compile(r"^const_def(?:\s+\$?([0-9A-Fa-f]+))?$")
CONST_NEXT_RE = re.compile(r"^const_next\s+\$?([0-9A-Fa-f]+)$")
CONST_SKIP_RE = re.compile(r"^const_skip\s+(\d+)$")
CONST_RE = re.compile(r"^const\s+(\w+)$")
NUM_EVENTS_RE = re.compile(r"^DEF\s+NUM_EVENTS\s+EQU\s+const_value$")


def _is_hex_next(raw_arg: str, line: str) -> bool:
    # const_next $28  -> hex;  const_next 40  -> decimal. Distinguish by the
    # literal '$' immediately preceding the captured digits in the source line.
    return f"${raw_arg}" in line


def parse_events(path: Path) -> dict[str, int]:
    idx = 0
    events: dict[str, int] = {}
    num_events: int | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(";"):
            continue

        m = CONST_DEF_RE.match(line)
        if m:
            idx = int(m.group(1), 16) if m.group(1) else 0
            continue

        m = CONST_NEXT_RE.match(line)
        if m:
            idx = int(m.group(1), 16) if _is_hex_next(m.group(1), line) else int(m.group(1))
            continue

        m = CONST_SKIP_RE.match(line)
        if m:
            idx += int(m.group(1))
            continue

        m = NUM_EVENTS_RE.match(line)
        if m:
            num_events = idx
            continue

        m = CONST_RE.match(line)
        if m:
            name = m.group(1)
            if name in events:
                raise ValueError(f"duplicate event constant {name!r} (idx {events[name]} vs {idx})")
            events[name] = idx
            idx += 1
            continue

    if num_events is None:
        raise ValueError("NUM_EVENTS not found -- did event_constants.asm's format change?")
    if num_events != idx:
        raise ValueError(f"NUM_EVENTS ({num_events}) != final parsed index ({idx})")

    return events


def render_module(events: dict[str, int], base_addr: int, pokered_commit: str) -> str:
    lines = [
        '"""Named wEventFlags bit indices, generated from pret/pokered.',
        "",
        f"Source: https://github.com/pret/pokered @ {pokered_commit}",
        "         constants/event_constants.asm",
        f"wEventFlags base address: 0x{base_addr:04X} (see src/pokemon/ram_constants.py)",
        "Regenerate with: python tools/gen_event_constants.py",
        "DO NOT EDIT BY HAND.",
        '"""',
        "",
        f"WEVENTFLAGS_BASE = 0x{base_addr:04X}",
        "",
        "# name -> global bit index (0..NUM_EVENTS-1).",
        "EVENTS: dict[str, int] = {",
    ]
    for name in sorted(events, key=lambda n: events[n]):
        lines.append(f"    {name!r}: {events[name]},")
    lines.append("}")
    lines.append("")
    lines.append(f"NUM_EVENTS = {len(events)}")
    lines.append("")
    lines.append("# index -> name, for reverse lookup.")
    lines.append("EVENTS_BY_INDEX: dict[int, str] = {v: k for k, v in EVENTS.items()}")
    lines.append("")
    lines.append("")
    lines.append("def event_address(name: str) -> int:")
    lines.append('    """Absolute WRAM byte address containing this event\'s bit."""')
    lines.append("    return WEVENTFLAGS_BASE + EVENTS[name] // 8")
    lines.append("")
    lines.append("")
    lines.append("def event_bit(name: str) -> int:")
    lines.append('    """Bit position (0-7) within event_address(name)."""')
    lines.append("    return EVENTS[name] % 8")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    events = parse_events(EVENT_CONSTANTS_ASM)

    from pokemon.ram_constants import RAM  # noqa: E402

    base_addr = RAM["wEventFlags"]

    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(REPO_ROOT / "reference" / "pokered"), "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip() or "unknown"

    OUTPUT_PATH.write_text(render_module(events, base_addr, commit), encoding="utf-8")
    print(f"wrote {len(events)} events to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
