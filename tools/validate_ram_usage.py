#!/usr/bin/env python
"""Cross-check hardcoded RAM addresses in src/pokemon/Data.py against the
named addresses generated from pret/pokered (src/pokemon/ram_constants.py).

For every `0xXXXX` literal found in Data.py, reports whether it matches a
known pokered symbol. Addresses used as the base of a computed offset
(e.g. `0xD188 + size * i` for party-mon array indexing) are flagged
separately since they legitimately won't match a single symbol name --
they still get looked up so you can eyeball whether the base looks right.

Usage: python tools/validate_ram_usage.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_PY = REPO_ROOT / "src" / "pokemon" / "Data.py"

sys.path.insert(0, str(REPO_ROOT / "src"))
from pokemon.ram_constants import RAM_BY_ADDRESS  # noqa: E402

HEX_RE = re.compile(r"0x([0-9A-Fa-f]{4})")
COMPUTED_RE = re.compile(r"0x[0-9A-Fa-f]{4}\s*[+\-]\s*\S|[+\-]\s*0x[0-9A-Fa-f]{4}")


def main() -> None:
    lines = DATA_PY.read_text(encoding="utf-8").splitlines()

    matched, unknown, computed = [], [], []
    seen_per_line: set[tuple[int, str]] = set()

    for lineno, line in enumerate(lines, start=1):
        for m in HEX_RE.finditer(line):
            addr = int(m.group(1), 16)
            key = (lineno, m.group(0))
            if key in seen_per_line:
                continue
            seen_per_line.add(key)

            is_computed = bool(COMPUTED_RE.search(line))
            names = RAM_BY_ADDRESS.get(addr)
            entry = (lineno, f"0x{addr:04X}", names, line.strip())

            if is_computed:
                computed.append(entry)
            elif names:
                matched.append(entry)
            else:
                unknown.append(entry)

    print(f"Data.py: {len(matched)} matched, {len(unknown)} unmatched, "
          f"{len(computed)} computed-offset addresses\n")

    if unknown:
        print(f"=== UNMATCHED addresses (no pokered symbol at this address) [{len(unknown)}] ===")
        for lineno, addr, _names, src in unknown:
            print(f"  Data.py:{lineno}  {addr}   {src}")
        print()

    print(f"=== Computed-offset base addresses (spot-check these) [{len(computed)}] ===")
    for lineno, addr, names, src in computed:
        label = ", ".join(names) if names else "??? no symbol at this base"
        print(f"  Data.py:{lineno}  {addr} -> {label}")
        print(f"      {src}")
    print()

    if matched:
        print(f"=== Matched (sample) [{len(matched)} total] ===")
        for lineno, addr, names, src in matched[:15]:
            print(f"  Data.py:{lineno}  {addr} -> {', '.join(names)}")
        if len(matched) > 15:
            print(f"  ... and {len(matched) - 15} more")


if __name__ == "__main__":
    main()
