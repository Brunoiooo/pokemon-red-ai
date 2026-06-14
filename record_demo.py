#!/usr/bin/env python3
"""Record a human demonstration for behavioral cloning pre-training.

Uses a global keyboard hook so keys are captured even when the game window
has focus.  PyBoy SDL2 internally maps Z→A and X→B, so we use A/B instead.

Controls (work in any focused window):
  Arrow keys  — movement
  A           — A button (confirm / interact)
  B           — B button (cancel)
  Enter       — Start
  Space       — Select
  ESC / Q     — stop recording

Usage:
  python record_demo.py
  python record_demo.py -o demos/my_demo.json -n 500
"""
import sys
sys.path.insert(0, "src")

import json
import time
import queue
import argparse
from collections import Counter
from pathlib import Path
from multiprocessing import RLock

import keyboard

from pokemon.Emulator import Emulator

BUTTON_NAMES = ["A", "B", "Start", "Select", "Left", "Right", "Up", "Down"]

# Keys that map to game actions.  Avoids Z/X (PyBoy SDL2 maps those to A/B).
KEY_MAP: dict[str, tuple[str, int | None]] = {
    "up":    ("Up",    6),
    "down":  ("Down",  7),
    "left":  ("Left",  4),
    "right": ("Right", 5),
    "a":     ("A",     0),
    "b":     ("B",     1),
    "enter": ("Start", 2),
    "space": ("Sel",   3),
    "esc":   ("ESC",   None),
    "q":     ("Q",     None),
}


def print_sep(char="=", width=60):
    print(char * width)


def run(args):
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Queue filled by the global keyboard hook; main loop drains it.
    key_queue: queue.Queue[tuple[str, int | None]] = queue.Queue()

    def on_key(event: keyboard.KeyboardEvent):
        if event.event_type != keyboard.KEY_DOWN:
            return
        entry = KEY_MAP.get(event.name.lower())
        if entry is not None:
            key_queue.put_nowait(entry)

    keyboard.hook(on_key, suppress=False)

    files_lock = RLock()
    emulator = Emulator(files_lock=files_lock)
    emulator.use_sdl = True

    print_sep()
    print("  Pokemon Red AI — Demo Recorder")
    print_sep()
    print("  Controls (work in any window — global hook):")
    print("    Arrow keys → movement")
    print("    A          → A button (confirm / interact)")
    print("    B          → B button (cancel)")
    print("    Enter      → Start")
    print("    Space      → Select")
    print("    ESC / Q    → stop recording")
    print_sep()
    print(f"  Checkpoint : saves/{args.checkpoint}")
    print(f"  Output     : {args.output}")
    print(f"  Max steps  : {args.max_steps}")
    print_sep()
    print("\nGame window open.  Keys are captured globally — no need to focus the terminal.\n")

    memory, inputs = emulator.reset(dir=args.checkpoint)
    emulator.pyboy.tick(1)  # force initial SDL2 render (avoid black screen)

    actions: list[int] = []
    total_reward = 0.0

    while len(actions) < args.max_steps:
        try:
            label, action = key_queue.get(timeout=0.016)
        except queue.Empty:
            # No key yet — tick one frame to keep SDL2 window alive
            emulator.pyboy.tick(1)
            continue

        if label in ("ESC", "Q"):
            print("\nStopped.")
            break

        if action is None:
            continue

        next_memory, next_inputs, reward, terminated, truncated = emulator.step(
            memory=memory, action=action
        )
        actions.append(action)
        total_reward += reward

        tiles = len(emulator.data.visited_positions)
        print(f"  step {len(actions):4d}: {label:6s}  "
              f"reward={reward:+.4f}  total={total_reward:+.3f}  tiles={tiles}")

        if terminated or truncated:
            print(f"\nEpisode ended ({'terminated' if terminated else 'truncated'}).")
            break

        memory, inputs = next_memory, next_inputs

    keyboard.unhook_all()

    try:
        emulator.pyboy.stop(False)
    except Exception:
        pass

    if not actions:
        print("No actions recorded.")
        return

    with open(args.output, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "actions": actions}, f)

    tiles = len(emulator.data.visited_positions)
    print_sep()
    print(f"  Saved {len(actions)} actions → {args.output}")
    print(f"  Total reward : {total_reward:.4f}")
    print(f"  Tiles visited: {tiles}")
    print()
    counts = Counter(BUTTON_NAMES[a] for a in actions)
    print("  Action breakdown:")
    for btn, cnt in counts.most_common():
        bar = "#" * int(cnt / len(actions) * 30)
        print(f"    {btn:8s}: {cnt:4d}  {bar}")
    print_sep()
    print(f"\n  Next: python pretrain_bc.py --demo {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Record human demo for BC pre-training")
    parser.add_argument("--checkpoint", "-c", default="start",
                        help="Save-state dir under saves/ (default: start)")
    parser.add_argument("--output", "-o", default="demos/demo.json",
                        help="Output JSON file (default: demos/demo.json)")
    parser.add_argument("--max-steps", "-n", type=int, default=500,
                        help="Max steps to record (default: 500)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
