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

from pokemon.Emulator import Emulator, DURATION_BINS

BUTTON_NAMES = ["A", "B", "Start", "Select", "Left", "Right", "Up", "Down", "None"]
BIN_16_IDX = 0  # index of 16-tick bin in DURATION_BINS
NONE_ACTION = 8  # "no button pressed" — same index the trainer uses for a no-op step

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

    # The "stuck" truncation budgets are tuned for a policy that decides every
    # step instantly. A human standing still to think, or navigating a menu,
    # burns through the same tick budget in real seconds, so scale it up
    # further for manual play on top of the global dialog patience in Data.py.
    patience = getattr(args, "patience", 8)
    if patience != 1:
        emulator.data.max_useless_ticks *= patience
        emulator.data.max_useless_dialog_ticks *= patience
        emulator.data.max_useless_battle_ticks *= patience

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
            # No key pressed — still counts as a step (a no-op), matching how
            # the trainer treats "no button" as a regular env step rather than
            # a skipped frame.
            label, action = "-", NONE_ACTION
        else:
            if label in ("ESC", "Q"):
                print("\nStopped.")
                break

            if action is None:
                continue

        meta_action = action * len(DURATION_BINS) + BIN_16_IDX
        # render_each=True paces frame-by-frame at real emulation speed; PyBoy
        # only rate-limits once per tick() call, so a single batched
        # tick(duration) call (the default, training-optimized path) blows
        # through all `duration` frames near-instantly instead of in real time.
        next_memory, next_inputs, reward, terminated, truncated = emulator.step(
            memory=memory, meta_action=meta_action, render_each=True
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
    parser.add_argument("--patience", "-p", type=int, default=8,
                        help="Multiplier on the stuck/idle truncation budget, "
                             "tuned up from the training default for human "
                             "reading/reaction speed (default: 8)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
