#!/usr/bin/env python3
"""Dry-run evaluation of a trained Pokemon Red AI model (no training)."""
import sys
sys.path.insert(0, "src")

import time
import argparse
import os
from pathlib import Path
from multiprocessing import RLock

import torch

from pokemon.Emulator import Emulator
from pokemon.ModelPokemon import get_model

os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")
torch.set_float32_matmul_precision("high")

BADGE_NAMES = ["Boulder", "Cascade", "Thunder", "Rainbow", "Soul", "Marsh", "Volcano", "Earth"]

ACTION_NAMES = ["A", "B", "Start", "Select", "Left", "Right", "Up", "Down"]


def fmt_time(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def load_model(model_path: str, device: torch.device, files_lock):
    model = get_model(device=device, files_lock=files_lock)
    data = torch.load(model_path, map_location=device, weights_only=False)
    state = data["model_state"] if isinstance(data, dict) and "model_state" in data else data
    model.load_state_dict(state)
    model.eval()
    return model


def run_episode(emulator: Emulator, model, device: torch.device, checkpoint: str,
                max_steps: int, verbose: bool):
    memory, inputs = emulator.reset(dir=checkpoint)
    total_reward = 0.0
    step = 0
    action_counts = [0] * 8
    start = time.time()

    while step < max_steps:
        step += 1

        with torch.inference_mode():
            out = model(inputs)
            q = out["q"] if isinstance(out, dict) else out
            q = q.squeeze(0)

        action = int(torch.argmax(q).item())
        action_counts[action] += 1

        next_memory, next_inputs, reward, terminated, truncated = emulator.step(
            memory=memory, action=action
        )
        total_reward += reward

        if verbose and step % 500 == 0:
            badges = sum(emulator.data.badges(emulator.pyboy.memory))
            tiles = len(emulator.data.visited_positions)
            elapsed = time.time() - start
            print(f"  step={step:5d}  reward={total_reward:8.3f}  badges={badges}  tiles={tiles}  {fmt_time(elapsed)}")

        if terminated or truncated:
            break

        memory, inputs = next_memory, next_inputs

    elapsed = time.time() - start
    badges_vec = emulator.data.badges(emulator.pyboy.memory)
    badges_earned = [BADGE_NAMES[i] for i, b in enumerate(badges_vec) if b]
    tiles_visited = len(emulator.data.visited_positions)

    return {
        "steps": step,
        "reward": total_reward,
        "time": elapsed,
        "badges": badges_earned,
        "tiles": tiles_visited,
        "action_counts": action_counts,
        "terminated": terminated,
        "truncated": truncated,
    }


def print_separator(char="=", width=64):
    print(char * width)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Pokemon Red AI model (dry run, no training)"
    )
    parser.add_argument("--model", "-m", default="models/best.pth",
                        help="Path to .pth checkpoint (default: models/best.pth)")
    parser.add_argument("--episodes", "-e", type=int, default=1,
                        help="Number of episodes (default: 1)")
    parser.add_argument("--checkpoint", "-c", default="start",
                        help="Save-state dir under saves/ to start from (default: start)")
    parser.add_argument("--max-steps", "-s", type=int, default=5000,
                        help="Max steps per episode (default: 5000)")
    parser.add_argument("--gui", action="store_true",
                        help="Show game window during evaluation")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress every 500 steps")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"[ERROR] Model not found: {args.model}")
        sys.exit(1)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print_separator()
    print("  Pokemon Red AI — Dry Evaluation")
    print_separator()
    print(f"  Model:       {args.model}")
    print(f"  Device:      {device}")
    print(f"  Episodes:    {args.episodes}")
    print(f"  Max steps:   {args.max_steps}")
    print(f"  Checkpoint:  saves/{args.checkpoint}")
    print(f"  GUI:         {'yes' if args.gui else 'no'}")
    print_separator()

    files_lock = RLock()

    print("Loading model... ", end="", flush=True)
    try:
        model = load_model(args.model, device, files_lock)
        print("OK")
    except Exception as exc:
        print(f"FAILED\n[ERROR] {exc}")
        sys.exit(1)

    emulator = Emulator(files_lock=files_lock)
    emulator.use_sdl = args.gui

    all_results = []

    for ep in range(1, args.episodes + 1):
        print(f"\n[Episode {ep}/{args.episodes}]")
        result = run_episode(
            emulator=emulator,
            model=model,
            device=device,
            checkpoint=args.checkpoint,
            max_steps=args.max_steps,
            verbose=args.verbose,
        )
        all_results.append(result)

        status = "terminated" if result["terminated"] else "truncated"
        badge_str = ", ".join(result["badges"]) if result["badges"] else "none"
        print(f"  Reward:  {result['reward']:8.3f}")
        print(f"  Steps:   {result['steps']}")
        print(f"  Tiles:   {result['tiles']}")
        print(f"  Badges:  {badge_str}")
        print(f"  Time:    {fmt_time(result['time'])}  ({status})")

        top_actions = sorted(enumerate(result["action_counts"]), key=lambda x: -x[1])[:3]
        top_str = "  ".join(f"{ACTION_NAMES[i]}:{c}" for i, c in top_actions)
        print(f"  Top actions: {top_str}")

    emulator.pyboy.stop(False)

    if args.episodes > 1:
        print()
        print_separator("-")
        rewards = [r["reward"] for r in all_results]
        tiles = [r["tiles"] for r in all_results]
        badge_counts = [len(r["badges"]) for r in all_results]
        print(f"  Summary over {args.episodes} episodes")
        print_separator("-")
        print(f"  Reward   avg={sum(rewards)/len(rewards):8.3f}  max={max(rewards):8.3f}  min={min(rewards):8.3f}")
        print(f"  Tiles    avg={sum(tiles)/len(tiles):.0f}  max={max(tiles)}  min={min(tiles)}")
        print(f"  Badges   avg={sum(badge_counts)/len(badge_counts):.2f}  max={max(badge_counts)}")
        print(f"  Total time: {fmt_time(sum(r['time'] for r in all_results))}")
        print_separator("-")


if __name__ == "__main__":
    main()
