#!/usr/bin/env python3
"""Benchmark training throughput (env steps/sec) across worker counts.

Sweeps SubprocVecEnv worker counts upward from --min-workers, measuring
steps/sec for each (the same per-second throughput SB3's progress bar
reports as "fps" -- env steps summed across all workers, per wall-clock
second). Stops once fps has declined for --patience consecutive worker
counts after the best one seen so far, i.e. once adding workers stops
helping (CPU contention outweighing added parallelism), and reports the
best worker count found.

Usage:
  python benchmark_workers.py
  python benchmark_workers.py --min-workers 4 --max-workers 24
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import set_start_method

sys.path.insert(0, "src")


def _make_env(rank, save_state, max_steps, frame_skip, goal, stage, seed):
    def _thunk():
        from env.pokemon_red_env import PokemonRedEnv

        env = PokemonRedEnv(
            save_state=save_state,
            max_steps=max_steps,
            frame_skip=frame_skip,
            goal=goal,
            stage=stage,
            curriculum_mix=0.0,
            curriculum_saves=[save_state],
            render_mode=None,
            auto_advance=True,
            worker_rank=rank,
            n_workers=1,
            collect_heatmap=False,
            save_checkpoints=False,
        )
        env.reset(seed=seed + rank)
        return env

    return _thunk


def measure_fps(
    n_workers: int,
    warmup_steps: int,
    measure_steps: int,
    save_state: str,
    max_steps: int,
    frame_skip: int,
    goal: str,
    stage: str,
    seed: int,
) -> float:
    """Random-action steps/sec for n_workers parallel envs.

    Actions are drawn uniformly at random rather than via action_masks():
    frame_skip ticks (not button legality) dominate per-step emulator cost,
    so this measures raw env-stepping throughput -- the thing that actually
    caps --workers -- without paying for a policy forward pass.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    thunks = [
        _make_env(i, save_state, max_steps, frame_skip, goal, stage, seed)
        for i in range(n_workers)
    ]
    vec_env = (
        DummyVecEnv(thunks)
        if n_workers <= 1
        else SubprocVecEnv(thunks, start_method="spawn")
    )
    try:
        vec_env.reset()
        action_space = vec_env.action_space

        for _ in range(warmup_steps):
            actions = [action_space.sample() for _ in range(n_workers)]
            vec_env.step(actions)

        start = time.perf_counter()
        for _ in range(measure_steps):
            actions = [action_space.sample() for _ in range(n_workers)]
            vec_env.step(actions)
        elapsed = time.perf_counter() - start
    finally:
        vec_env.close()

    return (measure_steps * n_workers) / elapsed


def run(args) -> None:
    set_start_method("spawn", force=True)

    from curriculum_config import (
        get_curriculum_saves,
        get_goal_for_stage,
        get_stage_max_steps,
        resolve_stage_name,
    )

    stage = resolve_stage_name(args.stage)
    goal = args.goal or get_goal_for_stage(stage)
    max_steps = args.max_steps or get_stage_max_steps(stage)
    save_state = get_curriculum_saves(stage)[-1]

    print("=" * 70)
    print("  Pokemon Red AI - Worker FPS Benchmark")
    print("=" * 70)
    print(f"Stage:        {stage}")
    print(f"Goal:         {goal}")
    print(f"Max steps:    {max_steps}")
    print(f"Frame skip:   {args.frame_skip}")
    print(f"Warmup steps: {args.warmup_steps}   Measure steps: {args.measure_steps}")
    print(f"Worker range: {args.min_workers}..{args.max_workers} (step {args.step})")
    print(f"Patience:     {args.patience}   Tolerance: {args.tolerance:.0%}")
    print("=" * 70)

    results: list[tuple[int, float]] = []
    best_fps = -1.0
    best_workers = args.min_workers
    declines = 0

    workers = args.min_workers
    while workers <= args.max_workers:
        print(f"[workers={workers:>3}] measuring...", end=" ", flush=True)
        fps = measure_fps(
            workers,
            args.warmup_steps,
            args.measure_steps,
            save_state,
            max_steps,
            args.frame_skip,
            goal,
            stage,
            args.seed,
        )
        print(f"{fps:.1f} steps/sec")
        results.append((workers, fps))

        if fps > best_fps * (1.0 + args.tolerance):
            best_fps = fps
            best_workers = workers
            declines = 0
        else:
            declines += 1

        if declines >= args.patience:
            print(
                f"\nFPS declined for {declines} consecutive worker count(s) "
                f"after the best (workers={best_workers}); stopping sweep."
            )
            break

        workers += args.step
    else:
        print(f"\nReached --max-workers {args.max_workers} without a decline; stopping sweep.")

    print("=" * 70)
    print(f"{'workers':>8}  {'steps/sec':>12}")
    for w, fps in results:
        marker = "  <-- best" if w == best_workers else ""
        print(f"{w:>8}  {fps:>12.1f}{marker}")
    print("=" * 70)
    print(f"Best: --workers {best_workers}  ({best_fps:.1f} steps/sec)")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark PokemonRedEnv steps/sec across worker counts "
        "to find the --workers value with peak training throughput."
    )
    p.add_argument("--min-workers", type=int, default=1)
    p.add_argument(
        "--max-workers", type=int, default=32,
        help="Safety cap on the sweep in case fps never declines (default: 32)",
    )
    p.add_argument(
        "--step", type=int, default=1,
        help="Worker-count increment per trial (default: 1)",
    )
    p.add_argument(
        "--warmup-steps", type=int, default=50,
        help="Steps discarded before timing, to let startup jitter settle (default: 50)",
    )
    p.add_argument(
        "--measure-steps", type=int, default=300,
        help="Timed steps per worker-count trial (default: 300)",
    )
    p.add_argument(
        "--patience", type=int, default=2,
        help="Consecutive declining trials after the best before stopping (default: 2)",
    )
    p.add_argument(
        "--tolerance", type=float, default=0.02,
        help="Relative fps improvement required to count as a new best, "
        "so measurement noise doesn't reset the decline counter (default: 0.02)",
    )
    p.add_argument("--stage", default="EVENT_GOT_STARTER",
                    help="Curriculum stage to benchmark against (see curriculum_config.STAGE_ORDER)")
    p.add_argument("--goal", default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--frame-skip", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
