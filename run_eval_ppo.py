#!/usr/bin/env python3
"""Evaluate a Stable-Baselines3 PPO checkpoint on Pokemon Red."""
from __future__ import annotations

import argparse
import sys

sys.path.insert(0, "src")

from ppo.checkpoints import resolve_model_path


def _raw_env(env):
    """Unwrap Monitor / wrappers to reach PokemonRedEnv."""
    cur = env
    while hasattr(cur, "env"):
        cur = cur.env
    return cur


def _apply_stage(raw, stage: str, checkpoint: str | None = None) -> None:
    from curriculum_config import get_curriculum_saves, get_goal_for_stage, get_stage_max_steps

    saves = get_curriculum_saves(stage)
    save = checkpoint if checkpoint else saves[-1]
    raw.set_curriculum(
        goal=get_goal_for_stage(stage),
        max_steps=get_stage_max_steps(stage),
        save_state=save,
        curriculum_saves=saves,
        curriculum_mix=0.0,
        reset_steps=True,
    )


def run(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from curriculum_config import (
        get_curriculum_saves,
        get_goal_for_stage,
        get_stage_max_steps,
        next_stage,
        resolve_stage_name,
    )
    from env.pokemon_red_env import PokemonRedEnv

    model_path = resolve_model_path(getattr(args, "model", None))
    auto = bool(getattr(args, "auto_curriculum", True))
    stage = resolve_stage_name(args.stage)

    if auto:
        goal = get_goal_for_stage(stage)
        max_steps = args.max_steps if args.max_steps else get_stage_max_steps(stage)
        saves = get_curriculum_saves(stage)
        save_state = args.checkpoint if args.checkpoint != "start" else saves[-1]
    else:
        goal = args.goal
        max_steps = args.max_steps if args.max_steps else 5000
        save_state = args.checkpoint
        saves = [save_state]

    env = PokemonRedEnv(
        save_state=save_state,
        max_steps=max_steps,
        frame_skip=args.frame_skip,
        goal=goal,
        render_mode="human" if args.gui else None,
        curriculum_mix=0.0,
        curriculum_saves=saves,
    )
    # Monitor forbids step() after terminated; auto-curriculum continues the
    # same playthrough across stages, so skip Monitor in that mode.
    if not auto:
        env = Monitor(env)
    raw = _raw_env(env)

    print(f"Loading {model_path}")
    print(f"Auto curriculum: {'ON' if auto else 'OFF'}")
    print(f"Start stage: {stage}  goal={goal}  max_steps={max_steps}")
    model = PPO.load(str(model_path), device="cpu" if args.cpu else "auto")

    for ep in range(args.episodes):
        stage = resolve_stage_name(args.stage)
        if auto:
            _apply_stage(raw, stage, checkpoint=args.checkpoint)

        obs, info = env.reset()
        total = 0.0
        steps = 0
        stages_cleared: list[str] = []
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += float(reward)
            steps += 1

            if truncated:
                done = True
                break

            if terminated:
                if auto:
                    nxt = next_stage(
                        stage,
                        is_satisfied=raw.emu.data.is_goal_satisfied,
                    )
                    if nxt is not None:
                        stages_cleared.append(stage)
                        print(
                            f"  ✓ {stage} cleared (goal={get_goal_for_stage(stage)}) "
                            f"→ advancing to {nxt} "
                            f"(goal={get_goal_for_stage(nxt)})"
                        )
                        stage = nxt
                        # Keep playing from current game state; only change goal/horizon.
                        raw.set_curriculum(
                            goal=get_goal_for_stage(stage),
                            max_steps=get_stage_max_steps(stage),
                            reset_steps=True,
                        )
                        # If a Monitor was wrapped somehow, allow further steps.
                        if hasattr(env, "needs_reset"):
                            env.needs_reset = False
                        continue
                    stages_cleared.append(stage)
                done = True

        print(
            f"Episode {ep + 1}/{args.episodes}: "
            f"return={total:.3f} steps={steps} "
            f"map={info.get('map_id')} badges={info.get('badges')} "
            f"milestone={info.get('milestone')} "
            f"stage={stage} cleared={stages_cleared or '-'} "
            f"loop={info.get('loop_flag')} "
            f"term={info.get('terminated')} trunc={info.get('truncated')}"
        )

    env.close()


def main():
    p = argparse.ArgumentParser(description="Evaluate PPO Pokemon Red agent")
    p.add_argument(
        "--model", "-m", default=None,
        help="PPO .zip checkpoint (default: newest models/ppo_*/best/best_model.zip, "
             "else ppo_latest.zip)",
    )
    p.add_argument("--episodes", "-e", type=int, default=3)
    p.add_argument("--checkpoint", "-c", default="start")
    p.add_argument(
        "--max-steps", type=int, default=None,
        help="Override max steps (default: per-stage curriculum limit)",
    )
    p.add_argument("--frame-skip", type=int, default=24)
    p.add_argument("--stage", default="stage_left_house", help="Starting curriculum stage")
    p.add_argument(
        "--goal", default="left_house",
        help="Fixed goal when --no-auto-curriculum",
    )
    p.add_argument(
        "--auto-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="On goal success, advance stage in-place (default: on)",
    )
    p.add_argument("--gui", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
