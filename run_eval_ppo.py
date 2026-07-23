#!/usr/bin/env python3
"""Evaluate a Stable-Baselines3 PPO checkpoint on Pokemon Red."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")


def run(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from env.pokemon_red_env import PokemonRedEnv

    env = PokemonRedEnv(
        save_state=args.checkpoint,
        max_steps=args.max_steps,
        frame_skip=args.frame_skip,
        goal=args.goal,
        render_mode="human" if args.gui else None,
        curriculum_mix=0.0,
    )
    env = Monitor(env)

    print(f"Loading {args.model}")
    model = PPO.load(args.model, device="cpu" if args.cpu else "auto")

    for ep in range(args.episodes):
        obs, info = env.reset()
        total = 0.0
        steps = 0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += float(reward)
            steps += 1
            done = terminated or truncated
        print(
            f"Episode {ep + 1}/{args.episodes}: "
            f"return={total:.3f} steps={steps} "
            f"map={info.get('map_id')} badges={info.get('badges')} "
            f"milestone={info.get('milestone')} loop={info.get('loop_flag')} "
            f"term={info.get('terminated')} trunc={info.get('truncated')}"
        )

    env.close()


def main():
    p = argparse.ArgumentParser(description="Evaluate PPO Pokemon Red agent")
    p.add_argument("--model", "-m", default="models/best/best_model.zip")
    p.add_argument("--episodes", "-e", type=int, default=3)
    p.add_argument("--checkpoint", "-c", default="start")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--frame-skip", type=int, default=24)
    p.add_argument("--goal", default="badge1")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
