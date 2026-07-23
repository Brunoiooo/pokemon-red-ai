#!/usr/bin/env python3
"""Evaluate a Stable-Baselines3 PPO checkpoint on Pokemon Red."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, "src")

_MODELS_ROOT = Path("models")
_SEARCH_BEST = "models/ppo_*/best/best_model.zip"
_SEARCH_LATEST = "models/ppo_*/ppo_latest.zip"


def resolve_model_path(model: str | None, models_root: Path = _MODELS_ROOT) -> Path:
    """Resolve a PPO checkpoint path.

    If ``model`` is given, use it. Otherwise pick the newest
    ``models/ppo_*/best/best_model.zip``, falling back to
    ``models/ppo_*/ppo_latest.zip``.
    """
    if model:
        path = Path(model)
        if not path.is_file():
            raise SystemExit(
                f"Model not found: {path}\n"
                f"Pass an existing checkpoint, e.g.\n"
                f"  python cli.py eval --model models/ppo_<timestamp>/best/best_model.zip --gui"
            )
        return path

    best = sorted(
        models_root.glob("ppo_*/best/best_model.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if best:
        return best[0]

    latest = sorted(
        models_root.glob("ppo_*/ppo_latest.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if latest:
        return latest[0]

    raise SystemExit(
        "No PPO checkpoint found.\n"
        f"Searched:\n"
        f"  {_SEARCH_BEST}\n"
        f"  {_SEARCH_LATEST}\n"
        "Train first, or pass an explicit path:\n"
        "  python cli.py eval --model models/ppo_<timestamp>/best/best_model.zip --gui\n"
        "  python cli.py eval --model models/ppo_<timestamp>/ppo_latest.zip --gui"
    )


def run(args):
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    from env.pokemon_red_env import PokemonRedEnv

    model_path = resolve_model_path(getattr(args, "model", None))

    env = PokemonRedEnv(
        save_state=args.checkpoint,
        max_steps=args.max_steps,
        frame_skip=args.frame_skip,
        goal=args.goal,
        render_mode="human" if args.gui else None,
        curriculum_mix=0.0,
    )
    env = Monitor(env)

    print(f"Loading {model_path}")
    print(f"Goal: {args.goal}")
    model = PPO.load(str(model_path), device="cpu" if args.cpu else "auto")

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
    p.add_argument(
        "--model", "-m", default=None,
        help="PPO .zip checkpoint (default: newest models/ppo_*/best/best_model.zip, "
             "else ppo_latest.zip)",
    )
    p.add_argument("--episodes", "-e", type=int, default=3)
    p.add_argument("--checkpoint", "-c", default="start")
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--frame-skip", type=int, default=24)
    p.add_argument(
        "--goal", default="left_house",
        help="Episode success goal (default: left_house; stage_0 early milestone)",
    )
    p.add_argument("--gui", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
