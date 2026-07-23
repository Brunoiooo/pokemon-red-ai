#!/usr/bin/env python3
"""Train Pokemon Red with Stable-Baselines3 PPO."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, "src")

from multiprocessing import set_start_method


def run(args):
    set_start_method("spawn", force=True)

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    from curriculum_config import (
        get_curriculum_saves,
        get_goal_for_stage,
        get_stage_max_steps,
    )
    from env.pokemon_red_env import PokemonRedEnv
    from ppo.callbacks import MilestoneCallback
    from ppo.features import PokemonFeaturesExtractor

    if args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    Path("logs").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / f"ppo_{session}"
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path("models") / f"ppo_{session}"
    model_dir.mkdir(parents=True, exist_ok=True)

    stage = args.stage
    goal = args.goal or get_goal_for_stage(stage)
    max_steps = args.max_steps or get_stage_max_steps(stage)
    curriculum_saves = get_curriculum_saves(stage)

    print("=" * 70)
    print("  Pokemon Red AI — PPO Training (Stable-Baselines3)")
    print("=" * 70)
    print(f"Device:     {device}")
    print(f"Workers:    {args.workers}")
    print(f"Stage:      {stage}")
    print(f"Goal:       {goal}")
    print(f"Max steps:  {max_steps}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"Saves:      {curriculum_saves}")
    print(f"Log dir:    {log_dir}")
    print(f"Total steps:{args.timesteps:,}")
    print("=" * 70)

    def _make(rank: int):
        def _thunk():
            env = PokemonRedEnv(
                save_state=curriculum_saves[-1],
                max_steps=max_steps,
                frame_skip=args.frame_skip,
                goal=goal,
                curriculum_mix=args.curriculum_mix,
                curriculum_saves=curriculum_saves,
                render_mode="human" if args.gui and rank == 0 else None,
            )
            env.reset(seed=args.seed + rank)
            return env

        return _thunk

    if args.workers <= 1:
        train_env = DummyVecEnv([_make(0)])
    else:
        train_env = SubprocVecEnv(
            [_make(i) for i in range(args.workers)], start_method="spawn"
        )
    train_env = VecMonitor(train_env, str(log_dir / "monitor"))

    eval_env = DummyVecEnv(
        [
            lambda: PokemonRedEnv(
                save_state=curriculum_saves[-1],
                max_steps=max_steps,
                frame_skip=args.frame_skip,
                goal=goal,
                curriculum_mix=0.0,
                curriculum_saves=curriculum_saves[-1:],
            )
        ]
    )
    eval_env = VecMonitor(eval_env, str(log_dir / "eval_monitor"))

    policy_kwargs = dict(
        features_extractor_class=PokemonFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
    )

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env, device=device)
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=train_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir),
            device=device,
            verbose=1,
            seed=args.seed,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // max(args.workers, 1), 1),
        save_path=str(model_dir),
        name_prefix="ppo_ckpt",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(args.eval_freq // max(args.workers, 1), 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    milestone_cb = MilestoneCallback(window=100)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList([checkpoint_cb, eval_cb, milestone_cb]),
            progress_bar=not args.no_progress,
        )
    except KeyboardInterrupt:
        print("\nInterrupted — saving latest model...")
    finally:
        out = model_dir / "ppo_latest.zip"
        model.save(str(out))
        print(f"Saved: {out}")
        train_env.close()
        eval_env.close()


def build_argparser(parent=None):
    p = argparse.ArgumentParser(
        description="Pokemon Red PPO training",
        parents=[parent] if parent else [],
    )
    p.add_argument("--workers", "-w", type=int, default=8)
    p.add_argument("--timesteps", "-t", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-epochs", type=int, default=4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--frame-skip", type=int, default=24)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--stage", default="stage_0",
                   help="Curriculum stage: stage_0 | stage_1 | stage_2")
    p.add_argument("--goal", default=None,
                   help="Override goal: left_house | route1 | oaks_lab | badge1 | all_badges")
    p.add_argument("--curriculum-mix", type=float, default=0.3,
                   help="Probability of resetting to an earlier curriculum save")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", default=None, help="Path to .zip SB3 checkpoint")
    p.add_argument("--checkpoint-freq", type=int, default=100_000)
    p.add_argument("--eval-freq", type=int, default=50_000)
    p.add_argument("--eval-episodes", type=int, default=3)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main():
    args = build_argparser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
