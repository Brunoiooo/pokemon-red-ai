#!/usr/bin/env python3
"""Pokemon Red AI — unified CLI.

Usage:
  python cli.py [--cpu] [--verbose] [--gui] <command> [command-args]

Commands:
  train       PPO training (Stable-Baselines3) — primary path
  train-dqn   Legacy Rainbow DQN training
  eval        Evaluate a PPO .zip checkpoint
  eval-dqn    Evaluate a legacy DQN .pth checkpoint
  pretrain    Behavioral-cloning pre-training from a human demo (legacy DQN)
  record      Record human gameplay for BC pre-training
  plot        Plot training metrics from CSV logs
  analyze     Analyze a training metrics JSON
  benchmark   Run inference / emulation / pipeline benchmarks
"""
import sys
sys.path.insert(0, "src")

import argparse

_global = argparse.ArgumentParser(add_help=False)
_g = _global.add_argument_group("global flags")
_g.add_argument("--cpu", action="store_true",
                help="Force CPU even if CUDA is available")
_g.add_argument("--verbose", "-v", action="store_true",
                help="Verbose output")
_g.add_argument("--gui", action="store_true",
                help="Show the game window (where applicable)")

parser = argparse.ArgumentParser(
    prog="pokemon-ai",
    description="Pokemon Red AI — PPO primary, legacy DQN available",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "Examples:\n"
        "  python cli.py train --workers 8 --stage stage_0\n"
        "  python cli.py eval --model models/ppo_*/best/best_model.zip\n"
        "  python cli.py train-dqn --workers 5\n"
        "  python cli.py eval-dqn --model models/best.pth\n"
    ),
    parents=[_global],
)
sub = parser.add_subparsers(dest="command", metavar="COMMAND")
sub.required = True

# ---------------------------------------------------------------------------
# train (PPO)
# ---------------------------------------------------------------------------
p_train = sub.add_parser(
    "train",
    parents=[_global],
    help="Run PPO training (Stable-Baselines3)",
)
p_train.add_argument("--workers", "-w", type=int, default=8)
p_train.add_argument("--timesteps", "-t", type=int, default=2_000_000)
p_train.add_argument("--lr", type=float, default=2.5e-4)
p_train.add_argument("--n-steps", type=int, default=2048)
p_train.add_argument("--batch-size", type=int, default=512)
p_train.add_argument("--n-epochs", type=int, default=4)
p_train.add_argument("--gamma", type=float, default=0.99)
p_train.add_argument("--ent-coef", type=float, default=0.01)
p_train.add_argument("--frame-skip", type=int, default=24)
p_train.add_argument("--max-steps", type=int, default=None)
p_train.add_argument("--stage", default="stage_0")
p_train.add_argument("--goal", default=None)
p_train.add_argument("--curriculum-mix", type=float, default=0.3)
p_train.add_argument("--seed", type=int, default=0)
p_train.add_argument("--resume", default=None)
p_train.add_argument("--checkpoint-freq", type=int, default=100_000)
p_train.add_argument("--eval-freq", type=int, default=50_000)
p_train.add_argument("--eval-episodes", type=int, default=3)
p_train.add_argument("--no-progress", action="store_true")

# ---------------------------------------------------------------------------
# train-dqn (legacy)
# ---------------------------------------------------------------------------
p_train_dqn = sub.add_parser(
    "train-dqn",
    parents=[_global],
    help="Legacy Rainbow DQN training",
)
p_train_dqn.add_argument("--workers", "-w", type=int, default=5)
p_train_dqn.add_argument("--eval-gui", "-eg", action="store_true")
p_train_dqn.add_argument("--reset-buffer", "-rb", action="store_true")

# ---------------------------------------------------------------------------
# eval (PPO)
# ---------------------------------------------------------------------------
p_eval = sub.add_parser(
    "eval",
    parents=[_global],
    help="Evaluate a PPO .zip checkpoint",
)
p_eval.add_argument("--model", "-m", default="models/best/best_model.zip")
p_eval.add_argument("--episodes", "-e", type=int, default=3)
p_eval.add_argument("--checkpoint", "-c", default="start")
p_eval.add_argument("--max-steps", "-s", type=int, default=5000)
p_eval.add_argument("--frame-skip", type=int, default=24)
p_eval.add_argument("--goal", default="badge1")
p_eval.add_argument("--stochastic", action="store_true")

# ---------------------------------------------------------------------------
# eval-dqn (legacy)
# ---------------------------------------------------------------------------
p_eval_dqn = sub.add_parser(
    "eval-dqn",
    parents=[_global],
    help="Evaluate a legacy DQN .pth checkpoint",
)
p_eval_dqn.add_argument("--model", "-m", default="models/best.pth")
p_eval_dqn.add_argument("--episodes", "-e", type=int, default=1)
p_eval_dqn.add_argument("--checkpoint", "-c", default="start")
p_eval_dqn.add_argument("--max-steps", "-s", type=int, default=5000)
p_eval_dqn.add_argument("--human-speed", action="store_true")

# ---------------------------------------------------------------------------
# pretrain / record / plot / analyze / benchmark
# ---------------------------------------------------------------------------
p_pretrain = sub.add_parser("pretrain", parents=[_global], help="BC pre-training (legacy)")
p_pretrain.add_argument("--demo", "-d", default="demos/demo.json")
p_pretrain.add_argument("--model-in", default=None, metavar="MODEL")
p_pretrain.add_argument("--model-out", default="models/bc_pretrained.pth", metavar="MODEL")
p_pretrain.add_argument("--epochs", type=int, default=100)
p_pretrain.add_argument("--lr", type=float, default=1e-4)
p_pretrain.add_argument("--batch-size", type=int, default=32)

p_record = sub.add_parser("record", parents=[_global], help="Record human demo")
p_record.add_argument("--checkpoint", "-c", default="start")
p_record.add_argument("--output", "-o", default="demos/demo.json")
p_record.add_argument("--max-steps", "-n", type=int, default=500)
p_record.add_argument("--patience", "-p", type=int, default=8)

p_plot = sub.add_parser("plot", parents=[_global], help="Plot training CSV metrics")
p_plot.add_argument("train_csv", nargs="?", default=None, metavar="TRAIN_CSV")

p_analyze = sub.add_parser("analyze", parents=[_global], help="Analyze metrics JSON")
_mex = p_analyze.add_mutually_exclusive_group(required=True)
_mex.add_argument("--file", "-f", metavar="FILE")
_mex.add_argument("--latest", "-l", action="store_true")

sub.add_parser("benchmark", parents=[_global], help="Run benchmarks")


def main():
    args = parser.parse_args()

    if args.command == "train":
        import train_ppo as _m
        _m.run(args)

    elif args.command == "train-dqn":
        import train as _m
        _m.run(args)

    elif args.command == "eval":
        import run_eval_ppo as _m
        _m.run(args)

    elif args.command == "eval-dqn":
        import run_eval as _m
        _m.run(args)

    elif args.command == "pretrain":
        import pretrain_bc as _m
        _m.run(args)

    elif args.command == "record":
        import record_demo as _m
        _m.run(args)

    elif args.command == "plot":
        import plot_metrics as _m
        _m.run(args)

    elif args.command == "analyze":
        import analyze_metrics as _m
        _m.run(args)

    elif args.command == "benchmark":
        import benchmark as _m
        _m.main()


if __name__ == "__main__":
    main()
