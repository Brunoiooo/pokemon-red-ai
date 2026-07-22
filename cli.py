#!/usr/bin/env python3
"""Pokemon Red AI — unified CLI.

Usage:
  python cli.py [--cpu] [--verbose] [--gui] <command> [command-args]
  python cli.py <command> [--cpu] [--verbose] [--gui] [command-args]

Commands:
  train       RL training loop
  eval        Evaluate a trained model (dry run, detailed stats)
  pretrain    Behavioral-cloning pre-training from a human demo
  record      Record human gameplay for BC pre-training
  plot        Plot training metrics from CSV logs
  analyze     Analyze a training metrics JSON
  benchmark   Run inference / emulation / pipeline benchmarks
"""
import sys
sys.path.insert(0, "src")

import argparse

# ---------------------------------------------------------------------------
# Global / shared flags — inherited by every subcommand
# ---------------------------------------------------------------------------
_global = argparse.ArgumentParser(add_help=False)
_g = _global.add_argument_group("global flags")
_g.add_argument("--cpu", action="store_true",
                help="Force CPU even if CUDA is available")
_g.add_argument("--verbose", "-v", action="store_true",
                help="Verbose output")
_g.add_argument("--gui", action="store_true",
                help="Show the game window (where applicable)")

# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    prog="pokemon-ai",
    description="Pokemon Red AI — unified CLI",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "Global flags (--cpu, --verbose, --gui) can be placed either before\n"
        "or after the subcommand name.\n\n"
        "Examples:\n"
        "  python cli.py train --workers 8 --verbose\n"
        "  python cli.py --cpu eval --model models/best.pth --episodes 5\n"
        "  python cli.py pretrain --demo demos/my_demo.json --epochs 200\n"
        "  python cli.py record --output demos/run1.json --max-steps 1000\n"
        "  python cli.py plot\n"
        "  python cli.py analyze --latest\n"
        "  python cli.py benchmark\n"
    ),
    parents=[_global],
)
sub = parser.add_subparsers(dest="command", metavar="COMMAND")
sub.required = True

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
p_train = sub.add_parser(
    "train",
    parents=[_global],
    help="Run the RL training loop",
    description="Start the full reinforcement-learning training loop with live monitoring.",
)
p_train.add_argument("--workers", "-w", type=int, default=5,
                     help="Number of experience workers (default: 5)")
p_train.add_argument("--eval-gui", "-eg", action="store_true",
                     help="Show GUI during evaluations only")
p_train.add_argument("--reset-buffer", "-rb", action="store_true",
                     help="Reset the replay buffer before starting")

# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
p_eval = sub.add_parser(
    "eval",
    parents=[_global],
    help="Evaluate a trained model (dry run, no training)",
    description="Run the greedy policy of a saved checkpoint and print detailed statistics.",
)
p_eval.add_argument("--model", "-m", default="models/best.pth",
                    help="Path to .pth checkpoint (default: models/best.pth)")
p_eval.add_argument("--episodes", "-e", type=int, default=1,
                    help="Number of episodes to evaluate (default: 1)")
p_eval.add_argument("--checkpoint", "-c", default="start",
                    help="Save-state directory under saves/ (default: start)")
p_eval.add_argument("--max-steps", "-s", type=int, default=5000,
                    help="Maximum steps per episode (default: 5000)")
p_eval.add_argument("--human-speed", action="store_true",
                    help="Run emulator at 1× real-time speed instead of max")

# ---------------------------------------------------------------------------
# pretrain
# ---------------------------------------------------------------------------
p_pretrain = sub.add_parser(
    "pretrain",
    parents=[_global],
    help="Behavioural-cloning pre-training from a recorded demo",
    description="Replay a human demo through the emulator and train the model via cross-entropy.",
)
p_pretrain.add_argument("--demo", "-d", default="demos/demo.json",
                        help="Demo JSON from `record` (default: demos/demo.json)")
p_pretrain.add_argument("--model-in", default=None, metavar="MODEL",
                        help="Checkpoint to fine-tune (default: fresh model)")
p_pretrain.add_argument("--model-out", default="models/bc_pretrained.pth", metavar="MODEL",
                        help="Where to save the result (default: models/bc_pretrained.pth)")
p_pretrain.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
p_pretrain.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
p_pretrain.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")

# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------
p_record = sub.add_parser(
    "record",
    parents=[_global],
    help="Record human gameplay for BC pre-training",
    description="Open the game window and record key presses to a demo JSON file.",
)
p_record.add_argument("--checkpoint", "-c", default="start",
                      help="Save-state directory under saves/ (default: start)")
p_record.add_argument("--output", "-o", default="demos/demo.json",
                      help="Output JSON file (default: demos/demo.json)")
p_record.add_argument("--max-steps", "-n", type=int, default=500,
                      help="Maximum steps to record (default: 500)")
p_record.add_argument("--patience", "-p", type=int, default=8,
                      help="Multiplier on the stuck/idle truncation budget, tuned up "
                           "from the training default for human reading/reaction speed (default: 8)")

# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
p_plot = sub.add_parser(
    "plot",
    parents=[_global],
    help="Plot training metrics from CSV logs",
    description="Generate a PNG chart from a train_steps CSV (auto-detects latest session).",
)
p_plot.add_argument("train_csv", nargs="?", default=None, metavar="TRAIN_CSV",
                    help="Path to a train_steps_*.csv file (default: latest in logs/)")

# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------
p_analyze = sub.add_parser(
    "analyze",
    parents=[_global],
    help="Analyze a training metrics JSON",
    description="Print a structured summary of a metrics_*.json file saved by the trainer.",
)
_mex = p_analyze.add_mutually_exclusive_group(required=True)
_mex.add_argument("--file", "-f", metavar="FILE",
                  help="Metrics JSON file to analyze")
_mex.add_argument("--latest", "-l", action="store_true",
                  help="Analyze the latest metrics file in logs/")

# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------
sub.add_parser(
    "benchmark",
    parents=[_global],
    help="Run inference / emulation / full-pipeline benchmarks",
)

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def main():
    args = parser.parse_args()

    if args.command == "train":
        import train as _m
        _m.run(args)

    elif args.command == "eval":
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
