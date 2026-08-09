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
  save-stage  Play manually and save a mid-game state for a curriculum stage
  debug-play  Human play with live reward / dialog / battle debug prints
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
_g.add_argument("--verbose", "-v", action="count", default=0,
                help="Verbose output; repeat (-vv) for eval to print every "
                     "single button press instead of a sampled subset")
_g.add_argument("--gui", action="store_true",
                help="Show the game window on every worker (slower)")

parser = argparse.ArgumentParser(
    prog="pokemon-ai",
    description="Pokemon Red AI — PPO primary, legacy DQN available",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "Examples:\n"
        "  python cli.py train --workers 8\n"
        "  python cli.py eval --gui\n"
        "  python cli.py eval --model models/ppo_<timestamp>/best/best_model.zip --gui\n"
        "  python cli.py eval --batch 300000 --batch-workers 8\n"
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
p_train.add_argument(
    "--ent-coef", type=float, default=0.05,
    help="PPO entropy bonus (default 0.05; higher fights action-sequence collapse)",
)
p_train.add_argument("--frame-skip", type=int, default=16)
p_train.add_argument("--max-steps", type=int, default=None)
p_train.add_argument("--stage", default="stage_left_house",
                     help="Starting curriculum stage (see curriculum_config.STAGE_ORDER)")
p_train.add_argument("--list-stages", action="store_true",
                     help="List all curriculum stages (id, goal, max_steps, save) and exit")
p_train.add_argument("--goal", default=None)
p_train.add_argument("--auto-curriculum", action=argparse.BooleanOptionalAction, default=True,
                     help="Auto-advance stages when success rate is high (default: on)")
p_train.add_argument("--curriculum-mix", type=float, default=0.3)
p_train.add_argument("--seed", type=int, default=0)
p_train.add_argument(
    "--resume",
    nargs="?",
    const="AUTO",
    default=None,
    help="Resume from .zip; omit path to auto-pick newest ppo_latest/best",
)
p_train.add_argument(
    "--migrate",
    nargs="?",
    const="AUTO",
    default=None,
    help="Like --resume, but for a checkpoint whose 'vector' observation "
         "width no longer matches (e.g. after a new curriculum goal grew "
         "GOAL_ORDER) — transplants shape-matching weights, remaps the "
         "goal one-hot by name. See ppo/migrate.py.",
)
p_train.add_argument("--checkpoint-freq", type=int, default=100_000)
p_train.add_argument("--eval-freq", type=int, default=150_000)
p_train.add_argument("--eval-episodes", type=int, default=3)
p_train.add_argument("--no-progress", action="store_true")
p_train.add_argument(
    "--heatmap", action="store_true",
    help="Open a live position-heatmap window (avg ticks/run per map tile, "
         "rolling window of --heatmap-frames). <-/-> switches map.",
)
p_train.add_argument(
    "--heatmap-frames", type=int, default=300_000,
    help="Rolling window size in frames, pooled across all runs (default: 300000)",
)

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
p_eval.add_argument(
    "--model", "-m", default=None,
    help="PPO .zip checkpoint (default: newest models/ppo_*/best/best_model.zip, "
         "else ppo_latest.zip)",
)
p_eval.add_argument("--episodes", "-e", type=int, default=3)
p_eval.add_argument("--checkpoint", "-c", default="start")
p_eval.add_argument("--max-steps", "-s", type=int, default=None,
                    help="Override max steps (default: per-stage limit)")
p_eval.add_argument("--frame-skip", type=int, default=16)
p_eval.add_argument("--stage", default="stage_left_house",
                    help="Starting curriculum stage (see curriculum_config.STAGE_ORDER)")
p_eval.add_argument(
    "--goal", default="left_house",
    help="Fixed goal when --no-auto-curriculum",
)
p_eval.add_argument(
    "--auto-curriculum",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="On goal success, advance to next stage without reset (default: on)",
)
p_eval.add_argument("--stochastic", action="store_true")
p_eval.add_argument(
    "--save-checkpoints",
    action="store_true",
    default=False,
    help="On goal success, overwrite saves/<new_stage>/checkpoint.state with "
         "the reached state (opt-in; off by default)",
)
p_eval.add_argument(
    "--heatmap", action="store_true",
    help="Open a live position-heatmap window (avg ticks/run per map tile, "
         "rolling window of --heatmap-frames). <-/-> switches map.",
)
p_eval.add_argument(
    "--heatmap-frames", type=int, default=300_000,
    help="Rolling window size in frames, pooled across all runs (default: 300000)",
)
p_eval.add_argument(
    "--batch", type=int, default=None, metavar="N",
    help="Batch mode: run N total episodes/curriculum-legs across "
         "--batch-workers parallel workers and save static heatmap PNGs to "
         "--batch-out at the end. Pass --heatmap too to also open the same "
         "live window as plain eval, fed live from the batch run — it just "
         "won't auto-close when the batch finishes. E.g. --batch 300000.",
)
p_eval.add_argument(
    "--batch-workers", type=int, default=8,
    help="Parallel PyBoy worker processes for --batch (default: 8)",
)
p_eval.add_argument(
    "--batch-out", default=None,
    help="Output directory for --batch heatmap PNGs "
         "(default: heatmaps/eval_<timestamp>)",
)
p_eval.add_argument(
    "--batch-metrics", nargs="+", choices=["ticks", "reward", "winrate"],
    default=["ticks"],
    help="Which heatmap metric(s) to render in --batch mode (default: ticks)",
)
p_eval.add_argument(
    "--batch-top-maps", type=int, default=6,
    help="Per stage, also render single-map views for the N most-visited "
         "maps, in addition to the combined view (default: 6)",
)
p_eval.add_argument(
    "--seed", type=int, default=0,
    help="Base worker seed for --batch mode (each worker gets seed + rank)",
)

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

p_save_stage = sub.add_parser(
    "save-stage",
    parents=[_global],
    help="Play manually and save a mid-game state for a curriculum stage",
)
p_save_stage.add_argument(
    "--stage",
    "-s",
    default=None,
    help="Stage / folder under saves/ (e.g. stage_oaks_lab)",
)
p_save_stage.add_argument(
    "--from",
    dest="from_checkpoint",
    default="start",
    help="Save dir to load first (default: start)",
)
p_save_stage.add_argument(
    "--list",
    "-l",
    action="store_true",
    help="List curriculum stages and which already have saves",
)

p_debug = sub.add_parser(
    "debug-play",
    parents=[_global],
    help="Human play with live reward / dialog / battle debug prints",
)
p_debug.add_argument(
    "--from",
    dest="from_checkpoint",
    default="start",
    help="Save dir to load first (default: start)",
)
p_debug.add_argument(
    "--save-as",
    default="manual_debug",
    help="Folder under saves/ for S key (default: manual_debug)",
)
p_debug.add_argument(
    "--goal",
    default=None,
    help="Active curriculum goal (default: left_house, same as train)",
)
p_debug.add_argument(
    "--stage",
    default=None,
    help="Optional stage — sets --goal from curriculum if --goal omitted",
)
p_debug.add_argument(
    "--frame-skip",
    type=int,
    default=16,
    help="Hold duration per action, same as PPO (default: 16)",
)
p_debug.add_argument(
    "--real-truncation",
    action="store_true",
    help="Keep agent stuck/loop fuses (default: disabled for human play)",
)
p_debug.add_argument(
    "--auto-curriculum",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="On goal success, advance stage like train (default: on)",
)
p_debug.add_argument(
    "--all-steps",
    action="store_true",
    help="Print every agent step (incl. unrecognized-key None)",
)

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
        if args.list_stages:
            _m.print_stage_list()
            return
        _m.run(args)

    elif args.command == "train-dqn":
        import train as _m
        _m.run(args)

    elif args.command == "eval":
        import run_eval_ppo as _m
        if args.batch:
            _m.run_batch(args)
        else:
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

    elif args.command == "save-stage":
        import create_stage_save as _m
        _m.run(args)

    elif args.command == "debug-play":
        import debug_play as _m
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
