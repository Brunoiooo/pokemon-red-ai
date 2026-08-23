#!/usr/bin/env python3
"""Pokemon Red AI — unified CLI.

Usage:
  python cli.py [--cpu] [--verbose] [--gui] <command> [command-args]

Commands:
  train       PPO training (Stable-Baselines3)
  eval        Evaluate a PPO .zip checkpoint
  save-stage  Play manually and save a mid-game state for a curriculum stage
  debug-play  Human play with live reward / dialog / battle debug prints
"""
import sys
sys.path.insert(0, "src")

# Windows consoles default stdout to the system codepage (cp1250/cp1252,
# ...), which can't encode compass arrow glyphs (debug-play/eval -vv) or
# other non-ASCII output -- crashes mid-run otherwise, including when a GUI
# (gui_app.py) pipes this process's stdout, which is never a real console.
# No-op where stdout is already utf-8 or doesn't support reconfigure.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse

import curriculum_config
import tunables
from pokemon.Data import Data
from pokemon.Emulator import Emulator

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
    description="Pokemon Red AI — PPO training",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=(
        "Examples:\n"
        "  python cli.py train --workers 8\n"
        "  python cli.py eval --gui\n"
        "  python cli.py eval --model models/ppo_<timestamp>/best/best_model.zip --gui\n"
        "  python cli.py eval --batch 300000 --batch-workers 8\n"
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
p_train.add_argument(
    "--auto-entropy", action="store_true",
    help="Auto-adjust ent_coef during training based on the rolling "
         "action-loop rate (bumps toward --ent-coef-max on collapse, "
         "decays back toward --ent-coef-min once it clears)",
)
p_train.add_argument("--ent-coef-min", type=float, default=0.01,
                     help="Floor for --auto-entropy (default: 0.01)")
p_train.add_argument("--ent-coef-max", type=float, default=0.3,
                     help="Ceiling for --auto-entropy (default: 0.3)")
p_train.add_argument("--frame-skip", type=int, default=16)
p_train.add_argument("--max-steps", type=int, default=None)
# Must match curriculum_config.DEFAULT_STAGE -- "stage_left_house" used to
# be here, a dead pre-generic-goal-system id that silently resolved to an
# arbitrary, unreachable, checkpoint-less goal (see curriculum_config.py's
# resolve_stage_name docstring).
p_train.add_argument("--stage", default="EVENT_GOT_STARTER",
                     help="Starting curriculum stage (see curriculum_config.STAGE_ORDER)")
p_train.add_argument("--list-stages", action="store_true",
                     help="List all curriculum stages (id, goal, max_steps, save) and exit")
p_train.add_argument("--goal", default=None)
p_train.add_argument("--auto-curriculum", action=argparse.BooleanOptionalAction, default=True,
                     help="Auto-advance stages when success rate is high (default: on)")
p_train.add_argument(
    "--save-checkpoints", action=argparse.BooleanOptionalAction, default=True,
    help="On every GOAL_CANDIDATES event/badge newly satisfied, overwrite "
         "saves/<name>/checkpoint.state with the reached state -- whichever "
         "goal an episode clears first, regardless of curriculum order "
         "(default: on; pass --no-save-checkpoints to disable)",
)
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
p_train.add_argument("--eval-freq", type=int, default=500_000)
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


def _int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


_train_model = p_train.add_argument_group("PPO / model hyperparameters")
_train_model.add_argument("--gae-lambda", type=float, default=0.95)
_train_model.add_argument("--clip-range", type=float, default=0.2)
_train_model.add_argument("--vf-coef", type=float, default=0.5)
_train_model.add_argument("--max-grad-norm", type=float, default=0.5)
_train_model.add_argument("--features-dim", type=int, default=256)
_train_model.add_argument(
    "--net-arch-pi", type=_int_list, default=[256, 256], metavar="N,N,...",
    help="Policy-head hidden layer sizes, comma-separated (default: 256,256)",
)
_train_model.add_argument(
    "--net-arch-vf", type=_int_list, default=[256, 256], metavar="N,N,...",
    help="Value-head hidden layer sizes, comma-separated (default: 256,256)",
)
_train_model.add_argument(
    "--screen-cnn-channels", type=_int_list, default=[32, 64, 64], metavar="N,N,...",
    help="screen_tiles CNN channel widths, comma-separated (default: 32,64,64)",
)
_train_model.add_argument(
    "--visit-cnn-channels", type=_int_list, default=[16, 32], metavar="N,N,...",
    help="visit_mask CNN channel widths, comma-separated (default: 16,32)",
)
_train_model.add_argument(
    "--vector-mlp-hidden", type=_int_list, default=[256, 128], metavar="N,N,...",
    help="Flat-vector MLP hidden layer sizes, comma-separated (default: 256,128)",
)
_train_model.add_argument(
    "--milestone-window", type=int, default=100,
    help="MilestoneCallback's rolling episode window (default: 100)",
)

_train_entropy = p_train.add_argument_group("--auto-entropy scheduler")
_train_entropy.add_argument("--entropy-window", type=int, default=50)
_train_entropy.add_argument("--entropy-check-every", type=int, default=20_000)
_train_entropy.add_argument("--loop-rate-high", type=float, default=0.3)
_train_entropy.add_argument("--loop-rate-low", type=float, default=0.08)
_train_entropy.add_argument("--entropy-increase-factor", type=float, default=1.5)
_train_entropy.add_argument("--entropy-decay-factor", type=float, default=0.9)
_train_entropy.add_argument("--entropy-cooldown-checks", type=int, default=5)

tunables.add_dataclass_args(p_train, Data, group_title="reward shaping (Data)")
tunables.add_dataclass_args(p_train, Emulator, group_title="emulator timing")
tunables.add_dataclass_args(
    p_train, curriculum_config.CurriculumConfig, group_title="curriculum advance"
)

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
# Same dead-id fallback problem as p_train's --stage above.
p_eval.add_argument("--stage", default="EVENT_GOT_STARTER",
                    help="Starting curriculum stage (see curriculum_config.STAGE_ORDER)")
p_eval.add_argument(
    "--goal", default="EVENT_GOT_STARTER",
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
    action=argparse.BooleanOptionalAction,
    default=True,
    help="On goal success, overwrite saves/<new_stage>/checkpoint.state with "
         "the reached state (default: on; pass --no-save-checkpoints to disable)",
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
tunables.add_dataclass_args(p_eval, Data, group_title="reward shaping")
tunables.add_dataclass_args(p_eval, Emulator, group_title="emulator timing")

# ---------------------------------------------------------------------------
# save-stage / debug-play
# ---------------------------------------------------------------------------
p_save_stage = sub.add_parser(
    "save-stage",
    parents=[_global],
    help="Play manually and save a mid-game state for a curriculum stage",
)
p_save_stage.add_argument(
    "--stage",
    "-s",
    default=None,
    help="Stage / folder under saves/ (e.g. stage_route1)",
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
tunables.add_dataclass_args(p_save_stage, Data, group_title="reward shaping")
tunables.add_dataclass_args(p_save_stage, Emulator, group_title="emulator timing")

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
p_debug.add_argument(
    "--model",
    nargs="?",
    const="",
    default=None,
    help="Load a PPO checkpoint and print its live button-probability "
         "table after every step, next to what you actually pressed. "
         "Bare --model auto-picks the newest best checkpoint.",
)
tunables.add_dataclass_args(p_debug, Data, group_title="reward shaping")
tunables.add_dataclass_args(p_debug, Emulator, group_title="emulator timing")

def main():
    args = parser.parse_args()

    if args.command == "train":
        import train_ppo as _m
        if args.list_stages:
            _m.print_stage_list()
            return
        _m.run(args)

    elif args.command == "eval":
        import run_eval_ppo as _m
        if args.batch:
            _m.run_batch(args)
        else:
            _m.run(args)

    elif args.command == "save-stage":
        import create_stage_save as _m
        _m.run(args)

    elif args.command == "debug-play":
        import debug_play as _m
        _m.run(args)


if __name__ == "__main__":
    main()
