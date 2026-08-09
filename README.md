# Pokemon Red AI

Train an agent to play **Pokémon Red** with **PPO** (Stable-Baselines3) on **PyBoy**.

The primary training path is PPO. The older Rainbow DQN stack remains available as `train-dqn` / `eval-dqn` for experiments, but is not the supported default.

## Requirements

- Python 3.10+
- A legally obtained Pokémon Red ROM named `rom.gb` in the repo root
- A starting save state at `saves/start/checkpoint.state`
- GPU recommended (CUDA); CPU works but is slow

## Setup

```bash
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
# Install a CUDA build of torch from https://pytorch.org if you have a GPU
```

Place `rom.gb` and ensure `saves/start/checkpoint.state` exists.

## Train (PPO)

```bash
python cli.py train --workers 8
# or
python train_ppo.py --workers 8 --timesteps 2000000
```

Auto-curriculum is **on by default**: when ~70% of recent episodes succeed at the current goal, training advances through a soft story order (leave house → Oak/Route 1 → parcel/map → each gym fight + badge → legendaries / extras → all badges). Stages whose flags are **already true** are skipped (flags can be earned out of order).

Disable with `--no-auto-curriculum` if you want a fixed goal.

Useful flags:

| Flag | Meaning |
|------|---------|
| `--stage stage_left_house` | Starting stage (auto-advances through STAGE_ORDER) |
| `--no-auto-curriculum` | Keep a fixed stage/goal |
| `--goal badge1` | Override episode success condition |
| `--workers 8` | Parallel `SubprocVecEnv` envs |
| `--gui` | Show SDL window on every worker (slower) |
| `--resume` | Resume newest `ppo_latest.zip` (or pass a path) |
| `--resume models/.../ppo_latest.zip` | Resume a specific checkpoint |

TensorBoard logs land under `logs/ppo_<timestamp>/`:

```bash
tensorboard --logdir logs
```

Watch especially:

- `pokemon/goal_success_rate` — success on the *current* curriculum goal
- `pokemon/curriculum_stage_idx` — index into `STAGE_ORDER` as auto-curriculum advances
- `pokemon/badges_mean` — average badges held at episode end
- `pokemon/loop_episode_rate` (target ≪ 0.10)
- `rollout/ep_rew_mean`

## Evaluate (PPO)

```bash
python cli.py eval --gui
```

Auto-picks the newest `best_model.zip` / `ppo_latest.zip`. With **auto-curriculum** (default), one episode keeps playing after each goal and skips already-satisfied flags/badges (prints `✓ stage_left_house cleared → advancing to stage_oaks_lab`).

```bash
python cli.py eval --gui --no-auto-curriculum --goal left_house   # fixed single goal
python cli.py eval --model models/ppo_<timestamp>/best/best_model.zip --gui
```

## Curriculum

Configured in [`curriculum_config.py`](curriculum_config.py) (~30 stages):

- Early nav: leave house, Oak's Lab, Route 1, Oak's Parcel, Town Map  
- Each gym: `fought_*` flag then corresponding badge bit (1–8)  
- Side / late: SS Anne, Lapras, Snorlax, birds, fossil, Mewtwo, all badges  

`STAGE_ORDER` is only a **recommended** order. Eval/train auto-advance skips any stage whose RAM goal is already satisfied, so out-of-order event flags do not block progress.

Create mid-game saves by copying `checkpoint.state` into `saves/<stage_name>/` (optional; missing saves fall back to `start`).

## Environment details

- **Obs**: `screen_tiles` (1×18×20), `visit_mask` (1×11×11), flat `vector` features from RAM  
- **Actions**: 9 discrete buttons (A/B/Start/Select/D-pad/None) with fixed `--frame-skip` (default 16)  
- **Rewards**: hierarchical micro/meso/macro + PokeRL-style anti-loop / menu-spam penalties  
- **Done**: curriculum goal (`left_house` / `route1` / `oaks_lab` / `badge1` / `all_badges`) or stuck / `max_steps`

## Legacy Rainbow DQN

```bash
python cli.py train-dqn --workers 5
python cli.py eval-dqn --model models/best.pth
```

See `TRAINING_IMPROVEMENTS.md` / `OPTIMIZATION_SUMMARY.md` for historical DQN notes (may be stale vs current code).

## Project layout

```
cli.py / train_ppo.py     # PPO entrypoints
src/env/                  # Gymnasium PokemonRedEnv
src/ppo/                  # Feature extractor + callbacks
src/pokemon/              # PyBoy Emulator + Data (RAM/rewards)
src/workers/              # Legacy DQN actor-learner
curriculum_config.py      # PPO curriculum stages
```

## Tk GUI

`main.py` (Tk) is **not** supported on the PPO path — the `TrainModel` module is missing. Use the CLI.
