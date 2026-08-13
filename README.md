# Pokemon Red AI

Train an agent to play **Pokémon Red** with **PPO** (Stable-Baselines3) on **PyBoy**.

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

Auto-curriculum is **on by default** and order-free: whenever ~70% of recent episodes clear *some* goal (in-game event order isn't reliable, so there's no fixed sequence to walk), training gets reassigned a fresh, randomly-picked not-yet-satisfied goal (see `curriculum_config.pick_new_goal`). Every goal an episode actually reaches — in whatever order it happens to reach it — gets its own `saves/<goal>/checkpoint.state` written on the spot (see `PokemonRedEnv._save_milestone_checkpoints`), so that pool of usable starting points just keeps growing.

Disable with `--no-auto-curriculum` if you want a fixed goal.

Useful flags:

| Flag | Meaning |
|------|---------|
| `--stage EVENT_GOT_STARTER` | Starting goal (auto-curriculum reassigns randomly from there, order-free) |
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

- `pokemon/goal_success_rate` — fraction of recent episodes that cleared *some* goal (any goal)
- `pokemon/curriculum_stage_idx` — informational only: STAGE_ORDER's list position of the currently-assigned goal (STAGE_ORDER no longer drives which goal comes next)
- `pokemon/badges_mean` — average badges held at episode end
- `pokemon/loop_episode_rate` (target ≪ 0.10)
- `rollout/ep_rew_mean`

## Evaluate (PPO)

```bash
python cli.py eval --gui
```

Auto-picks the newest `best_model.zip` / `ppo_latest.zip`. With **auto-curriculum** (default), one episode keeps playing after each goal — whichever fires next, in whatever order — and each one saves its own `saves/<goal>/checkpoint.state` when `--save-checkpoints` is on (prints `[ok] <goal> cleared -> advancing to <next-random-goal>`).

```bash
python cli.py eval --gui --no-auto-curriculum --goal left_house   # fixed single goal
python cli.py eval --model models/ppo_<timestamp>/best/best_model.zip --gui
```

## Curriculum

Configured in [`curriculum_config.py`](curriculum_config.py) (~30 stages):

- Early nav: leave house, Oak's Lab, Route 1, Oak's Parcel, Town Map  
- Each gym: `fought_*` flag then corresponding badge bit (1–8)  
- Side / late: SS Anne, Lapras, Snorlax, birds, fossil, Mewtwo, all badges  

`STAGE_ORDER` is **not** a progression order — it only backs the goal one-hot's indexing and cosmetic listings (`--list-stages`). Which goal gets assigned next (in training, eval, and debug-play) is a random, order-free pick among not-yet-satisfied goals (`curriculum_config.pick_new_goal`), restricted to goals that either already have a `saves/<goal>/checkpoint.state` written or whose `event_graph.py` parent events all do (i.e. whatever leads up to it has already happened once). In training, the pick is further restricted to goals on a map some episode has actually visited this run (`MilestoneCallback._visited_maps`, persisted to `saves/visited_maps.json` and reused for the rest of the run).

`saves/<goal_name>/checkpoint.state` gets written automatically the moment any episode reaches that goal (see `--save-checkpoints`, on by default); missing saves fall back to `start`.

## Environment details

- **Obs**: `screen_tiles` (1×18×20), `visit_mask` (1×11×11), flat `vector` features from RAM  
- **Actions**: 9 discrete buttons (A/B/Start/Select/D-pad/None) with fixed `--frame-skip` (default 16)  
- **Rewards**: hierarchical micro/meso/macro + PokeRL-style anti-loop / menu-spam penalties  
- **Done**: curriculum goal (`left_house` / `route1` / `gave_parcel` / `badge1` / `all_badges`) or stuck / `max_steps`

## Project layout

```
cli.py / train_ppo.py     # PPO entrypoints
src/env/                  # Gymnasium PokemonRedEnv
src/ppo/                  # Feature extractor + callbacks
src/pokemon/              # PyBoy Emulator + Data (RAM/rewards)
curriculum_config.py      # PPO curriculum stages
```
