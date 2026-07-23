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
python cli.py train --workers 8 --stage stage_0
# or
python train_ppo.py --workers 8 --stage stage_0 --timesteps 2000000
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--stage stage_0` | Leave the house (`goal=left_house`) |
| `--stage stage_1` | Reach Route 1 |
| `--stage stage_2` | Push toward Badge 1 |
| `--goal badge1` | Override episode success condition |
| `--workers 8` | Parallel `SubprocVecEnv` envs |
| `--gui` | Show window on worker 0 |
| `--resume models/.../ppo_latest.zip` | Continue training |

TensorBoard logs land under `logs/ppo_<timestamp>/`:

```bash
tensorboard --logdir logs
```

Watch especially:

- `pokemon/loop_episode_rate` (target ≪ 0.10)
- `pokemon/left_house_rate`, `pokemon/route1_rate`, `pokemon/badge1_rate`
- `rollout/ep_rew_mean`

## Evaluate (PPO)

```bash
python cli.py eval --model models/ppo_<timestamp>/best/best_model.zip --episodes 5 --gui
```

## Curriculum

Configured in [`curriculum_config.py`](curriculum_config.py):

1. **stage_0** — start save, succeed by leaving Red's house  
2. **stage_1** — optional `saves/stage_1`, succeed on Route 1  
3. **stage_2** — optional `saves/stage_2`, succeed on Badge 1  

Create mid-game saves by playing (`python cli.py record` or manual PyBoy) and copying `checkpoint.state` into `saves/stage_N/`. Missing stage saves fall back to `start`.

## Environment details

- **Obs**: `screen_tiles` (1×18×20), `visit_mask` (1×11×11), flat `vector` features from RAM  
- **Actions**: 9 discrete buttons (A/B/Start/Select/D-pad/None) with fixed `--frame-skip` (default 24)  
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
