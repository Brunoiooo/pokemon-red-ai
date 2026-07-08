# Pokemon Red AI

A reinforcement learning project for training an agent to play Pokémon Red using PyBoy.

## Features

- Unified CLI for training, evaluation, recording demos, pretraining, plotting, and benchmarking
- Headless optimized training workflow
- Experience replay with prioritized sampling
- Training and episode metrics logging

## Requirements

- Python 3.10+
- A Pokémon Red ROM at `./rom.gb` (project root)

Install dependencies:

```bash
cd <project-root>
python -m pip install -r requirements.txt
```

## Quick Start

From the project root:

```bash
python cli.py train
```

Common commands:

```bash
python cli.py train --workers 5
python cli.py eval --model models/best.pth --episodes 3
python cli.py benchmark
python cli.py plot
python cli.py analyze --latest
```

## Alternative Entry Points

```bash
python train_headless.py
python run_eval.py
python benchmark.py
```

## Project Structure

- `src/pokemon` - model, emulator, replay buffers
- `src/workers` - training and experience workers
- `logs` - generated training logs and CSV metrics
- `saves` - emulator checkpoints

## Notes

- Additional training and metrics details are documented in:
  - [`HEADLESS_TRAINING.md`](./HEADLESS_TRAINING.md)
  - [`METRICS_SYSTEM.md`](./METRICS_SYSTEM.md)
  - [`TRAINING_IMPROVEMENTS.md`](./TRAINING_IMPROVEMENTS.md)
