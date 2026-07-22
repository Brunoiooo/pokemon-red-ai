#!/usr/bin/env python3
"""Behavioral cloning pre-training from a recorded human demonstration.

Replays the demo through the emulator to collect (observation, action) pairs,
then trains the model to imitate the expert via cross-entropy on Q-values.

Usage:
  # Pre-train a fresh model from a demo:
  python pretrain_bc.py

  # Fine-tune an existing model:
  python pretrain_bc.py --model-in models/latest.pth

  # Then start RL training from the BC checkpoint:
  python train.py --model models/bc_pretrained.pth
"""
import sys
sys.path.insert(0, "src")

import json
import argparse
from collections import Counter
from pathlib import Path
from multiprocessing import RLock

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from pokemon.Emulator import Emulator, DURATION_BINS
from pokemon.ModelPokemon import get_model

BUTTON_NAMES = ["A", "B", "Start", "Select", "Left", "Right", "Up", "Down", "None"]
BIN_16_IDX = 0  # index of 16-tick bin in DURATION_BINS


def print_sep(char="=", width=60):
    print(char * width)


def replay_demo(demo_path: str, device: torch.device, files_lock) -> list[tuple]:
    """Run the demo actions through the emulator and return (inputs, action) pairs."""
    with open(demo_path) as f:
        demo = json.load(f)

    checkpoint = demo["checkpoint"]
    actions: list[int] = demo["actions"]
    print(f"  Demo actions : {len(actions)}")

    emulator = Emulator(files_lock=files_lock)
    emulator.use_sdl = False

    memory, inputs = emulator.reset(dir=checkpoint)
    samples: list[tuple] = []

    for step, action in enumerate(actions):
        meta_action = action * len(DURATION_BINS) + BIN_16_IDX
        inp = {k: v.to(device) if isinstance(v, torch.Tensor) else v
               for k, v in inputs.items()}
        samples.append((inp, meta_action))

        next_memory, next_inputs, _reward, terminated, truncated = emulator.step(
            memory=memory, meta_action=meta_action
        )
        if terminated or truncated:
            print(f"  Episode ended at step {step + 1}")
            break
        memory, inputs = next_memory, next_inputs

    try:
        emulator.pyboy.stop(False)
    except Exception:
        pass

    print(f"  Samples      : {len(samples)}")
    counts = Counter(BUTTON_NAMES[a // len(DURATION_BINS)] for _, a in samples)
    print("  Action distribution:")
    for btn, cnt in counts.most_common():
        bar = "#" * int(cnt / len(samples) * 30)
        print(f"    {btn:8s}: {cnt:4d}  {bar}")

    return samples


def train_bc(
    model,
    samples: list[tuple],
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> None:
    """Train model via cross-entropy between Q-values and expert actions."""
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()

    n = len(samples)
    print()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0
        n_batches = 0

        for start in range(0, n, batch_size):
            batch = samples[start : start + batch_size]
            if not batch:
                continue

            optimizer.zero_grad()
            batch_loss = 0.0

            for inp, act in batch:
                out = model(inp)
                # out["q"] shape: [1, N_META_ACTIONS] — Q-value per (button, duration) pair
                q = out["q"].squeeze(0)   # [N_META_ACTIONS]
                target = torch.tensor([act], dtype=torch.long, device=device)
                loss = F.cross_entropy(q.unsqueeze(0), target) / len(batch)
                # backward immediately — NoisyLinear resamples noise in-place,
                # so we must not accumulate losses across forward passes
                loss.backward()
                batch_loss += loss.item()
                if q.argmax().item() == act:
                    correct += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += batch_loss
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        accuracy = correct / n * 100
        print(f"  Epoch {epoch:4d}/{epochs} | loss={avg_loss:.4f} | acc={accuracy:.1f}%")

    model.eval()


def run(args):
    if not Path(args.demo).exists():
        print(f"[ERROR] Demo file not found: {args.demo}")
        sys.exit(1)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print_sep()
    print("  Pokemon Red AI — Behavioral Cloning Pre-training")
    print_sep()
    print(f"  Demo       : {args.demo}")
    print(f"  Model in   : {args.model_in or '(fresh)'}")
    print(f"  Model out  : {args.model_out}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  LR         : {args.lr}")
    print(f"  Batch size : {args.batch_size}")
    print(f"  Device     : {device}")
    print_sep()

    files_lock = RLock()
    model = get_model(device=device, files_lock=files_lock)

    if args.model_in and Path(args.model_in).exists():
        data = torch.load(args.model_in, map_location=device, weights_only=False)
        state = data["model_state"] if isinstance(data, dict) and "model_state" in data else data
        model.load_state_dict(state)
        print(f"Loaded weights from {args.model_in}")

    print("\nReplaying demo through emulator...")
    samples = replay_demo(args.demo, device, files_lock)

    if not samples:
        print("[ERROR] No samples collected.")
        sys.exit(1)

    print(f"\nTraining for {args.epochs} epochs on {len(samples)} samples...")
    train_bc(model, samples, device, args.epochs, args.lr, args.batch_size)

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict()}, args.model_out)

    print_sep()
    print(f"  Saved pre-trained model → {args.model_out}")
    print()
    print("  Next steps:")
    print(f"    python train.py --model {args.model_out}")
    print_sep()


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral cloning pre-training from a human demo"
    )
    parser.add_argument("--demo", "-d", default="demos/demo.json",
                        help="Demo file from record_demo.py (default: demos/demo.json)")
    parser.add_argument("--model-in", default=None,
                        help="Existing .pth to fine-tune (default: fresh model)")
    parser.add_argument("--model-out", default="models/bc_pretrained.pth",
                        help="Output model path (default: models/bc_pretrained.pth)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
