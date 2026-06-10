#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
import torch
import argparse
from pathlib import Path
from datetime import datetime
from multiprocessing import RLock
from pokemon.Emulator import Emulator
from pokemon.ModelPokemon import get_model

def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def print_header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")

def evaluate_model(model_path: str, num_episodes: int = 1, checkpoint: str = "start"):
    print_header(f"Pokemon Red AI - Model Evaluation")
    print(f"Model: {model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")

    files_lock = RLock()
    emulator = Emulator(files_lock=files_lock)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device=device, files_lock=files_lock)

    # Load model
    try:
        checkpoint_data = torch.load(model_path, map_location=device)
        if isinstance(checkpoint_data, dict) and "model_state" in checkpoint_data:
            model.load_state_dict(checkpoint_data["model_state"])
        else:
            model.load_state_dict(checkpoint_data)
        print(f"✓ Model loaded successfully\n")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return

    model.eval()

    results = []

    for episode in range(num_episodes):
        print(f"[Episode {episode + 1}/{num_episodes}] ", end="", flush=True)

        memory, inputs = emulator.reset(dir=checkpoint)
        total_reward = 0.0
        steps = 0
        import time
        start_time = time.time()

        while True:
            with torch.inference_mode():
                q = model(inputs)
                q = q.squeeze(0)
                action = int(torch.argmax(q).item())

            next_memory, next_inputs, reward, terminated, truncated = emulator.step(
                memory=memory, action=action
            )

            total_reward += reward
            steps += 1

            if terminated or truncated:
                elapsed = time.time() - start_time
                results.append({
                    "episode": episode + 1,
                    "reward": total_reward,
                    "steps": steps,
                    "time": elapsed
                })
                print(f"Return: {total_reward:7.2f} | Steps: {steps:4d} | Time: {format_time(elapsed)}")
                break

            memory, inputs = next_memory, next_inputs

    # Print summary
    print_header("Evaluation Summary")

    total_return = sum(r["reward"] for r in results)
    avg_return = total_return / len(results)
    max_return = max(r["reward"] for r in results)
    min_return = min(r["reward"] for r in results)

    print(f"Episodes: {len(results)}")
    print(f"Average return: {avg_return:.4f}")
    print(f"Max return:     {max_return:.4f}")
    print(f"Min return:     {min_return:.4f}")
    print(f"Total time:     {sum(r['time'] for r in results):.2f}s")

    emulator.pyboy.stop(False)
    print("\n✓ Evaluation complete")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pokemon Red AI model")
    parser.add_argument("--model", "-m", type=str, default="models/best.pth",
                        help="Path to model file (default: models/best.pth)")
    parser.add_argument("--episodes", "-e", type=int, default=1,
                        help="Number of episodes to run (default: 1)")
    parser.add_argument("--checkpoint", "-c", type=str, default="start",
                        help="Starting checkpoint name (default: start)")

    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"✗ Model file not found: {args.model}")
        return

    evaluate_model(args.model, num_episodes=args.episodes, checkpoint=args.checkpoint)

if __name__ == "__main__":
    main()
