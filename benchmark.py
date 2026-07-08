#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
import time
import torch
import numpy as np
from multiprocessing import RLock
from pokemon.Emulator import Emulator
from pokemon.ModelPokemon import get_model

def benchmark_inference(num_steps: int = 1000):
    """Test model inference speed"""
    print("\n" + "="*60)
    print("  Inference Speed Benchmark")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    files_lock = RLock()
    model = get_model(device=device, files_lock=files_lock)
    model.eval()

    dummy_input = {
        "screen": torch.randn(1, 4, 144, 160, device=device).float(),
        "map_id": torch.tensor([[0]], dtype=torch.long, device=device),
        "position": torch.tensor([[0, 0]], dtype=torch.long, device=device),
    }

    # Warmup
    with torch.inference_mode():
        for _ in range(5):
            try:
                _ = model(dummy_input)
            except Exception as e:
                print(f"Model inference issue: {e}")
                return

    # Benchmark
    start = time.time()
    with torch.inference_mode():
        for i in range(num_steps):
            try:
                _ = model(dummy_input)
            except Exception:
                break
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                fps = (i + 1) / elapsed
                print(f"Step {i+1}/{num_steps}: {fps:.1f} inferences/sec")

    total_time = time.time() - start
    fps = num_steps / total_time
    print(f"\nTotal: {fps:.1f} inferences/sec ({total_time:.2f}s for {num_steps} steps)")

def benchmark_emulation(num_steps: int = 1000):
    """Test emulation speed"""
    print("\n" + "="*60)
    print("  Emulation Speed Benchmark")
    print("="*60)

    files_lock = RLock()
    emulator = Emulator(files_lock=files_lock)
    emulator.use_sdl = False

    memory, inputs = emulator.reset(dir="start")
    print(f"Initialized emulator\n")

    start = time.time()
    for i in range(num_steps):
        action = np.random.randint(0, 8)
        memory, inputs, reward, terminated, truncated = emulator.step(
            memory=memory, action=action
        )

        if terminated or truncated:
            memory, inputs = emulator.reset(dir="start")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            steps_per_sec = (i + 1) / elapsed
            print(f"Step {i+1}/{num_steps}: {steps_per_sec:.1f} steps/sec")

    total_time = time.time() - start
    steps_per_sec = num_steps / total_time
    print(f"\nTotal: {steps_per_sec:.1f} steps/sec ({total_time:.2f}s for {num_steps} steps)")

    emulator.pyboy.stop(False)

def benchmark_full_pipeline(num_steps: int = 500):
    """Test full training pipeline speed"""
    print("\n" + "="*60)
    print("  Full Pipeline Benchmark")
    print("="*60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    files_lock = RLock()
    emulator = Emulator(files_lock=files_lock)
    model = get_model(device=device, files_lock=files_lock)
    model.eval()

    memory, inputs = emulator.reset(dir="start")

    start = time.time()
    for i in range(num_steps):
        # Inference
        with torch.inference_mode():
            q = model(inputs)
            action = int(torch.argmax(q.squeeze(0)).item())

        # Emulation
        memory, inputs, reward, terminated, truncated = emulator.step(
            memory=memory, action=action
        )

        if terminated or truncated:
            memory, inputs = emulator.reset(dir="start")

        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            steps_per_sec = (i + 1) / elapsed
            print(f"Step {i+1}/{num_steps}: {steps_per_sec:.1f} steps/sec")

    total_time = time.time() - start
    steps_per_sec = num_steps / total_time
    print(f"\nTotal: {steps_per_sec:.1f} steps/sec ({total_time:.2f}s for {num_steps} steps)")

    emulator.pyboy.stop(False)

def main():
    print("\n" + "="*60)
    print("  Pokemon Red AI - Benchmarks")
    print("  (Run this once to profile performance)")
    print("="*60)

    try:
        benchmark_inference(num_steps=1000)
    except Exception as e:
        print(f"Inference benchmark failed: {e}")

    try:
        benchmark_emulation(num_steps=1000)
    except Exception as e:
        print(f"Emulation benchmark failed: {e}")

    try:
        benchmark_full_pipeline(num_steps=500)
    except Exception as e:
        print(f"Full pipeline benchmark failed: {e}")

    print("\n" + "="*60)
    print("  Benchmarks Complete")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
