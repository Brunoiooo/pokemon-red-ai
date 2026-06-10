#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
import time
import torch
import os
from datetime import datetime
from pathlib import Path
from multiprocessing import set_start_method
from workers.TrainWorker import TrainWorker

def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    set_start_method("spawn", force=True)
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")

    print("\n" + "="*70)
    print("  Pokemon Red AI - Test Training (15 minutes)")
    print("="*70)
    print(f"Batch size: 256 | LR: 1e-3 | Workers: 5 | Device: cuda\n")

    trainer = TrainWorker(max_workers=5)
    trainer.run()

    print("Training started. Collecting data...\n")
    start_time = time.time()
    last_print = time.time()
    last_count = 0
    last_opt = 0
    evals = []

    try:
        while time.time() - start_time < 900:  # 15 minutes
            time.sleep(0.5)

            # Print metrics every 30 seconds
            if time.time() - last_print > 30:
                elapsed = time.time() - start_time
                current_count = trainer.count
                opt_steps = trainer._opt_steps
                buffer_size = len(trainer.buffer) if trainer.buffer else 0

                steps_per_sec = (current_count - last_count) / (time.time() - last_print)
                opt_per_sec = (opt_steps - last_opt) / (time.time() - last_print)
                buffer_pct = (buffer_size / trainer.buffer_capacity) * 100

                print(f"[{format_time(elapsed)}] Steps: {current_count:7d} ({steps_per_sec:5.1f} s/s) | "
                      f"Opt: {opt_steps:6d} ({opt_per_sec:4.1f} opt/s) | "
                      f"Buf: {buffer_pct:5.1f}% ({buffer_size:6d})")

                last_count = current_count
                last_opt = opt_steps
                last_print = time.time()

            # Collect evaluation results
            while not trainer.queue_dots.empty():
                count, avg_ret = trainer.queue_dots.get_nowait()
                evals.append((count, avg_ret))
                elapsed = time.time() - start_time
                print(f"  [EVAL] Return={avg_ret:8.4f} at step {count:6d} ({format_time(elapsed)})")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        trainer.event_start.clear()
        time.sleep(1)

        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"  Test Complete")
        print(f"{'='*70}")
        print(f"Total time:        {format_time(total_time)}")
        print(f"Total steps:       {trainer.count}")
        print(f"Steps/sec avg:     {trainer.count / total_time:.1f}")
        print(f"Optimization steps:{trainer._opt_steps}")
        print(f"Evaluations:       {len(evals)}")
        if evals:
            returns = [r[1] for r in evals]
            best_ret = max(returns)
            avg_ret = sum(returns) / len(returns)
            print(f"Best eval return:  {best_ret:.4f}")
            print(f"Avg eval return:   {avg_ret:.4f}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
