#!/usr/bin/env python3
import sys
import warnings
warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`")
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
    # Setup
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    set_start_method("spawn", force=True)
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")

    print("\n" + "="*70)
    print("  Pokemon Red AI - Quick Test (5 minutes)")
    print("="*70 + "\n")

    trainer = TrainWorker(max_workers=5)
    trainer.run()

    print("Training started. Running for 5 minutes...\n")
    start_time = time.time()
    last_print = time.time()
    last_count = 0
    evals = []
    last_train_stat = {}
    last_ep_stat = {}
    global_ep = 0

    try:
        while time.time() - start_time < 300:  # 5 minutes
            time.sleep(0.2)

            # Collect training/episode stats
            while not trainer.stats_queue.empty():
                stat = trainer.stats_queue.get_nowait()
                if stat.get("type") == "train_step":
                    last_train_stat = stat
                elif stat.get("type") == "episode":
                    last_ep_stat = stat
                    global_ep += 1

            # Print metrics every 30 seconds
            if time.time() - last_print > 30:
                elapsed = time.time() - start_time
                current_count = trainer.count
                opt_steps = trainer._opt_steps
                buffer_size = len(trainer.buffer) if trainer.buffer else 0

                steps_per_sec = (current_count - last_count) / (time.time() - last_print)
                buffer_pct = (buffer_size / trainer.buffer_capacity) * 100

                print(f"[{format_time(elapsed)}] Steps: {current_count:6d} | "
                      f"Opt: {opt_steps:5d} | "
                      f"Buffer: {buffer_pct:5.1f}% | "
                      f"Speed: {steps_per_sec:.1f} steps/s")

                if last_train_stat:
                    print(f"           "
                          f"Loss: {last_train_stat.get('loss', 0):.4f} | "
                          f"TD_err: {last_train_stat.get('td_error_mean', 0):.3f} | "
                          f"Q_mean: {last_train_stat.get('q_mean', 0):.3f} | "
                          f"GradNorm: {last_train_stat.get('grad_norm', 0):.3f}")

                if last_ep_stat:
                    ac = last_ep_stat.get("action_counts", [0]*8)
                    dominant = max(range(8), key=lambda i: ac[i]) if ac else 0
                    action_names = ["A", "B", "Start", "Sel", "L", "R", "Up", "Dn"]
                    print(f"           "
                          f"Ep#{global_ep} len={last_ep_stat.get('episode_length', 0)} | "
                          f"Eps={last_ep_stat.get('epsilon', 0):.4f} | "
                          f"Rew={last_ep_stat.get('total_reward', 0):.3f} | "
                          f"Top: {action_names[dominant]}({ac[dominant]})")

                last_count = current_count
                last_print = time.time()

            # Collect evaluation results
            while not trainer.queue_dots.empty():
                count, avg_ret = trainer.queue_dots.get_nowait()
                evals.append((count, avg_ret))
                print(f"  [EVAL] return={avg_ret:.4f} at step {count}")

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
            print(f"Best eval return:  {max(evals, key=lambda x: x[1])[1]:.4f}")
        print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
