#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
import time
import argparse
import torch
import os
from datetime import datetime
from pathlib import Path
from multiprocessing import set_start_method
from workers.TrainWorker import TrainWorker
from utils.MetricsCollector import MetricsCollector

def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def print_header(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(description="Pokemon Red AI Headless Training")
    parser.add_argument("--gui", "-g", action="store_true",
                        help="Enable GUI (show game window)")
    parser.add_argument("--workers", "-w", type=int, default=5,
                        help="Number of experience workers (default: 5)")
    parser.add_argument("--eval-gui", "-eg", action="store_true",
                        help="Show GUI during evaluations only")

    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    set_start_method("spawn", force=True)
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")

    gui_mode = "GUI ON" if args.gui else "Headless"
    eval_gui_mode = "ON" if args.eval_gui else "OFF"

    print_header(f"Pokemon Red AI - Training ({gui_mode})")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {gui_mode}")
    print(f"Eval GUI: {eval_gui_mode}")

    Path("logs").mkdir(exist_ok=True)
    log_file = Path("logs") / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    trainer = TrainWorker(max_workers=args.workers)

    # Set GUI mode
    trainer.train_use_sdl.value = args.gui
    trainer.is_evaluation_window.value = args.eval_gui

    print(f"Device: {trainer.device}")
    print(f"Workers: {trainer.max_workers}")
    print(f"Batch size: {trainer.batch_size}")
    print(f"Learning rate: {trainer.lr:.2e}")
    print(f"Buffer capacity: {trainer.buffer_capacity}")
    print(f"Episode steps: 5000 (UPDATED)")
    print(f"Base reward: -0.0001 (UPDATED)")
    print(f"Exploration: 50-80% (UPDATED)\n")

    trainer.run()

    print("Training threads started. Monitoring...\n")
    print("Press Ctrl+C to stop gracefully.\n")

    start_time = time.time()
    last_count = 0
    last_print = time.time()
    last_opt = 0
    log_messages = []
    evals = []

    metrics = MetricsCollector()

    try:
        while True:
            time.sleep(0.2)

            # Collect logs
            while not trainer.queue_logs.empty():
                msg = trainer.queue_logs.get_nowait()
                log_messages.append(msg)

            # Collect evaluation results
            while not trainer.queue_dots.empty():
                count, avg_ret = trainer.queue_dots.get_nowait()
                elapsed = time.time() - start_time
                evals.append((count, avg_ret, elapsed))

                # Collect eval metrics
                metrics.add_eval_metrics(
                    eval_num=len(evals),
                    step=count,
                    timestamp=elapsed,
                    return_value=avg_ret,
                )

                print(f"\n{'='*70}")
                print(f"[EVAL #{len(evals)}] Step {count} | Return: {avg_ret:.4f} | Time: {format_time(elapsed)}")
                if len(evals) > 1:
                    prev_ret = evals[-2][1]
                    delta = avg_ret - prev_ret
                    trend = "↑" if delta > 0 else "↓"
                    print(f"Delta: {trend} {abs(delta):+.4f}")
                print(f"{'='*70}\n")

            # Print metrics every 30 seconds
            if time.time() - last_print > 30:
                elapsed = time.time() - start_time
                current_count = trainer.count
                opt_steps = trainer._opt_steps
                buffer_size = len(trainer.buffer) if trainer.buffer else 0

                if current_count > last_count:
                    steps_per_sec = (current_count - last_count) / (time.time() - last_print)
                    opt_per_sec = (opt_steps - last_opt) / (time.time() - last_print)
                    buffer_pct = (buffer_size / trainer.buffer_capacity) * 100

                    print(f"[{format_time(elapsed)}] "
                          f"Steps: {current_count:7d} ({steps_per_sec:5.1f} s/s) | "
                          f"Opt: {opt_steps:6d} ({opt_per_sec:4.1f} opt/s) | "
                          f"Buf: {buffer_pct:5.1f}%")

                    # Collect metrics
                    metrics.add_step_metrics(
                        step=current_count,
                        timestamp=elapsed,
                        buffer_size=buffer_size,
                        buffer_pct=buffer_pct,
                        steps_per_sec=steps_per_sec,
                    )

                    last_count = current_count
                    last_opt = opt_steps

                last_print = time.time()

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("  Shutdown requested...")
        print("="*70)
        trainer.event_start.clear()

        print("Waiting for threads to finish...")
        time.sleep(2)

        total_time = time.time() - start_time

        # Finalize metrics
        metrics.finalize(trainer.count, trainer._opt_steps, total_time)
        metrics_file = metrics.save_json()

        print(f"\n{'='*70}")
        print(f"  Training Summary")
        print(f"{'='*70}")
        print(f"Total time:        {format_time(total_time)}")
        print(f"Total steps:       {trainer.count}")
        print(f"Total opt steps:   {trainer._opt_steps}")
        print(f"Steps/sec avg:     {trainer.count / total_time:.1f}")
        print(f"Evaluations:       {len(evals)}")

        if evals:
            best_eval = max(evals, key=lambda x: x[1])
            last_eval = evals[-1]
            print(f"Best eval return:  {best_eval[1]:.4f} (step {best_eval[0]})")
            print(f"Last eval return:  {last_eval[1]:.4f}")
            if best_eval[1] > 0:
                print(f"\n✓ Model achieved positive returns!")

        print(f"{'='*70}\n")

        # Print analysis and recommendations
        metrics.print_summary()
        print(f"Detailed metrics saved to: {metrics_file}\n")

    finally:
        # Save logs
        with open(log_file, "w") as f:
            for msg in log_messages:
                f.write(msg + "\n")
        print(f"Logs saved to: {log_file}")

if __name__ == "__main__":
    main()
