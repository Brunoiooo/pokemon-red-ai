#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
import time
from datetime import datetime
from pathlib import Path
from workers.TrainWorker import TrainWorker

def format_time(seconds):
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def print_header(msg):
    print(f"\n{'='*70}")
    print(f"  {msg}")
    print(f"{'='*70}\n")

def main():
    print_header("Pokemon Red AI - Headless Training (Optimized)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    Path("logs").mkdir(exist_ok=True)
    log_file = Path("logs") / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    trainer = TrainWorker(max_workers=5)

    print(f"Device: {trainer.device}")
    print(f"Workers: {trainer.max_workers}")
    print(f"Batch size: {trainer.batch_size}")
    print(f"Learning rate: {trainer.lr:.2e}")
    print(f"Buffer capacity: {trainer.buffer_capacity}")
    print(f"PER alpha: {trainer.per_alpha}")
    print(f"Target update interval: {trainer.target_update_interval}\n")

    trainer.run()

    print("Training threads started. Monitoring...\n")

    start_time = time.time()
    last_count = 0
    last_print = time.time()
    log_messages = []
    evals = []

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
                print(f"\n{'='*70}")
                print(f"✓ EVAL #{len(evals)}")
                print(f"{'='*70}")
                print(f"Steps:      {count}")
                print(f"Return:     {avg_ret:.4f}")
                print(f"Time:       {format_time(elapsed)}")
                if len(evals) > 1:
                    prev_ret = evals[-2][1]
                    delta = avg_ret - prev_ret
                    trend = "↑" if delta > 0 else "↓"
                    print(f"Delta:      {trend} {abs(delta):+.4f}")
                print(f"{'='*70}\n")

            # Print metrics every 15 seconds
            if time.time() - last_print > 15:
                elapsed = time.time() - start_time
                current_count = trainer.count
                opt_steps = trainer._opt_steps
                buffer_size = len(trainer.buffer) if trainer.buffer else 0

                if current_count > last_count:
                    steps_per_sec = (current_count - last_count) / (time.time() - last_print)
                    buffer_pct = (buffer_size / trainer.buffer_capacity) * 100
                    print(f"[{format_time(elapsed)}] Steps: {current_count:6d} | "
                          f"Opt: {opt_steps:5d} | "
                          f"Buffer: {buffer_size:6d}/{trainer.buffer_capacity} ({buffer_pct:5.1f}%) | "
                          f"Speed: {steps_per_sec:.1f} steps/s")
                    last_count = current_count

                last_print = time.time()

    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("  Shutdown requested...")
        print("="*70)
        trainer.event_start.clear()

        print("Waiting for threads to finish...")
        time.sleep(2)

        total_time = time.time() - start_time
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
            print(f"Best eval return:  {best_eval[1]:.4f} (step {best_eval[0]})")
            print(f"Last eval return:  {evals[-1][1]:.4f}")

        print(f"{'='*70}\n")

    finally:
        # Save logs
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            for msg in log_messages:
                f.write(msg + "\n")
        print(f"Logs saved to: {log_file}")

if __name__ == "__main__":
    main()

