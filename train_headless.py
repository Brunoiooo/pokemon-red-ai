#!/usr/bin/env python3
import sys
sys.path.insert(0, "src")
import csv
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
    session_name = datetime.now().strftime('%Y%m%d_%H%M%S')

    train_csv_path = Path("logs") / f"train_steps_{session_name}.csv"
    episode_csv_path = Path("logs") / f"episodes_{session_name}.csv"
    train_csv_file = open(train_csv_path, "w", newline="")
    episode_csv_file = open(episode_csv_path, "w", newline="")
    train_csv_writer = csv.DictWriter(train_csv_file, fieldnames=[
        "opt_step", "timestamp", "loss", "td_error_mean", "td_error_max", "q_mean", "grad_norm"])
    episode_csv_writer = csv.DictWriter(episode_csv_file, fieldnames=[
        "episode", "timestamp", "length", "epsilon", "total_reward",
        "action_0", "action_1", "action_2", "action_3", "action_4", "action_5", "action_6", "action_7"])
    train_csv_writer.writeheader()
    episode_csv_writer.writeheader()

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
    global_ep = 0
    last_train_stat = {}
    last_ep_stat = {}

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

            # Collect training/episode stats
            elapsed = time.time() - start_time
            while not trainer.stats_queue.empty():
                stat = trainer.stats_queue.get_nowait()
                if stat.get("type") == "train_step":
                    last_train_stat = stat
                    train_csv_writer.writerow({
                        "opt_step": stat["opt_step"],
                        "timestamp": f"{elapsed:.1f}",
                        "loss": f"{stat['loss']:.6f}",
                        "td_error_mean": f"{stat['td_error_mean']:.6f}",
                        "td_error_max": f"{stat['td_error_max']:.6f}",
                        "q_mean": f"{stat['q_mean']:.6f}",
                        "grad_norm": f"{stat['grad_norm']:.6f}",
                    })
                    train_csv_file.flush()
                elif stat.get("type") == "episode":
                    last_ep_stat = stat
                    global_ep += 1
                    ac = stat.get("action_counts", [0]*8)
                    episode_csv_writer.writerow({
                        "episode": global_ep,
                        "timestamp": f"{elapsed:.1f}",
                        "length": stat.get("episode_length", 0),
                        "epsilon": f"{stat.get('epsilon', 0):.5f}",
                        "total_reward": f"{stat.get('total_reward', 0):.4f}",
                        **{f"action_{i}": ac[i] if i < len(ac) else 0 for i in range(8)},
                    })
                    episode_csv_file.flush()

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

                    if last_train_stat:
                        print(f"           "
                              f"Loss: {last_train_stat.get('loss', 0):.4f} | "
                              f"TD_err: {last_train_stat.get('td_error_mean', 0):.3f} "
                              f"(max {last_train_stat.get('td_error_max', 0):.2f}) | "
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
                              f"Top action: {action_names[dominant]}({ac[dominant]})")

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
        train_csv_file.close()
        episode_csv_file.close()
        print(f"Training stats: {train_csv_path}")
        print(f"Episode stats:  {episode_csv_path}")

if __name__ == "__main__":
    main()

