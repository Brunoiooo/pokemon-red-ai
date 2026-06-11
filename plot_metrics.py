#!/usr/bin/env python3
"""
Usage:
  python plot_metrics.py                        # auto-picks latest session
  python plot_metrics.py logs/train_steps_*.csv
"""
import sys
import csv
import glob
import os
from pathlib import Path

def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: float(v) if v else 0.0 for k, v in row.items()})
    return rows

def find_latest_session():
    train_files = sorted(glob.glob("logs/train_steps_*.csv"))
    if not train_files:
        return None, None
    latest_train = train_files[-1]
    session = Path(latest_train).stem.replace("train_steps_", "")
    episode_file = f"logs/episodes_{session}.csv"
    return latest_train, episode_file if os.path.exists(episode_file) else None

def plot(train_file, episode_file=None):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        print_summary(train_file, episode_file)
        return

    train_rows = load_csv(train_file)
    if not train_rows:
        print("No training data found.")
        return

    steps = [r["opt_step"] for r in train_rows]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Training metrics — {Path(train_file).stem}", fontsize=12)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Loss
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(steps, [r["loss"] for r in train_rows], color="tab:blue", linewidth=0.8)
    ax.set_title("Loss")
    ax.set_xlabel("Opt step")
    ax.set_ylabel("Loss")

    # TD error
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(steps, [r["td_error_mean"] for r in train_rows], label="mean", color="tab:orange", linewidth=0.8)
    ax.plot(steps, [r["td_error_max"] for r in train_rows], label="max", color="tab:red", linewidth=0.5, alpha=0.6)
    ax.set_title("TD Error")
    ax.set_xlabel("Opt step")
    ax.legend(fontsize=8)

    # Q-mean
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(steps, [r["q_mean"] for r in train_rows], color="tab:green", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Q-value mean")
    ax.set_xlabel("Opt step")

    # Gradient norm
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(steps, [r["grad_norm"] for r in train_rows], color="tab:purple", linewidth=0.8)
    ax.axhline(1.0, color="red", linewidth=0.5, linestyle="--", label="clip threshold")
    ax.set_title("Gradient norm")
    ax.set_xlabel("Opt step")
    ax.legend(fontsize=8)

    if episode_file:
        ep_rows = load_csv(episode_file)
        if ep_rows:
            eps_nums = [r["episode"] for r in ep_rows]

            # Episode reward
            ax = fig.add_subplot(gs[2, 0])
            ax.plot(eps_nums, [r["total_reward"] for r in ep_rows], color="tab:brown", linewidth=0.8)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_title("Episode total reward")
            ax.set_xlabel("Episode")

            # Action distribution (stacked)
            ax = fig.add_subplot(gs[2, 1])
            action_names = ["A", "B", "Start", "Sel", "L", "R", "Up", "Dn"]
            totals = [sum(r[f"action_{i}"] for r in ep_rows) for i in range(8)]
            total_sum = max(sum(totals), 1)
            fracs = [t / total_sum for t in totals]
            bars = ax.bar(action_names, fracs, color=plt.cm.tab10.colors[:8])
            ax.set_title("Action distribution (overall)")
            ax.set_ylabel("Fraction")
            for bar, frac in zip(bars, fracs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{frac*100:.1f}%", ha="center", va="bottom", fontsize=7)

    out_path = Path(train_file).parent / f"plot_{Path(train_file).stem}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()

def print_summary(train_file, episode_file=None):
    train_rows = load_csv(train_file)
    if not train_rows:
        return
    losses = [r["loss"] for r in train_rows if r["loss"] > 0]
    grad_norms = [r["grad_norm"] for r in train_rows if r["grad_norm"] > 0]
    q_means = [r["q_mean"] for r in train_rows]
    td_means = [r["td_error_mean"] for r in train_rows]

    print(f"\n=== Training stats ({len(train_rows)} records) ===")
    if losses:
        print(f"Loss:      min={min(losses):.4f}  max={max(losses):.4f}  last={losses[-1]:.4f}")
    if grad_norms:
        clipped = sum(1 for g in grad_norms if g >= 0.95)
        print(f"GradNorm:  min={min(grad_norms):.3f}  max={max(grad_norms):.3f}  clipped={clipped/len(grad_norms)*100:.0f}%")
    if q_means:
        print(f"Q_mean:    min={min(q_means):.3f}  max={max(q_means):.3f}  last={q_means[-1]:.3f}")
    if td_means:
        print(f"TD_err:    min={min(td_means):.3f}  max={max(td_means):.3f}  last={td_means[-1]:.3f}")

    if episode_file:
        ep_rows = load_csv(episode_file)
        if ep_rows:
            rewards = [r["total_reward"] for r in ep_rows]
            lengths = [r["length"] for r in ep_rows]
            print(f"\n=== Episode stats ({len(ep_rows)} episodes) ===")
            print(f"Reward:    min={min(rewards):.3f}  max={max(rewards):.3f}  avg={sum(rewards)/len(rewards):.3f}")
            print(f"Length:    min={min(lengths):.0f}  max={max(lengths):.0f}  avg={sum(lengths)/len(lengths):.0f}")
            totals = [sum(r[f"action_{i}"] for r in ep_rows) for i in range(8)]
            total_sum = max(sum(totals), 1)
            action_names = ["A", "B", "Start", "Sel", "L", "R", "Up", "Dn"]
            print("Actions:   " + "  ".join(f"{n}={t/total_sum*100:.0f}%" for n, t in zip(action_names, totals)))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_file = sys.argv[1]
        session = Path(train_file).stem.replace("train_steps_", "")
        episode_file = f"logs/episodes_{session}.csv"
        if not os.path.exists(episode_file):
            episode_file = None
    else:
        train_file, episode_file = find_latest_session()
        if not train_file:
            print("No training CSV files found in logs/")
            sys.exit(1)
        print(f"Using: {train_file}")

    plot(train_file, episode_file)
