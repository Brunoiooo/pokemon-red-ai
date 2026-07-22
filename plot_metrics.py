#!/usr/bin/env python3
"""
Usage:
  python plot_metrics.py                        # loads ALL sessions, oldest → newest
  python plot_metrics.py logs/train_steps_*.csv # single session
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
            parsed = {}
            for k, v in row.items():
                if v is None or v == "":
                    parsed[k] = 0.0
                    continue
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows

def find_all_sessions():
    """Return list of (train_file, episode_file|None) sorted oldest → newest."""
    train_files = sorted(glob.glob("logs/train_steps_*.csv"))
    sessions = []
    for tf in train_files:
        session = Path(tf).stem.replace("train_steps_", "")
        ep_file = f"logs/episodes_{session}.csv"
        sessions.append((tf, ep_file if os.path.exists(ep_file) else None))
    return sessions

def find_latest_session():
    sessions = find_all_sessions()
    return sessions[-1] if sessions else (None, None)

def load_all_train(sessions):
    """Concatenate all train CSVs with continuous opt_step; None = gap between sessions."""
    all_rows = []
    step_offset = 0
    for train_file, _ in sessions:
        rows = load_csv(train_file)
        if not rows:
            continue
        if all_rows:
            all_rows.append(None)
        for r in rows:
            r = dict(r)
            r["opt_step"] += step_offset
            all_rows.append(r)
        step_offset = max(r["opt_step"] for r in all_rows if r is not None) + 1
    return all_rows

def load_all_episodes(sessions):
    """Concatenate all episode CSVs with continuous episode numbers; None = gap."""
    truncate_modes = ["position", "dialog", "battle", "menu"]
    all_rows = []
    ep_offset = 0
    trunc_offsets = {m: 0 for m in truncate_modes}
    for _, ep_file in sessions:
        if not ep_file:
            continue
        rows = load_csv(ep_file)
        if not rows:
            continue
        if all_rows:
            all_rows.append(None)
        for r in rows:
            r = dict(r)
            r["episode"] += ep_offset
            for m in truncate_modes:
                key = f"truncate_{m}"
                if key in r:
                    r[key] = r[key] + trunc_offsets[m]
            all_rows.append(r)
        ep_offset = max(r["episode"] for r in all_rows if r is not None) + 1
        last = rows[-1]
        for m in truncate_modes:
            key = f"truncate_{m}"
            if key in last:
                trunc_offsets[m] += int(last[key])
    return all_rows

def _plot_segments(ax, rows, x_key, y_key, **kwargs):
    """Plot rows that may contain None gap-markers as separate line segments."""
    xs, ys = [], []
    for r in rows:
        if r is None:
            if xs:
                ax.plot(xs, ys, **kwargs)
            xs, ys = [], []
        else:
            xs.append(r[x_key])
            ys.append(r[y_key])
    if xs:
        ax.plot(xs, ys, **kwargs)

def plot(train_rows, ep_rows, title):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    real_train = [r for r in train_rows if r is not None]
    if not real_train:
        print("No training data found.")
        return

    real_ep = [r for r in ep_rows if r is not None] if ep_rows else []
    truncate_modes = ["position", "dialog", "battle", "menu"]
    has_truncate = bool(real_ep) and any(f"truncate_{m}" in real_ep[0] for m in truncate_modes)

    n_rows = 4 if (real_ep and has_truncate) else (3 if real_ep else 2)
    fig = plt.figure(figsize=(14, 3.2 * n_rows))
    fig.suptitle(title, fontsize=12)
    gs = gridspec.GridSpec(n_rows, 2, figure=fig, hspace=0.45, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    _plot_segments(ax, train_rows, "opt_step", "loss", color="tab:blue", linewidth=0.8)
    ax.set_title("Loss")
    ax.set_xlabel("Opt step")
    ax.set_ylabel("Loss")

    ax = fig.add_subplot(gs[0, 1])
    _plot_segments(ax, train_rows, "opt_step", "td_error_mean", label="mean", color="tab:orange", linewidth=0.8)
    _plot_segments(ax, train_rows, "opt_step", "td_error_max", label="max", color="tab:red", linewidth=0.5, alpha=0.6)
    ax.set_title("TD Error")
    ax.set_xlabel("Opt step")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    _plot_segments(ax, train_rows, "opt_step", "q_mean", color="tab:green", linewidth=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_title("Q-value mean")
    ax.set_xlabel("Opt step")

    ax = fig.add_subplot(gs[1, 1])
    _plot_segments(ax, train_rows, "opt_step", "grad_norm", color="tab:purple", linewidth=0.8)
    ax.axhline(1.0, color="red", linewidth=0.5, linestyle="--", label="clip threshold")
    ax.set_title("Gradient norm")
    ax.set_xlabel("Opt step")
    ax.legend(fontsize=8)

    if real_ep:
        ax = fig.add_subplot(gs[2, 0])
        _plot_segments(ax, ep_rows, "episode", "total_reward", color="tab:brown", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title("Episode total reward")
        ax.set_xlabel("Episode")

        ax = fig.add_subplot(gs[2, 1])
        action_names = ["A", "B", "Start", "Sel", "L", "R", "Up", "Dn"]
        totals = [sum(r.get(f"action_{i}", 0) for r in real_ep) for i in range(8)]
        total_sum = max(sum(totals), 1)
        fracs = [t / total_sum for t in totals]
        bars = ax.bar(action_names, fracs, color=plt.cm.tab10.colors[:8])
        ax.set_title("Action distribution (overall)")
        ax.set_ylabel("Fraction")
        for bar, frac in zip(bars, fracs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{frac*100:.1f}%", ha="center", va="bottom", fontsize=7)

        if has_truncate:
            colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
            ax = fig.add_subplot(gs[3, 0])
            for mode, color in zip(truncate_modes, colors):
                key = f"truncate_{mode}"
                _plot_segments(ax, ep_rows, "episode", key, label=mode, color=color, linewidth=0.8)
            ax.set_title("Cumulative truncates by mode")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)

            ax = fig.add_subplot(gs[3, 1])
            mode_counts = {m: 0 for m in truncate_modes}
            for r in real_ep:
                m = r.get("truncate_mode", "")
                if isinstance(m, str) and m in mode_counts:
                    mode_counts[m] += 1
            # Fallback: derive from cumulative columns if truncate_mode missing
            if not any(mode_counts.values()):
                last = real_ep[-1]
                mode_counts = {m: int(last.get(f"truncate_{m}", 0)) for m in truncate_modes}
            total_trunc = max(sum(mode_counts.values()), 1)
            fracs = [mode_counts[m] / total_trunc for m in truncate_modes]
            bars = ax.bar(truncate_modes, fracs, color=colors)
            top_mode = max(mode_counts, key=mode_counts.get) if any(mode_counts.values()) else "-"
            ax.set_title(f"Truncate mode distribution (top={top_mode})")
            ax.set_ylabel("Fraction")
            for bar, frac, mode in zip(bars, fracs, truncate_modes):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{mode_counts[mode]}\n{frac*100:.0f}%", ha="center", va="bottom", fontsize=7)

    out_path = Path("logs") / f"plot_{title.replace(' ', '_').replace('/', '-')}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()

def print_summary(train_rows, ep_rows):
    real_train = [r for r in train_rows if r is not None]
    if not real_train:
        return
    losses = [r["loss"] for r in real_train if r["loss"] > 0]
    grad_norms = [r["grad_norm"] for r in real_train if r["grad_norm"] > 0]
    q_means = [r["q_mean"] for r in real_train]
    td_means = [r["td_error_mean"] for r in real_train]

    print(f"\n=== Training stats ({len(real_train)} records) ===")
    if losses:
        print(f"Loss:      min={min(losses):.4f}  max={max(losses):.4f}  last={losses[-1]:.4f}")
    if grad_norms:
        clipped = sum(1 for g in grad_norms if g >= 0.95)
        print(f"GradNorm:  min={min(grad_norms):.3f}  max={max(grad_norms):.3f}  clipped={clipped/len(grad_norms)*100:.0f}%")
    if q_means:
        print(f"Q_mean:    min={min(q_means):.3f}  max={max(q_means):.3f}  last={q_means[-1]:.3f}")
    if td_means:
        print(f"TD_err:    min={min(td_means):.3f}  max={max(td_means):.3f}  last={td_means[-1]:.3f}")

    real_ep = [r for r in ep_rows if r is not None] if ep_rows else []
    if real_ep:
        rewards = [r["total_reward"] for r in real_ep]
        lengths = [r["length"] for r in real_ep]
        print(f"\n=== Episode stats ({len(real_ep)} episodes) ===")
        print(f"Reward:    min={min(rewards):.3f}  max={max(rewards):.3f}  avg={sum(rewards)/len(rewards):.3f}")
        print(f"Length:    min={min(lengths):.0f}  max={max(lengths):.0f}  avg={sum(lengths)/len(lengths):.0f}")
        totals = [sum(r.get(f"action_{i}", 0) for r in real_ep) for i in range(8)]
        total_sum = max(sum(totals), 1)
        action_names = ["A", "B", "Start", "Sel", "L", "R", "Up", "Dn"]
        print("Actions:   " + "  ".join(f"{n}={t/total_sum*100:.0f}%" for n, t in zip(action_names, totals)))

        truncate_modes = ["position", "dialog", "battle", "menu"]
        if any(f"truncate_{m}" in real_ep[0] for m in truncate_modes):
            last = real_ep[-1]
            counts = {m: int(last.get(f"truncate_{m}", 0)) for m in truncate_modes}
            if not any(counts.values()):
                counts = {m: 0 for m in truncate_modes}
                for r in real_ep:
                    m = r.get("truncate_mode", "")
                    if isinstance(m, str) and m in counts:
                        counts[m] += 1
            top = max(counts, key=counts.get) if any(counts.values()) else "-"
            print("Truncates: " + "  ".join(f"{m}={counts[m]}" for m in truncate_modes) + f"  top={top}")

def run(args):
    train_csv = getattr(args, "train_csv", None)
    if train_csv:
        session = Path(train_csv).stem.replace("train_steps_", "")
        ep_file = f"logs/episodes_{session}.csv"
        sessions = [(train_csv, ep_file if os.path.exists(ep_file) else None)]
        title = Path(train_csv).stem
    else:
        sessions = find_all_sessions()
        if not sessions:
            print("No training CSV files found in logs/")
            return
        print(f"Loading {len(sessions)} session(s): {[s[0] for s in sessions]}")
        title = f"All sessions ({len(sessions)})"

    train_rows = load_all_train(sessions)
    ep_rows = load_all_episodes(sessions)
    plot(train_rows, ep_rows, title)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        train_file = sys.argv[1]
        session = Path(train_file).stem.replace("train_steps_", "")
        ep_file = f"logs/episodes_{session}.csv"
        sessions = [(train_file, ep_file if os.path.exists(ep_file) else None)]
        title = Path(train_file).stem
    else:
        sessions = find_all_sessions()
        if not sessions:
            print("No training CSV files found in logs/")
            sys.exit(1)
        print(f"Loading {len(sessions)} session(s): {[s[0] for s in sessions]}")
        title = f"All sessions ({len(sessions)})"

    train_rows = load_all_train(sessions)
    ep_rows = load_all_episodes(sessions)
    plot(train_rows, ep_rows, title)
