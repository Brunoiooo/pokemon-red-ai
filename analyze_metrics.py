#!/usr/bin/env python3
"""Analyze saved training metrics"""
import sys
sys.path.insert(0, "src")
import json
import argparse
from pathlib import Path
from datetime import datetime

def analyze_metrics(metrics_file):
    """Load and analyze metrics from file"""

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    print(f"\n{'='*70}")
    print(f"  Training Metrics Analysis")
    print(f"{'='*70}\n")

    session = data['session']
    print(f"Session Duration:")
    print(f"  Start: {session['start_time']}")
    print(f"  End:   {session['end_time']}")
    print(f"  Steps: {session['total_steps']}")
    print(f"  Opt steps: {session['total_opt_steps']}")
    print(f"  Evals: {session['total_evals']}")
    print(f"  Throughput: {session['avg_steps_per_sec']:.1f} steps/sec\n")

    # Analysis
    if data['issues']:
        print(f"Issues Found:")
        for issue in data['issues']:
            print(f"  ⚠ {issue}")
        print()

    if data['recommendations']:
        print(f"Recommendations:")
        for i, rec in enumerate(data['recommendations'], 1):
            print(f"  {i}. {rec}")
        print()

    # Eval details
    if data['eval_metrics']:
        evals = data['eval_metrics']
        print(f"Evaluation Details:")
        print(f"  Total: {len(evals)}")
        returns = [e['return_value'] for e in evals]
        print(f"  Best: {max(returns):+.4f}")
        print(f"  Worst: {min(returns):+.4f}")
        print(f"  Avg: {sum(returns)/len(returns):+.4f}\n")

    print(f"{'='*70}\n")

def run(args):
    if args.latest:
        log_dir = Path("logs")
        metrics_files = list(log_dir.glob("metrics_*.json"))
        if not metrics_files:
            print("No metrics files found!")
            return
        metrics_file = max(metrics_files)
        print(f"Analyzing: {metrics_file.name}")
    elif args.file:
        metrics_file = Path(args.file)
    else:
        print("Specify --file or --latest")
        return

    if not metrics_file.exists():
        print(f"File not found: {metrics_file}")
        return

    analyze_metrics(metrics_file)


def main():
    parser = argparse.ArgumentParser(description="Analyze training metrics")
    parser.add_argument("--file", "-f", type=str,
                        help="Metrics JSON file to analyze")
    parser.add_argument("--latest", "-l", action="store_true",
                        help="Analyze latest metrics file")

    run(parser.parse_args())

if __name__ == "__main__":
    main()
