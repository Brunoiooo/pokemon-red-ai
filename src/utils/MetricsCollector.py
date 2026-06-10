#!/usr/bin/env python3
"""Metrics collector for training analysis and automatic improvements"""
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import statistics

@dataclass
class StepMetrics:
    """Metrics collected per optimization step"""
    step: int
    timestamp: float
    loss: float = 0.0
    td_error: float = 0.0
    buffer_size: int = 0
    buffer_pct: float = 0.0
    epsilon: float = 0.0
    steps_per_sec: float = 0.0

@dataclass
class EvalMetrics:
    """Metrics collected per evaluation"""
    eval_num: int
    step: int
    timestamp: float
    return_value: float
    episode_length: int = 0
    badges_collected: int = 0
    delta_from_prev: float = 0.0

@dataclass
class TrainingSession:
    """Complete training session metrics"""
    start_time: datetime
    end_time: datetime | None = None
    total_steps: int = 0
    total_opt_steps: int = 0
    total_evals: int = 0
    avg_steps_per_sec: float = 0.0

    step_metrics: List[StepMetrics] = field(default_factory=list)
    eval_metrics: List[EvalMetrics] = field(default_factory=list)

    config: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class MetricsCollector:
    """Collects and analyzes training metrics"""

    def __init__(self, log_dir: Path = Path("logs")):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.session = TrainingSession(start_time=datetime.now())
        self.last_eval_return = None

    def add_step_metrics(self, **kwargs):
        """Add metrics for a training step"""
        metrics = StepMetrics(**kwargs)
        self.session.step_metrics.append(metrics)

    def add_eval_metrics(self, eval_num: int, step: int, timestamp: float,
                         return_value: float, **kwargs):
        """Add metrics for an evaluation"""
        delta = 0.0
        if self.last_eval_return is not None:
            delta = return_value - self.last_eval_return
        self.last_eval_return = return_value

        metrics = EvalMetrics(
            eval_num=eval_num,
            step=step,
            timestamp=timestamp,
            return_value=return_value,
            delta_from_prev=delta,
            **kwargs
        )
        self.session.eval_metrics.append(metrics)

    def finalize(self, total_steps: int, total_opt_steps: int, elapsed_time: float):
        """Finalize session and analyze"""
        self.session.end_time = datetime.now()
        self.session.total_steps = total_steps
        self.session.total_opt_steps = total_opt_steps
        self.session.avg_steps_per_sec = total_steps / max(1, elapsed_time)

        self._analyze_and_recommend()

    def _analyze_and_recommend(self):
        """Analyze metrics and generate recommendations"""

        # Check exploration
        if self.session.eval_metrics:
            avg_return = statistics.mean([e.return_value for e in self.session.eval_metrics])
            max_return = max([e.return_value for e in self.session.eval_metrics])
            min_return = min([e.return_value for e in self.session.eval_metrics])

            if max_return < 0.1:
                self.session.issues.append("Model achieving only negative returns")
                self.session.recommendations.append(
                    "Increase exploration (--workers 8) or decrease base_reward penalty"
                )

            if max_return > 0.5 and min_return > 0.2:
                self.session.recommendations.append(
                    "Model converged well! Can reduce exploration epsilon"
                )

            # Check delta trend
            deltas = [e.delta_from_prev for e in self.session.eval_metrics[1:]]
            if deltas:
                avg_delta = statistics.mean(deltas)
                if avg_delta < -0.01:
                    self.session.issues.append("Learning rate too high (negative delta trend)")
                    self.session.recommendations.append("Reduce learning rate from 5e-4 to 1e-4")
                elif avg_delta > 0.05:
                    self.session.recommendations.append("Learning rate good! Can increase for faster convergence")

        # Check buffer fill rate
        if self.session.step_metrics:
            final_buffer_pct = self.session.step_metrics[-1].buffer_pct
            if final_buffer_pct < 50:
                self.session.issues.append(f"Buffer only {final_buffer_pct:.1f}% full after session")
                self.session.recommendations.append(
                    "Run longer or increase --workers for faster buffer fill"
                )

        # Check throughput
        if self.session.avg_steps_per_sec < 5:
            self.session.issues.append(f"Low throughput: {self.session.avg_steps_per_sec:.1f} steps/sec")
            self.session.recommendations.append("Check GPU usage or reduce batch_size")
        elif self.session.avg_steps_per_sec > 20:
            self.session.recommendations.append("High throughput! Can increase batch_size for faster learning")

        # Check optimization frequency
        if self.session.total_opt_steps > 0:
            opt_ratio = self.session.total_opt_steps / max(1, self.session.total_steps)
            if opt_ratio < 0.8:
                self.session.issues.append("Low optimization frequency")
                self.session.recommendations.append(
                    "Increase buffer capacity or reduce batch_size for more frequent updates"
                )

    def save_json(self):
        """Save metrics to JSON"""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        filename = self.log_dir / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert to serializable format
        data = {
            "session": {
                "start_time": self.session.start_time.isoformat(),
                "end_time": self.session.end_time.isoformat() if self.session.end_time else None,
                "total_steps": self.session.total_steps,
                "total_opt_steps": self.session.total_opt_steps,
                "total_evals": len(self.session.eval_metrics),
                "avg_steps_per_sec": self.session.avg_steps_per_sec,
            },
            "step_metrics": [asdict(m) for m in self.session.step_metrics],
            "eval_metrics": [asdict(m) for m in self.session.eval_metrics],
            "issues": self.session.issues,
            "recommendations": self.session.recommendations,
        }

        with open(filename, "w") as f:
            json.dump(data, f, indent=2)

        return filename

    def print_summary(self):
        """Print analysis summary"""
        print(f"\n{'='*70}")
        print(f"  Training Analysis & Recommendations")
        print(f"{'='*70}\n")

        if self.session.eval_metrics:
            returns = [e.return_value for e in self.session.eval_metrics]
            print(f"Evaluation Statistics:")
            print(f"  Total evals:    {len(returns)}")
            print(f"  Best return:    {max(returns):+.4f}")
            print(f"  Worst return:   {min(returns):+.4f}")
            print(f"  Avg return:     {statistics.mean(returns):+.4f}")

            if len(returns) > 1:
                stdev = statistics.stdev(returns)
                print(f"  Std deviation:  {stdev:.4f}")

                deltas = [returns[i] - returns[i-1] for i in range(1, len(returns))]
                avg_delta = statistics.mean(deltas)
                print(f"  Avg delta:      {avg_delta:+.4f}")
            print()

        if self.session.issues:
            print(f"Issues Found ({len(self.session.issues)}):")
            for issue in self.session.issues:
                print(f"  ⚠ {issue}")
            print()

        if self.session.recommendations:
            print(f"Recommendations ({len(self.session.recommendations)}):")
            for i, rec in enumerate(self.session.recommendations, 1):
                print(f"  {i}. {rec}")
            print()

        print(f"{'='*70}\n")
