# Headless Training Guide (Optimized)

## Performance Improvements

**Optimizations made:**
- ✓ Batch size: 128 → 256 (better GPU utilization)
- ✓ Learning rate: 5e-4 → 1e-3 (faster convergence)
- ✓ Grad accum: 2 → 1 (faster updates)
- ✓ Target update: 1000 → 500 (more stable Q-values)
- ✓ Tau: 0.0005 → 0.001 (faster target network sync)
- ✓ PER alpha: 0.6 → 0.7 (better prioritization)
- ✓ Buffer sleep: 1.0s → 0.1s (faster fill)
- ✓ Episode steps: 100 → 1000 (longer episodes)
- ✓ PyBoy: "headless" → "null" + cgb=False (compatibility fix)

**Benchmark results:**
- Pipeline speed: 44.7 → 50.6 steps/sec (+13%)
- GPU inference: Working with CUDA
- Emulation: 37-42 steps/sec (CPU baseline)

## Quick Start

### 1. Benchmark (Optional - verify setup)
```bash
./venv/Scripts/python benchmark.py
```

Expected output:
- Emulation: 35-45 steps/sec
- Full pipeline: 40-55 steps/sec

### 2. Test Training (5 minutes)
```bash
./venv/Scripts/python test_training.py
```

### 3. Test Training (15 minutes)
```bash
./venv/Scripts/python test_training_15min.py
```

### 4. Full Training (Headless, no GUI)
```bash
./venv/Scripts/python train_headless.py
```

Features:
- No GUI overhead = max speed
- Real-time metrics every 30s
- Evaluation results highlighted
- Auto-logging to `logs/train_*.log`
- Graceful shutdown with Ctrl+C

### 5. Evaluate Model
```bash
# Test best model
./venv/Scripts/python eval_model.py

# Test specific model
./venv/Scripts/python eval_model.py --model models/latest.pth --episodes 3
```

## Training Configuration

### Default (Optimized)
```
Batch size:       256
Learning rate:    1e-3
Workers:          5
Device:           auto (CUDA if available)
Update interval:  500 steps
Buffer capacity:  200,000
```

### Conservative (More stable)
In `src/workers/TrainWorker.py`:
```python
batch_size = 128
lr = 5e-4
grad_accum_steps = 2
target_update_interval = 1000
per_alpha = 0.6
```

### Aggressive (Faster learning, might be unstable)
```python
batch_size = 512
lr = 2e-3
grad_accum_steps = 1
target_update_interval = 250
per_alpha = 0.8
```

## Monitoring Training

### Live monitoring
```bash
# Terminal 1 - Training
./venv/Scripts/python train_headless.py

# Terminal 2 - Watch logs in real-time
tail -f logs/train_*.log
```

### Metrics explained
- `Steps`: Total game steps taken by workers
- `Opt`: Optimization/training steps completed
- `Buffer`: Replay buffer fill percentage
- `Speed`: Steps per second (game simulation speed)

### Expected progress
- First 30s: Buffer filling (slow)
- 30s-2min: Buffer reaches 50%, training starts accelerating
- 2min+: Steady optimization loop

## Troubleshooting

### "Buffer too small" error
- Normal on startup - workers collecting data
- Should resolve within 30-60 seconds

### Low steps/sec
- Check `top` or Task Manager for CPU/GPU usage
- Expected: ~40-50 steps/sec on modern GPU
- On CPU only: ~35-40 steps/sec

### CUDA errors
- Falls back to CPU automatically
- Check `trainer.device` output

## Advanced Tips

### Using multiple workers
```python
# In train_headless.py, change:
trainer = TrainWorker(max_workers=8)  # Use 8 workers instead of 5
```

### Adjusting evaluation frequency
The model evaluates based on training steps. To evaluate more/less frequently, edit `src/workers/TrainWorker.py`:
```python
# Current: every time run_workers runs
# Modify evaluate_greedy() method to add counter
```

### Resuming from checkpoint
Models auto-save to `models/latest.pth` and best to `models/best.pth`
- Training automatically loads latest.pth if it exists

## File Structure

```
root/
├── train_headless.py          # Main training script
├── eval_model.py              # Model evaluation
├── benchmark.py               # Performance benchmarks
├── test_training.py           # 5-minute test run
├── test_training_15min.py     # 15-minute test run
├── HEADLESS_TRAINING.md       # This file
├── src/
│   ├── workers/
│   │   ├── TrainWorker.py     # Main training loop (optimized)
│   │   └── ExperienceWorker.py # Experience collection
│   └── pokemon/
│       ├── Emulator.py        # PyBoy wrapper (optimized)
│       └── ModelPokemon.py    # DQN model
├── models/
│   ├── latest.pth             # Latest checkpoint
│   └── best.pth               # Best performing model
├── logs/
│   └── train_*.log            # Training logs (auto-generated)
└── saves/                     # Game state checkpoints
```

## Performance Notes

- GPU is essential for good performance
- CPU-only: ~35-40 steps/sec
- Modern GPU (RTX 3060+): 50-100+ steps/sec
- Multi-core CPU helps with game emulation parallelization

## Next Steps

1. Run 15-min test to verify setup works
2. If stable, run full training
3. Monitor evaluation returns for improvement
4. Adjust hyperparameters if needed based on learning curves

