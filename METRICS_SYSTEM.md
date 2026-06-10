# Metrics Collection & Auto-Recommendations

## Overview

System automatycznie zbiera metryki treningu i proponuje ulepszenia.

## Zbierane Parametry

### Training Metrics (co 30 sekund)
- **step**: Numer kroku treningu
- **timestamp**: Czas w sekundach od startu
- **buffer_size**: Liczba próbek w replay buffer
- **buffer_pct**: Procent zapełnienia bufora
- **steps_per_sec**: Przepustowość (game steps/sec)
- **loss**: Funkcja straty (gdy dostępna)
- **td_error**: Temporal Difference error

### Evaluation Metrics (po każdej ewaluacji)
- **eval_num**: Numer ewaluacji
- **step**: Na jakim kroku
- **timestamp**: Czas od startu
- **return_value**: Całkowity reward z epizodu
- **delta_from_prev**: Zmiana zwrotu względem poprzedniej ewaluacji
- **badges_collected**: Liczba zdobytych badge'ów
- **episode_length**: Długość epizodu

## Automatyczne Rekomendacje

System analizuje metryki i proponuje konkretne ulepszenia:

### Jeśli returns < 0.1
```
Issue: Model achieving only negative returns
Recommendation: Increase exploration or decrease penalty
Action: Add --workers 8 or modify base_reward
```

### Jeśli negatywny trend delta
```
Issue: Learning rate too high (negative delta trend)
Recommendation: Reduce learning rate from 5e-4 to 1e-4
Action: Edit TrainWorker.py line 77
```

### Jeśli buffer < 50% po sesji
```
Issue: Buffer only 30% full after session
Recommendation: Run longer or increase workers
Action: ./train.py --workers 10
```

### Jeśli low throughput
```
Issue: Low throughput: 3.2 steps/sec
Recommendation: Check GPU or reduce batch_size
Action: Check nvidia-smi or modify batch_size
```

### Jeśli high throughput
```
Issue: None
Recommendation: High throughput! Can increase batch_size
Action: Modify batch_size in TrainWorker.py
```

## Usage

### Trening z automatyczną analizą
```bash
./venv/Scripts/python train.py --eval-gui
# Po wciśnięciu Ctrl+C zobaczy metryki i rekomendacje
```

### Analiza zapisanych metryk
```bash
# Analiza ostatniego treningu
./venv/Scripts/python analyze_metrics.py --latest

# Analiza konkretnego pliku
./venv/Scripts/python analyze_metrics.py --file logs/metrics_20260610_150000.json
```

## Output Example

```
======================================================================
  Training Analysis & Recommendations
======================================================================

Evaluation Statistics:
  Total evals:    15
  Best return:    0.3468
  Worst return:   -0.1424
  Avg return:     -0.0425
  Std deviation:  0.1234
  Avg delta:      +0.0156

Issues Found (1):
  ⚠ Low optimization frequency

Recommendations (2):
  1. Increase buffer capacity or reduce batch_size for more frequent updates
  2. Learning rate good! Can increase for faster convergence

======================================================================
```

## Metrics Files

Metryki zapisywane automatycznie do:
```
logs/metrics_YYYYMMDD_HHMMSS.json
```

Format JSON:
```json
{
  "session": {
    "start_time": "2026-06-10T18:00:00",
    "total_steps": 11079,
    "total_opt_steps": 10993,
    "avg_steps_per_sec": 12.3
  },
  "step_metrics": [...],
  "eval_metrics": [...],
  "issues": [...],
  "recommendations": [...]
}
```

## Interpreting Recommendations

| Recommendation | Means | Action |
|---|---|---|
| "Negative delta trend" | Learning rate too high | Reduce lr: 5e-4 → 1e-4 |
| "Low throughput" | GPU underutilized | Check GPU / reduce batch |
| "Buffer slow to fill" | Workers can't keep up | Add --workers 8+ |
| "High throughput" | Good GPU utilization | Can increase batch_size |
| "Only negative returns" | Model not learning | More exploration, easier task |
| "Already converged" | Learning is done | Can reduce epsilon |

## Next Steps

1. Run training: `./train.py --eval-gui`
2. Let it run 30+ minutes
3. Press Ctrl+C
4. Read recommendations
5. Apply suggestions:
   - Modify command line args
   - Edit hyperparameters
   - Run again
6. Track progress with: `analyze_metrics.py --latest`

## Data-Driven Improvement Loop

```
1. Run training with metrics
   ↓
2. Get automatic recommendations
   ↓
3. Apply ONE change at a time
   ↓
4. Re-run and compare metrics
   ↓
5. Keep changes that improve, revert others
   ↓
6. Repeat with next recommendation
```

This ensures we improve systematically based on actual data, not guesses!
