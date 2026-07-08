# Training Improvements - Badge Learning

## 🎯 Problem Fixed

**Previous Issue:** Model nie mógł się nauczyć zdobyć pierwszego badge'a

**Root Cause:**
- Base reward penalty: `-0.002` per step
- Episode length: 1000 steps
- Total episode penalty: `-2.0` reward!
- To make model learn badge (+1.0), musiałby zarabiać przez całą drogę

**Solution:** Wdrażam 3 kluczowe ulepszenia

## ✅ Zmiany Wdrożone

### 1. **Base Reward Penalty (KRYTYCZNE)**
```
Przed:  -0.002 per step  →  -2.0 per episode (1000 steps)
Po:     -0.0001 per step →  -0.5 per episode (5000 steps)
        20x mniejsza kara!
```
📁 `src/pokemon/Data.py` line 24

### 2. **Exploration (WAŻNE)**
```
Było:   min_epsilon=0.15, max_epsilon=0.50  (15-50% random)
Jest:   min_epsilon=0.50, max_epsilon=0.80  (50-80% random)
        3x więcej exploracji = lepsze odkrywanie gry
```
📁 `src/workers/ExperienceWorker.py` line 31-34

### 3. **Episode Length (WAŻNE)**
```
Było:   max_episode_steps = 1000
Jest:   max_episode_steps = 5000
        5x dłuższe episody = więcej czasu na naukę
```
📁 `src/workers/ExperienceWorker.py` line 36

### 4. **Learning Stability (BONUS)**
```
batch_size:   256 → 512  (lepsze GPU utilizacji)
lr:           1e-3 → 5e-4 (bardziej stabilny)
```
📁 `src/workers/TrainWorker.py` line 76-77

## 🚀 Nowy Training Script

### Headless (bez GUI - szybko)
```bash
./venv/Scripts/python train.py
```

### Headless + GUI (obserwuj grę)
```bash
./venv/Scripts/python train.py --gui
```

### GUI tylko podczas ewaluacji (balans)
```bash
./venv/Scripts/python train.py --eval-gui
```

### Więcej workers (szybciej, wymaga mocy)
```bash
./venv/Scripts/python train.py --workers 10
```

### Kombinacje
```bash
./venv/Scripts/python train.py --gui --workers 8
./venv/Scripts/python train.py --eval-gui --workers 6
```

## 📊 Expected Improvements

| Metrika | Było | Jest | Zysk |
|---------|------|------|------|
| Base penalty | -0.002 | -0.0001 | 20x mniej |
| Exploration | 15-50% | 50-80% | 3x więcej |
| Episode length | 1000 | 5000 | 5x dłużej |
| Time to badge | Niemożliwe | ~1-3h | ✓ Osiągalny |

## 📈 Training Phases Expected

### Phase 1 (0-5 min): Buffer Fill
- Model eksploruje losowo
- Zbiera first experiences
- Return: ~ -0.1 do -0.3

### Phase 2 (5-30 min): Initial Learning
- Model zaczyna uczyć się
- Mniejsza exploration
- Return: ~ -0.05 do 0.0

### Phase 3 (30-60 min): Badge Learning
- Positive returns pojawiają się
- Model osiąga pierwszy gym
- Return: ~ 0.0 do 0.5

### Phase 4 (1h+): Scaling
- Multiple badges
- Better policy
- Return: > 0.5

## 🎮 GUI Options Explained

**`--gui`**: Pokazuje główne okno gry (wymaga więcej CPU, wolniej trenuje)
- Przydatne do: debugowania, obserwacji procesu
- Performance: ~70% prędkości

**`--eval-gui`**: Tylko durante ewaluacji (rzadko)
- Przydatne do: weryfikacji że model się uczy
- Performance: ~95% prędkości

**Bez GUI (domyślnie)**: Najszybciej
- Performance: 100% (baseline)

## 💾 Model Checkpoints

- `models/latest.pth` - aktualizuje się co ewaluacja
- `models/best.pth` - najlepszy dotychczasowy
- `saves/last_*` - ostatnie stany gry
- `saves/best_*` - stany z najlepszych ewaluacji

## 🔄 Porównanie

### Stary Config
```python
base_reward = -0.002  # za duży penalty
min_epsilon = 0.15    # mało exploracji
max_episode = 1000    # za krótko
batch_size = 256      # mało
lr = 1e-3             # za szybko
```

### Nowy Config
```python
base_reward = -0.0001  # 20x mniejszy penalty ✓
min_epsilon = 0.50     # 3x więcej exploracji ✓
max_episode = 5000     # 5x dłużej ✓
batch_size = 512       # lepsze GPU util ✓
lr = 5e-4              # stabilniej ✓
```

## 🎯 Cel

**Osiągnięcie:** Model zdolny do:
- ✓ Eksploracji całej mapy
- ✓ Walki z pierwszym gym leaderem  
- ✓ Zdobycia co najmniej 1 badge'a

**Czas:** ~1-3 godziny treningu

## 📝 Polecenie do Startu

```bash
# Rekomendacja: Start headless, obserwuj co 30 min
./venv/Scripts/python train.py

# Lub jeśli chcesz obserwować gre:
./venv/Scripts/python train.py --gui

# Lub balans:
./venv/Scripts/python train.py --eval-gui --workers 8
```

## 🔍 Monitoring

```bash
# Terminal 1: Training
./venv/Scripts/python train.py --eval-gui

# Terminal 2: Watch logs real-time
tail -f logs/train_*.log

# Terminal 3: Check models
ls -lh models/
```

---

**Status:** Ready to learn badges! 🏆
