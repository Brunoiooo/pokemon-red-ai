# Legacy Rainbow DQN

The PPO path (`cli.py train`) is the primary training stack.

This directory marker documents that the following modules belong to the
archived Rainbow DQN pipeline and are kept for reference / ablation only:

- `src/workers/TrainWorker.py`, `ExperienceWorker.py`
- `src/pokemon/ModelPokemon.py` (C51 + Noisy nets)
- `src/pokemon/PrioritizedReplayBuffer.py`, `SumTree.py`
- `src/utils/BufferManager.py`
- `train.py`, `run_eval.py`, `pretrain_bc.py`

Invoke with:

```bash
python cli.py train-dqn
python cli.py eval-dqn --model models/best.pth
```
