import random
import numpy as np
from pokemon.SumTree import SumTree


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha=0.6):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.epsilon = 0.01
        self._max_priority = 1.0

    def add(self, data):
        # Running max instead of np.max(tree[-capacity:]) on every insert — that
        # scan is O(capacity) and was re-run for every single experience pushed
        # by every actor, holding buffer_lock the whole time.
        self.tree.add(self._max_priority, data)

    def recompute_max_priority(self):
        """Call once after restoring tree state from disk, since the running
        max isn't persisted — a single O(capacity) scan here is fine."""
        max_p = float(np.max(self.tree.tree[-self.tree.capacity:]))
        self._max_priority = max_p if max_p > 0 else 1.0

    def sample(self, n: int, beta=0.4):
        batch = []
        idxs = []
        segment = self.tree.total_p / n
        priorities = []

        self.beta = beta

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_leaf(s)

            if data is None:
                (idx, p, data) = self.tree.get_leaf(a)

            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = np.array(priorities) / self.tree.total_p
        is_weights = np.power(self.tree.capacity * sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()

        return batch, idxs, is_weights

    def update_priorities(self, idxs: list[int], errors: list[float]):
        for idx, error in zip(idxs, errors):
            p = (error + self.epsilon) ** self.alpha
            self.tree.update(idx, p)
            if p > self._max_priority:
                self._max_priority = p

    def __len__(self) -> int:
        return self.tree.n_entries
