import pickle
import numpy as np
from pathlib import Path
from pokemon.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from pokemon.SumTree import SumTree


class BufferManager:
    """Handles saving and loading of experience replay buffer to/from disk"""

    def __init__(self, buffer_dir: str = "checkpoints/buffer"):
        self.buffer_dir = Path(buffer_dir)
        self.buffer_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_path = self.buffer_dir / "replay_buffer.pkl"

    def save_buffer(self, buffer: PrioritizedReplayBuffer) -> bool:
        """Save buffer to disk. Returns True if successful."""
        try:
            buffer_state = {
                'tree_data': buffer.tree.data.copy(),
                'tree_array': buffer.tree.tree.copy(),
                'tree_capacity': buffer.tree.capacity,
                'tree_data_pointer': buffer.tree.data_pointer,
                'tree_n_entries': buffer.tree.n_entries,
                'alpha': buffer.alpha,
                'epsilon': buffer.epsilon,
            }
            with open(self.buffer_path, 'wb') as f:
                pickle.dump(buffer_state, f)
            return True
        except Exception as e:
            print(f"Error saving buffer: {e}")
            return False

    def load_buffer(self, capacity: int, alpha: float = 0.6) -> PrioritizedReplayBuffer | None:
        """Load buffer from disk. Returns None if file doesn't exist or fails."""
        if not self.buffer_path.exists():
            return None

        try:
            with open(self.buffer_path, 'rb') as f:
                buffer_state = pickle.load(f)

            buffer = PrioritizedReplayBuffer(capacity=capacity, alpha=alpha)

            buffer.tree.data = buffer_state['tree_data']
            buffer.tree.tree = buffer_state['tree_array']
            buffer.tree.data_pointer = buffer_state['tree_data_pointer']
            buffer.tree.n_entries = buffer_state['tree_n_entries']
            buffer.epsilon = buffer_state['epsilon']

            return buffer
        except Exception as e:
            print(f"Error loading buffer: {e}")
            return None

    def delete_buffer(self) -> bool:
        """Delete saved buffer file."""
        try:
            if self.buffer_path.exists():
                self.buffer_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting buffer: {e}")
            return False

    def buffer_exists(self) -> bool:
        """Check if buffer file exists."""
        return self.buffer_path.exists()

    def get_buffer_size(self) -> int:
        """Get size of buffer file in MB."""
        if self.buffer_path.exists():
            return self.buffer_path.stat().st_size / (1024 * 1024)
        return 0
