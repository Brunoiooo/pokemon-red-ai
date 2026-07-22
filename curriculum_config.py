"""
Curriculum Learning Configuration for Pokemon Red AI

Defines progressive training stages with different starting checkpoints.
Agent starts from simpler stages and progresses to harder ones.

Stages:
  - stage_0: Start from beginning (Pallet Town)
  - stage_1: After starter selection (unlock first battle, leveling)
  - stage_2: After Badge 1 (more gym battles, exploration)
  - stage_3: After Badge 3 (mid-game progression)

NOTE: ranges are in raw environment *steps* (ExperienceWorker._total_steps),
not episodes — that's the unit get_checkpoint_for_episode()/get_current_stage()
are actually called with from ExperienceWorker.random_save_path. They used to
be labeled/scaled as episode counts (e.g. "0-5000"), which made stage_0 expire
after well under a single episode. Also note that once a worker saves its own
checkpoint (saves/worker_N), ExperienceWorker.random_save_path prefers that over
any curriculum stage — curriculum only applies before a worker has found its
own checkpoint.
"""

CURRICULUM = {
    # stage_0: steps 0-500k, start from "start" checkpoint
    "stage_0": {
        "step_range": (0, 500_000),
        "checkpoint": "start",
        "description": "Beginning: Pallet Town, learn to navigate and select starter",
        "min_reward": 0.5,  # Minimum average episode reward to advance
        "enabled": True,
    },
    # stage_1: steps 500k-2M, start from mid-game checkpoint
    # Requires: creates saves/stage_1 with game progress to first gym
    "stage_1": {
        "step_range": (500_000, 2_000_000),
        "checkpoint": "stage_1",
        "description": "After starter: explore, level up, reach first gym",
        "min_reward": 0.7,
        "enabled": False,  # Set to True once saves/stage_1 exists
    },
    # stage_2: steps 2M+, start from early-mid game
    "stage_2": {
        "step_range": (2_000_000, 10_000_000),
        "checkpoint": "stage_2",
        "description": "After Badge 1: broader exploration, multiple gyms",
        "min_reward": 0.8,
        "enabled": False,  # Set to True once saves/stage_2 exists
    },
}

def get_checkpoint_for_episode(total_steps: int) -> str | None:
    """
    Get appropriate checkpoint for the worker's total environment steps so far.
    Returns checkpoint name, or None if curriculum not applicable.
    """
    for stage_name, config in sorted(CURRICULUM.items(),
                                     key=lambda x: x[1]["step_range"][0],
                                     reverse=True):
        if not config["enabled"]:
            continue
        step_start, step_end = config["step_range"]
        if step_start <= total_steps < step_end:
            return config["checkpoint"]
    return None

def get_current_stage(total_steps: int) -> str | None:
    """Get current curriculum stage name."""
    for stage_name, config in CURRICULUM.items():
        if not config["enabled"]:
            continue
        step_start, step_end = config["step_range"]
        if step_start <= total_steps < step_end:
            return stage_name
    return None
