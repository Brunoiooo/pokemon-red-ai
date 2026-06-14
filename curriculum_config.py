"""
Curriculum Learning Configuration for Pokemon Red AI

Defines progressive training stages with different starting checkpoints.
Agent starts from simpler stages and progresses to harder ones.

Stages:
  - stage_0: Start from beginning (Pallet Town)
  - stage_1: After starter selection (unlock first battle, leveling)
  - stage_2: After Badge 1 (more gym battles, exploration)
  - stage_3: After Badge 3 (mid-game progression)
"""

CURRICULUM = {
    # stage_0: Episodes 0-5000, start from "start" checkpoint
    "stage_0": {
        "episode_range": (0, 5000),
        "checkpoint": "start",
        "description": "Beginning: Pallet Town, learn to navigate and select starter",
        "min_reward": 0.5,  # Minimum average episode reward to advance
        "enabled": True,
    },
    # stage_1: Episodes 5000-15000, start from mid-game checkpoint
    # Requires: creates saves/stage_1 with game progress to first gym
    "stage_1": {
        "episode_range": (5000, 15000),
        "checkpoint": "stage_1",
        "description": "After starter: explore, level up, reach first gym",
        "min_reward": 0.7,
        "enabled": False,  # Set to True once stage_1 save exists
    },
    # stage_2: Episodes 15000+, start from early-mid game
    "stage_2": {
        "episode_range": (15000, 50000),
        "checkpoint": "stage_2",
        "description": "After Badge 1: broader exploration, multiple gyms",
        "min_reward": 0.8,
        "enabled": False,
    },
}

def get_checkpoint_for_episode(episode: int) -> str | None:
    """
    Get appropriate checkpoint for training episode number.
    Returns checkpoint name, or None if curriculum not applicable.
    """
    for stage_name, config in sorted(CURRICULUM.items(),
                                     key=lambda x: x[1]["episode_range"][0],
                                     reverse=True):
        if not config["enabled"]:
            continue
        ep_start, ep_end = config["episode_range"]
        if ep_start <= episode < ep_end:
            return config["checkpoint"]
    return None

def get_current_stage(episode: int) -> str | None:
    """Get current curriculum stage name."""
    for stage_name, config in CURRICULUM.items():
        if not config["enabled"]:
            continue
        ep_start, ep_end = config["episode_range"]
        if ep_start <= episode < ep_end:
            return stage_name
    return None
