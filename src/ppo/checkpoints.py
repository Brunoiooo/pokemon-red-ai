"""Shared PPO checkpoint path resolution."""
from __future__ import annotations

from pathlib import Path

MODELS_ROOT = Path("models")
SEARCH_BEST = "models/ppo_*/best/best_model.zip"
SEARCH_LATEST = "models/ppo_*/ppo_latest.zip"


def resolve_model_path(
    model: str | None,
    models_root: Path = MODELS_ROOT,
    *,
    prefer_latest: bool = False,
) -> Path:
    """Resolve a PPO checkpoint path.

    If ``model`` is given, use it. Otherwise pick the newest checkpoint:
    - default: ``models/ppo_*/best/best_model.zip``, else ``ppo_latest.zip``
    - ``prefer_latest=True``: ``ppo_latest.zip`` first, then best
    """
    if model:
        path = Path(model)
        if not path.is_file():
            raise SystemExit(
                f"Model not found: {path}\n"
                f"Pass an existing checkpoint, e.g.\n"
                f"  --resume models/ppo_<timestamp>/ppo_latest.zip\n"
                f"  --model models/ppo_<timestamp>/best/best_model.zip"
            )
        return path

    patterns = (
        ("ppo_*/ppo_latest.zip", "ppo_*/best/best_model.zip")
        if prefer_latest
        else ("ppo_*/best/best_model.zip", "ppo_*/ppo_latest.zip")
    )
    for pattern in patterns:
        found = sorted(
            models_root.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if found:
            return found[0]

    raise SystemExit(
        "No PPO checkpoint found.\n"
        f"Searched:\n"
        f"  {SEARCH_BEST}\n"
        f"  {SEARCH_LATEST}\n"
        "Train first, or pass an explicit path:\n"
        "  python cli.py train --resume models/ppo_<timestamp>/ppo_latest.zip\n"
        "  python cli.py eval --model models/ppo_<timestamp>/best/best_model.zip"
    )
