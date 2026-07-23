"""Shared PPO checkpoint path resolution."""
from __future__ import annotations

import zipfile
from pathlib import Path

MODELS_ROOT = Path("models")
SEARCH_BEST = "models/ppo_*/best/best_model.zip"
SEARCH_LATEST = "models/ppo_*/ppo_latest.zip"
SEARCH_CKPT = "models/ppo_*/ppo_ckpt_*.zip"


def is_valid_zip(path: Path) -> bool:
    """True if ``path`` looks like a readable zip (SB3 checkpoint)."""
    try:
        return path.is_file() and path.stat().st_size > 0 and zipfile.is_zipfile(path)
    except OSError:
        return False


def resolve_model_path(
    model: str | None,
    models_root: Path = MODELS_ROOT,
    *,
    prefer_latest: bool = False,
) -> Path:
    """Resolve a PPO checkpoint path.

    If ``model`` is given, use it. Otherwise pick the newest *valid* checkpoint:
    - default: ``models/ppo_*/best/best_model.zip``, else ``ppo_latest.zip``, else ckpt
    - ``prefer_latest=True``: ``ppo_latest.zip`` first, then best, then ckpt

    Corrupt / empty / non-zip files are skipped (common after interrupted saves).
    """
    if model:
        path = Path(model)
        if not path.is_file():
            raise SystemExit(
                f"Model not found: {path}\n"
                f"Pass an existing checkpoint, e.g.\n"
                f"  --resume models/ppo_<timestamp>/best/best_model.zip\n"
                f"  --resume models/ppo_<timestamp>/ppo_ckpt_*.zip"
            )
        if not is_valid_zip(path):
            raise SystemExit(
                f"Checkpoint is corrupt or not a zip: {path}\n"
                f"Size: {path.stat().st_size} bytes\n"
                f"Try best_model or a ppo_ckpt_* instead:\n"
                f"  python cli.py train --resume {path.parent / 'best' / 'best_model.zip'}"
            )
        return path

    patterns = (
        ("ppo_*/ppo_latest.zip", "ppo_*/best/best_model.zip", "ppo_*/ppo_ckpt_*.zip")
        if prefer_latest
        else ("ppo_*/best/best_model.zip", "ppo_*/ppo_latest.zip", "ppo_*/ppo_ckpt_*.zip")
    )
    skipped: list[str] = []
    for pattern in patterns:
        found = sorted(
            models_root.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for path in found:
            if is_valid_zip(path):
                if skipped:
                    print(
                        f"Note: skipped {len(skipped)} corrupt checkpoint(s); "
                        f"using {path}"
                    )
                return path
            skipped.append(str(path))

    hint = ""
    if skipped:
        hint = "\nCorrupt/empty files skipped:\n  " + "\n  ".join(skipped[:10])
        if len(skipped) > 10:
            hint += f"\n  ... and {len(skipped) - 10} more"

    raise SystemExit(
        "No valid PPO checkpoint found.\n"
        f"Searched:\n"
        f"  {SEARCH_BEST}\n"
        f"  {SEARCH_LATEST}\n"
        f"  {SEARCH_CKPT}"
        f"{hint}\n"
        "Train first, or pass an explicit path:\n"
        "  python cli.py train --resume models/ppo_<timestamp>/best/best_model.zip"
    )


def atomic_model_save(model, path: Path) -> Path:
    """Save SB3 model via temp file then replace (avoids corrupt ppo_latest.zip)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    model.save(str(tmp))
    tmp.replace(path)
    return path
