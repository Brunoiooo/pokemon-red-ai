"""Unit checks for prompt-arrow dialog helpers (no emulator required)."""
import sys
import tempfile
from multiprocessing import RLock
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pokemon.Data import Data


def _make_data() -> Data:
    return Data(pyboy=MagicMock(), files_lock=RLock())


def _blank_tilemap() -> bytearray:
    mem = bytearray(0xC508)
    for i in range(0xC3A0, 0xC508):
        mem[i] = 0x7F
    return mem


def test_arrow_on_and_body_hash_ignores_blink():
    data = _make_data()
    mem = _blank_tilemap()

    assert data.prompt_arrow_on(mem) is False
    h0 = data.textbox_body_hash(mem)

    mem[Data.PROMPT_ARROW_ADDR] = Data.PROMPT_ARROW_TILE
    assert data.prompt_arrow_on(mem) is True
    assert data.textbox_body_hash(mem) == h0, "blink must not change textbox_body_hash"

    mem[0xC3A0 + 14 * 20 + 2] = 0x80
    assert data.textbox_body_hash(mem) != h0


def test_truncate_uses_prompt_fuse_not_visited_dialogs():
    data = _make_data()
    data.visited_dialogs[(1, 0)] = 9999
    data.useless_prompt_ticks = 0
    data.useless_printing_ticks = 0
    assert data.truncated_mode(b"") is None

    data.useless_prompt_ticks = data.max_useless_dialog_ticks
    assert data.truncated_mode(b"") == "dialog"

    data.useless_prompt_ticks = 0
    data.useless_printing_ticks = data.max_useless_printing_ticks
    assert data.truncated_mode(b"") == "dialog"


def test_load_resets_prompt_fuse():
    data = _make_data()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        data.visited_dialogs[(3, 1)] = 1
        data.awaiting_prompt = True
        data.useless_prompt_ticks = 400
        data.save(str(path))

        data2 = _make_data()
        data2.useless_prompt_ticks = 999
        data2.awaiting_prompt = True
        data2.load(str(path))
        assert data2.visited_dialogs.get((3, 1)) == 1
        assert data2.useless_prompt_ticks == 0
        assert data2.awaiting_prompt is False


if __name__ == "__main__":
    test_arrow_on_and_body_hash_ignores_blink()
    test_truncate_uses_prompt_fuse_not_visited_dialogs()
    test_load_resets_prompt_fuse()
    print("ok")
