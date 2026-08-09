"""Tile PyBoy SDL windows across the primary monitor work area."""
from __future__ import annotations

import math
from dataclasses import dataclass

GB_WIDTH = 160
GB_HEIGHT = 144
# Approximate Windows window chrome so tiles don't spill into the taskbar.
_FRAME_W = 16
_FRAME_H = 40


@dataclass(frozen=True)
class WindowSlot:
    scale: int
    x: int
    y: int
    cols: int
    rows: int


def get_work_area() -> tuple[int, int, int, int]:
    """Return (left, top, right, bottom) of the primary monitor work area."""
    try:
        import ctypes
        from ctypes import wintypes

        rect = wintypes.RECT()
        if ctypes.windll.user32.SystemParametersInfoW(48, 0, ctypes.byref(rect), 0):
            return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
    except Exception:
        pass
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w, h = int(root.winfo_screenwidth()), int(root.winfo_screenheight())
        root.destroy()
        return 0, 0, w, h
    except Exception:
        return 0, 0, 1920, 1080


def _best_grid(n: int, area_w: int, area_h: int) -> tuple[int, int, int]:
    """Pick (cols, rows, scale) maximizing scale for ``n`` Game Boy windows."""
    best: tuple[tuple, int, int, int] | None = None
    for cols in range(1, n + 1):
        rows = math.ceil(n / cols)
        cell_w = area_w // cols
        cell_h = area_h // rows
        usable_w = max(GB_WIDTH, cell_w - _FRAME_W)
        usable_h = max(GB_HEIGHT, cell_h - _FRAME_H)
        scale = max(1, min(usable_w // GB_WIDTH, usable_h // GB_HEIGHT))
        empty = cols * rows - n
        squareness = -abs(cols - rows)
        score = (scale, -empty, squareness)
        if best is None or score > best[0]:
            best = (score, cols, rows, scale)
    assert best is not None
    _, cols, rows, scale = best
    return cols, rows, scale


def window_slot(rank: int, n_windows: int = 1) -> WindowSlot:
    """Compute scale + top-left position for worker ``rank`` of ``n_windows``."""
    n = max(1, int(n_windows))
    rank = max(0, min(int(rank), n - 1))
    left, top, right, bottom = get_work_area()
    area_w = max(GB_WIDTH, right - left)
    area_h = max(GB_HEIGHT, bottom - top)
    cols, rows, scale = _best_grid(n, area_w, area_h)
    cell_w = area_w // cols
    cell_h = area_h // rows
    col = rank % cols
    row = rank // cols
    win_w = GB_WIDTH * scale
    win_h = GB_HEIGHT * scale
    x = left + col * cell_w + max(0, (cell_w - win_w - _FRAME_W) // 2)
    y = top + row * cell_h + max(0, (cell_h - win_h - _FRAME_H) // 2)
    return WindowSlot(scale=scale, x=x, y=y, cols=cols, rows=rows)


def _sdl_main_window():
    """Resolve the SDL2 window pointer (Cython PyBoy hides ``_window``)."""
    try:
        import sdl2
    except ImportError:
        return None
    # PyBoy's SDL plugin creates the game window as ID 1.
    window = sdl2.SDL_GetWindowFromID(1)
    return window if window else None


def apply_pyboy_window(
    pyboy,
    *,
    x: int | None = None,
    y: int | None = None,
    title: str | None = None,
) -> None:
    """Move / retitle an already-created SDL2 PyBoy window."""
    try:
        import sdl2
    except ImportError:
        return

    window = _sdl_main_window()
    if window is None:
        return

    if x is not None and y is not None:
        try:
            sdl2.SDL_SetWindowPosition(window, int(x), int(y))
        except Exception:
            pass

    if not title:
        return

    encoded = title.encode("utf-8", errors="replace")
    try:
        sdl2.SDL_SetWindowTitle(window, encoded)
    except Exception:
        return

    # PyBoy resets the title every tick; keep our label afterwards.
    if getattr(pyboy, "_prai_title_patched", False):
        return
    try:
        orig = pyboy._update_window_title

        def _update_window_title():
            orig()
            win = _sdl_main_window()
            if win is not None:
                sdl2.SDL_SetWindowTitle(win, encoded)

        pyboy._update_window_title = _update_window_title
        pyboy._prai_title_patched = True
    except Exception:
        pass
