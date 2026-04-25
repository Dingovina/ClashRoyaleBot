from __future__ import annotations

import ctypes
import sys


def foreground_title_lower() -> str | None:
    """Return lowercase foreground window title on Windows; None if unavailable."""
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32
    except (AttributeError, OSError):
        return None
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = user32.GetWindowTextLengthW(hwnd)
    if length <= 0:
        return ""
    buffer = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buffer, length + 1)
    return buffer.value.lower()


def foreground_matches(title_lower: str | None, substrings: tuple[str, ...]) -> bool:
    if title_lower is None:
        return True
    if not substrings:
        return True
    return any(s and s in title_lower for s in substrings)
