from __future__ import annotations

import logging


def send_slot_hotkey(hotkey: str, logger: logging.Logger) -> None:
    """Send one deck-slot hotkey. Prefer pynput on Windows; fall back to pyautogui."""
    try:
        from pynput.keyboard import Controller

        controller = Controller()
        if len(hotkey) == 1:
            controller.press(hotkey)
            controller.release(hotkey)
        else:
            controller.type(hotkey)
        return
    except ModuleNotFoundError:
        logger.warning("keyboard_backend reason=pynput_not_installed fallback=pyautogui")
    except Exception as exc:
        logger.warning("keyboard_backend reason=pynput_failed err=%s fallback=pyautogui", exc)

    import pyautogui

    pyautogui.press(hotkey)
