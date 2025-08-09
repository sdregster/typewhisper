"""Пакет CLI для локального диктовщика на Whisper."""

from __future__ import annotations

try:
    from importlib.metadata import version as _version

    __version__ = _version("typewhisper")
except Exception:
    __version__ = "0.0.0"

__all__ = ["__version__"]
