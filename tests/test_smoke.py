from __future__ import annotations

"""Простейший smoke-тест импорта пакета.

Проверяет наличие атрибута версии, чтобы убедиться, что пакет собирается.
"""


def test_import() -> None:
    """Импорт пакета и проверка атрибута `__version__`."""
    import typewhisper

    assert hasattr(typewhisper, "__version__")
