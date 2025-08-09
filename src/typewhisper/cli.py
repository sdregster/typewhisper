"""
Простейший локальный диктовщик на базе faster-whisper.

Горячие клавиши:
  - Ctrl+Alt+H — старт/стоп записи
  - Ctrl+Alt+Q — выход

По завершении распознавания текст копируется в буфер обмена и вставляется в активное окно (Ctrl+V).
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # noqa: BLE001
    print("[Ошибка] Не удалось импортировать sounddevice:", exc)
    sys.exit(1)

try:
    from faster_whisper import WhisperModel
except Exception as exc:  # noqa: BLE001
    print("[Ошибка] Не удалось импортировать faster-whisper:", exc)
    sys.exit(1)

try:
    import keyboard
except Exception as exc:  # noqa: BLE001
    print(
        "[Ошибка] Не удалось импортировать keyboard:\n"
        "Убедитесь, что скрипт запущен с достаточными правами (иногда нужны права администратора на Windows).\n",
        exc,
    )
    sys.exit(1)

try:
    import pyperclip
except Exception as exc:  # noqa: BLE001
    print("[Ошибка] Не удалось импортировать pyperclip:", exc)
    sys.exit(1)

try:
    import winsound
except Exception:
    winsound = None  # не критично на не-Windows


SAMPLE_RATE = 16000  # Гц, соответствует ожиданиям Whisper
CHANNELS = 1
DTYPE = "float32"


class AudioRecorder:
    """Простейший накопитель аудио из микрофона.

    - Открывает `InputStream` при старте, закрывает при стопе
    - Буферизует кадры в список numpy-массивов
    """

    def __init__(
        self,
        samplerate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        dtype: str = DTYPE,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self.dtype = dtype
        self._lock = threading.Lock()
        self._chunks: List[np.ndarray] = []
        self._stream: Optional[sd.InputStream] = None
        self._active = False

    def _callback(self, indata: np.ndarray, frames: int, time_info: Dict[str, float], status: Any) -> None:
        """Колбэк аудиопотока. Складывает копии входящих кадров в буфер.

        Параметры соответствуют сигнатуре колбэка `sounddevice.InputStream`.
        """
        if status:
            # печатаем предупреждения из драйвера
            print("[sounddevice]", status)
        if not self._active:
            return
        with self._lock:
            # indata.shape == (frames, channels)
            data = np.asarray(indata, dtype=np.float32)
            if data.ndim == 2 and data.shape[1] == 1:
                data = data[:, 0]
            self._chunks.append(data.copy())

    def start(self) -> None:
        """Открывает поток записи и начинает накопление кадров."""
        if self._active:
            return
        self._chunks.clear()
        self._stream = sd.InputStream(
            samplerate=self.samplerate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._callback,
        )
        self._stream.start()
        self._active = True

    def stop(self) -> np.ndarray:
        """Останавливает поток и возвращает накопленную дорожку как `np.ndarray`."""
        if not self._active:
            return np.zeros(0, dtype=np.float32)
        self._active = False
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
        with self._lock:
            if not self._chunks:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(self._chunks, axis=0).astype(np.float32, copy=False)
            self._chunks.clear()
            return audio


def beep(freq: int, dur_ms: int) -> None:
    """Короткий звуковой сигнал (если доступен)."""
    if winsound is None:
        return
    try:
        winsound.Beep(freq, dur_ms)
    except Exception:
        pass


def paste_text_via_clipboard(text: str) -> None:
    """Копирует в буфер и имитирует Ctrl+V для вставки."""
    if not text:
        return
    pyperclip.copy(text)
    time.sleep(0.05)
    keyboard.press_and_release("ctrl+v")


def build_model(model_name: str, device: str, compute: str) -> WhisperModel:
    """Создаёт и настраивает модель Whisper."""
    print(
        f"[Модель] Загрузка: model={model_name}, device={device}, compute={compute} ..."
    )
    model = WhisperModel(model_name, device=device, compute_type=compute)
    print("[Модель] Готово")
    return model


def transcribe_audio(
    model: WhisperModel, audio: np.ndarray, language: Optional[str], use_vad: bool
) -> str:
    """Распознаёт массив 16 кГц float32. Возвращает текст."""
    if audio.size == 0:
        return ""
    # segments — генератор; соберём полностью
    segments, info = model.transcribe(
        audio,
        language=language,
        vad_filter=use_vad,
    )
    try:
        text = " ".join(seg.text.strip() for seg in segments)
    except Exception:
        # если генератор уже исчерпан
        text = ""
    if language is None:
        print(
            f"[Язык] Определён: {getattr(info, 'language', '?')} (p={getattr(info, 'language_probability', 0):.3f})"
        )
    return text.strip()


def main() -> None:
    """Точка входа CLI."""
    parser = argparse.ArgumentParser(description="Локальный диктовщик (Whisper)")
    parser.add_argument(
        "--model",
        default="small",
        help="Имя модели (напр. small, base, large-v3, distil-large-v3)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"], help="Устройство: cpu|cuda"
    )
    parser.add_argument(
        "--compute",
        default="int8",
        help="Тип вычислений (cpu: int8; gpu: float16/int8_float16/float32)",
    )
    parser.add_argument(
        "--lang", default=None, help="Код языка (напр. ru, en); по умолчанию авто"
    )
    parser.add_argument(
        "--vad", action="store_true", help="Фильтровать тишину (Silero VAD)"
    )
    args = parser.parse_args()

    try:
        model = build_model(args.model, args.device, args.compute)
    except Exception as exc:  # noqa: BLE001
        print("[Ошибка] Не удалось инициализировать модель: ", exc)
        sys.exit(1)

    recorder = AudioRecorder()
    state = {"recording": False}

    def toggle_recording() -> None:
        # переключатель по горячей клавише
        if not state["recording"]:
            print("[Запись] Старт… Говорите. Нажмите Ctrl+Alt+H, чтобы закончить.")
            beep(880, 120)
            try:
                recorder.start()
                state["recording"] = True
            except Exception as exc:  # noqa: BLE001
                print("[Ошибка] Не удалось начать запись: ", exc)
        else:
            print("[Запись] Стоп. Распознаю…")
            beep(440, 150)
            audio = recorder.stop()
            state["recording"] = False
            try:
                text = transcribe_audio(model, audio, args.lang, args.vad)
            except Exception as exc:  # noqa: BLE001
                print("[Ошибка] Распознавание завершилось с ошибкой:", exc)
                return
            if text:
                print(f"[Текст] {text}")
                paste_text_via_clipboard(text)
            else:
                print("[Текст] (пусто)")

    def quit_app() -> None:
        print("[Выход] Завершение работы…")
        if state["recording"]:
            try:
                recorder.stop()
            except Exception:
                pass
        # Небольшая задержка, чтобы отпустить модификаторы горячих клавиш
        time.sleep(0.1)
        sys.exit(0)

    # Регистрируем глобальные горячие клавиши
    try:
        keyboard.add_hotkey("ctrl+alt+h", toggle_recording, suppress=False)
        keyboard.add_hotkey("ctrl+alt+q", quit_app, suppress=False)
    except Exception as exc:  # noqa: BLE001
        print(
            "[Ошибка] Регистрация горячих клавиш не удалась. Попробуйте запустить PowerShell от имени администратора.\n",
            exc,
        )
        sys.exit(1)

    print(
        "\nГотово. Горячие клавиши:\n  Ctrl+Alt+H — старт/стоп записи\n  Ctrl+Alt+Q — выход\n"
    )
    # Ожидаем выхода
    keyboard.wait()


if __name__ == "__main__":
    main()
