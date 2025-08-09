"""
Простейший локальный диктовщик на базе faster-whisper.

Горячие клавиши:
  - Ctrl+Alt+H — старт/стоп записи
  - Ctrl+Alt+Q — выход

По завершении распознавания текст копируется в буфер обмена и вставляется в активное окно (Ctrl+V).
"""

from __future__ import annotations

import argparse
import glob
import os
import platform
import shutil
import site
import subprocess
import sys
import threading
import time
import warnings
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # noqa: BLE001
    print("[Ошибка] Не удалось импортировать sounddevice:", exc)
    sys.exit(1)


def _augment_path_for_cuda_dlls() -> None:
    """Добавляет в PATH типичные каталоги с CUDA/cuDNN DLL на Windows.

    Ищем каталоги наподобие:
      - Program Files\\NVIDIA\\CUDNN\\v*\\bin\\*
      - Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin
      - site-packages/nvidia/{cudnn,cublas,cuda_runtime,nvrtc}/bin
    """
    if platform.system() != "Windows":
        return

    candidate_dirs: List[str] = []

    # Program Files cuDNN
    candidate_dirs.extend(glob.glob(r"C:\\Program Files\\NVIDIA\\CUDNN\\v*\\bin\\*"))

    # CUDA Toolkit bin
    candidate_dirs.extend(
        glob.glob(r"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v*\\bin")
    )

    # Python-installed NVIDIA runtime packages
    site_dirs = []
    try:
        site_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        site_dirs.append(site.getusersitepackages())
    except Exception:
        pass
    for base in site_dirs:
        for sub in [
            os.path.join(base, "nvidia", "cudnn", "bin"),
            os.path.join(base, "nvidia", "cublas", "bin"),
            os.path.join(base, "nvidia", "cuda_runtime", "bin"),
            os.path.join(base, "nvidia", "nvrtc", "bin"),
        ]:
            candidate_dirs.append(sub)

    # Уникализируем и фильтруем существующие
    existing = []
    seen = set()
    for d in candidate_dirs:
        if d and os.path.isdir(d) and d not in seen:
            existing.append(d)
            seen.add(d)

    if not existing:
        return

    current = os.environ.get("PATH", "")
    new_path = ";".join(existing + [current]) if current else ";".join(existing)
    os.environ["PATH"] = new_path


try:
    _augment_path_for_cuda_dlls()
except Exception:
    pass

try:
    # Подавляем известный UserWarning из ctranslate2 (pkg_resources deprecation)
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        module=r"ctranslate2(\.|$)",
    )
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message=r"pkg_resources is deprecated as an API.*",
    )
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

    def _callback(
        self, indata: np.ndarray, frames: int, time_info: Dict[str, float], status: Any
    ) -> None:
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


def _resolve_compute_type(device: str, compute: str) -> str:
    """Возвращает подходящий compute_type.

    - auto: cpu→int8, cuda→float16
    - иначе возвращает указанное значение
    """
    if compute.lower() == "auto":
        return "float16" if device.lower() == "cuda" else "int8"
    return compute


def _cuda_is_available() -> bool:
    """Проверяет доступность CUDA для CTranslate2 упрощённо.

    На Windows пытается загрузить ключевые DLL (cudart и cuDNN).
    На других ОС возвращает True, чтобы не блокировать авто-режим.
    """
    if platform.system() != "Windows":
        return True
    try:
        import ctypes  # noqa: WPS433

        for name in ("cudart64_12.dll", "cudnn_ops64_9.dll"):
            try:
                ctypes.WinDLL(name)
            except Exception:
                return False
        return True
    except Exception:
        return False


def _resolve_device(device: str) -> str:
    """Определяет устройство при значении auto."""
    if device.lower() != "auto":
        return device
    return "cuda" if _cuda_is_available() else "cpu"


def build_model(model_name: str, device: str, compute: str) -> WhisperModel:
    """Создаёт и настраивает модель Whisper."""
    resolved_compute = _resolve_compute_type(device, compute)
    print(
        f"[Модель] Загрузка: model={model_name}, device={device}, compute={resolved_compute} ..."
    )
    model = WhisperModel(model_name, device=device, compute_type=resolved_compute)
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


def _safe_run(cmd: List[str]) -> str:
    """Возвращает stdout команды или пустую строку."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return (out.stdout or "").strip()
    except Exception:
        return ""


def run_diagnostics(preferred_device: str, preferred_compute: str) -> None:
    """Печатает диагностику окружения для GPU/CUDA и выходит.

    Показывает базовую системную информацию, наличие `nvidia-smi`,
    доступность CUDA в PyTorch (как индикатор корректной CUDA-цепочки),
    а также рекомендуемые флаги запуска для faster-whisper.
    """
    print("[Diag] Система:")
    print(f"  OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"  Python: {platform.python_version()}")
    print(f"  Preferred device: {preferred_device}")
    resolved = _resolve_compute_type(preferred_device, preferred_compute)
    print(f"  Preferred compute: {preferred_compute} -> {resolved}")

    exe = shutil.which("nvidia-smi")
    print("[Diag] nvidia-smi:")
    if exe:
        out = _safe_run([exe])
        print("  found in PATH")
        if out:
            head = "\n".join(out.splitlines()[:6])
            print("  output:")
            print("  " + head.replace("\n", "\n  "))
    else:
        print("  not found in PATH")

    print("[Diag] PyTorch CUDA:")
    try:
        import torch  # type: ignore

        print(f"  torch: {torch.__version__}")
        print(f"  cuda.is_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                count = torch.cuda.device_count()
                print(f"  cuda.device_count: {count}")
                for idx in range(count):
                    name = torch.cuda.get_device_name(idx)
                    print(f"  device[{idx}]: {name}")
            except Exception:
                pass
    except Exception as exc:  # noqa: BLE001
        print(f"  torch import failed: {exc}")

    print("[Diag] Рекомендации:")
    print("  - Для GPU запустите: --device cuda (compute по умолчанию auto->float16)")
    print(
        "  - Убедитесь, что PyTorch с поддержкой CUDA установлен и совпадает с версией CUDA драйвера"
    )
    print(
        "  - Примеры команд установки PyTorch под CUDA см. обсуждение Whisper (GitHub Discussions #47)"
    )


def _register_hotkeys(on_toggle: Any, on_quit: Any) -> None:
    """Регистрирует глобальные горячие клавиши и обрабатывает возможные ошибки."""
    try:
        keyboard.add_hotkey("ctrl+alt+h", on_toggle, suppress=False)
        keyboard.add_hotkey("ctrl+alt+q", on_quit, suppress=False)
    except Exception as exc:  # noqa: BLE001
        print(
            "[Ошибка] Регистрация горячих клавиш не удалась. Попробуйте запустить PowerShell от имени администратора.\n",
            exc,
        )
        sys.exit(1)


def _print_shortcuts() -> None:
    """Печатает список горячих клавиш."""
    print(
        "\nГотово. Горячие клавиши:\n  Ctrl+Alt+H — старт/стоп записи\n  Ctrl+Alt+Q — выход\n"
    )


def _parse_args() -> argparse.Namespace:
    """Создаёт и парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description="Локальный диктовщик (Whisper)")
    parser.add_argument(
        "--model",
        default="small",
        help="Имя модели (напр. small, base, large-v3, distil-large-v3)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Устройство: auto|cpu|cuda (auto: попытка использовать CUDA, иначе CPU)",
    )
    parser.add_argument(
        "--compute",
        default="auto",
        help="Тип вычислений: auto|int8|float16|int8_float16|float32 (auto: cpu->int8, cuda->float16)",
    )
    parser.add_argument(
        "--lang", default=None, help="Код языка (напр. ru, en); по умолчанию авто"
    )
    parser.add_argument(
        "--vad", action="store_true", help="Фильтровать тишину (Silero VAD)"
    )
    parser.add_argument(
        "--diag",
        action="store_true",
        help="Режим диагностики GPU/окружения и выход",
    )
    return parser.parse_args()


def main() -> None:
    """Точка входа CLI."""
    args = _parse_args()

    if args.diag:
        run_diagnostics(preferred_device=args.device, preferred_compute=args.compute)
        return

    try:
        if args.device.lower() == "auto":
            print("[Устройство] Автоопределение: пробую CUDA…")
            try:
                model = build_model(args.model, "cuda", args.compute)
                print("[Устройство] Выбрано: cuda")
            except Exception as cuda_exc:  # noqa: BLE001
                print("[Устройство] CUDA недоступна, переключаюсь на CPU:", cuda_exc)
                model = build_model(args.model, "cpu", args.compute)
                print("[Устройство] Выбрано: cpu")
        else:
            model = build_model(args.model, _resolve_device(args.device), args.compute)
            print(f"[Устройство] Выбрано: {args.device}")
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

    _register_hotkeys(toggle_recording, quit_app)
    _print_shortcuts()
    # Ожидаем выхода
    keyboard.wait()


if __name__ == "__main__":
    main()
