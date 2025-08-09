## Локальный диктовщик (Whisper, Windows)

Простой оффлайн-аналог Win+H на базе `faster-whisper`.

### Возможности
- Старт/стоп записи по горячей клавише
- Локальное распознавание (Whisper через `faster-whisper`)
- Автовставка текста в активное окно

### Требования
- Windows 10/11, Python 3.10+
- Рекомендуется `uv` для управления зависимостями

Установка `uv` (PowerShell 7+):
```powershell
iwr -useb https://astral.sh/uv/install.ps1 | iex
```

### Установка и запуск (uv)
```powershell
uv sync
uv run typewhisper
# либо
uv run python -m typewhisper
```

### Глобальная установка CLI (uv tool)
Чтобы команда `typewhisper` была доступна из любого каталога:

```powershell
cd C:\python\typewhisper
uv tool install --force .

# проверка
Get-Command typewhisper
typewhisper --help
```

Обновление после изменений в исходниках:

```powershell
cd C:\python\typewhisper
uv tool install --force .
```

Удаление глобальной команды:

```powershell
uv tool uninstall typewhisper
```

Если команда не находится сразу, перезапустите PowerShell окно.

Поддержка GPU (CUDA) настраивается библиотекой `faster-whisper` отдельно; по умолчанию работает на CPU.

### Горячие клавиши
- Ctrl+Alt+H — старт/стоп записи
- Ctrl+Alt+Q — выход

Примечание: для `keyboard` на Windows иногда требуются права администратора для глобальных горячих клавиш.

### Параметры запуска
```powershell
typewhisper --model MODEL --device {cpu|cuda} --compute COMPUTE --lang LANG --vad
```
- `--model` (по умолчанию `small`): `tiny|base|small|medium|large-v3|distil-large-v3` и др.
- `--device` (по умолчанию `cpu`): `cpu` или `cuda`
- `--compute` (CPU: `int8`; GPU: `float16|int8_float16|float32`)
- `--lang` (например `ru`, `en`), по умолчанию авто
- `--vad` — фильтрация тишины (Silero VAD)

### Структура проекта
```
src/
  typewhisper/
    __init__.py
    __main__.py
    cli.py
tests/
  test_smoke.py
pyproject.toml
.gitignore
README.md
```

### Лицензии
Whisper и `faster-whisper` — см. соответствующие репозитории.
