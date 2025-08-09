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
typewhisper --model MODEL --device {auto|cpu|cuda} --compute {auto|int8|float16|int8_float16|float32} --lang LANG --vad
```
- `--model` (по умолчанию `small`): `tiny|base|small|medium|large-v3|distil-large-v3` и др.
- `--device` (по умолчанию `auto`): `auto|cpu|cuda` (auto выбирает CUDA при доступности)
- `--compute` (по умолчанию `auto`): `auto|int8|float16|int8_float16|float32` (auto: cpu→int8, cuda→float16)
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


### Модели и очистка скачанных моделей

Рекомендуемые варианты моделей:
- `tiny`/`base`: очень быстро, ниже качество;
- `small`: хороший базовый баланс скорость/качество;
- `distil-large-v3`: лучший баланс (рекомендуется);
- `large-v3`: максимальное качество, требует больше VRAM/времени.

Примеры запуска:
```powershell
typewhisper --model distil-large-v3 --lang ru --vad
# GPU: автоматически выберется при доступности (или укажите явно)
typewhisper --device cuda --model large-v3 --lang ru --vad
```

Где лежат скачанные модели и как их удалить (Windows, PowerShell 7+):
```powershell
# Путь к кэшу Hugging Face Hub (по умолчанию):
$cache = if ($env:HF_HOME) { Join-Path $env:HF_HOME 'hub' } else { Join-Path $env:USERPROFILE '.cache\huggingface\hub' }
$cache

# Посмотреть модели faster-whisper в кэше:
Get-ChildItem -Path $cache -Directory -Depth 1 -Filter 'models--*faster-whisper*' | Select-Object FullName

# Удалить только модели faster-whisper:
Get-ChildItem -Path $cache -Directory -Depth 1 -Filter 'models--*faster-whisper*' | Remove-Item -Recurse -Force

# Полностью очистить кэш Hugging Face (удалит ВСЕ скачанные модели/веса из Hub):
# Внимание: действие необратимо и затронет и другие проекты
# Remove-Item -Path $cache -Recurse -Force
```

Подсказки:
- Можно задать собственный каталог кэша через переменную окружения `HF_HOME`.
- Скачанные модели переиспользуются между виртуальными окружениями.
