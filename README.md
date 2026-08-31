# hermes-skill-mlx-whisper

Скилл [Hermes Agent](https://hermes-agent.nousresearch.com/docs) для локального распознавания речи (ru/en) на Apple Silicon через MLX Whisper. Всё считается на машине: ни аудио, ни текст не покидают компьютер.

## Что умеет

- **Локальный STT** — whisper-podlodka-turbo fp16 (специализирована на русском, база bond005/whisper-podlodka-turbo, MLX-конверсия evilfreelancer); английский тоже поддерживается
- **Полный пайплайн `vad_transcribe.py`**: Silero VAD (паузы/шумы отрезаются) → нарезка на сегменты ≤28 с → Whisper → LLM-коррекция терминов
- **LLM-коррекция терминов** — glm-5.3-flash (`reasoning_effort=low`): regex-препасс по встроенному словарю + LLM-чанки ~1200 слов × 10 потоков с word-diff верификацией; без ключа честно деградирует до regex-результата
- **Готовый Markdown** — транскрипт блоками ~60 с (`**mm:ss** текст`), метаданные (дата, источник, длительность, модель, время этапов) + таблица сегментов; файл `~/result-mlx-whisper/YYYY-MM-DD_<имя>.md`
- **Субтитры и форматы** — srt/vtt/txt/tsv/json, таймстампы слов, перевод ru→en (`--task translate`)
- **Экономия RAM** — load-run-exit: модель занимает память только на время процесса (~1.85 ГБ пик), демонов нет

Измеренная скорость (Mac Mini M2): 15.7 с русской речи → чистый текст за 4.6 с; 43-минутная запись → ~12 минут полной обработки с VAD и LLM-коррекцией.

## Установка на чистый Mac

Требуется Apple Silicon (MLX не работает на Intel).

### 1. uv, ffmpeg

```bash
brew install uv ffmpeg
uv tool install mlx-whisper
```

`uv tool install` ставит CLI `mlx_whisper` в изолированное окружение (~/.local/share/uv/tools/mlx-whisper/). Обновление: `uv tool upgrade mlx-whisper`.

### 2. Модель whisper-podlodka-turbo fp16 (1.5 ГБ)

```bash
mkdir -p ~/.local/share/models/whisper-podlodka-turbo-MLX-fp16
cd ~/.local/share/models/whisper-podlodka-turbo-MLX-fp16
curl -LO https://huggingface.co/evilfreelancer/whisper-podlodka-turbo-MLX/resolve/main/fp16/config.json
curl -LO https://huggingface.co/evilfreelancer/whisper-podlodka-turbo-MLX/resolve/main/fp16/weights.safetensors
```

Важно: модель передаётся в скрипты **путём к папке**, не HF-id — работает офлайн и без сюрпризов кэша HuggingFace. Если нужно меньше RAM — в том же репозитории есть `q4/` и `q8/`.

### 3. Окружение VAD-пайплайна

Основной скрипт `vad_transcribe.py` использует Silero VAD:

```bash
uv venv ~/.local/share/stt-vad/venv
uv pip install --python ~/.local/share/stt-vad/venv/bin/python onnxruntime numpy
mkdir -p ~/.local/share/models/silero-vad
curl -L -o ~/.local/share/models/silero-vad/silero_vad.onnx \
  https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx
```

### 4. Скилл в Hermes Agent

```bash
mkdir -p ~/.hermes/skills/media
git clone https://github.com/vokasug/hermes-skill-mlx-whisper ~/.hermes/skills/media/mlx-whisper
```

### 5. Опционально: LLM-коррекция терминов

Ключ GLM (z.ai) в `~/.hermes/.env`:

```
GLM_API_KEY=sk-...
```

Без ключа пайплайн работает — просто пропускает LLM-этап и фиксирует это в MD.

### 6. Проверка

```bash
~/.local/share/uv/tools/mlx-whisper/bin/python \
  ~/.hermes/skills/media/mlx-whisper/scripts/vad_transcribe.py <аудио-файл> --language ru
```

Результат: `~/result-mlx-whisper/YYYY-MM-DD_<имя>.md`.

## Использование

Основной путь — полный пайплайн (запускать именно питоном uv-tool, там mlx):

```bash
~/.local/share/uv/tools/mlx-whisper/bin/python \
  ~/.hermes/skills/media/mlx-whisper/scripts/vad_transcribe.py <аудио> [ещё...] --language ru
```

- `--language ru` указывать явно — на коротких клипах авто-детект иногда ошибается
- `--no-llm` — пропустить LLM-коррекцию
- `--terms "Имя, Ещё Имя"` — канонические написания терминов сверх встроенного словаря

Быстрый MD без VAD и LLM:

```bash
python3 ~/.hermes/skills/media/mlx-whisper/scripts/transcribe_to_md.py <аудио>
```

Сырой mlx_whisper (srt, перевод, отладка):

```bash
# субтитры + таймстампы слов
mlx_whisper --model ~/.local/share/models/whisper-podlodka-turbo-MLX-fp16 \
  --language ru --condition-on-previous-text False \
  --output-format srt --word-timestamps True --output-dir /tmp/stt <аудио>

# перевод ru→en (обратно en→ru модель не умеет)
mlx_whisper ... --task translate <аудио>
```

В сырых вызовах всегда передавайте `--condition-on-previous-text False` — убирает галлюцинации-петли на тишине/музыке (проверено экспериментом). Подробные подводные камни — в [SKILL.md](SKILL.md).

## Настройка под себя

Скрипты содержат пути, захардкоженные под конкретную машину: `OUT_DIR` (`/Users/alexander/result-mlx-whisper`) в `scripts/vad_transcribe.py` и `scripts/transcribe_to_md.py`. Пути моделей и VAD-окружения строятся от `Path.home()` — они переносимы. На своём Mac замените `OUT_DIR` на свой.

## Структура репозитория

```
├── README.md                  # этот файл
├── LICENSE                    # MIT
├── SKILL.md                   # скилл: frontmatter + инструкции для агента
└── scripts/
    ├── vad_transcribe.py      # полный пайплайн: VAD → STT → LLM-коррекция → MD
    ├── vad_segments.py        # Silero VAD → интервалы речи
    └── transcribe_to_md.py    # быстрый путь без VAD/LLM
```

## Лицензия

MIT — см. [LICENSE](LICENSE).
