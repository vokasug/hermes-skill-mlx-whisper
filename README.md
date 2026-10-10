# hermes-skill-mlx-whisper

Скилл [Hermes Agent](https://hermes-agent.nousresearch.com/docs) для локального распознавания речи на Apple Silicon через MLX Whisper. Всё считается на машине: ни аудио, ни текст не покидают компьютер.

## Что умеет

- **Локальный STT** — модель выбирается автоматически по языку: русский — whisper-podlodka-turbo q8 (специализирована на русском, база bond005/whisper-podlodka-turbo, MLX-конверсия evilfreelancer), все остальные языки — whisper-large-v3-turbo-8bit (mlx-community)
- **Пайплайн `vad_transcribe.py`**: Silero VAD (паузы/шумы отрезаются) → нарезка на сегменты ≤28 с → Whisper → готовый Markdown. Один вызов — один файл `~/result-mlx-whisper/YYYY-MM-DD_<имя>.md`: шапка с контент-метаданными (`--meta-*`) и транскрипт блоками ~60 с (`**mm:ss** текст`). Коррекции текста в скрипте нет — правки ослышек и терминов вносит вызывающий агент в своём пайплайне (например, brain-summary)
- **Субтитры и форматы** — srt/vtt/txt/tsv/json, таймстампы слов, перевод ru→en (`--task translate`) через сырой CLI
- **Экономия RAM** — load-run-exit: модель занимает память только на время процесса (~1.85 ГБ пик), демонов нет

## Установка на чистый Mac

Требуется Apple Silicon (MLX не работает на Intel).

### 1. uv, ffmpeg

```bash
brew install uv ffmpeg
uv tool install mlx-whisper
```

`uv tool install` ставит CLI `mlx_whisper` в изолированное окружение (~/.local/share/uv/tools/mlx-whisper/). Обновление: `uv tool upgrade mlx-whisper`.

### 2. Модель whisper-podlodka-turbo q8 (824 МБ)

```bash
mkdir -p ~/.local/share/models/whisper-podlodka-turbo-MLX-q8
cd ~/.local/share/models/whisper-podlodka-turbo-MLX-q8
curl -LO https://huggingface.co/evilfreelancer/whisper-podlodka-turbo-MLX/resolve/main/q8/config.json
curl -LO https://huggingface.co/evilfreelancer/whisper-podlodka-turbo-MLX/resolve/main/q8/weights.safetensors
```

Важно: модель передаётся в скрипты **путём к папке**, не HF-id — работает офлайн и без сюрпризов кэша HuggingFace. q8 быстрее fp16 при совпадении текста 99.8%. Если нужна максимальная точность или ещё меньше RAM — в том же репозитории есть `fp16/` и `q4/`.

### 2b. Модель whisper-large-v3-turbo-8bit (~809 МБ) — для всех языков кроме русского

```bash
mkdir -p ~/.local/share/models/whisper-large-v3-turbo-8bit
cd ~/.local/share/models/whisper-large-v3-turbo-8bit
curl -LO https://huggingface.co/mlx-community/whisper-large-v3-turbo-8bit/resolve/main/config.json
# в репозитории веса называются model.safetensors — сохраняем как weights.safetensors (такого имени ждёт mlx_whisper)
curl -L -o weights.safetensors https://huggingface.co/mlx-community/whisper-large-v3-turbo-8bit/resolve/main/model.safetensors
```

Без этой папки транскрибация не-ru языков упадёт на старте с понятной ошибкой.

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
mkdir -p $HERMES_HOME/skills/media
git clone https://github.com/vokasug/hermes-skill-mlx-whisper $HERMES_HOME/skills/media/mlx-whisper
```

### 5. Проверка

```bash
~/.local/share/uv/tools/mlx-whisper/bin/python \
  $HERMES_HOME/skills/media/mlx-whisper/scripts/vad_transcribe.py <аудио-файл> --language ru
```

Результат: `~/result-mlx-whisper/YYYY-MM-DD_<имя>.md`.

## Использование

Основной путь — пайплайн VAD+STT (запускать именно питоном uv-tool, там mlx):

```bash
~/.local/share/uv/tools/mlx-whisper/bin/python \
  $HERMES_HOME/skills/media/mlx-whisper/scripts/vad_transcribe.py <аудио> [ещё...] --language ru \
  [--meta-title "..." --meta-author "..." --meta-date YYYY-MM-DD --meta-url "..."]
```

- `--language ru` указывать явно — на коротких клипах авто-детект иногда ошибается. От языка зависит и модель: `ru` → podlodka q8, любой другой → whisper-large-v3-turbo-8bit; `--model <папка>` перекрывает автовыбор
- `--meta-*` — контент-метаданные шапки MD (название, автор, дата публикации, каноническая ссылка); пустые значения запрещены (скрипт падает)
- `--debug-segments` — дополнительно пишет `<имя>.segments.json` (`[{start, end, text, logprob}]`) — контракт для скиллов-потребителей сегментов
- `--no-correct` — флаг-заглушка для совместимости вызовов, ни на что не влияет
- `--for <brain-summary|multilingual-audio-replacement|translated-video-subtitles>` — специфика запуска под скилл-потребителя одним флагом; что делает каждый режим — в [SKILL.md](SKILL.md)

Быстрый MD без VAD (сплошной текст + таблица сегментов):

```bash
python3 $HERMES_HOME/skills/media/mlx-whisper/scripts/transcribe_to_md.py <аудио>
```

Сырой mlx_whisper (srt, перевод, отладка):

```bash
# субтитры + таймстампы слов
mlx_whisper --model ~/.local/share/models/whisper-podlodka-turbo-MLX-q8 \
  --language ru --condition-on-previous-text False \
  --output-format srt --word-timestamps True --output-dir /tmp/stt <аудио>

# перевод ru→en (обратно en→ru модель не умеет)
mlx_whisper ... --task translate <аудио>
```

В сырых вызовах всегда передавайте `--condition-on-previous-text False` — убирает галлюцинации-петли на тишине/музыке. Подробные подводные камни — в [SKILL.md](SKILL.md).

## Настройка под себя

Скрипты пишут вывод в `~/result-mlx-whisper` — путь строится от домашней папки текущего пользователя (`OUT_DIR` в `scripts/vad_transcribe.py` и `scripts/transcribe_to_md.py` — от `pathlib.Path.home()`), ничего менять не нужно. Пути моделей и VAD-окружения тоже строятся от `Path.home()` — они переносимы.

## Структура репозитория

```
├── README.md                  # этот файл
├── LICENSE                    # MIT
├── SKILL.md                   # скилл: frontmatter + инструкции для агента
└── scripts/
    ├── vad_transcribe.py      # пайплайн: VAD → STT → MD; --debug-segments
    ├── vad_segments.py        # Silero VAD → интервалы речи
    └── transcribe_to_md.py    # быстрый путь без VAD
```

## Лицензия

MIT — см. [LICENSE](LICENSE).
