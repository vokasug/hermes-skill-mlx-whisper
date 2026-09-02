---
name: mlx-whisper
description: Local ru/en speech-to-text via MLX Whisper; load-run-exit.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [macos]
metadata:
  hermes:
    tags: [stt, transcription, mlx, whisper, audio]
    related_skills: []
---

# MLX Whisper Skill

Локальное распознавание речи (ru/en) на Apple Silicon через mlx-whisper. Оптимум по скорости
и качеству для русского. Загрузка → распознавание → выгрузка: модель занимает RAM только
на время процесса mlx_whisper (~1.85 ГБ пик), демонов и резидентных процессов нет.

## Environment (проверено 2026-08-30)

- CLI: `mlx_whisper` (uv tool `mlx-whisper` 0.4.3, окружение изолировано, обновление: `uv tool upgrade mlx-whisper`)
- Модель: **whisper-podlodka-turbo fp16 (16 бит)** локально в
  `~/.local/share/models/whisper-podlodka-turbo-MLX-fp16/` (config.json + weights.safetensors, 1.5 ГБ)
  Источник: evilfreelancer/whisper-podlodka-turbo-MLX (fp16/), база bond005/whisper-podlodka-turbo
- Тянет ffmpeg для декодирования не-wav аудио; wav 24k mono берёт как есть
- Тест-эталон: 15.7 c русская речь → чистый текст за 4.6 c (RAM 1.85 ГБ)

## When to Use

- «распознай/транскрибируй аудио/голосовое/запись» — любой локальный файл (wav/mp3/m4a/mp4/ogg)
- русская и английская речь; длинные записи; нужны субтитры srt/vtt или таймстампы
- Don't use for: диаризация спикеров (кто говорил), живые стримы, языки кроме ru/en (модель специализирована)

## Quick Reference

**Основной путь — полный пайплайн VAD+LLM (скрипт `vad_transcribe.py`):**

```bash
~/.local/share/uv/tools/mlx-whisper/bin/python \
  ~/.hermes/skills/media/mlx-whisper/scripts/vad_transcribe.py <аудио> [ещё...] --language ru
# результат: /Users/alexander/result-mlx-whisper/YYYY-MM-DD_<имя-аудио>.md
```

Этапы: Silero VAD (паузы/эффекты отрезаются) → нарезка ≤28 с → mlx_whisper.transcribe() в одном
процессе (`condition_on_previous_text=False`) → LLM-коррекция терминов (glm-5.3-flash,
`reasoning_effort=low`: regex-препасс по встроенному словарю гарантированно, затем LLM чанками ~1200
слов ×10 потоков, word-diff верификация, при ошибке чанка — regex-результат) → Канон-термины: словарь mishear +
`--terms "Имя, Ещё Имя"`. Ключ — GLM_API_KEY из ~/.hermes/.env; нет ключа/терминов — honest-degrade.

**Формат MD-результата (спецификация, 2026-08-30):** транскрипт = блоки ~60 с. Новый блок начинается
на предложении, старт которого БЛИЖЕ всего к «старт предыдущего блока + 60 с» (детерминированный
`group_into_blocks`, без LLM: сравнение |старт − target| текущей и следующей границы). Каждый блок —
одна строка `**mm:ss** текст` + ПУСТАЯ СТРОКА после (всегда, без исключений). Хвост <20 с вливается
в предыдущий блок (если результат ≤80 с). Коротких блоков нет по построению; единственный случай
блока >60 с — предложение длиннее 60 с (по правилу «заканчивается ближайшее предложение»).

- `--no-llm` — пропустить коррекцию; `--terms` — добавить канонические написания сверх словаря.
- Быстрый MD без LLM (старый путь, без VAD): `python3 ~/.hermes/skills/media/mlx-whisper/scripts/transcribe_to_md.py <аудио>`
- MD-файл: метаданные (дата, источник, длительность, модель, время этапов, LLM-правки) + полный текст + таблица сегментов.

**Сырой mlx_whisper** (когда MD не нужен — srt, перевод, отладка):

```bash
# субтитры + таймстампы слов
mlx_whisper --model ~/.local/share/models/whisper-podlodka-turbo-MLX-fp16 \
  --language ru --condition-on-previous-text False \
  --output-format srt --word-timestamps True --output-dir /tmp/stt <аудио>

# форматы вывода: txt, vtt, srt, tsv, json, all (json содержит сегменты с вероятностями)
# перевод ru→en: --task translate (обратно en→ru НЕ умеет)
# фрагмент файла: --clip-timestamps "30"  (формат mm:ss CLI НЕ понимает — только секунды)
# промпт-контекст для имён/терминов: --initial-prompt "..." (термины НЕ исправляет, только стиль)
```

## Procedure

1. Определить, что скачиваем (видео/mp3/плейлист/субтитры). Основной путь — скрипт `vad_transcribe.py`
   (VAD → STT → LLM-коррекция, датированный MD в `/Users/alexander/result-mlx-whisper/`).
2. Запустить: `~/.local/share/uv/tools/mlx-whisper/bin/python ~/.hermes/skills/media/mlx-whisper/scripts/vad_transcribe.py <аудио> --language ru`;
   для длинных файлов — `terminal(background=true)` + `process wait`. Скрипт печатает прогресс по этапам и `OK <путь>`.
3. Прочитать созданный MD (`read_file`), показать пользователю текст и список LLM-правок.
4. **ГЕЙТ — определить язык ДО запуска, иначе не запускать вообще** (инцидент 2026-09-02: 39-мин английское видео прогнали с `--language ru` «по привычке» → полкаша, полный перезапуск). Порядок:
   а. Взять язык из самого источника: заголовок/описание (для YouTube — `yt-dlp --print "%(title)s | %(description)s"` — доступно ДО скачивания), метаданные, субтитры.
   б. Если источник неинформативен — прогнать детект на первых 30 с сырым mlx_whisper БЕЗ `--language` (или short-фрагмент) и прочитать `language` из результата.
   в. Перед запуском полного файла убедиться: в команде стоит явный `--language <detected>` и ты можешь назвать, ИЗ КАКОГО ИСТОЧНИКА взят этот язык («так в заголовке», «детект на 30 с дал en»). Не можешь назвать источник — стоп, гейт не пройден.
   г. Язык из памяти о прошлых задачах / привычки / дефолта («обычно всё русское») — ЗАПРЕЩЁН как источник.
5. Нужен srt/перевод/фрагмент — сырой mlx_whisper из Quick Reference (не забывать `--condition-on-previous-text False`).

## Pitfalls

- **Путь модели — папка, не HF-id**: с локальной папкой работает офлайн и без сюрпризов кэша HF.
- `--language ru` указывать явно: без него Whisper детектит язык сам и на коротких клипах иногда ошибается.
- Галлюцинации на тишине/музыке: повторяющиеся фразы = признак. Лечится `--condition-on-previous-text False`:
  в скриптах скилла включено всегда; в сырых вызовах передавать руками (проверено экспериментом 2026-08-30:
  убирает петли-повторы и восстанавливает фразы, потерянные после музыкальных вставок).
- VAD-пайплайн требует: venv `~/.local/share/stt-vad/` (onnxruntime+numpy), Silero onnx
  `~/.local/share/models/silero-vad/silero_vad.onnx`, ffmpeg. Запускать именно uv-tool питоном
  (`~/.local/share/uv/tools/mlx-whisper/bin/python`) — там mlx. LLM-этап деградирует честно:
  без ключа/терминов или при ошибке чанка — raw-текст, ошибка фиксируется в MD.
- glm-5.3-flash: thinking не отключается, но `reasoning_effort: "low"` работает (в ~30 раз меньше
  reasoning-токенов, 6× быстрее) — использовать всегда; `max_tokens: 65536` с запасом (юзер разрешил).
  Concurrency limit провайдера = 50, чанки коррекции гонять параллельно (10 потоков).
- initial-prompt НЕ исправляет термины (codecs/DeepSeek остаются) и даёт регрессы (Ox→Aux) —
  только для стиля; орфографию терминов чинит LLM-этап по канон-списку.
- `--fp16` флаг CLI не трогать: он про float16 при декодировании в оригинальном whisper, веса тут уже fp16 в safetensors.
- Русские таймстампы srt сдвигаются на паузах — для точного монтажа включать `--word-timestamps True`.
- Если нужны q4/q8 (меньше RAM): скачать соотв. папку из того же HF-репо рядом и передать её путь.

## Verification

1. Скрипт напечатал `OK /Users/alexander/result-mlx-whisper/YYYY-MM-DD_<имя>.md`; файл существует и непустой (`read_file`): заголовок с метаданными, текст, таблица сегментов.
2. Имя файла начинается с сегодняшней даты `YYYY-MM-DD_`.
3. Процесс завершился: `pgrep -fl mlx_whisper` пуст — модель выгружена из RAM (норма CLI-процесса).
4. Для длинного аудио сверить длительность последнего сегмента с ffprobe-duration.
