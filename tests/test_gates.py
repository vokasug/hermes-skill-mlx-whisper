#!/usr/bin/env python3
"""Offline-тесты гейтов коррекции vad_transcribe (без аудио и модели).

Запуск:  ~/.local/share/uv/tools/mlx-whisper/bin/python tests/test_gates.py
Выход:   ALL TESTS OK — иначе AssertionError с описанием кейса.

Покрыто:
  1. md_to_text: шапка игнорируется, таймкоды mm:ss и h:mm:ss парсятся;
  2. word-diff гейт: вставка >3 слов подряд -> REJECT;
  3. word-diff гейт: сумма вставок/удалений сверх бюджета -> REJECT;
  4. легальная замена термина проходит (OK);
  5. --verify: чистая правка -> exit 0, ничего не пишется (ни шапки, ни sidecar);
  6. --verify: правка числа -> exit 1, «ОТКЛОНЕНО (числа)», MD не тронут;
  7. --verify: изменён таймкод -> exit 1 «структура блоков»;
  8. --verify --finalize: шапка проштампована, .corrections.md с правкой и regex-логом.
"""
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import vad_transcribe as vt  # noqa: E402

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "vad_transcribe.py"

# 1. md_to_text: только строки-транскрипт, таймкоды обоих форматов
tmp = pathlib.Path(tempfile.mkdtemp())
md = tmp / "t.md"
md.write_text(
    "# Транскрипция — t.mp3\n\n- **Название:** t.mp3\n\n- **Дата:** 2026-10-07\n"
    "- **Коррекция терминов:** regex-база\n\n## Транскрипт\n\n"
    "**00:00** alpha astro beta\n\n**01:00** gamma 326 delta\n\n**1:02:03** omega tail\n",
    encoding="utf-8")
text, ts = vt.md_to_text(md)
assert text == "alpha astro beta gamma 326 delta omega tail", text
assert ts == ["00:00", "01:00", "1:02:03"], ts

# 2. вставка >3 слов подряд -> REJECT (длинный текст: баланс в бюджете, ловит лимит рывка)
base400 = " ".join(f"w{i}" for i in range(400))  # бюджет 4
out404 = base400.replace("w100", "w100 extra1 extra2 extra3 extra4", 1)  # рывок 4 при бюджете 4
ver, diag = vt._verify(base400, out404)
assert ver is None and diag.startswith("REJECT") and "вставка 4 слов подряд" in diag, diag

# 3. сумма вставок/удалений сверх бюджета -> REJECT (короткий текст: бюджет 2)
ver, diag = vt._verify("alpha beta gamma delta epsilon",
                       "alpha X beta Y gamma delta epsilon Z")
assert ver is None and diag.startswith("REJECT"), diag

# 4. легальная замена — OK
ver, diag = vt._verify("alpha astro beta", "alpha Astra beta")
assert ver is not None and diag.startswith("OK"), diag


def run_verify(base, md_text, *extra):
    """Пишет base.json + MD во временную папку, гоняет --verify CLI, возвращает (r, md_path)."""
    d = pathlib.Path(tempfile.mkdtemp())
    m = d / "t.md"
    m.write_text(md_text, encoding="utf-8")
    b = d / "t.base.json"
    b.write_text(json.dumps(base, ensure_ascii=False), encoding="utf-8")
    r = subprocess.run([sys.executable, str(SCRIPT), "--verify", str(b), str(m), *extra],
                       capture_output=True, text=True)
    return r, m


MD_BASE = ("# Транскрипция — t.mp3\n\n- **Название:** t.mp3\n\n- **Дата:** 2026-10-07\n"
           "- **Коррекция терминов:** regex-база — ожидает правок основной модели\n\n"
           "## Транскрипт\n\n**00:00** alpha astro beta\n\n**01:00** gamma 326 delta\n")
BASE = {"version": 2, "out": "t.md", "src_name": "t.mp3",
        "base_text": "alpha astro beta gamma 326 delta", "canonical": ["Astra"],
        "regex_log": [["omlx → MLX", 1]], "ts_marks": ["00:00", "01:00"]}

# 5. чистая правка термина -> exit 0, MD и sidecar не тронуты
r, m = run_verify(BASE, MD_BASE.replace("astro", "Astra"))
assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
assert "правок: 1, отклонено числовым гейтом: 0" in r.stdout, r.stdout
assert "regex-база" in m.read_text(encoding="utf-8"), "verify без --finalize пишет в MD!"
assert not m.with_suffix(".corrections.md").exists(), "verify без --finalize создал sidecar!"

# 6. правка числа -> exit 1, причина в выводе, MD не тронут
r, m = run_verify(BASE, MD_BASE.replace("326", "327"))
assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
assert "ОТКЛОНЕНО (числа)" in r.stdout and "326" in r.stdout, r.stdout
assert "327" in m.read_text(encoding="utf-8"), "verify не должен откатывать файл сам"

# 7. изменён таймкод -> exit 1 «структура блоков» (REJECT уходит в stderr через sys.exit)
r, m = run_verify(BASE, MD_BASE.replace("**01:00**", "**02:00**"))
assert r.returncode == 1 and "структура блоков" in r.stderr and "01:00 -> 02:00" in r.stderr, \
    (r.returncode, r.stdout, r.stderr)

# 8. --finalize: шапка проштампована, sidecar с правкой и regex-логом
r, m = run_verify(BASE, MD_BASE.replace("astro", "Astra"), "--finalize")
assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
md_out = m.read_text(encoding="utf-8")
assert "- **Коррекция терминов:** основная модель; канонов 1, правок 1" in md_out, md_out
assert "alpha Astra beta" in md_out, md_out
side = m.with_suffix(".corrections.md").read_text(encoding="utf-8")
assert "`astro` → `Astra`" in side and "omlx → MLX` ×1 (regex" in side, side

print("ALL TESTS OK")
