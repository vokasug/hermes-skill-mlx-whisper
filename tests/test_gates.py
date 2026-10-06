#!/usr/bin/env python3
"""Offline-тесты гейтов коррекции vad_transcribe (без аудио и модели).

Запуск:  ~/.local/share/uv/tools/mlx-whisper/bin/python tests/test_gates.py
Выход:   ALL TESTS OK — иначе AssertionError с описанием кейса.

Покрыто:
  1. числовой гейт: delete-опкод с цифрами откатывается позиционно (исторический баг:
     out.replace(b, a, 1) с b=="" вставлял текст в позицию 0);
  2. числовой гейт: откат правки с дублирующимся токеном — в правильной позиции,
     а не на первом вхождении;
  3. легальная замена термина проходит и логируется;
  4. галлюцинационная вставка (>3 слов подряд) -> verify отклоняет весь вывод, база цела;
  5. вывод длиннее входа (в пределах бюджета) -> хвост приписан к последнему сегменту;
  6. apply_mode end-to-end: MD переписан с правками, шапка «Коррекция терминов», sidecar;
  7. apply_mode с отклонённым выводом -> exit 1, MD с пометкой отклонения, текст базовый.
"""
import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import vad_transcribe as vt  # noqa: E402

SCRIPT = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "vad_transcribe.py"


def mk_payload(base, terms=("Astra",)):
    segs = [{"text": t, "start": float(i), "end": float(i + 1)}
            for i, t in enumerate(base.split("|"))]
    return {"segments": segs, "base_text": " ".join(s["text"] for s in segs),
            "canonical": list(terms)}


def run(base, out):
    return vt.apply_corrections(mk_payload(base), out)


def text_of(res):
    return " ".join(s["text"] for s in res["segments"])


# 1. delete-опкод с цифрами -> позиционный откат, текст не портится
res = run("it costs 3 dollars per|million now", "it costs per million now")
assert res["status"] == "ok", res
assert text_of(res) == "it costs 3 dollars per million now", text_of(res)
assert res["rejected"] and not res["changes"], res

# 2. откат при дублирующемся токене — позиционно (старое replace() било по первому вхождению)
res = run("alpha 326 beta|gamma 326 delta", "alpha 326 beta gamma 327 delta")
assert res["status"] == "ok"
assert text_of(res) == "alpha 326 beta gamma 326 delta", text_of(res)

# 3. легальная замена без цифр — принята и залогирована
res = run("alpha 326 beta|gamma astro delta", "alpha 326 beta gamma Astra delta")
assert res["status"] == "ok"
assert text_of(res) == "alpha 326 beta gamma Astra delta", text_of(res)
assert ("astro", "Astra") in res["changes"], res["changes"]

# 4. вставка 5 слов подряд -> verify отклоняет весь вывод, остаётся база
res = run("one two three four five",
          "one two three four five and then some more extra words")
assert res["status"] == "rejected", res
assert text_of(res) == "one two three four five", text_of(res)

# 5. вывод длиннее входа (в пределах бюджета) -> хвост приписан к последнему сегменту
res = run("one two|three four", "one two three four five six")
assert res["status"] == "ok"
assert text_of(res) == "one two three four five six", text_of(res)
assert res["note"] and "хвост приписан" in res["note"][1], res["note"]

# 6/7. apply_mode end-to-end через CLI
tmp = pathlib.Path(tempfile.mkdtemp())
payload = mk_payload("alpha astro beta", ("Astra",))
payload.update({
    "version": 1, "out": str(tmp / "t.md"), "src_name": "t.mp3", "src": "/tmp/t.mp3",
    "meta": {"title": None, "author": None, "date": None, "url": None},
    "date": "2026-09-24", "duration": 2.0, "speech_s": 2.0, "speech_ratio": 1.0,
    "model": "m", "language": "ru", "subs_line": "не использовались",
    "t_vad": 0.1, "t_stt": 0.2, "regex_log": [["x → y", 1]], "spans": [], "subs_text": "",
    "instructions": "…",
})
p_json = tmp / "t.correct-payload.json"
p_json.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

# 6. принятые правки: MD переписан, шапка и sidecar на месте
c_txt = tmp / "t.corrected.txt"
c_txt.write_text("alpha Astra beta", encoding="utf-8")
r = subprocess.run([sys.executable, str(SCRIPT), "--apply-corrections", str(p_json), str(c_txt)],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr + r.stdout
md = (tmp / "t.md").read_text(encoding="utf-8")
assert "alpha Astra beta" in md, md
assert "- **Коррекция терминов:** основная модель; канонов 1, правок 1" in md, md
assert "- **Название:** t.mp3" in md, md  # fallback на имя файла с расширением
side = (tmp / "t.corrections.md").read_text(encoding="utf-8")
assert "`astro` → `Astra`" in side and "×1 (regex" in side, side

# 7. отклонённый вывод: exit 1, MD с пометкой, текст базовый
c_txt.write_text("alpha Astra beta plus five more extra words here", encoding="utf-8")
r = subprocess.run([sys.executable, str(SCRIPT), "--apply-corrections", str(p_json), str(c_txt)],
                   capture_output=True, text=True)
assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
md = (tmp / "t.md").read_text(encoding="utf-8")
assert "ОТКЛОНЕНА верификацией" in md, md
assert "alpha astro beta" in md, md

# 8. apply_edit_list: базовая замена
txt, app, skip = vt.apply_edit_list("alpha astro beta gamma",
                                    [{"old": "astro", "new": "Astra"}])
assert txt == "alpha Astra beta gamma" and len(app) == 1 and not skip, (txt, app, skip)

# 9. apply_edit_list: old не найден -> пропуск с причиной
txt, app, skip = vt.apply_edit_list("alpha beta", [{"old": "zzz", "new": "yyy"}])
assert txt == "alpha beta" and not app and "не найден" in skip[0][2], (txt, app, skip)

# 10. apply_edit_list: неоднозначный old без all -> пропуск; с all -> обе замены
txt, app, skip = vt.apply_edit_list("орнит и орнит", [{"old": "орнит", "new": "Ornith"}])
assert txt == "орнит и орнит" and not app and "встречается 2 раз" in skip[0][2], (txt, app, skip)
txt, app, skip = vt.apply_edit_list("орнит и орнит", [{"old": "орнит", "new": "Ornith", "all": True}])
assert txt == "Ornith и Ornith" and len(app) == 1 and not skip, (txt, app, skip)

# 11. apply_edit_list: числовое изменение -> пропуск до гейтов
txt, app, skip = vt.apply_edit_list("версия полтора вышла", [{"old": "полтора", "new": "1.5"}])
assert txt == "версия полтора вышла" and "ОТКЛОНЕНО: числа" in skip[0][2], (txt, app, skip)

# 12. apply_edit_list: нормализация пробелов (old скопирован из MD с переносами)
txt, app, skip = vt.apply_edit_list("alpha astro beta",
                                    [{"old": "alpha   astro\nbeta", "new": "alpha Astra beta"}])
assert txt == "alpha Astra beta" and len(app) == 1, (txt, app, skip)

# 13. apply_edit_list: таймкод в old запрещён (правка через границу блока)
txt, app, skip = vt.apply_edit_list("alpha beta", [{"old": "alpha **01:00** beta", "new": "x"}])
assert "таймкод" in skip[0][2], skip

# 14/15. --apply-edits end-to-end через CLI
e_json = tmp / "t.edits.json"
e_json.write_text(json.dumps([{"old": "astro", "new": "Astra"},
                              {"old": "zzz", "new": "yyy"}], ensure_ascii=False), encoding="utf-8")
r = subprocess.run([sys.executable, str(SCRIPT), "--check-edits", str(p_json), str(e_json)],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr + r.stdout
assert "замен применится: 1, пропущено: 1" in r.stdout, r.stdout
r = subprocess.run([sys.executable, str(SCRIPT), "--apply-edits", str(p_json), str(e_json)],
                   capture_output=True, text=True)
assert r.returncode == 0, r.stderr + r.stdout
md = (tmp / "t.md").read_text(encoding="utf-8")
assert "alpha Astra beta" in md and "правок 1" in md, md
side = (tmp / "t.corrections.md").read_text(encoding="utf-8")
assert "Пропущенные замены" in side and "old не найден" in side, side

# 15. --apply-edits с выводом за гейтами: вставка >3 слов подряд -> exit 1, база цела
e_json.write_text(json.dumps([{"old": "beta", "new": "beta plus five more extra words here"}]),
                  encoding="utf-8")
r = subprocess.run([sys.executable, str(SCRIPT), "--apply-edits", str(p_json), str(e_json)],
                   capture_output=True, text=True)
assert r.returncode == 1, (r.returncode, r.stdout, r.stderr)
md = (tmp / "t.md").read_text(encoding="utf-8")
assert "ОТКЛОНЕНА верификацией" in md and "alpha astro beta" in md, md

print("ALL TESTS OK")
