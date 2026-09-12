#!/usr/bin/env python3
"""Offline-тесты LLM-гейтов vad_transcribe.correct_stage (без API, llm_call подменён).

Запуск:  ~/.local/share/uv/tools/mlx-whisper/bin/python tests/test_llm_gates.py
Выход:   ALL TESTS OK — иначе AssertionError с описанием кейса.

Покрыто:
  1. числовой гейт: delete-опкод с цифрами откатывается позиционно (исторический баг:
     out.replace(b, a, 1) с b=="" вставлял текст в позицию 0);
  2. числовой гейт: откат правки с дублирующимся токеном — в правильной позиции,
     а не на первом вхождении;
  3. легальная замена термина проходит и логируется;
  4. галлюцинационная вставка (>3 слов подряд) -> verify FAIL -> regex-only база;
  5. вывод длиннее входа -> хвост приписывается к последнему сегменту, не теряется;
  6. транскрипт > WHOLE_MAX_WORDS -> рекурсивное уполовинивание, каждый вызов <= лимита;
  7. кусок, проваливший LLM-вызов, делится пополам и обе половины обрабатываются;
  8. finish_reason=length -> TruncatedError сразу, без повторного запроса (retry — только
     сетевые/HTTP ошибки);
  9. уполовинивание после ошибки не дублирует regex-правки в логе (записи провалившегося
     куска удаляются перед делением).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
import vad_transcribe as vt  # noqa: E402

# --- подмена LLM ---
FAKE = {"out": "", "fn": None, "calls": []}


def fake_llm(system, user, cfg, attempts=2):
    FAKE["calls"].append(user)
    if FAKE["fn"]:
        return FAKE["fn"](user)
    return FAKE["out"]


REAL_LLM_CALL = vt.llm_call  # настоящий, до подмены (нужен тесту 8)
vt.llm_call = fake_llm


def mk(text):
    return [{"text": t, "start": float(i), "end": float(i + 1)}
            for i, t in enumerate(text.split("|"))]


def run(base, out, terms=("Astra",)):
    FAKE["out"] = out
    FAKE["fn"] = None
    FAKE["calls"] = []
    fixed, log, errors = vt.correct_stage(mk(base), list(terms), ("k", "http://x", "m"), None)
    return " ".join(s["text"] for s in fixed), log, errors


# 1. delete-опкод с цифрами -> позиционный откат, текст не портится
res, _, _ = run("it costs 3 dollars per|million now", "it costs per million now")
assert res == "it costs 3 dollars per million now", res

# 2. откат при дублирующемся токене — позиционно (старое replace() било по первому вхождению)
res, _, _ = run("alpha 326 beta|gamma 326 delta", "alpha 326 beta gamma 327 delta")
assert res == "alpha 326 beta gamma 326 delta", res

# 3. легальная замена без цифр — принята и залогирована
res, log, _ = run("alpha 326 beta|gamma astro delta", "alpha 326 beta gamma Astra delta")
assert res == "alpha 326 beta gamma Astra delta", res
assert ("astro", "Astra") in log, log

# 4. вставка 5 слов подряд -> verify отклоняет весь вывод, остаётся regex-база
res, _, errors = run("one two three four five",
                     "one two three four five and then some more extra words")
assert res == "one two three four five", res
assert errors and "verify failed" in errors[0], errors

# 5. вывод длиннее входа (в пределах бюджета) -> хвост приписан к последнему сегменту
res, log, _ = run("one two|three four", "one two three four five six")
assert res == "one two three four five six", res
assert any("хвост приписан" in str(b) for _, b in log), log

# 6. уполовинивание: 500 слов при лимите 100 -> ровно 8 вызовов, каждый <= 100 слов
orig_limit = vt.WHOLE_MAX_WORDS
vt.WHOLE_MAX_WORDS = 100
try:
    base = " ".join(f"w{i}" for i in range(500))
    # mk() делит на сегменты по "|": сегменты по 10 слов, иначе один 500-словный сегмент
    base = "|".join(" ".join(f"w{i}" for i in range(k, k + 10)) for k in range(0, 500, 10))
    FAKE["calls"] = []

    def echo(user):  # вернуть базу как есть (0 правок)
        return user.rsplit("BASE TRANSCRIPT TO CORRECT (output only this, corrected):\n", 1)[1]
    FAKE["fn"] = echo
    fixed, log, errors = vt.correct_stage(mk(base), ["Astra"], ("k", "http://x", "m"), None)
    assert not errors, errors
    assert " ".join(s["text"] for s in fixed) == base.replace("|", " ")
    n_calls = len(FAKE["calls"])
    assert n_calls == 8, f"ожидалось 8 вызовов (500->2x250->4x125->8x<=100), получено {n_calls}"
    for c in FAKE["calls"]:
        piece = c.rsplit("BASE TRANSCRIPT TO CORRECT (output only this, corrected):\n", 1)[1]
        assert len(piece.split()) <= 100, len(piece.split())

    # 7. провал LLM на большом куске -> уполовинивание; половины обрабатываются успешно
    FAKE["calls"] = []

    def fail_big(user):
        piece = user.rsplit("BASE TRANSCRIPT TO CORRECT (output only this, corrected):\n", 1)[1]
        if len(piece.split()) > 50:
            raise RuntimeError("boom")
        return piece
    FAKE["fn"] = fail_big
    fixed, log, errors = vt.correct_stage(mk(base), ["Astra"], ("k", "http://x", "m"), None)
    assert " ".join(s["text"] for s in fixed) == base.replace("|", " ")
    assert any("halving" in e for e in errors), errors

    # 9. уполовинивание после ошибки НЕ дублирует regex-правки в логе
    #    (правило "codex belts" живёт в одной половине -> ровно одна запись в логе)
    base2 = "|".join(" ".join(f"w{i}" for i in range(k, k + 10)) for k in range(0, 490, 10))
    base2 += "|codex belts happened here just now"  # 50-й сегмент, 5 слов с правилом
    FAKE["calls"] = []

    def fail_big2(user):
        piece = user.rsplit("BASE TRANSCRIPT TO CORRECT (output only this, corrected):\n", 1)[1]
        if len(piece.split()) > 60:
            raise RuntimeError("boom")
        return piece
    FAKE["fn"] = fail_big2
    fixed, log, errors = vt.correct_stage(mk(base2), ["Astra"], ("k", "http://x", "m"), None)
    n_regex = sum(1 for a, b in log if "codex belts" in str(a) and "regex" in str(b))
    assert n_regex == 1, f"regex-правка в логе {n_regex} раз (дубли при halving): {log}"
finally:
    vt.WHOLE_MAX_WORDS = orig_limit

# 8. finish_reason=length -> TruncatedError СРАЗУ, без повторного запроса (retry только сеть)
http_calls = {"n": 0}


class FakeResp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b'{"choices": [{"finish_reason": "length", "message": {"content": "x"}}]}'


def fake_urlopen(req, timeout=None):
    http_calls["n"] += 1
    return FakeResp()


orig_urlopen = vt.urllib.request.urlopen
vt.urllib.request.urlopen = fake_urlopen
try:
    try:
        REAL_LLM_CALL("s", "u", ("k", "http://x", "m"))
        raise AssertionError("ожидался TruncatedError")
    except vt.TruncatedError:
        pass
    assert http_calls["n"] == 1, f"усечённый запрос повторён {http_calls['n']} раз"
finally:
    vt.urllib.request.urlopen = orig_urlopen

# 10. корректор (боевой ~/.hermes/.env): glm-5.3-flash для всех языков
cfg_en = vt.read_llm_config("en")
cfg_ru = vt.read_llm_config("ru")
cfg_xx = vt.read_llm_config("de")
assert cfg_en and cfg_en[2] == "glm-5.3-flash" and "z.ai" in cfg_en[1], cfg_en
assert cfg_ru and cfg_ru[2] == "glm-5.3-flash" and "z.ai" in cfg_ru[1], cfg_ru
assert cfg_xx and cfg_xx[2] == "glm-5.3-flash", cfg_xx
assert cfg_en[1].endswith("/chat/completions") and cfg_ru[1].endswith("/chat/completions")
print("case10 corrector OK:", cfg_en[2], "|", cfg_ru[2])

print("ALL TESTS OK")
