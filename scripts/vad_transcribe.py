#!/usr/bin/env python3
"""Full STT pipeline: Silero VAD -> mlx_whisper -> term correction -> paragraph transcript.

Run with the mlx-whisper uv tool python:
  ~/.local/share/uv/tools/mlx-whisper/bin/python vad_transcribe.py <audio> [audio2...] [--language ru]

Stages:
  1. VAD   (silero venv + onnx model)          -> speech intervals
  2. STT   (segments <=28s, one process, condition_on_previous_text=False)
  3. Correction prep: regex prepass (mishear dictionary) + junction fix. When canonical
     terms or subtitles exist, writes <stem>.base.json (base snapshot for the judge)
     next to the MD.

Correction is an advisor/judge loop with the MAIN AGENT MODEL as advisor:
  - advisor (main model): reads the finished MD transcript + the subtitle file (when
    present) + the content description it already has in context, and edits the MD
    IN PLACE with a patch tool — only mishearings and term spelling;
  - judge (this script):  --verify BASE_JSON MD [--finalize] diffs the edited transcript
    against the base snapshot with deterministic gates (block structure; word-diff
    budget; numeric gate) — no writes, exit 1 with reasons on violation. On a clean
    pass --finalize stamps the header correction line and writes <stem>.corrections.md.

Rendering: transcript is cut into sentences (by terminal punctuation of the word stream), then
sentences are grouped into ~60 s blocks — a new block starts at the sentence boundary nearest to
(block start + 60 s). Every block renders as one line "**mm:ss** text" followed by a blank line.
Deterministic; no model involved.

Main output: ~/result-mlx-whisper/YYYY-MM-DD_<stem>.md  — readable transcript with timestamps.
Sidecars (when applicable): <stem>.base.json (judge data), <stem>.corrections.md
(applied change log), <stem>.segments.json (--debug-segments).
--no-correct skips stage 3 (no base snapshot, regex-only text).
"""
import argparse
import collections
import datetime
import difflib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time

HOME = pathlib.Path.home()
VAD_PY = HOME / ".local/share/stt-vad/venv/bin/python"
VAD_SCRIPT = pathlib.Path(__file__).parent / "vad_segments.py"   # sibling in skill scripts/
VAD_MODEL = HOME / ".local/share/models/silero-vad/silero_vad.onnx"
OUT_DIR = HOME / "result-mlx-whisper"

# Model by language: podlodka q8 is ru-specialized; every other language goes
# to the multilingual whisper-large-v3-turbo-8bit.
MODELS = {
    "ru": HOME / ".local/share/models/whisper-podlodka-turbo-MLX-q8",
}
MODEL_FALLBACK = HOME / ".local/share/models/whisper-large-v3-turbo-8bit"


def default_model_for_language(language: str) -> pathlib.Path:
    return MODELS.get(language, MODEL_FALLBACK)

MAX_SEG = 28.0
GAP_MERGE = 0.25
PAD = 0.15
PARA_TARGET_S = 60.0  # block target: a new block starts at the sentence boundary nearest to +60s

# Built-in mishear -> canonical map (term extraction from the draft).
MISHEAR_MAP = {
    "codecs": "Codex", "codex belts": "Codex builds", "quad code": "Claude Code", "cloud code": "Claude Code",
    "deep-swee": "DeepSeek", "deep swee": "DeepSeek", "deep sweep": "DeepSeek",
    "deepseq": "DeepSeek", "deep seek": "DeepSeek", "d-seq": "DeepSeek", "dcp4": "DeepSeek V4",
    "zii": "ZAI", "cash tokens": "cache tokens", "cashed": "cached",
    "aux alpha": "Ox Alpha", "auxalpha": "Ox Alpha", "oxalpha": "Ox Alpha",
    "whisperflow": "WhisperFlow", "vibe proxy": "VibeProxy",
    "t3 code": "T3 Code", "base 10": "Base10",
}

# Common function words safe to lowercase after a comma at a segment junction.
LOWER_STARTERS = {
    "the", "a", "an", "and", "but", "or", "so", "it", "we", "they", "he", "she", "you",
    "this", "that", "there", "then", "these", "those", "when", "what", "which", "who",
    "with", "in", "on", "for", "to", "of", "my", "his", "her", "its", "if", "because",
    "as", "at", "from", "like", "also", "just", "now", "here", "is", "are", "was", "were",
    "и", "а", "но", "что", "это", "он", "она", "мы", "вы", "они", "когда", "если",
    "как", "в", "на", "с", "для", "к", "по", "у", "за", "от", "там", "тут", "ещё", "тоже",
}

# The correction payload is judge-side data (segments + base_text for gates and MD
# rebuild). The advisor model never reads it: it works from the finished MD, the
# subtitle file and the content description it already has in context.


def fmt_ts(sec) -> str:
    if sec is None:
        return "—"
    m, s = divmod(int(float(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_short(sec) -> str:
    sec = int(float(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def run_vad(src: pathlib.Path) -> dict:
    r = subprocess.run([str(VAD_PY), str(VAD_SCRIPT), str(src)],
                       capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def cut_and_transcribe(src: pathlib.Path, vad: dict, model: str, language: str) -> tuple[list, dict]:
    import mlx_whisper  # in-process: model loads once (ModelHolder cache)

    intervals = vad["intervals"]
    merged = []
    for s, e in intervals:
        if merged and s - merged[-1][1] <= GAP_MERGE and (e - merged[-1][0]) <= MAX_SEG:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    segs = []
    for s, e in merged:
        while e - s > MAX_SEG:
            segs.append((s, s + MAX_SEG))
            s += MAX_SEG
        segs.append((s, e))

    all_segments = []
    with tempfile.TemporaryDirectory() as td:
        for i, (s, e) in enumerate(segs):
            cs, ce = max(0.0, s - PAD), min(vad["duration"], e + PAD)
            p = pathlib.Path(td) / f"seg_{i:04d}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{cs:.3f}", "-to", f"{ce:.3f}",
                            "-i", str(src), "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", str(p)], check=True)
            res = mlx_whisper.transcribe(
                str(p), path_or_hf_repo=model, language=language,
                condition_on_previous_text=False, fp16=True, verbose=False)
            for seg in res["segments"]:
                txt = seg["text"].strip()
                # drop content-free segments: Whisper on tiny non-speech VAD blips
                # (music/applause) hallucinates pure punctuation runs ("! ! !", "♪");
                # they carry no speech and would explode into one line per token downstream
                if txt and re.search(r"\w", txt):
                    all_segments.append({"start": round(seg["start"] + cs, 2),
                                         "end": round(seg["end"] + cs, 2), "text": txt,
                                         "logprob": seg.get("avg_logprob")})
    return all_segments, {"sent": len(segs)}


def fix_segment_junctions(segments: list) -> None:
    """Repair false sentence breaks at segment boundaries. Words are never changed
    except forced lowercasing of a sentence-starter after removing a false period.
    In-place."""
    for i in range(len(segments) - 1):
        a, b = segments[i]["text"], segments[i + 1]["text"]
        if not a or not b:
            continue
        first_w = b.split()[0]
        stripped = first_w.lstrip("«\"'(")
        low = stripped.lower()
        if a.endswith(".") and stripped[:1].islower():
            segments[i]["text"] = a[:-1].rstrip()
        elif a.endswith(".") and low in LOWER_STARTERS and stripped[:1].isupper():
            # false period + false capital: "…per day. Is just…" -> "…per day, is just…"
            segments[i]["text"] = a[:-1].rstrip() + ","
            segments[i + 1]["text"] = b.replace(first_w, first_w[:1].lower() + first_w[1:], 1)
        elif a.endswith(",") and low in LOWER_STARTERS and stripped[:1].isupper():
            segments[i + 1]["text"] = b.replace(first_w, first_w[:1].lower() + first_w[1:], 1)


def extract_canonical(full_text: str, extra_terms: list[str]) -> list[str]:
    low = full_text.lower()
    canon, seen = [], set()
    for wrong, right in MISHEAR_MAP.items():
        if wrong in low and right not in seen:
            canon.append(right)
            seen.add(right)
    for t in extra_terms:
        if t and t not in seen:
            canon.append(t)
            seen.add(t)
    return canon


def regex_prepass(text: str) -> tuple[str, list]:
    """Deterministic mishear fixes by word-boundary, case-insensitive regex.
    Guaranteed (advisor-invisible) corrections; returns (text, applied [(rule, n)])."""
    applied = []
    for wrong, right in MISHEAR_MAP.items():
        pat = r"\b" + re.escape(wrong) + r"\b"
        text, n = re.subn(pat, right, text, flags=re.IGNORECASE)
        if n:
            applied.append((f"{wrong} → {right}", n))
    return text, applied


def parse_subs(path: pathlib.Path) -> list[tuple[float, float, str]]:
    """Parse .srt/.vtt into [(start, end, text)]. Tags stripped, rolling auto-caption
    duplicates removed (YouTube auto-subs repeat the previous line in each cue)."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    ts = r"(\d+):(\d+):(\d+)[,.](\d+)"
    cues = []
    for m in re.finditer(ts + r"\s*-->\s*" + ts + r"[^\n]*\n(.*?)(?=\n\s*\n|\n\d+\s*\n|\Z)",
                         raw, flags=re.DOTALL):
        h, mnt, s, ms = (int(m.group(i)) for i in range(1, 5))
        h2, mnt2, s2, ms2 = (int(m.group(i)) for i in range(5, 9))
        start = h * 3600 + mnt * 60 + s + ms / 1000
        end = h2 * 3600 + mnt2 * 60 + s2 + ms2 / 1000
        lines = [re.sub(r"<[^>]+>", "", ln).strip() for ln in m.group(9).strip().splitlines()]
        lines = [ln for ln in lines if ln]
        # rolling dedup: drop lines identical to the tail of the previous cue
        if cues:
            prev_lines = cues[-1][2].split("\n")
            while lines and prev_lines and lines[0] == prev_lines[-1]:
                lines.pop(0)
                prev_lines = prev_lines[:-1]
        if lines:
            cues.append((start, end, "\n".join(lines)))
    return cues


def extract_terms_from_subs(sub_cues: list, draft_text: str, max_terms: int = 25) -> list[str]:
    """Auto canonical terms from subtitles: capitalized words (>=2 occurrences in subs)
    that the draft either lowercased or spelled similarly-but-wrong (fuzzy >=0.8).
    Conservative: term is always the subs casing; never invent words absent from subs."""
    subs_text = " ".join(t for _, _, t in sub_cues)
    cand = collections.Counter(m.group(0) for m in re.finditer(r"\b[A-Z][\w.+-]{2,}\b", subs_text))
    draft_words = set(draft_text.split())
    draft_low_map = {}
    for w in draft_words:
        draft_low_map.setdefault(w.lower(), w)
    draft_lows = set(draft_low_map)
    terms, seen = [], set()
    for w, n in cand.most_common():
        if len(terms) >= max_terms:
            break
        if n < 2 or w.lower() in seen:
            continue
        lw = w.lower()
        if lw in LOWER_STARTERS:
            continue
        # sentence-start filter: proper nouns stay capitalized mid-sentence; words that are
        # capitalized ONLY at sentence starts (Yeah, However…) are not terms. Also, if the
        # lowercase form occurs in subs at all, it's a common word (deep, flash…), skip.
        if re.search(r"\b" + re.escape(lw) + r"\b", subs_text):
            continue
        mid_caps = [m for m in re.finditer(r"\b" + re.escape(w) + r"\b", subs_text)
                    if m.start() > 1 and not re.search(r"[.!?…]\s+$", subs_text[:m.start()])]
        if len(mid_caps) < 2:  # one-off mid-sentence cap is too weak a signal
            continue
        # multiword-brand fragment filter: "Deep" in "Deep Seek" — if the candidate is
        # usually followed by another capitalized word, it's a name fragment, not a term
        followed_cap = sum(1 for m in re.finditer(r"\b" + re.escape(w) + r"\b", subs_text)
                           if re.match(r"\s+[A-Z]", subs_text[m.end():]))
        if followed_cap * 2 >= n:
            continue
        hit = None
        if lw in draft_lows and w not in draft_words:
            dv = draft_low_map[lw]
            if dv.islower():
                hit = dv                     # draft has it fully lowercased -> fix case
            # draft already carries a capitalized variant (DeepSeek): trust it, skip
        elif lw not in draft_lows:
            close = difflib.get_close_matches(lw, draft_lows, n=1, cutoff=0.8)
            if close:
                hit = draft_low_map[close[0]]  # likely mishearing of this subs word
        if hit and hit != w:
            terms.append(w)
            seen.add(lw)
    return terms


def numeric_tokens(s: str) -> list[str]:
    """All digit runs incl. separated forms ($3.12 -> ['3.12']); the numeric gate
    guarantees corrections never alter numbers regardless of advisor intent."""
    return re.findall(r"\d+(?:[.,:]\d+)*", s)


LOGPROB_LOW = -0.7  # avg_logprob below this = Whisper itself was unsure about the segment


def uncertain_spans(segments: list, max_spans: int = 40) -> list[str]:
    return [s["text"] for s in segments
            if s.get("logprob") is not None and s["logprob"] < LOGPROB_LOW][:max_spans]


def _verify(pre_text: str, out: str) -> tuple:
    rw, ow = pre_text.split(), out.split()
    budget = max(2, min(25, len(rw) // 100))
    delta = len(ow) - len(rw)
    if abs(delta) > budget:
        return None, f"REJECT: баланс слов {delta:+d} при бюджете {budget} (слов {len(rw)} → {len(ow)})"
    sm = difflib.SequenceMatcher(None, rw, ow)
    ops = sm.get_opcodes()
    id_total = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag in ("insert", "delete"):
            run = max(i2 - i1, j2 - j1)
            if run > 3:            # single run of >3 inserted/deleted words = hallucination
                words = rw[i1:i2] if tag == "delete" else ow[j1:j2]
                return None, f"REJECT: {'удаление' if tag == 'delete' else 'вставка'} {run} слов подряд (лимит 3): «{' '.join(words)}»"
            id_total += run
    if id_total > budget:
        return None, f"REJECT: вставок/удалений суммарно {id_total} при бюджете {budget} (баланс слов {delta:+d})"
    return (ops, rw, ow), f"OK: баланс слов {delta:+d}, вставок/удалений {id_total}, бюджет {budget}"


def redistribute_words(chunk: list, text: str) -> tuple[list, tuple | None]:
    """Map a corrected word stream back onto segments by original word counts. A longer
    output's tail is appended to the last segment (never silently dropped).
    Returns (fixed_texts, note_or_None)."""
    outw = text.split()
    fixed, wpos = [], 0
    for s in chunk:
        n = len(s["text"].split())
        fixed.append(" ".join(outw[wpos:wpos + n]))
        wpos += n
    note = None
    if fixed and wpos < len(outw):
        extra = len(outw) - wpos
        fixed[-1] = (fixed[-1] + " " + " ".join(outw[wpos:])).strip()
        note = (f"+{extra} слов", "вывод длиннее входа — хвост приписан к концу")
    return fixed, note


TS_LINE = re.compile(r"^\*\*(\d+:\d{2}(?::\d{2})?)\*\* (.*)$")


def md_to_text(md_path: pathlib.Path) -> tuple[str, list]:
    """Transcript body of an MD -> (plain word stream, block timestamps). Only
    '**mm:ss** text' lines are read, so header edits never reach the gates."""
    parts, ts = [], []
    for line in pathlib.Path(md_path).read_text(encoding="utf-8").splitlines():
        m = TS_LINE.match(line)
        if m:
            ts.append(m.group(1))
            parts.append(m.group(2))
    return " ".join(" ".join(parts).split()), ts


def verify_mode(base_path: pathlib.Path, md_path: pathlib.Path, finalize: bool = False):
    """Post-check (judge): the advisor (main model) edits the MD in place with a patch
    tool; this mode diffs the edited transcript against the base snapshot with the
    deterministic gates — block structure, word-diff budget, numeric gate. Nothing is
    written unless --finalize: on a clean pass it stamps the header correction line
    and writes <stem>.corrections.md. Exit 1 with reasons on any violation."""
    base = json.loads(base_path.read_text(encoding="utf-8"))
    text, ts = md_to_text(md_path)
    if ts != base["ts_marks"]:
        diff = next(((a, b) for a, b in zip(base["ts_marks"], ts) if a != b), None)
        hint = f", первое расхождение: {diff[0]} -> {diff[1]}" if diff else ""
        sys.exit(f"REJECT: структура блоков изменена (таймкодов было {len(base['ts_marks'])}, "
                 f"стало {len(ts)}{hint}) — восстановите строки **mm:ss** в исходном виде")
    ver, diag = _verify(base["base_text"], text)
    print(diag)
    if ver is None:
        sys.exit(1)
    ops, rw, ow = ver
    changes, rejected = [], []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            continue
        a, b = " ".join(rw[i1:i2]), " ".join(ow[j1:j2])
        (rejected if numeric_tokens(a) != numeric_tokens(b) else changes).append((a, b))
    print(f"правок: {len(changes)}, отклонено числовым гейтом: {len(rejected)}")
    for a, b in rejected:
        print(f"  ОТКЛОНЕНО (числа): `{a}` -> `{b}` — откатите эту правку в MD и повторите --verify")
    if rejected:
        sys.exit(1)
    if not finalize:
        return
    md_path = pathlib.Path(md_path)
    lines = md_path.read_text(encoding="utf-8").splitlines()
    corr = f"основная модель; канонов {len(base['canonical'])}, правок {len(changes)}"
    lines = [f"- **Коррекция терминов:** {corr}" if ln.startswith("- **Коррекция терминов:**")
             else ln for ln in lines]
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"OK {md_path}")
    clines = [f"# Правки — {base['src_name']}", "",
              "Советчик: основная модель агента — правки внесены прямо в MD. Пост-чек — "
              "детерминированные гейты этого скрипта (структура блоков, word-diff бюджет, "
              "числовой гейт).", ""]
    for rule, n in base.get("regex_log", []):
        clines.append(f"- `{rule}` ×{n} (regex, гарантированно)")
    for a, b in changes:
        clines.append(f"- `{a}` → `{b}`")
    cpath = md_path.with_suffix(".corrections.md")
    cpath.write_text("\n".join(clines) + "\n", encoding="utf-8")
    print(f"OK {cpath}")


def group_into_blocks(sents: list[dict], audio_end: float | None = None) -> list[list[dict]]:
    """Group sentences into ~PARA_TARGET_S blocks: a new block starts at the sentence
    boundary NEAREST to (block start + PARA_TARGET_S) — the distance |start - target|
    decreases while boundaries approach the target and increases after it, so cutting at
    the first local minimum picks exactly the nearest boundary. The final (tail) block is
    whatever remains; it merges into the previous block when tiny (<20 s) and the merge
    keeps the result <= 80 s. Tail duration uses audio_end (real end of audio) when given:
    interpolated word times of the last Whisper segment can overshoot the file end, which
    would fake a long tail and skip the merge."""
    n = len(sents)
    virtual_next = audio_end if audio_end is not None else sents[-1]["start"] + 1.0
    blocks, cur, t0 = [], [], None
    for i, s in enumerate(sents):
        if cur:
            target = t0 + PARA_TARGET_S
            d_here = abs(s["start"] - target)
            # next boundary = next sentence, or the end of audio for the final sentence
            d_next = abs(sents[i + 1]["start"] - target) if i + 1 < n else abs(virtual_next - target)
            if d_here <= d_next:
                blocks.append(cur)
                cur, t0 = [], None
        if not cur:
            t0 = s["start"]
        cur.append(s)
    if cur:
        blocks.append(cur)
    if len(blocks) >= 2:
        tail, prev = blocks[-1], blocks[-2]
        # sents carry no "end" — tail duration = real audio end (or last boundary + 1s)
        tail_end = audio_end if audio_end is not None else sents[-1]["start"] + 1.0
        tail_s = tail_end - tail[0]["start"]
        prev_s = tail[0]["start"] - prev[0]["start"]
        if tail_s < 20 and prev_s + tail_s <= 80:
            blocks[-2] = prev + tail
            blocks.pop()
    return blocks


SENT_END = re.compile(r"[.!?…][\"'»)\]]*$")


def split_into_sentences(segments: list) -> list[dict]:
    """Build a global word stream with interpolated times, then cut it into sentences.
    Whisper puts several sentences inside one segment and often NO period at segment
    junctions, so sentences are cut by terminal punctuation of the *word* stream, not segment ends. A sentence never
 breaks: block timestamps attach to sentence starts only."""
    # word stream: (word, t)
    words = []
    for seg in segments:
        wl = seg["text"].split()
        n = max(1, len(wl))
        dur = seg["end"] - seg["start"]
        for k, w in enumerate(wl):
            words.append((w, seg["start"] + dur * (k + 0.5) / n))

    sents, cur = [], []
    for w, t in words:
        cur.append((w, t))
        if SENT_END.search(w):
            sents.append(cur)
            cur = []
    if cur:
        if sents and len(cur) <= 3:
            sents[-1].extend(cur)  # tiny dangling tail joins last sentence
        elif sents:
            # long tail: emit as its own sentences (rare no-punctuation run-on)
            sents.append(cur)
        else:
            sents.append(cur)

    return [{"start": sent[0][1], "text": " ".join(w for w, _ in sent)} for sent in sents]


def build_md_lines(info: dict, segments: list, corr_line: str) -> list[str]:
    """Full MD: header (content metadata block first, then technical lines) + transcript
    rendered into ~60 s blocks. Shared by the transcribe run and --apply-corrections."""
    lines = [f"# Транскрипция — {info['src_name']}", ""]
    # content metadata block (always present; Название falls back to file name with extension)
    m = info["meta"]
    lines.append(f"- **Название:** {m.get('title') or info['src_name']}")
    if m.get("author"):
        lines.append(f"- **Автор:** {m['author']}")
    if m.get("date"):
        lines.append(f"- **Дата публикации:** {m['date']}")
    if m.get("url"):
        lines.append(f"- **Ссылка:** {m['url']}")
    lines += [
        "",
        f"- **Дата:** {info['date']}",
        f"- **Источник:** `{info['src']}`",
        f"- **Субтитры:** {info['subs_line']}",
        f"- **Длительность:** {fmt_ts(info['duration'])} | речь (VAD): {info['speech_s']/60:.1f} мин ({info['speech_ratio']*100:.0f}%)",
        f"- **Модель:** {info['model']}, язык: {info['language']}, VAD: Silero",
        f"- **Коррекция терминов:** {corr_line}",
        f"- **Время:** VAD {info['t_vad']:.0f}с + STT {info['t_stt']:.0f}с = {(info['t_vad'] + info['t_stt'])/60:.1f} мин",
        "", "## Транскрипт", "",
    ]
    # render: sentences -> ~60 s blocks; each block = one "**mm:ss** text" line + blank line
    sents = split_into_sentences(segments)
    for block in group_into_blocks(sents, audio_end=float(info["duration"])):
        ts = fmt_short(block[0]["start"])
        lines.append(f"**{ts}** " + " ".join(s["text"] for s in block))
        lines.append("")
    lines.append("")
    return lines


def process_one(src: pathlib.Path, args):
    t0 = time.time()
    print(f"[1/3] VAD: {src.name}", flush=True)
    vad = run_vad(src)
    t_vad = time.time() - t0
    speech_s = sum(e - s for s, e in vad["intervals"])
    print(f"      {len(vad['intervals'])} интервалов, речь {speech_s/60:.1f} мин "
          f"({vad['speech_ratio']*100:.0f}%) — {t_vad:.1f}с", flush=True)

    t1 = time.time()
    print(f"[2/3] STT ({args.language})...", flush=True)
    segments, stt_info = cut_and_transcribe(src, vad, args.model, args.language)
    t_stt = time.time() - t1
    draft_words = sum(len(s["text"].split()) for s in segments)
    print(f"      {stt_info['sent']} сегментов, {draft_words} слов — {t_stt:.0f}с", flush=True)

    sub_cues = None
    if args.subs:
        sp = pathlib.Path(args.subs).expanduser()
        if sp.exists():
            sub_cues = parse_subs(sp)
            print(f"      субтитры: {sp.name} — {len(sub_cues)} реплик", flush=True)
            if not sub_cues:
                sub_cues = None
                print(f"      субтитры: {sp.name} распарсились в 0 реплик — игнорируются", flush=True)
        else:
            print(f"      субтитры: файл не найден: {sp} — игнорируются", flush=True)
    subs_line = (f"`{args.subs}` ({len(sub_cues)} реплик) — второе мнение для коррекции"
                 if sub_cues else "не использовались")

    stem = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", src.stem)
    out = OUT_DIR / f"{datetime.date.today().isoformat()}_{stem}.md"
    info = {
        "src_name": src.name, "src": str(src),
        "meta": {"title": args.meta_title, "author": args.meta_author,
                 "date": args.meta_date, "url": args.meta_url},
        "date": datetime.date.today().isoformat(),
        "duration": vad["duration"], "speech_s": speech_s, "speech_ratio": vad["speech_ratio"],
        "model": pathlib.Path(args.model).name, "language": args.language,
        "subs_line": subs_line, "t_vad": t_vad, "t_stt": t_stt,
    }

    canonical, regex_log, base_path = [], [], None
    if args.no_correct:
        fix_segment_junctions(segments)
        corr_line = "выключена (--no-correct)"
        print("[3/3] Коррекция терминов: выключена (--no-correct)", flush=True)
    else:
        draft_text = " ".join(s["text"] for s in segments)
        canonical = extract_canonical(draft_text,
                                      [t.strip() for t in args.terms.split(",") if t.strip()])
        if sub_cues:
            auto = [t for t in extract_terms_from_subs(sub_cues, draft_text) if t not in canonical]
            if auto:
                canonical += auto
                print(f"      авто-термины из субтитров ({len(auto)}): {', '.join(auto)}", flush=True)
        if canonical or sub_cues:
            pre_text, pre_rules = regex_prepass(draft_text)
            regex_log = [[rule, n] for rule, n in pre_rules]
            fixed, _note = redistribute_words(segments, pre_text)
            for s, fx in zip(segments, fixed):
                s["text"] = fx
            fix_segment_junctions(segments)
            base_path = out.with_name(out.stem + ".base.json")
            src_note = " + субтитры-второе-мнение" if sub_cues else ""
            print(f"[3/3] Коррекция терминов: канонов {len(canonical)}{src_note}"
                  + (f": {', '.join(canonical)}" if canonical else ""), flush=True)
            spans = uncertain_spans(segments)
            if spans:
                print(f"      низкоуверенные спаны ({len(spans)}): " + " | ".join(spans), flush=True)
            corr_line = "regex-база — ожидает правок основной модели"
        else:
            fix_segment_junctions(segments)
            corr_line = "не потребовалась (терминов не найдено)"
            print("[3/3] Коррекция терминов: терминов не найдено — пропуск", flush=True)

    out.write_text("\n".join(build_md_lines(info, segments, corr_line)).rstrip() + "\n",
                   encoding="utf-8")
    print(f"OK {out}")
    if base_path:
        _txt, ts_marks = md_to_text(out)
        base = dict(version=2, out=str(out), src_name=info["src_name"],
                    base_text=" ".join(s["text"] for s in segments),
                    canonical=canonical, regex_log=regex_log, ts_marks=ts_marks)
        base_path.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"BASE {base_path}")
        print(f"      следующий шаг: основная модель правит {out.name} напрямую (patch)"
              + (f" с опорой на субтитры ({args.subs})" if sub_cues else "")
              + f", затем: --verify {base_path.name} {out.name} [--finalize]", flush=True)

    if args.debug_segments:
        spath = out.with_suffix(".segments.json")
        spath.write_text(json.dumps(
            {"file": str(src), "segments": segments}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"OK {spath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="*")
    ap.add_argument("--verify", nargs=2, metavar=("BASE_JSON", "MD"),
                    help="judge step (post-check): verify advisor edits made directly in the "
                         "MD against the base snapshot (block structure, word-diff budget, "
                         "numeric gate); no writes, exit 1 with reasons on violation")
    ap.add_argument("--finalize", action="store_true",
                    help="with --verify: on a clean pass stamp the header correction line "
                         "and write <stem>.corrections.md")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--model", default=None,
                    help="model folder path; default: auto by --language "
                         "(ru → podlodka q8, other → whisper-large-v3-turbo-8bit)")
    ap.add_argument("--terms", default="", help="extra canonical terms, comma-separated")
    ap.add_argument("--subs", default="", help="subtitle file (srt/vtt) as second opinion for correction")
    ap.add_argument("--no-correct", action="store_true", help="skip correction prep (no payload, regex-only text)")
    ap.add_argument("--debug-segments", action="store_true", help="save raw segments JSON sidecar")
    ap.add_argument("--meta-title", default=None, help="content title (video name); default: file name with extension")
    ap.add_argument("--meta-author", default=None, help="content author/channel")
    ap.add_argument("--meta-date", default=None, help="content publication date (YYYY-MM-DD)")
    ap.add_argument("--meta-url", default=None, help="canonical content URL (no tracking/time params)")
    args = ap.parse_args()

    # ГЕЙТ: пустые метаданные запрещены — --meta-* либо не передаётся, либо несёт непустое значение
    for name in ("meta_title", "meta_author", "meta_date", "meta_url"):
        v = getattr(args, name)
        if v is not None:
            v = v.strip()
            if not v:
                sys.exit(f"error: --{name.replace('_', '-')} передан с пустым значением — "
                         "пустые метаданные запрещены; запросите значение у пользователя")
            setattr(args, name, v)

    if args.verify:
        b, m = (pathlib.Path(x).expanduser() for x in args.verify)
        if not b.exists():
            sys.exit(f"error: base snapshot not found: {b}")
        if not m.exists():
            sys.exit(f"error: MD not found: {m}")
        verify_mode(b, m, finalize=args.finalize)
        return
    if not args.audio:
        ap.error("нужны аудиофайлы или --verify BASE_JSON MD [--finalize]")

    if args.model is None:
        args.model = str(default_model_for_language(args.language))
    if not pathlib.Path(args.model).is_dir():
        sys.exit(f"error: папка модели не найдена: {args.model} "
                 f"(выбор по языку '{args.language}'; скачайте по инструкции README "
                 f"или передайте --model)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for raw in args.audio:
        src = pathlib.Path(raw).expanduser().resolve()
        if not src.exists():
            print(f"SKIP (not found): {src}")
            continue
        process_one(src, args)


if __name__ == "__main__":
    main()
