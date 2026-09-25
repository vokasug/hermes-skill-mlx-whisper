#!/usr/bin/env python3
"""Full STT pipeline: Silero VAD -> mlx_whisper -> term correction -> paragraph transcript.

Run with the mlx-whisper uv tool python:
  ~/.local/share/uv/tools/mlx-whisper/bin/python vad_transcribe.py <audio> [audio2...] [--language ru]

Stages:
  1. VAD   (silero venv + onnx model)          -> speech intervals
  2. STT   (segments <=28s, one process, condition_on_previous_text=False)
  3. Correction prep: regex prepass (mishear dictionary) + junction fix. When canonical
     terms or subtitles exist, writes <stem>.correct-payload.json next to the MD.

Correction is an advisor/judge loop with the MAIN AGENT MODEL as advisor:
  - advisor (main model): rewrites the payload's base_text fixing only mishearings and term
    spelling -> <stem>.corrected.txt (full text, whole transcript at once);
  - judge (this script):  --apply-corrections PAYLOAD CORRECTED re-verifies the output with
    deterministic gates (word-diff budget; positional numeric gate) and rewrites the MD.
    Rejected output leaves the MD on the regex-only base.

Rendering: transcript is cut into sentences (by terminal punctuation of the word stream), then
sentences are grouped into ~60 s blocks — a new block starts at the sentence boundary nearest to
(block start + 60 s). Every block renders as one line "**mm:ss** text" followed by a blank line.
Deterministic; no model involved.

Main output: ~/result-mlx-whisper/YYYY-MM-DD_<stem>.md  — readable transcript with timestamps.
Sidecars (when applicable): <stem>.correct-payload.json, <stem>.corrected.txt,
<stem>.corrections.md (applied change log), <stem>.segments.json (--debug-segments).
--no-correct skips stage 3 (no payload, regex-only text).
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
DEFAULT_MODEL = str(HOME / ".local/share/models/whisper-podlodka-turbo-MLX-q8")
OUT_DIR = HOME / "result-mlx-whisper"

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

# Instructions embedded into every correction payload — the advisor (main agent model)
# reads them from the payload file itself, so the payload is self-contained.
CORRECTOR_INSTRUCTIONS = """\
You are correcting a speech-to-text transcript. This payload gives you:
- base_text: BASE TRANSCRIPT (Whisper, local model) — its word stream, word order and
  structure are authoritative;
- canonical: canonical spellings of terms that must be fixed;
- spans: LOW-CONFIDENCE BASE SPANS — places where Whisper itself reported uncertainty;
  these are the prime suspects for mishearings. Every span NOT listed was recognized
  with high confidence;
- subs_text (may be empty): ALT-TRANSCRIPT (auto-captions) — a SECOND OPINION, NOT a
  reference and NOT necessarily accurate. It has its own systematic errors (numbers often
  mangled, words dropped or merged). Never trust it over the base without a knowledge-based
  reason.
Decide each divergence on its merits: if your own knowledge (technical terms, brands,
product names, version numbers) tells you the correct form — use it. Where knowledge does
not help and Whisper was confident about the span — prefer the base.
Fix ONLY clear mishearings and wrong spellings/casing of terms, including the canonical
list. WHEN IN DOUBT, KEEP THE BASE TEXT UNCHANGED.
Rules: (1) The base word stream is preserved: replace words in place; never add, drop,
merge or reorder words. (2) Do NOT rephrase or restyle. (3) Do NOT change punctuation or
grammar. (4) Numbers: keep the base transcript's numeric forms exactly; never convert
between words and digits. (5) Output ONLY the full corrected base text, no comments."""


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


def subs_for_range(cues: list, t0: float, t1: float, max_words: int = 700) -> str:
    """Plain subtitle text overlapping [t0, t1] (5s padding), capped by word count."""
    out, words = [], 0
    for s, e, txt in cues:
        if e < t0 - 5 or s > t1 + 5:
            continue
        w = len(txt.split())
        if words + w > max_words:
            break
        out.append(txt)
        words += w
    return " ".join(out)


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


def verify_correction_out(pre_text: str, out: str) -> tuple | None:
    """Acceptance gate. Allows replace opcodes of any span plus tiny insert/delete wiggle;
    rejects on large drift. Returns (opcodes, base_words, out_words) or None.
    Reject reason with numbers: verify_correction_diag()."""
    return _verify(pre_text, out)[0]


def verify_correction_diag(pre_text: str, out: str) -> str:
    """Human-readable gate verdict with numbers: 'OK: ...' or 'REJECT: <причина>'."""
    return _verify(pre_text, out)[1]


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


def apply_corrections(payload: dict, corrected_text: str) -> dict:
    """Judge: verify the advisor's corrected text against payload["base_text"] with the
    deterministic gates (word-diff budget; positional numeric gate — any edit whose digit
    tokens differ is rolled back to the base words in place). Returns a dict:
    status "ok" with corrected segments/changes/rejected, or "rejected" with the base kept."""
    base = payload["base_text"]
    segments = payload["segments"]
    ver = verify_correction_out(base, corrected_text)
    if ver is None:
        return {"status": "rejected", "reason": verify_correction_diag(base, corrected_text),
                "segments": segments, "changes": [], "rejected": [], "note": None}
    ops, rw, ow = ver
    final, changes, rejected = [], [], []
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            final.extend(ow[j1:j2])
            continue
        a, b = " ".join(rw[i1:i2]), " ".join(ow[j1:j2])
        if numeric_tokens(a) != numeric_tokens(b):
            final.extend(rw[i1:i2])
            rejected.append((a, b))
        else:
            final.extend(ow[j1:j2])
            changes.append((a, b))
    fixed, note = redistribute_words(segments, " ".join(final))
    out_segments = [dict(s, text=fx) for s, fx in zip(segments, fixed)]
    return {"status": "ok", "segments": out_segments, "changes": changes,
            "rejected": rejected, "note": note}


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

    canonical, regex_log, payload_path = [], [], None
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
            fixed, note = redistribute_words(segments, pre_text)
            for s, fx in zip(segments, fixed):
                s["text"] = fx
            fix_segment_junctions(segments)
            base_text = " ".join(s["text"] for s in segments)
            subs_text = (subs_for_range(sub_cues, segments[0]["start"], segments[-1]["end"],
                                        max(700, int(draft_words * 1.2))) if sub_cues else "")
            payload_path = out.with_name(out.stem + ".correct-payload.json")
            corrected_path = out.with_name(out.stem + ".corrected.txt")
            payload = dict(info, version=1, out=str(out), segments=segments,
                           base_text=base_text, canonical=canonical,
                           spans=uncertain_spans(segments), subs_text=subs_text,
                           regex_log=regex_log, instructions=CORRECTOR_INSTRUCTIONS)
            payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
            src_note = " + субтитры-второе-мнение" if sub_cues else ""
            print(f"[3/3] Коррекция терминов: канонов {len(canonical)}{src_note}"
                  + (f": {', '.join(canonical)}" if canonical else ""), flush=True)
            print(f"      payload → {payload_path}", flush=True)
            print(f"      следующий шаг: основная модель правит base_text → {corrected_path.name}, "
                  f"затем --apply-corrections", flush=True)
            corr_line = f"payload `{payload_path.name}` — ожидает правок основной модели"
        else:
            fix_segment_junctions(segments)
            corr_line = "не потребовалась (терминов не найдено)"
            print("[3/3] Коррекция терминов: терминов не найдено — пропуск", flush=True)

    out.write_text("\n".join(build_md_lines(info, segments, corr_line)).rstrip() + "\n",
                   encoding="utf-8")
    print(f"OK {out}")
    if payload_path:
        print(f"PAYLOAD {payload_path}")

    if args.debug_segments:
        spath = out.with_suffix(".segments.json")
        spath.write_text(json.dumps(
            {"file": str(src), "segments": segments}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"OK {spath}")


def apply_mode(payload_path: pathlib.Path, corrected_path: pathlib.Path):
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    corrected = corrected_path.read_text(encoding="utf-8").strip()
    res = apply_corrections(payload, corrected)
    out = pathlib.Path(payload["out"])

    if res["status"] != "ok":
        corr_line = f"ОТКЛОНЕНА верификацией ({res['reason']}) — текст оставлен без правок"
        out.write_text("\n".join(build_md_lines(payload, payload["segments"], corr_line)).rstrip() + "\n",
                       encoding="utf-8")
        print(f"REJECTED {corrected_path}: {res['reason']}; MD без правок: {out}")
        sys.exit(1)

    changes, rejected = res["changes"], res["rejected"]
    corr_line = f"основная модель; канонов {len(payload['canonical'])}, правок {len(changes)}"
    if rejected:
        corr_line += f", отклонено числовым гейтом: {len(rejected)}"
    out.write_text("\n".join(build_md_lines(payload, res["segments"], corr_line)).rstrip() + "\n",
                   encoding="utf-8")
    print(f"OK {out}")

    clines = [f"# Правки — {payload['src_name']}", "",
              "Советчик: основная модель агента. Принятие каждой правки — детерминированные "
              "гейты этого скрипта (word-diff бюджет, позиционный числовой гейт).", ""]
    for rule, n in payload.get("regex_log", []):
        clines.append(f"- `{rule}` ×{n} (regex, гарантированно)")
    for a, b in changes:
        clines.append(f"- `{a}` → `{b}`")
    if res["note"]:
        clines.append(f"- {res['note'][0]}: {res['note'][1]}")
    if rejected:
        clines += ["", "## Отклонено числовым гейтом", ""]
        clines += [f"- `{a}` → `{b}` (ОТКЛОНЕНО: числа)" for a, b in rejected]
    cpath = out.with_suffix(".corrections.md")
    cpath.write_text("\n".join(clines) + "\n", encoding="utf-8")
    print(f"OK {cpath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="*")
    ap.add_argument("--apply-corrections", nargs=2, metavar=("PAYLOAD_JSON", "CORRECTED_TXT"),
                    help="judge step: verify advisor corrections with the deterministic gates "
                         "and rewrite the MD")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--model", default=DEFAULT_MODEL)
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

    if args.apply_corrections:
        p, c = (pathlib.Path(x).expanduser() for x in args.apply_corrections)
        if not p.exists():
            sys.exit(f"error: payload not found: {p}")
        if not c.exists():
            sys.exit(f"error: corrected text not found: {c}")
        apply_mode(p, c)
        return
    if not args.audio:
        ap.error("нужны аудиофайлы или --apply-corrections PAYLOAD_JSON CORRECTED_TXT")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for raw in args.audio:
        src = pathlib.Path(raw).expanduser().resolve()
        if not src.exists():
            print(f"SKIP (not found): {src}")
            continue
        process_one(src, args)


if __name__ == "__main__":
    main()
