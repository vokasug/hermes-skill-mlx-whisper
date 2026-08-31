#!/usr/bin/env python3
"""Full STT pipeline: Silero VAD -> mlx_whisper -> LLM term correction -> paragraph transcript.

Run with the mlx-whisper uv tool python:
  ~/.local/share/uv/tools/mlx-whisper/bin/python vad_transcribe.py <audio> [audio2...] [--language ru]

Stages:
  1. VAD   (silero venv + onnx model)          -> speech intervals
  2. STT   (segments <=28s, one process, condition_on_previous_text=False)
  3. LLM   (glm-5.3-flash, reasoning_effort=low) -> term spelling correction (chunks ~1200 words,
           10 parallel, word-diff verified per chunk; failed chunk keeps regex-only result)

Rendering: transcript is cut into sentences (by terminal punctuation of the word stream), then
sentences are grouped into ~60 s blocks — a new block starts at the sentence boundary nearest to
(block start + 60 s). Every block renders as one line "**mm:ss** text" followed by a blank line.
Deterministic; no LLM involved.

Main output: ~/result-mlx-whisper/YYYY-MM-DD_<stem>.md  — readable transcript with timestamps.
Sidecars (only when applicable): <stem>.corrections.md (LLM change log), <stem>.segments.json (--debug-segments).
--no-llm skips stage 3.
"""
import argparse
import collections
import concurrent.futures
import datetime
import difflib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import time
import urllib.request

HOME = pathlib.Path.home()
VAD_PY = HOME / ".local/share/stt-vad/venv/bin/python"
VAD_SCRIPT = pathlib.Path(__file__).parent / "vad_segments.py"   # sibling in skill scripts/
VAD_MODEL = HOME / ".local/share/models/silero-vad/silero_vad.onnx"
DEFAULT_MODEL = str(HOME / ".local/share/models/whisper-podlodka-turbo-MLX-fp16")
OUT_DIR = pathlib.Path("/Users/alexander/result-mlx-whisper")
ENV_FILE = HOME / ".hermes/.env"
ZAI_URL = "https://api.z.ai/api/paas/v4/chat/completions"

MAX_SEG = 28.0
GAP_MERGE = 0.25
PAD = 0.15
CHUNK_WORDS = 1200
MAX_WORKERS = 10
PARA_TARGET_S = 60.0  # block target: a new block starts at the sentence boundary nearest to +60s

# Built-in mishear -> canonical map (source-2 term extraction from the draft).
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


def read_glm_key() -> str | None:
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("GLM_API_KEY="):
            return line.split("=", 1)[1].strip() or None
    return None


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


def clean_word(w: str) -> str:
    return re.sub(r"[^\w'-]", "", w.lower())


def word_counter(text: str) -> collections.Counter:
    return collections.Counter(w for w in (clean_word(t) for t in text.split()) if w)


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
                if txt:
                    all_segments.append({"start": round(seg["start"] + cs, 2),
                                         "end": round(seg["end"] + cs, 2), "text": txt})
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


def build_chunks(segments: list, target_words: int = CHUNK_WORDS) -> list[list]:
    chunks, cur, words = [], [], 0
    for seg in segments:
        cur.append(seg)
        words += len(seg["text"].split())
        if words >= target_words:
            chunks.append(cur)
            cur, words = [], 0
    if cur:
        chunks.append(cur)
    return chunks


def glm_call(system: str, user: str, key: str, attempts: int = 2) -> str:
    """zai chat completion. Short timeout + bounded retries: a hung/dropped call
    must never stall the pipeline (parallel chunks wait for the slowest one)."""
    body = {"model": "glm-5.3-flash",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.1, "max_tokens": 65536, "reasoning_effort": "low"}
    last_exc = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            ZAI_URL, data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
            if d["choices"][0].get("finish_reason") == "length":
                raise RuntimeError("LLM output truncated (finish_reason=length)")
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            last_exc = e
            if attempt + 1 < attempts:
                time.sleep(3 * (attempt + 1))
    raise last_exc


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
    Guaranteed (LLM-invisible) corrections; returns (text, applied [(rule, n)])."""
    applied = []
    for wrong, right in MISHEAR_MAP.items():
        pat = r"\b" + re.escape(wrong) + r"\b"
        text, n = re.subn(pat, right, text, flags=re.IGNORECASE)
        if n:
            applied.append((f"{wrong} → {right}", n))
    return text, applied


def correct_stage(segments: list, canonical: list[str], key: str) -> tuple[list, list, list]:
    """Term spelling correction: regex prepass (guaranteed) + LLM for the rest.
    Retries each chunk once on LLM error. Returns (segments, change_log, errors)."""
    chunks = build_chunks(segments)
    seg_fixed = [None] * len(segments)
    log, errors = [], []
    seg_idx_ranges, pos = [], 0
    for chunk in chunks:
        seg_idx_ranges.append((pos, pos + len(chunk)))
        pos += len(chunk)

    def do_chunk(ci: int):
        lo, hi = seg_idx_ranges[ci]
        chunk = segments[lo:hi]
        raw_text = " ".join(s["text"] for s in chunk)
        # deterministic pass first (guaranteed fixes)
        pre_text, pre_rules = regex_prepass(raw_text)
        for rule, n in pre_rules:
            log.append((rule, f"×{n} (regex, гарантированно)"))
        pre_words = len(pre_text.split())

        def redistribute(text: str) -> list[str]:
            outw = text.split()
            fixed, wpos = [], 0
            for s in chunk:
                n = len(s["text"].split())
                fixed.append(" ".join(outw[wpos:wpos + n]))
                wpos += n
            return fixed

        if not canonical:  # no LLM terms: regex pass is all we need
            return ci, redistribute(pre_text), [], None

        canon_str = ", ".join(canonical)
        system = (
            "You are a transcript spelling corrector. Fix ONLY technical term spellings AND capitalization "
            "per this canonical list: " + canon_str + ". "
            "IMPORTANT: also fix CAPITALIZATION of listed terms wherever they appear. "
            "Rules: (1) Change ONLY terms from the list (spelling or casing). "
            "(2) Do NOT rephrase, add, remove, or reorder any other words. "
            "(3) Do NOT change punctuation or grammar. "
            "(4) Output ONLY the corrected text, no comments."
        )
        try:
            out = glm_call(system, pre_text, key)
        except Exception as e:
            return ci, redistribute(pre_text), [], f"chunk {ci+1}/{len(chunks)}: LLM error ({type(e).__name__}); regex-only"
        if abs(len(out.split()) - pre_words) > 2:
            return ci, redistribute(pre_text), [], f"chunk {ci+1}/{len(chunks)}: word parity {pre_words}->{len(out.split())}; regex-only"
        rw, ow = pre_text.split(), out.split()
        sm = difflib.SequenceMatcher(None, rw, ow)
        changes = [(" ".join(rw[i1:i2]), " ".join(ow[j1:j2]))
                   for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal"]
        return ci, redistribute(out), changes, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for fut in concurrent.futures.as_completed([ex.submit(do_chunk, i) for i in range(len(chunks))]):
            ci, fixed, changes, err = fut.result()
            if err:
                errors.append(err)
                continue
            lo, hi = seg_idx_ranges[ci]
            for k, si in enumerate(range(lo, hi)):
                seg_fixed[si] = fixed[k]
            log.extend(changes)

    out_segments = []
    for s, fx in zip(segments, seg_fixed):
        s2 = dict(s)
        if fx is not None:
            s2["text"] = fx
        out_segments.append(s2)
    return out_segments, log, errors


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

    key = read_glm_key()
    change_log, llm_errors, canonical = [], [], []

    if args.no_llm:
        print("[3/3] LLM-коррекция пропущена (--no-llm)", flush=True)
        fix_segment_junctions(segments)
    elif key is None:
        llm_errors.append("GLM_API_KEY not found in ~/.hermes/.env; LLM correction skipped")
        print("[3/3] LLM-коррекция: нет ключа — пропуск", flush=True)
        fix_segment_junctions(segments)
    else:
        draft_text = " ".join(s["text"] for s in segments)
        canonical = extract_canonical(draft_text, [t.strip() for t in args.terms.split(",") if t.strip()])
        if canonical:
            t2 = time.time()
            print(f"[3/3] LLM-коррекция: канонов {len(canonical)}: {', '.join(canonical)}", flush=True)
            segments, change_log, llm_errors = correct_stage(segments, canonical, key)
            t_llm = time.time() - t2
            print(f"      правок {len(change_log)} — {t_llm:.0f}с", flush=True)
        else:
            print("[3/3] LLM-коррекция: терминов не найдено — пропуск", flush=True)
        fix_segment_junctions(segments)
    total = time.time() - t0
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", src.stem)
    out = OUT_DIR / f"{datetime.date.today().isoformat()}_{stem}.md"

    llm_line = "выключена (--no-llm)" if args.no_llm else (
        f"glm-5.3-flash (reasoning_effort=low); канонов {len(canonical)}, правок {len(change_log)}, {t_llm:.0f}с"
        if (canonical or change_log) else "не потребовалась (терминов не найдено)")

    lines = [
        f"# Транскрипция — {src.name}", "",
        f"- **Дата:** {datetime.date.today().isoformat()}",
        f"- **Источник:** `{src}`",
        f"- **Длительность:** {fmt_ts(vad['duration'])} | речь (VAD): {speech_s/60:.1f} мин ({vad['speech_ratio']*100:.0f}%)",
        f"- **Модель:** whisper-podlodka-turbo-MLX (fp16), язык: {args.language}, VAD: Silero",
        f"- **LLM-коррекция:** {llm_line}",
        f"- **Время:** VAD {t_vad:.0f}с + STT {t_stt:.0f}с + LLM {t_llm:.0f}с = {total/60:.1f} мин",
        "", "## Транскрипт", "",
    ]
    # render: sentences -> ~60 s blocks; each block = one "**mm:ss** text" line + blank line
    sents = split_into_sentences(segments)
    for block in group_into_blocks(sents, audio_end=float(vad["duration"])):
        ts = fmt_short(block[0]["start"])
        lines.append(f"**{ts}** " + " ".join(s["text"] for s in block))
        lines.append("")
    lines.append("")
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"OK {out}")

    if args.save_corrections and change_log:
        cpath = out.with_suffix(".corrections.md")
        clines = [f"# LLM-правки — {src.name}", "",
                  f"Модель: glm-5.3-flash (reasoning_effort=low), канон-терминов: {len(canonical)}. "
                  "Каждая правка проверена word-diff'ом; чанки с ошибкой оставлены без правок.", ""]
        for a, b in change_log:
            clines.append(f"- `{a}` → `{b}`")
        if llm_errors:
            clines += ["", "## Ошибки (чанки оставлены без правок)", ""] + [f"- {e}" for e in llm_errors]
        cpath.write_text("\n".join(clines) + "\n", encoding="utf-8")
        print(f"OK {cpath}")
    if args.debug_segments:
        spath = out.with_suffix(".segments.json")
        spath.write_text(json.dumps(
            {"file": str(src), "segments": segments}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"OK {spath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--terms", default="", help="extra canonical terms, comma-separated")
    ap.add_argument("--no-llm", action="store_true", help="skip LLM correction (regex prepass only)")
    ap.add_argument("--save-corrections", action="store_true", help="save corrections sidecar (default: off)")
    ap.add_argument("--debug-segments", action="store_true", help="save raw segments JSON sidecar")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for raw in args.audio:
        src = pathlib.Path(raw).expanduser().resolve()
        if not src.exists():
            print(f"SKIP (not found): {src}")
            continue
        process_one(src, args)


if __name__ == "__main__":
    main()
