#!/usr/bin/env python3
"""Full STT pipeline: Silero VAD -> mlx_whisper -> LLM term correction -> paragraph transcript.

Run with the mlx-whisper uv tool python:
  ~/.local/share/uv/tools/mlx-whisper/bin/python vad_transcribe.py <audio> [audio2...] [--language ru]

Stages:
  1. VAD   (silero venv + onnx model)          -> speech intervals
  2. STT   (segments <=28s, one process, condition_on_previous_text=False)
  3. LLM   (deepseek-flash, reasoning_effort=low) -> term spelling correction. One call for
           the whole transcript up to WHOLE_MAX_WORDS; larger transcripts are halved recursively
           (halves in parallel). A piece failing the LLM call or word-diff verification is halved
           again; a single failing segment keeps its regex-only result.

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
DEFAULT_MODEL = str(HOME / ".local/share/models/whisper-podlodka-turbo-MLX-q8")
OUT_DIR = pathlib.Path("/Users/alexander/result-mlx-whisper")
ENV_FILE = HOME / ".hermes/.env"
LLM_MODEL = "deepseek-flash"   # fallback default (DeepSeek V4.1-Flash; thinking on, reasoning_effort=low)

# Language-routed corrector (measured 2026-09-11: 3 runs per cell, two videos, same raw input):
#   en -> glm-5.3-flash (z.ai): stable core 116 fixes vs deepseek's 45; catches Soul/cache/Tibo
#         on every run (deepseek never caught Tibo and actively broke Soul->Sol on one run).
#   ru -> deepseek-flash: catches Wildberries/Телеграме on every run (GLM never did); GLM drifts
#         into rephrasing on Russian at any reasoning level (tested reasoning_effort=low and
#         full thinking). GLM-en quirks: follows subtitle typos (atvive.link, BB1 corpus).
# Env overrides: CORRECT_MODEL_<LANG> (e.g. CORRECT_MODEL_EN). Keys: GLM_API_KEY (+GLM_BASE_URL),
# DEEPSEEK_API_KEY (+DEEPSEEK_BASE_URL). Routed provider without a key -> the other provider;
# none -> LLM stage skipped honestly.
LLM_PROVIDERS = {
    "glm": ("GLM_API_KEY", "GLM_BASE_URL", "https://api.z.ai/api/paas/v4", "glm-5.3-flash"),
    "deepseek": ("DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "https://api.deepseek.com", LLM_MODEL),
}
LLM_ROUTE = {"en": "glm"}   # every other language defaults to deepseek


def read_llm_config(lang: str = "") -> tuple[str, str, str] | None:
    """Corrector (key, url, model) for the transcript language, from ~/.hermes/.env.
    Routing via LLM_ROUTE with per-language CORRECT_MODEL_<LANG> override; falls back to
    the other provider if the routed one has no key. None when no provider key exists."""
    if not ENV_FILE.exists():
        return None
    env = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    order = [LLM_ROUTE.get(lang, "deepseek")]
    order += [p for p in LLM_PROVIDERS if p not in order]
    for pname in order:
        key_env, url_env, default_url, default_model = LLM_PROVIDERS[pname]
        if not env.get(key_env):
            continue
        model = env.get(f"CORRECT_MODEL_{lang.upper()}") or default_model
        url = (env.get(url_env) or default_url).rstrip("/") + "/chat/completions"
        return env[key_env], url, model
    return None

MAX_SEG = 28.0
GAP_MERGE = 0.25
PAD = 0.15
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


class TruncatedError(RuntimeError):
    """finish_reason=length. Deterministic for a given input size: retrying the SAME
    request is a guaranteed repeat truncation (~2x cost burned for nothing) — the caller
    halves the piece instead. Network/HTTP errors below are still retried."""


def llm_call(system: str, user: str, cfg: tuple[str, str, str], attempts: int = 2) -> str:
    """Chat completion for the corrector (reasoning_effort=low; temperature is a no-op
    while thinking is on — kept for compat). cfg = (api_key, chat_completions_url, model).
    Short timeout + bounded retries: a hung/dropped call must never stall the pipeline.
    Truncation raises TruncatedError immediately (no retry — same size truncates again)."""
    key, url, model = cfg
    body = {"model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0.1, "max_tokens": 131072, "reasoning_effort": "low"}
    last_exc = None
    for attempt in range(attempts):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                d = json.load(r)
        except Exception as e:
            last_exc = e
            if attempt + 1 < attempts:
                time.sleep(3 * (attempt + 1))
            continue
        if d["choices"][0].get("finish_reason") == "length":
            raise TruncatedError("LLM output truncated (finish_reason=length); halve the piece")
        return d["choices"][0]["message"]["content"]
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


WHOLE_MAX_WORDS = 10000    # <= this many words: one LLM call; larger transcripts are halved
                           # recursively (measured 2026-09-11: 14826 words = 45.6k completion
                           # tokens of the 131072 cap, finish=stop, verify ok)


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


def build_corrector_prompts(canonical: list[str], sub_ref: str) -> tuple[str, str]:
    canon_str = ", ".join(canonical)
    system = (
        "You are correcting a speech-to-text transcript. You are given TWO imperfect machine "
        "transcripts of the same audio:\n"
        "1. BASE TRANSCRIPT (Whisper, local model) — the base text. Its word stream, word order "
        "and structure are authoritative.\n"
    )
    if sub_ref:
        system += (
            "2. ALT-TRANSCRIPT (auto-captions downloaded from the internet) — a SECOND OPINION from "
            "another recognizer, NOT a reference and NOT necessarily accurate. It has its own "
            "systematic errors (numbers often mangled, words dropped or merged). Never trust it "
            "over the base without a knowledge-based reason.\n"
        )
    system += (
        "The user message may also list LOW-CONFIDENCE BASE SPANS — places where Whisper itself "
        "reported uncertainty; these are the prime suspects for mishearings, check them first "
        "against the alt-transcript and your own knowledge. Every base span NOT listed there was "
        "recognized by Whisper with high confidence.\n"
        "Decide each divergence on its merits: if your own knowledge (technical terms, brands, "
        "product names, version numbers) tells you the correct form — use it. Where knowledge "
        "does not help and Whisper was confident about the span — prefer the base.\n"
        "Fix ONLY clear mishearings and wrong spellings/casing of terms"
        + (", including this canonical list: " + canon_str + ". " if canon_str else ". ")
        + "WHEN IN DOUBT, KEEP THE BASE TEXT UNCHANGED.\n"
        "Rules: (1) The base word stream is preserved: replace words in place; never add, drop, "
        "merge or reorder words. (2) Do NOT rephrase or restyle. (3) Do NOT change punctuation or "
        "grammar. (4) Numbers: keep the base transcript's numeric forms exactly; never convert "
        "between words and digits. (5) Output ONLY the corrected base text, no comments."
    )
    return system, sub_ref


def numeric_tokens(s: str) -> list[str]:
    """All digit runs incl. separated forms ($3.12 -> ['3.12']); used to guarantee the
    LLM never alters numbers regardless of what the prompt says."""
    return re.findall(r"\d+(?:[.,:]\d+)*", s)


LOGPROB_LOW = -0.7  # avg_logprob below this = Whisper itself was unsure about the segment


def uncertain_spans(chunk: list, max_spans: int = 40) -> list[str]:
    return [s["text"] for s in chunk
            if s.get("logprob") is not None and s["logprob"] < LOGPROB_LOW][:max_spans]


def verify_llm_out(pre_text: str, out: str) -> tuple | None:
    """Acceptance gate. Allows replace opcodes of any span (number merges like
    'V four point five' -> 'V 4.5' are legitimate) plus tiny insert/delete wiggle;
    rejects on large drift. Returns (opcodes, base_words, out_words) or None."""
    rw, ow = pre_text.split(), out.split()
    budget = min(10, max(2, len(rw) // 100))
    if abs(len(rw) - len(ow)) > budget:
        return None
    sm = difflib.SequenceMatcher(None, rw, ow)
    ops = sm.get_opcodes()
    id_total = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag in ("insert", "delete"):
            run = max(i2 - i1, j2 - j1)
            if run > 3:            # single run of >3 inserted/deleted words = hallucination
                return None
            id_total += run
    if id_total > budget:
        return None
    return ops, rw, ow


def correct_stage(segments: list, canonical: list[str], cfg: tuple,
                  sub_cues: list | None = None) -> tuple[list, list, list]:
    """Term spelling correction: regex prepass (guaranteed) + LLM for the rest.
    A piece of <=WHOLE_MAX_WORDS words is corrected in ONE call — consistent term casing
    across it. Larger pieces are halved recursively (halves run in parallel, each half is
    re-checked against the limit). A piece that fails the LLM call or strict verification
    is halved again; a single failing segment keeps its regex-only result. There is no
    fixed small-chunk fallback: halving keeps pieces as large as possible, which preserves
    cross-piece term consistency and costs ~4-5x fewer completion tokens than 1200-word
    chunking (measured 2026-09-11). Returns (segments, change_log, errors)."""
    log, errors = [], []
    seg_fixed: list = [None] * len(segments)

    def redistribute(chunk, text: str) -> list[str]:
        outw = text.split()
        fixed, wpos = [], 0
        for s in chunk:
            n = len(s["text"].split())
            fixed.append(" ".join(outw[wpos:wpos + n]))
            wpos += n
        if fixed and wpos < len(outw):
            # LLM вернул больше слов, чем во входе (verify-бюджет это разрешает): хвост
            # приписываем к последнему сегменту куска, а не отбрасываем молча
            extra = len(outw) - wpos
            fixed[-1] = (fixed[-1] + " " + " ".join(outw[wpos:])).strip()
            log.append((f"+{extra} слов", "вывод длиннее входа — хвост приписан к концу куска"))
        return fixed

    def llm_correct(chunk, sub_ref: str, label: str):
        """Returns (fixed_texts, changes, error). regex prepass always applied."""
        raw_text = " ".join(s["text"] for s in chunk)
        pre_text, pre_rules = regex_prepass(raw_text)
        for rule, n in pre_rules:
            log.append((rule, f"×{n} (regex, гарантированно)"))
        if not canonical and not sub_ref:
            return redistribute(chunk, pre_text), [], None
        system, sub_ref = build_corrector_prompts(canonical, sub_ref)
        parts = []
        if sub_ref:
            parts.append("ALT-TRANSCRIPT (second opinion, error-prone):\n" + sub_ref)
        spans = uncertain_spans(chunk)
        if spans:
            parts.append("LOW-CONFIDENCE BASE SPANS (Whisper unsure, check first):\n"
                         + "\n".join("- " + t for t in spans))
        parts.append("BASE TRANSCRIPT TO CORRECT (output only this, corrected):\n" + pre_text)
        user_msg = "\n\n".join(parts)
        try:
            out = llm_call(system, user_msg, cfg)
        except Exception as e:
            return redistribute(chunk, pre_text), [], f"{label}: LLM error ({type(e).__name__}); regex-only"
        ver = verify_llm_out(pre_text, out)
        if ver is None:
            return redistribute(chunk, pre_text), [], f"{label}: strict verify failed (word count/insert-delete); regex-only"
        # deterministic numeric guard, POSITIONAL: rebuild the output from opcodes,
        # taking base words for any opcode whose digit tokens differ ($3.12->$312,
        # words->$3.60, digit-bearing deletions). Prompts are advisory, this is enforced.
        # (Previously done via out.replace(b, a, 1): reverted the FIRST occurrence of b
        # anywhere in the text and corrupted the output on delete opcodes where b == "".)
        ops, rw, ow = ver
        final, kept = [], []
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                final.extend(ow[j1:j2])
                continue
            a, b = " ".join(rw[i1:i2]), " ".join(ow[j1:j2])
            if numeric_tokens(a) != numeric_tokens(b):
                final.extend(rw[i1:i2])
                log.append((a, f"{b} (ОТКЛОНЕНО: числа)"))
            else:
                final.extend(ow[j1:j2])
                kept.append((a, b))
        return redistribute(chunk, " ".join(final)), kept, None

    def word_count(idxs) -> int:
        return sum(len(segments[i]["text"].split()) for i in idxs)

    def split_half(idxs) -> tuple[list, list]:
        """Split segment index list into two halves by word count (both non-empty)."""
        half_words, acc, cut = word_count(idxs) / 2, 0, len(idxs) // 2
        for k, i in enumerate(idxs):
            acc += len(segments[i]["text"].split())
            if acc >= half_words:
                cut = k + 1
                break
        cut = min(max(cut, 1), len(idxs) - 1)
        return idxs[:cut], idxs[cut:]

    def process(idxs, label: str):
        """Correct segments[idxs]; returns error string or None. Halves on overflow/failure."""
        if word_count(idxs) <= WHOLE_MAX_WORDS or len(idxs) == 1:
            chunk = [segments[i] for i in idxs]
            # subs reference scaled to piece size (was a flat 700-word cap / 20000 whole)
            max_sub_words = max(700, int(word_count(idxs) * 1.2))
            sub_ref = (subs_for_range(sub_cues, chunk[0]["start"], chunk[-1]["end"],
                                      max_sub_words) if sub_cues else "")
            log_mark = len(log)  # regex-prepass entries of THIS piece (llm_correct logs them
                                 # before knowing acceptance); dropped if we fall to halving
            fixed, changes, err = llm_correct(chunk, sub_ref, label)
            if err is None or len(idxs) == 1:
                for k, i in enumerate(idxs):
                    seg_fixed[i] = fixed[k]
                if err is None:
                    log.extend(changes)
                return err
            del log[log_mark:]  # halves redo the regex prepass — keep the log free of duplicates
            errors.append(err + " -> halving")
        a, b = split_half(idxs)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            results = list(ex.map(lambda p: process(*p),
                                  [(a, label + "a"), (b, label + "b")]))
        errs = [e for e in results if e]
        return "; ".join(errs) if errs else None

    total_words = word_count(list(range(len(segments))))
    err = process(list(range(len(segments))), "whole")
    if err:
        errors.append(err)
    mode = ("целиком, один вызов" if total_words <= WHOLE_MAX_WORDS
            else "рекурсивное уполовинивание")
    print(f"      режим: {mode} ({total_words} слов)", flush=True)

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
    t_llm = 0.0
    draft_words = sum(len(s["text"].split()) for s in segments)
    print(f"      {stt_info['sent']} сегментов, {draft_words} слов — {t_stt:.0f}с", flush=True)

    cfg = read_llm_config(args.language)
    llm_model = cfg[2] if cfg else LLM_MODEL
    change_log, llm_errors, canonical = [], [], []
    sub_cues = None
    if args.subs:
        sp = pathlib.Path(args.subs).expanduser()
        if sp.exists():
            sub_cues = parse_subs(sp)
            print(f"      субтитры: {sp.name} — {len(sub_cues)} реплик", flush=True)
            if not sub_cues:
                sub_cues = None
                llm_errors.append(f"subs file {sp.name} parsed to 0 cues; ignored")
        else:
            llm_errors.append(f"subs file not found: {sp}; ignored")

    if args.no_llm:
        print("[3/3] LLM-коррекция пропущена (--no-llm)", flush=True)
        fix_segment_junctions(segments)
    elif cfg is None:
        llm_errors.append("no corrector key (DEEPSEEK_API_KEY/GLM_API_KEY) in ~/.hermes/.env; LLM correction skipped")
        print("[3/3] LLM-коррекция: нет ключа — пропуск", flush=True)
        fix_segment_junctions(segments)
    else:
        draft_text = " ".join(s["text"] for s in segments)
        canonical = extract_canonical(draft_text, [t.strip() for t in args.terms.split(",") if t.strip()])
        if sub_cues:
            auto = [t for t in extract_terms_from_subs(sub_cues, draft_text) if t not in canonical]
            if auto:
                canonical += auto
                print(f"      авто-термины из субтитров ({len(auto)}): {', '.join(auto)}", flush=True)
        if canonical or sub_cues:
            t2 = time.time()
            src_note = " + субтитры-референс" if sub_cues else ""
            print(f"[3/3] LLM-коррекция ({llm_model}): канонов {len(canonical)}{src_note}"
                  + (f": {', '.join(canonical)}" if canonical else ""), flush=True)
            segments, change_log, llm_errors2 = correct_stage(segments, canonical, cfg, sub_cues)
            llm_errors.extend(llm_errors2)
            t_llm = time.time() - t2
            print(f"      правок {len(change_log)} — {t_llm:.0f}с", flush=True)
        else:
            print("[3/3] LLM-коррекция: терминов не найдено — пропуск", flush=True)
        fix_segment_junctions(segments)
    total = time.time() - t0
    stem = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", src.stem)
    out = OUT_DIR / f"{datetime.date.today().isoformat()}_{stem}.md"

    llm_line = "выключена (--no-llm)" if args.no_llm else (
        f"{llm_model} (reasoning_effort=low); канонов {len(canonical)}, правок {len(change_log)}, {t_llm:.0f}с"
        if (canonical or change_log or sub_cues) else "не потребовалась (терминов не найдено)")
    subs_line = (f"`{args.subs}` ({len(sub_cues)} реплик) — референс LLM-коррекции"
                 if sub_cues else "не использовались")

    lines = [
        f"# Транскрипция — {src.name}", "",
        f"- **Дата:** {datetime.date.today().isoformat()}",
        f"- **Источник:** `{src}`",
        f"- **Субтитры:** {subs_line}",
        f"- **Длительность:** {fmt_ts(vad['duration'])} | речь (VAD): {speech_s/60:.1f} мин ({vad['speech_ratio']*100:.0f}%)",
        f"- **Модель:** {pathlib.Path(args.model).name}, язык: {args.language}, VAD: Silero",
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
                  f"Модель: {llm_model} (reasoning_effort=low), канон-терминов: {len(canonical)}. "
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
    ap.add_argument("--subs", default="", help="subtitle file (srt/vtt) as LLM-correction reference")
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
