#!/usr/bin/env python3
"""Silero VAD -> mlx_whisper -> датированный MD-транскрипт.

Скрипт только распознаёт речь и пишет ОДИН файл:
~/result-mlx-whisper/YYYY-MM-DD_<имя>.md — транскрипт блоками "**mm:ss** текст".
Коррекции текста здесь нет: правки ослышек и терминов вносит вызывающий агент
в своём пайплайне поверх копии этого файла.

Запуск питоном uv-tool (там mlx):
  ~/.local/share/uv/tools/mlx-whisper/bin/python vad_transcribe.py <аудио> [ещё...] --language ru

Рендеринг: словопоток режется на предложения по финальной пунктуации, предложения
группируются в блоки ~60 с — новый блок на предложении, старт которого ближе всего
к (старт блока + 60 с); хвост <20 с вливается в предыдущий блок (если итог <=80 с).
Каждый блок — строка "**mm:ss** текст" + пустая строка. Детерминированно, без модели.

Sidecar пишется только по запросу: <имя>.segments.json (--debug-segments) —
контракт для других скиллов: {"file", "segments": [{"start","end","text","logprob"}]}.

Специфика запуска под скиллы-потребители — ОДИН флаг `--for <skill>`:
  --for brain-summary                   MD-транскрипт (поведение по умолчанию; флаг
                                        опционален, только фиксирует режим).
  --for multilingual-audio-replacement  MD + <имя>.segments.json (эквивалент
                                        --debug-segments).
  --for translated-video-subtitles      без VAD и MD: весь файл целиком с word
                                        timestamps, один JSON
                                        (~/result-mlx-whisper/<имя>.json) —
                                        сегменты + пословные тайминги.
"""
import argparse
import datetime
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time

HOME = pathlib.Path.home()
VAD_PY = HOME / ".local/share/stt-vad/venv/bin/python"
VAD_SCRIPT = pathlib.Path(__file__).parent / "vad_segments.py"   # sibling in skill scripts/
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


def group_into_blocks(sents: list[dict], audio_end=None) -> list[list[dict]]:
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


def build_md_lines(info: dict, segments: list) -> list[str]:
    """Full MD: header (content metadata block first, then technical lines) + transcript
    rendered into ~60 s blocks."""
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
        f"- **Длительность:** {fmt_ts(info['duration'])} | речь (VAD): {info['speech_s']/60:.1f} мин ({info['speech_ratio']*100:.0f}%)",
        f"- **Модель:** {info['model']}, язык: {info['language']}, VAD: Silero",
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


def run_tvs(src: pathlib.Path, model: str, language: str) -> None:
    """Launch spec for translated-video-subtitles: whole-file transcription with word
    timestamps (no VAD); one JSON artifact (<stem>.json) with segments + word timings."""
    cli = shutil.which("mlx_whisper") or str(HOME / ".local/bin/mlx_whisper")
    subprocess.run([cli, "--model", model, "--language", language,
                    "--condition-on-previous-text", "False",
                    "--word-timestamps", "True", "--output-format", "json",
                    "--output-dir", str(OUT_DIR), str(src)], check=True)
    p = OUT_DIR / f"{src.stem}.json"
    if p.exists():
        print(f"OK {p}")


def process_one(src: pathlib.Path, args):
    if args.for_skill == "translated-video-subtitles":
        print(f"[1/1] STT с word timestamps ({args.language}): {src.name}", flush=True)
        run_tvs(src, args.model, args.language)
        return

    t0 = time.time()
    print(f"[1/2] VAD: {src.name}", flush=True)
    vad = run_vad(src)
    t_vad = time.time() - t0
    speech_s = sum(e - s for s, e in vad["intervals"])
    print(f"      {len(vad['intervals'])} интервалов, речь {speech_s/60:.1f} мин "
          f"({vad['speech_ratio']*100:.0f}%) — {t_vad:.1f}с", flush=True)

    t1 = time.time()
    print(f"[2/2] STT ({args.language})...", flush=True)
    segments, stt_info = cut_and_transcribe(src, vad, args.model, args.language)
    t_stt = time.time() - t1
    draft_words = sum(len(s["text"].split()) for s in segments)
    print(f"      {stt_info['sent']} сегментов, {draft_words} слов — {t_stt:.0f}с", flush=True)

    stem = re.sub(r"^\d{4}-\d{2}-\d{2}_", "", src.stem)
    out = OUT_DIR / f"{datetime.date.today().isoformat()}_{stem}.md"
    info = {
        "src_name": src.name, "src": str(src),
        "meta": {"title": args.meta_title, "author": args.meta_author,
                 "date": args.meta_date, "url": args.meta_url},
        "date": datetime.date.today().isoformat(),
        "duration": vad["duration"], "speech_s": speech_s, "speech_ratio": vad["speech_ratio"],
        "model": pathlib.Path(args.model).name, "language": args.language,
        "t_vad": t_vad, "t_stt": t_stt,
    }
    out.write_text("\n".join(build_md_lines(info, segments)).rstrip() + "\n",
                   encoding="utf-8")
    print(f"OK {out}")

    if args.debug_segments:
        spath = out.with_suffix(".segments.json")
        spath.write_text(json.dumps(
            {"file": str(src), "segments": segments}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"OK {spath}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="*")
    ap.add_argument("--for", dest="for_skill",
                    choices=["brain-summary", "multilingual-audio-replacement",
                             "translated-video-subtitles"],
                    help="специфика запуска под скилл-потребителя (один флаг, см. docstring); "
                         "без флага — обычный MD-транскрипт")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--model", default=None,
                    help="model folder path; default: auto by --language "
                         "(ru → podlodka q8, other → whisper-large-v3-turbo-8bit)")
    ap.add_argument("--debug-segments", action="store_true", help="save raw segments JSON sidecar")
    ap.add_argument("--no-correct", action="store_true",
                    help="no-op: коррекции в скрипте нет; флаг принимается для совместимости вызовов")
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
                         f"пустые метаданные запрещены; запросите значение у пользователя")
            setattr(args, name, v)

    if not args.audio:
        ap.error("нужны аудиофайлы")

    # пресеты --for <skill>
    if args.for_skill == "multilingual-audio-replacement":
        args.debug_segments = True

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
