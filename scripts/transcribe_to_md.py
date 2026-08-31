#!/usr/bin/env python3
"""Transcribe audio via mlx_whisper (podlodka-turbo fp16) -> dated Markdown file.

Output: /Users/alexander/result-mlx-whisper/YYYY-MM-DD_<audio-basename>.md
Only stdlib + mlx_whisper CLI. Model lives in RAM only while mlx_whisper runs.
"""
import argparse
import datetime
import json
import pathlib
import subprocess
import tempfile

MODEL = pathlib.Path.home() / ".local/share/models/whisper-podlodka-turbo-MLX-fp16"
OUT_DIR = pathlib.Path("/Users/alexander/result-mlx-whisper")


def ffprobe_duration(path: pathlib.Path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=30)
        return float(r.stdout.strip())
    except Exception:
        return None


def fmt_ts(sec) -> str:
    if sec is None:
        return "—"
    sec = int(float(sec))
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_one(src: pathlib.Path, model: str, language: str) -> pathlib.Path:
    dur = ffprobe_duration(src)
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(
            ["mlx_whisper", "--model", model, "--language", language,
             "--output-format", "json", "--output-dir", td,
             "--condition-on-previous-text", "False", str(src)],
            check=True, capture_output=True, text=True)
        data = json.loads((pathlib.Path(td) / f"{src.stem}.json").read_text())

    text = (data.get("text") or "").strip()
    segments = data.get("segments") or []
    out = OUT_DIR / f"{datetime.date.today().isoformat()}_{src.stem}.md"

    lines = [
        f"# Распознавание речи — {src.name}",
        "",
        f"- **Дата:** {datetime.date.today().isoformat()}",
        f"- **Источник:** `{src}`",
        f"- **Длительность:** {fmt_ts(dur)}",
        f"- **Модель:** whisper-podlodka-turbo-MLX (fp16), язык: {language}",
        "",
        "## Текст",
        "",
        text,
    ]
    if segments:
        lines += ["", "## Сегменты", "", "| Начало | Конец | Текст |", "|---|---|---|"]
        for seg in segments:
            lines.append(
                f"| `{fmt_ts(seg.get('start'))}` | `{fmt_ts(seg.get('end'))}` "
                f"| {str(seg.get('text', '')).strip()} |")
        lines.append("")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("audio", nargs="+", help="audio/video file(s)")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--model", default=str(MODEL))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for raw in args.audio:
        src = pathlib.Path(raw).expanduser().resolve()
        if not src.exists():
            print(f"SKIP (not found): {src}")
            continue
        out = transcribe_one(src, args.model, args.language)
        print(f"OK {out}")


if __name__ == "__main__":
    main()
