#!/usr/bin/env python3
"""Silero VAD speech segmentation: print speech intervals (seconds) as JSON.

Usage: vad_segments.py <audio> [--threshold 0.5] [--min-speech 0.25] [--min-silence 0.1]
Output: [["start","end"],...] on stdout (seconds, float).
"""
import argparse
import json
import subprocess
import sys
import pathlib
import numpy as np
import onnxruntime as ort

MODEL = pathlib.Path.home() / ".local/share/models/silero-vad/silero_vad.onnx"
SR = 16000


def load_audio_mono16k(path: str) -> np.ndarray:
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SR),
         "-f", "f32le", "-"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--min-speech", type=float, default=0.25)
    ap.add_argument("--min-silence", type=float, default=0.1)
    args = ap.parse_args()

    audio = load_audio_mono16k(args.audio)
    total = len(audio) / SR

    opts = ort.SessionOptions()
    opts.inter_op_num_threads, opts.intra_op_num_threads = 1, 1
    sess = ort.InferenceSession(str(MODEL), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    state = np.zeros((2, 1, 128), dtype=np.float32)
    ctx = np.zeros(64, dtype=np.float32)  # 64-sample context (1-D)

    chunk, hop = 512, 512  # silero expects 512-sample frames at 16k
    speech = np.zeros(len(audio) // hop + 1, dtype=bool)
    probs = []

    for i in range(0, len(audio) - chunk + 1, hop):
        frame = np.concatenate([ctx, audio[i:i + chunk]])
        ctx = frame[-64:]
        out, state = sess.run(
            None, {"input": frame.reshape(1, -1).astype(np.float32),
                   "state": state, "sr": np.array(SR, dtype=np.int64)})
        p = float(out.item())
        probs.append(p)
        speech[i // hop] = p >= args.threshold

    # merge frames into intervals with hysteresis
    min_speech_frames = int(args.min_speech * SR / hop)
    min_sil_frames = int(args.min_silence * SR / hop)
    intervals, start, sil = [], None, 0
    for fi in range(len(speech)):
        if speech[fi]:
            if start is None:
                start = fi
            sil = 0
        elif start is not None:
            sil += 1
            if sil >= min_sil_frames:
                end = fi - sil + 1
                if end - start >= min_speech_frames:
                    intervals.append((start, end))
                start, sil = None, 0
    if start is not None:
        end = len(speech)
        if end - start >= min_speech_frames:
            intervals.append((start, end))

    out = [[round(s * hop / SR, 3), round(e * hop / SR, 3)] for s, e in intervals]
    print(json.dumps({"duration": round(total, 2), "speech_ratio": round(float(np.mean(speech)), 3), "intervals": out}))


if __name__ == "__main__":
    main()
