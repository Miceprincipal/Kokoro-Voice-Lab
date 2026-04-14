#!/usr/bin/env python3
"""
Kokoro persistent synthesis server.

Loads the ONNX model ONCE on startup, then handles requests from stdin.
Eliminates cold-start overhead (Python + model load) on every synthesis call.

Protocol (newline-delimited JSON on stdin/stdout):
  Request:  {"voice": "/path/to/voice.bin", "text": "...", "out": "/path/to/out.wav", "speed": 1.0}
  Response: {"status": "ok"} | {"error": "message"}

Startup line (written to stdout before entering request loop):
  {"ready": true, "providers": ["CUDAExecutionProvider", ...]}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from misaki import en
from onnxruntime import InferenceSession

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH     = APP_DIR / "onnx" / "model_uint8.onnx"
TOKENIZER_PATH = APP_DIR / "tokenizer.json"


def _load_vocab() -> dict[str, int]:
    tok = json.loads(TOKENIZER_PATH.read_text(encoding="utf-8"))
    vocab = tok.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise RuntimeError("No model.vocab in tokenizer.json")
    return {str(k): int(v) for k, v in vocab.items()}


def _phonemes_to_ids(phonemes: str, vocab: dict[str, int]) -> list[int]:
    return [vocab[ch] for ch in phonemes if ch in vocab]


def _load_style(voice_path: Path, token_count: int) -> np.ndarray:
    raw = np.fromfile(voice_path, dtype=np.float32)
    if raw.size % 256 != 0:
        raise ValueError(f"Invalid voice bin (not 256-dim): {voice_path.name}")
    voices = raw.reshape(-1, 1, 256)
    idx = min(token_count, len(voices) - 1)
    return voices[idx].astype(np.float32)


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def main() -> None:
    # ── Load model with GPU preference ──────────────────────────────────────
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = InferenceSession(str(MODEL_PATH), providers=providers)
    except Exception:
        # CUDA not available or onnxruntime-gpu not installed — CPU only
        sess = InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])

    active_providers = sess.get_providers()
    input_names = {inp.name for inp in sess.get_inputs()}

    vocab = _load_vocab()

    # Cache G2P objects (one per accent — constructing is expensive)
    _g2p: dict[bool, en.G2P] = {}

    _emit({"ready": True, "providers": active_providers})

    # ── Request loop ─────────────────────────────────────────────────────────
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req   = json.loads(line)
            voice = Path(req["voice"])
            text  = req["text"]
            out   = Path(req["out"])
            speed = float(req.get("speed", 1.0))

            british = voice.stem.startswith(("bf_", "bm_"))
            if british not in _g2p:
                _g2p[british] = en.G2P(trf=False, british=british, fallback=None)

            phonemes, _ = _g2p[british](text)
            if not phonemes:
                raise ValueError("G2P returned no phonemes.")

            token_ids = _phonemes_to_ids(phonemes, vocab)
            if len(token_ids) > 510:
                token_ids = token_ids[:510]

            ref_s     = _load_style(voice, len(token_ids))
            input_ids = np.array([[0, *token_ids, 0]], dtype=np.int64)
            speed_arr = np.array([speed], dtype=np.float32)

            feed: dict[str, np.ndarray] = {"input_ids": input_ids}
            if "style" in input_names:
                feed["style"] = ref_s
            elif "ref_s" in input_names:
                feed["ref_s"] = ref_s
            if "speed" in input_names:
                feed["speed"] = speed_arr

            audio = sess.run(None, feed)[0]
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(out), audio[0], 24000)

            _emit({"status": "ok"})

        except Exception as exc:
            _emit({"error": str(exc)})


if __name__ == "__main__":
    main()
