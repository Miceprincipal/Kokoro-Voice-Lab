from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from misaki import en
from onnxruntime import InferenceSession


def build_g2p(voice_name: str) -> en.G2P:
    british = voice_name.startswith(("bf_", "bm_"))
    return en.G2P(trf=False, british=british, fallback=None)


def load_vocab(tokenizer_path: Path) -> dict[str, int]:
    tok = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab = tok.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise ValueError(f"No model.vocab dictionary found in {tokenizer_path}")
    return {str(k): int(v) for k, v in vocab.items()}


def phonemes_to_ids(phonemes: str, vocab: dict[str, int]) -> list[int]:
    ids: list[int] = []
    unknown: list[str] = []

    for ch in phonemes:
        if ch in vocab:
            ids.append(vocab[ch])
        else:
            unknown.append(ch)

    if unknown:
        uniq = "".join(sorted(set(unknown)))
        raise ValueError(f"Unknown phoneme characters not in tokenizer vocab: {uniq!r}")

    return ids


def load_style(voice_path: Path, token_count: int) -> np.ndarray:
    voices = np.fromfile(voice_path, dtype=np.float32)
    if voices.size % 256 != 0:
        raise ValueError(f"Voice file is not a valid 256-dim float32 tensor: {voice_path}")

    voices = voices.reshape(-1, 1, 256)

    if token_count >= len(voices):
        raise ValueError(
            f"Text too long for style lookup in {voice_path.name}: "
            f"{token_count=} but only {len(voices)} style rows available"
        )

    return voices[token_count]


def synthesize(
    root: Path,
    voice_path: Path,
    text: str,
    output_path: Path,
    speed: float,
) -> None:
    model_path = root / "onnx" / "model_uint8.onnx"
    tokenizer_path = root / "tokenizer.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_path}")
    if not voice_path.exists():
        raise FileNotFoundError(f"Missing voice: {voice_path}")

    voice_name = voice_path.stem
    g2p = build_g2p(voice_name)
    vocab = load_vocab(tokenizer_path)

    phonemes, _tokens = g2p(text)
    if not phonemes:
        raise ValueError("Misaki returned no phonemes for the input text.")

    token_ids = phonemes_to_ids(phonemes, vocab)

    # tokenizer post-processing uses $ at both ends; model card guidance leaves room for padding
    if len(token_ids) > 510:
        raise ValueError(f"Text too long: {len(token_ids)} token IDs (max 510 before padding)")

    ref_s = load_style(voice_path, len(token_ids))
    input_ids = np.array([[0, *token_ids, 0]], dtype=np.int64)
    style = ref_s.astype(np.float32)
    speed_arr = np.array([speed], dtype=np.float32)

    sess = InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in sess.get_inputs()}

    feed: dict[str, np.ndarray] = {}
    if "input_ids" in input_names:
        feed["input_ids"] = input_ids
    else:
        raise RuntimeError(f"Model inputs do not include 'input_ids': {sorted(input_names)}")

    if "style" in input_names:
        feed["style"] = style
    elif "ref_s" in input_names:
        feed["ref_s"] = style
    else:
        raise RuntimeError(f"Model inputs do not include 'style' or 'ref_s': {sorted(input_names)}")

    if "speed" in input_names:
        feed["speed"] = speed_arr

    audio = sess.run(None, feed)[0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio[0], 24000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voice", required=True, help="Path to voice .bin")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--output", required=True, help="Output wav path")
    parser.add_argument("--speed", type=float, default=1.0, help="Model speed input")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    synthesize(
        root=root,
        voice_path=Path(args.voice),
        text=args.text,
        output_path=Path(args.output),
        speed=args.speed,
    )


if __name__ == "__main__":
    main()