"""Extend cache/voice-audition/voice-analysis.csv with voices not yet analysed.

Reads the existing CSV, finds .bin files without entries, synthesises each one
using the same test sentence, runs acoustic analysis, and appends rows.

Usage:
    python extend_voice_analysis.py

Requires: librosa, soundfile, numpy, onnxruntime, misaki
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

# ── paths ────────────────────────────────────────────────────────────────────
APP_DIR       = Path(__file__).resolve().parent
VOICES_DIR    = APP_DIR / "voices"
AUDITION_DIR  = APP_DIR / "cache" / "voice-audition"
CSV_PATH      = AUDITION_DIR / "voice-analysis.csv"
TEMP_WAV      = AUDITION_DIR / "_extend_tmp.wav"

# Same sentence as the original analysis
TEST_SENTENCE = "I am Batman. Stay calm, kid. You've got a lot to learn."

VOICE_GROUPS = {
    "af": "US Female",   "am": "US Male",
    "bf": "UK Female",   "bm": "UK Male",
    "ef": "Euro Female", "em": "Euro Male",
    "ff": "French Female",
    "hf": "Hindi Female","hm": "Hindi Male",
    "if": "Italian Female", "im": "Italian Male",
    "jf": "Japanese Female", "jm": "Japanese Male",
    "pf": "Portuguese Female", "pm": "Portuguese Male",
    "zf": "Chinese Female", "zm": "Chinese Male",
}


def group_for(voice_id: str) -> str:
    prefix = voice_id[:2]
    return VOICE_GROUPS.get(prefix, "Other")


def load_existing_ids() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return {row["id"].strip() for row in csv.DictReader(f) if row.get("id")}


def synthesise_voice(voice_id: str, voice_path: Path, out_wav: Path) -> None:
    """Synthesise TEST_SENTENCE for one voice using infer.py's logic directly."""
    from misaki import en as misaki_en
    from onnxruntime import InferenceSession

    # Build G2P — British variant for bf_/bm_, US for everything else
    british = voice_id.startswith(("bf_", "bm_"))
    g2p = misaki_en.G2P(trf=False, british=british, fallback=None)

    tokenizer_path = APP_DIR / "tokenizer.json"
    vocab_raw = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    vocab: dict[str, int] = {str(k): int(v) for k, v in vocab_raw["model"]["vocab"].items()}

    phonemes, _ = g2p(TEST_SENTENCE)
    if not phonemes:
        raise RuntimeError(f"G2P returned no phonemes for {voice_id}")

    token_ids = [vocab[ch] for ch in phonemes if ch in vocab]
    if not token_ids:
        raise RuntimeError(f"No valid token IDs for {voice_id}")

    voices_arr = np.fromfile(voice_path, dtype=np.float32)
    if voices_arr.size % 256 != 0:
        raise RuntimeError(f"Invalid .bin size for {voice_id}")
    voices_arr = voices_arr.reshape(-1, 1, 256)

    tok_idx = min(len(token_ids), len(voices_arr) - 1)
    style = voices_arr[tok_idx].astype(np.float32)
    input_ids = np.array([[0, *token_ids, 0]], dtype=np.int64)
    speed_arr = np.array([1.0], dtype=np.float32)

    model_path = APP_DIR / "onnx" / "model_uint8.onnx"
    sess = InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in sess.get_inputs()}

    feed: dict[str, np.ndarray] = {"input_ids": input_ids}
    feed["style" if "style" in input_names else "ref_s"] = style
    if "speed" in input_names:
        feed["speed"] = speed_arr

    audio = sess.run(None, feed)[0]
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), audio[0], 24000)


def analyse_wav(wav_path: Path) -> dict:
    """Return acoustic metrics matching the existing CSV columns."""
    try:
        import librosa
    except ImportError:
        print("ERROR: librosa not installed. Run: pip install librosa")
        sys.exit(1)

    y, sr = librosa.load(str(wav_path), sr=24000, mono=True)
    duration = float(len(y) / sr)
    rms = float(np.sqrt(np.mean(y ** 2)))

    # F0 via pyin — voiced frames median
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=50, fmax=500, sr=sr,
        frame_length=2048, hop_length=512,
    )
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else f0
    f0_hz = float(np.median(voiced_f0[voiced_f0 > 0])) if voiced_f0 is not None and np.any(voiced_f0 > 0) else 0.0

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)))

    # Brightness: spectral centroid (Hz) / 1000
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=2048, hop_length=512)
    brightness_ratio = float(np.mean(centroid) / 1000.0)

    return {
        "duration_s":       round(duration, 2),
        "rms":              round(rms, 4),
        "f0_hz":            round(f0_hz, 1),
        "zcr":              round(zcr, 4),
        "brightness_ratio": round(brightness_ratio, 3),
    }


def describe(f0_hz: float, brightness: float, rms: float) -> str:
    """Generate a rough description string from acoustic metrics."""
    if f0_hz < 120:
        pitch_label = "low"
    elif f0_hz < 165:
        pitch_label = "mid"
    elif f0_hz < 200:
        pitch_label = "high"
    else:
        pitch_label = "very high"

    if brightness < 0.6:
        bright_label = "dark"
    elif brightness < 1.0:
        bright_label = "balanced"
    elif brightness < 1.4:
        bright_label = "bright"
    else:
        bright_label = "very bright"

    if rms < 0.05:
        energy_label = "soft"
    elif rms < 0.09:
        energy_label = "medium"
    else:
        energy_label = "strong"

    return f"{pitch_label}, {bright_label}, {energy_label}"


def append_to_csv(row: dict) -> None:
    fieldnames = ["group", "id", "file", "duration_s", "rms", "f0_hz",
                  "zcr", "brightness_ratio", "description", "error"]
    write_header = not CSV_PATH.exists()
    with open(CSV_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    existing = load_existing_ids()
    bins = sorted(VOICES_DIR.glob("*.bin"))
    missing = [b for b in bins if b.stem not in existing and b.stem != "test" and b.stem != "af"]

    if not missing:
        print("Nothing to do — all voices already in CSV.")
        return

    print(f"Found {len(missing)} voices to analyse:\n  " +
          "\n  ".join(b.stem for b in missing))
    print()

    # Load ONNX model once by doing a warm-up before the loop.
    # We rely on InferenceSession caching within the process — each call in
    # synthesise_voice creates a new session, but for 29 voices this is acceptable.
    # A future optimisation would thread the session through, but that requires
    # refactoring infer.py. Not worth it for a one-shot script.

    AUDITION_DIR.mkdir(parents=True, exist_ok=True)

    for i, bin_path in enumerate(missing, 1):
        voice_id = bin_path.stem
        wav_dest = AUDITION_DIR / f"{voice_id}.wav"
        print(f"[{i}/{len(missing)}] {voice_id} ...", end=" ", flush=True)

        error_str = ""
        metrics = {}
        try:
            if not wav_dest.exists():
                synthesise_voice(voice_id, bin_path, wav_dest)
            metrics = analyse_wav(wav_dest)
            desc = describe(metrics["f0_hz"], metrics["brightness_ratio"], metrics["rms"])
            print(f"f0={metrics['f0_hz']} Hz  brightness={metrics['brightness_ratio']}  → {desc}")
        except Exception as exc:
            error_str = str(exc)
            print(f"ERROR: {exc}")
            metrics = {"duration_s": 0, "rms": 0, "f0_hz": 0, "zcr": 0, "brightness_ratio": 0}
            desc = ""

        append_to_csv({
            "group":            group_for(voice_id),
            "id":               voice_id,
            "file":             f"{voice_id}.wav",
            "duration_s":       metrics["duration_s"],
            "rms":              metrics["rms"],
            "f0_hz":            metrics["f0_hz"],
            "zcr":              metrics["zcr"],
            "brightness_ratio": metrics["brightness_ratio"],
            "description":      desc,
            "error":            error_str,
        })

    if TEMP_WAV.exists():
        TEMP_WAV.unlink()

    print(f"\nDone. CSV updated: {CSV_PATH}")
    print("Re-load the voice folder in Voice Lab to rebuild the pitch axis.")


if __name__ == "__main__":
    main()
