# Kokoro Voice Lab

A GUI tool for rating, blending, and previewing voices from the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS model.

![logo](logo.png)

---

## What it does

**Ratings tab** — Work through all 54 Kokoro voices and rate each one on 8 perceptual axes: age, authority, clarity, energy, gender presentation, pitch, roughness, and warmth. Navigate with arrow keys. Ratings auto-save as you go.

**Mixer tab** — Blend up to 3 voices by weight. Per-slot pitch shift (semitones) and speed control for individual slot previews. Global output speed (0.25–2.0×) and pitch shift (±12 semitones) applied to the final blended preview and export. Export the result as a `.bin` file that Kokoro can use directly as a custom voice.

**Trait Match Assist** — Set target trait values and the tool suggests the 3 closest voices from your rated library. Uses the same axis scores, so suggestions get better the more voices you rate.

**Persistent synth server** — Loads the ONNX model once on startup instead of reloading it every preview. Eliminates the cold-start delay (~3s per call without this). Uses GPU automatically if `onnxruntime-gpu` is installed.

**Voice pre-cache** — After the server starts, all voice WAVs are generated in the background. Once cached, every preview in the Ratings tab is an instant click with no synthesis delay.

---

## Setup

### 1. Get the Kokoro model files

Download the ONNX release from HuggingFace:

```
https://huggingface.co/hexgrad/Kokoro-82M/tree/main
```

You need these files/folders placed **in the same directory as the scripts**:

```
kokoro_voice_lab.py   ← this repo
synth_server.py       ← this repo
infer.py              ← this repo
config.json           ← from HuggingFace
tokenizer.json        ← from HuggingFace
tokenizer_config.json ← from HuggingFace
onnx/
  model_uint8.onnx    ← from HuggingFace
voices/
  af_alloy.bin        ← from HuggingFace
  af_bella.bin
  ... (54 voices total)
```

### 2. Install Python dependencies

```bash
pip install onnxruntime misaki[en] soundfile numpy pillow
```

For GPU support (optional — CUDA required):
```bash
pip install onnxruntime-gpu misaki[en] soundfile numpy pillow
```

`ffmpeg` is optional — only needed if you use pitch shift or speed controls that deviate from default (pitch ≠ 0, speed ≠ 1.0). If you don't have it, leave those at their defaults and it's bypassed entirely.

### 3. Run

```bash
python kokoro_voice_lab.py
```

The tool auto-detects the `voices/` folder and working directory on first launch.

---

## Quick start

1. **Start the server** — Config tab → **▶ Start Server**. Takes ~3 seconds to load the model. The status shows `Server: running [CPU]` or `Server: running [GPU]`.

2. **Pre-cache voices** — Happens automatically after the server starts. Watch the progress in the Ratings tab (`Pre-caching 1/54…`). Takes a minute or two on CPU, much faster on GPU.

3. **Rate voices** — Ratings tab. Click a voice, hit **▶ Preview**, adjust the 8 sliders, add notes, hit **✓ Save Voice**. Or use `←` / `→` arrow keys to navigate — it auto-saves as you move.

4. **Blend voices** — Mixer tab. Pick voices for each slot, set weights, preview the mix. Use the **Output Speed** and **Output Pitch** sliders to tune the final result. Export as `.bin` when you're happy.

5. **Use Trait Match** — Set target trait sliders in the Mixer tab → **Suggest 3 Voices**. Loads the best matches into the slots.

---

## Mixer: how weighting works

Blending mixes the raw voice embedding tensors (256-dim float32 vectors) by weighted average. Weights are always normalized to sum to 1 before the blend is applied — this keeps the mixed tensor at the same scale as the original voices, which is required for the ONNX model to produce valid audio. A 50/30/20 split is identical to a 5/3/2 split.

**Per-slot speed and pitch** (in each slot card) apply when previewing that slot individually. They are averaged across active slots for mix previews.

**Output Speed and Output Pitch** (in the Mix Output panel) apply to the final blended synthesis — use these to tune the whole mix without touching individual slots.

Exported `.bin` files are always normalized. They match the format of the vendor voices and can be dropped into the `voices/` folder for use with any Kokoro-compatible tool.

---

## Files

| File | Purpose |
|------|---------|
| `kokoro_voice_lab.py` | Main GUI application |
| `synth_server.py` | Persistent synthesis server (model loaded once) |
| `infer.py` | Standalone synthesis script (also used as fallback) |
| `voice_lab_config.json` | Session config — auto-saved, contains your local paths |
| `voice_ratings.json` | Your voice ratings — created when you first save |
| `logo.png` | The rat |

---

## Exported `.bin` files

Blended voice files are saved to the `exports/` folder by default. They're the same format as the voices in `voices/` — raw little-endian float32, 256 values per row, no header — and can be dropped in alongside them to use with any Kokoro-compatible tool.

The sidecar JSON (saved alongside the `.bin`) records what voices were blended and at what weights, so you can recreate or adjust the mix later.

---

## Notes

- The tool saves your config (paths, slot settings, session state) to `voice_lab_config.json` automatically. This file is gitignored by default since it contains your local paths.
- `voice_ratings.json` is also gitignored — if you want to share your ratings with someone, commit it manually or send it directly.
- The `preview_cache/` folder holds pre-generated WAVs keyed to the test sentence. Delete it to force regeneration (e.g. after changing the test sentence).
- Ratings use a 1–5 scale on each axis. Trait Match distance is Manhattan distance across all rated axes.

---

## Requirements

| Package | Required | Notes |
|---------|----------|-------|
| `onnxruntime` | Yes | Or `onnxruntime-gpu` for CUDA acceleration |
| `misaki[en]` | Yes | G2P phonemiser — converts text to phoneme tokens |
| `soundfile` | Yes | WAV read/write |
| `numpy` | Yes | Voice bin arithmetic and tensor operations |
| `Pillow` | Recommended | Logo transparency support; gracefully absent otherwise |
| `ffmpeg` | Optional | Required only for non-default pitch or speed values |

Python 3.10 or later.

---

## Acknowledgements

This tool is built on the work of several projects:

**[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** by [hexgrad](https://huggingface.co/hexgrad) — the TTS model this tool is built around. Kokoro is an open-weight, Apache 2.0 licensed text-to-speech model. All voice `.bin` files and the ONNX model are from this release.

**[misaki](https://github.com/hexgrad/misaki)** by hexgrad — the G2P (grapheme-to-phoneme) library used to convert input text into phoneme token sequences before synthesis. Handles both American and British English pronunciation.

**[ONNX Runtime](https://onnxruntime.ai/)** by Microsoft — cross-platform inference engine used to run the Kokoro model. The `onnxruntime-gpu` variant enables CUDA acceleration where available.

**[soundfile](https://python-soundfile.readthedocs.io/)** / **[libsndfile](https://libsndfile.github.io/libsndfile/)** — WAV I/O used for reading and writing synthesis output. soundfile is the Python wrapper; the underlying work is libsndfile by Erik de Castro Lopo.

**[NumPy](https://numpy.org/)** — voice bin loading, tensor arithmetic, and weight normalization.

**[Pillow](https://python-pillow.org/)** — used for logo transparency and image handling in the GUI watermark.

**[ffmpeg](https://ffmpeg.org/)** — optional post-processing for pitch shift and speed transforms applied to synthesized audio.

**[tkinter](https://docs.python.org/3/library/tkinter.html)** — Python's standard GUI toolkit, used for the entire interface.

---

*Built for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad.*
