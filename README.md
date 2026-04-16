# Kokoro Voice Lab

A GUI tool for rating, blending, and approximating voices from the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS model.

![logo](logo.png)

---

## What it does

**Ratings tab** — Work through all 54 Kokoro voices and rate each one on 8 perceptual axes: age, authority, clarity, energy, gender presentation, pitch, roughness, and warmth. Navigate with arrow keys. Ratings auto-save as you go. Scale is 1–10.

**Mixer tab** — Blend up to 3 voices by weight. Per-slot pitch and speed sliders let you audition each voice individually. Separate Output Speed (0.25–2.0×) and Output Pitch (±12 semitones) controls apply to the final blended preview and export. Export the result as a `.bin` file that Kokoro can use directly as a custom voice.

**Bake slot pitch into .bin** — Checkbox in the Mixer tab. When enabled, per-slot pitch offsets are baked directly into the voice embeddings before blending, rather than applied as audio post-processing. The exported `.bin` will already contain the pitch-shifted voice — no ffmpeg required at inference time. Uses Ridge regression over the acoustic analysis CSV to find the pitch direction in embedding space.

**Trait Match Assist** — Set target trait values and the tool suggests the 3 closest voices from your rated library. Uses the same axis scores, so suggestions get better the more voices you rate.

**Persistent synth server** — Loads the ONNX model once on startup instead of reloading it every preview. Eliminates the cold-start delay (~3s per call without this). Uses GPU automatically if `onnxruntime-gpu` is installed.

**Voice pre-cache** — After the server starts, all voice WAVs are generated in the background. Once cached, every preview in the Ratings tab is an instant click with no synthesis delay. Use **Re-cache All Voices** in the Config tab to force regeneration (e.g. after changing the test sentence or adding new voices).

**Acoustic slot info** — Each mixer slot shows the selected voice's measured F0 (Hz) and brightness ratio from the acoustic analysis CSV, so you know what you're blending before you preview.

**Voice list auto-refresh** — After exporting a `.bin`, the voice dropdowns automatically rescan the voices folder. Newly exported voices are immediately available in all slots without restarting.

**Voice Match tab** — Drop in any WAV or MP3 of a real voice and approximate it from the Kokoro voice pool. Full pipeline:

1. **Build Fingerprints** — synthesises a coverage sentence for every voice, extracts an 86-dim audio feature vector and a 256-dim canonical embedding, trains a Ridge regressor (audio → embedding space), and saves everything to `voice_match_mfcc.json`. Only needs to run once, or after adding new voices.
2. **Find Match** — measures reference F0, gates the pool to same-pitch voices, ranks by embedding-space cosine distance (preferred) or audio-space cosine distance, then synthesises the top 12 saying your transcript and re-ranks by audio cosine distance. Eliminates content-mismatch contamination from the coverage sentence.
3. **Optimise Blend** — seeds 5 starting candidates (singles and blends of top pool voices), hill-climbs weights and pitch against synthesised audio using cosine distance, runs a fine pass (±5% weights, ±0.5st pitch), then Nelder-Mead continuous refinement (scipy, optional). Result scored: excellent / good / fair / rough.
4. **Fine Tune** — the optimiser result auto-loads here. Three slots with voice dropdown, weight slider (normalised contribution %), and pitch slider (±5st). Speed slider (0.5–2.0×) for pacing control. Per-slot preview and full blend preview. Export directly or send to the Mixer.
5. **Blend Explorer** — each round injects 3 pool voices (18% each) into your current blend and offers them as A / B / C choices. Play each (synthesised saying your transcript), pick the closest — Fine Tune updates and the next round tests 3 more voices. Export button included.

Requires `librosa` for fingerprints and matching. Transcription requires `faster-whisper` or `openai-whisper`.

---

## Setup

### 1. Get the Kokoro model files

Download the ONNX release from HuggingFace:

```
https://huggingface.co/hexgrad/Kokoro-82M/tree/main
```

You need these files/folders placed **in the same directory as the scripts**:

```
kokoro_voice_lab.py        ← this repo
extend_voice_analysis.py   ← this repo
synth_server.py            ← this repo
infer.py                   ← this repo
config.json                ← from HuggingFace
tokenizer.json             ← from HuggingFace
tokenizer_config.json      ← from HuggingFace
onnx/
  model_uint8.onnx         ← from HuggingFace
voices/
  af_alloy.bin             ← from HuggingFace
  af_bella.bin
  ... (54 voices total)
```

### 2. Install Python dependencies

```bash
pip install onnxruntime misaki[en] soundfile numpy pillow scikit-learn librosa
```

For GPU support (optional — CUDA required):
```bash
pip install onnxruntime-gpu misaki[en] soundfile numpy pillow scikit-learn librosa
```

`ffmpeg` is optional — only needed if you use pitch shift or speed controls that deviate from default (pitch ≠ 0, speed ≠ 1.0) **and** the Bake slot pitch checkbox is **off**. With baking enabled, pitch is applied at the embedding level and ffmpeg is not required.

`scikit-learn` is required for the Bake slot pitch feature (Ridge regression to build the pitch direction vector). If not installed, the checkbox is disabled and a status message explains why.

`librosa` is required for Voice Match (MFCC fingerprinting, F0 measurement, acoustic analysis). Also needed for `extend_voice_analysis.py`.

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

6. **Match a real voice** — Voice Match tab. Browse to a WAV or MP3, transcribe or paste the text, **Build Fingerprints** (once), **Find Match**, **Optimise Blend**. Use Fine Tune and Blend Explorer to converge by ear.

---

## Voice Match: how it works

### Architecture overview

The matching pipeline operates in Kokoro's embedding space rather than raw audio feature space. MFCC similarity in audio space doesn't correlate reliably with Kokoro identity similarity — one voice can produce similar MFCCs for different texts, and different voices can produce similar MFCCs for the same text. The solution: predict where the reference sits in Kokoro's 256-dim voice embedding space and compare directly.

```
Reference audio
  → 86-dim features (MFCC + Δ + ΔΔ + log-mel + spectral contrast)
  → Ridge regressor (trained on voice corpus)
  → predicted 256-dim Kokoro embedding
  → cosine distance to each voice's canonical embedding
  → top 12 → synthesis re-rank → result
```

Each voice `.bin` has a **canonical embedding**: the mean of all its rows, giving a stable single identity vector used for matching.

### Find Match pipeline

1. Loads reference audio, trims silence, extracts the best 8-second window by RMS energy. Peak-normalises before analysis.
2. Measures reference F0 with `librosa.pyin`. Filters the voice pool to voices within ±40% of that F0 — prevents cross-gender matches.
3. **Mode A (preferred)** — extracts 86-dim features from the reference, multiplies by the Ridge regressor to predict a Kokoro embedding, ranks all voices by cosine distance to their canonical embeddings. No synthesis needed for the coarse pass.
4. **Mode B (fallback)** — if the regressor is absent, z-normalises features across the population and ranks by cosine distance in feature space.
5. **Mode C (legacy)** — if fingerprints are pre-v2, falls back to weighted MFCC L2 distance with a rebuild prompt.
6. Takes the top 12, synthesises each saying the transcript, re-ranks by audio cosine distance. Eliminates content-mismatch contamination from the coverage sentence.

For reference audio longer than 20 seconds, the pipeline averages feature measurements across three windows (start, middle, end).

### Optimise Blend

Seeds 5 starting candidates (the top single voice, two 70/30 blends, a 60/25/15 three-way, and a balanced three-way). Three optimisation stages run in sequence:

1. **Hill-climb** — random weight and pitch mutations (±15%), stops after 2 consecutive no-improve rounds (max 10 iterations).
2. **Fine pass** — deterministic ±5% on each weight, ±0.5st on each pitch, up to 5 full sweeps.
3. **Nelder-Mead** (requires `scipy`) — continuous gradient-free refinement from the best candidate, capped at 50 synthesis evaluations.

All scoring uses cosine distance (78-dim MFCC+Δ+ΔΔ mean+std) between synthesised candidate and reference audio.

Speed calibration: measures reference audio duration vs. Kokoro's natural synthesis pace for the transcript, seeds the speed slider with the ratio. Override manually — slider range is 0.5–2.0×.

Quality thresholds (cosine distance, lower = closer):
- **excellent** < 0.15 — **good** < 0.30 — **fair** < 0.50 — **rough** ≥ 0.50

### Fine Tune

Sliders for each voice slot: weight (0–100, normalised to show actual contribution %) and pitch (±5 semitones). Weights are always normalised before use — 80/40/0 and 2/1/0 produce identical blends. Speed slider (0.5–2.0×) applies to all previews and exports. Export name field (shared with Mixer tab) sets the filename for all Voice Match exports. Sidecar JSON records `matched_speed` alongside the blend recipe.

Preview individual slots or the full blend. Export directly or load into the Mixer for further work.

### Blend Explorer

After optimising, the Explorer tests whether adding a small amount of another voice improves the match. Each round:

1. Takes the current Fine Tune blend as a single unit.
2. Picks the next 3 MFCC-ranked pool voices not already in the blend.
3. For each: creates a variant = current blend (82%) + new voice (18%). Because embedding mixing is linear, this is mathematically identical to a flat multi-component recipe — no temp file needed.
4. Synthesises A / B / C saying your transcript and offers them for comparison.

Pick the closest to your target → Fine Tune updates, next round tests 3 more voices from the pool. Repeat until satisfied. Export button saves the current Fine Tune state directly from the Explorer panel.

---

## Mixer: how it works

### Weights

Blending mixes the raw voice embedding tensors (256-dim float32 vectors) by weighted average. Weights are always normalized to sum to 1 before the blend is applied — this keeps the mixed tensor at the same scale as the original voices, which is required for the ONNX model to produce valid audio. A 50/30/20 split is identical to a 5/3/2 split.

### Speed and pitch — two separate sets of controls

**Per-slot Pitch and Speed** apply only in **▶ Preview Slot**. They let you audition each voice individually at different tempos and registers before committing to a blend.

**▶ Preview Mix** works differently: the slot embeddings are blended into a single voice bin, synthesised once, and then the Output Speed / Output Pitch controls are applied as a final stage. Per-slot pitch and speed are not applied during mix preview — mixing separately-synthesised clips of different durations causes overlap, and there is no clean way to time-align them at the audio level.

**Output Speed / Output Pitch** (in the Mix Output panel) control the final result — both for **▶ Preview Mix** and for the exported `.bin` synthesis. Use these to tune overall tempo and register of the blended voice.

- Use **Reset Output FX** to snap both back to default (speed 1.0×, pitch 0 semitones).
- All pitch and speed transforms use ffmpeg. **ffmpeg must be on your PATH** for any non-default value. At defaults (pitch 0, speed 1.0) ffmpeg is not called. (Exception: if Bake slot pitch is enabled, per-slot pitch is applied to the embeddings and does not require ffmpeg.)

### Bake slot pitch

When the **Bake slot pitch into .bin** checkbox is on, each slot's pitch offset is applied directly to the voice embedding before blending. The export `.bin` already encodes the pitch shift — no ffmpeg is needed at playback time, and the pitch is permanent rather than a post-processing effect.

How it works: `extend_voice_analysis.py` measures the F0 of each voice from its synthesised WAV. The tool runs Ridge regression over all measured voices (embedding → log F0) to find the direction in the 256-dim embedding space that corresponds to pitch change. Shifting a `.bin` N semitones means moving it `N * log(2)/12` units along this direction (properly scaled by the regression vector norm). The direction is computed once from the acoustic CSV on startup and reused for all exports.

If `scikit-learn` is not installed, or the acoustic CSV has fewer than 10 voices with valid F0 measurements, the checkbox is shown as disabled with a status explaining the limitation.

### Exported `.bin` files

Always normalized. Match the raw float32 format of the vendor voices and can be dropped into the `voices/` folder for use with any Kokoro-compatible tool. The sidecar JSON records the blend recipe so you can recreate it later.

---

## Style extraction (all synthesis paths)

All synthesis — in the persistent server, the standalone `infer.py`, and the analysis script — uses **multi-frame Hanning-pooled style selection** instead of single-timestep lookup.

For a given token count `n`, a window of ±4 frames is selected from the `.bin`, weighted by a Hanning window (tapered edges, peak at centre), and the weighted mean is used as the style vector. This removes the token-length dependency that previously caused the same voice to sound slightly different at different text lengths.

---

## Rebuilding fingerprints (if you had a previous version)

If you have a `voice_match_mfcc.json` built before this update, the tool will detect it as legacy format and fall back to weighted MFCC distance. A status message says `Legacy fingerprints — rebuild for embedding-space matching`.

Click **Build Fingerprints** in the Voice Match tab (server must be running) to rebuild. The status will show `Ready — N/N voices fingerprinted [regressor ✓]` when complete. The old file is overwritten in place.

---

## Extending voice analysis coverage

The tool ships with acoustic analysis data for voices that were cached in the original run. If you have voices that are missing from the analysis CSV (or you've added custom `.bin` files), run:

```bash
python extend_voice_analysis.py
```

This script:
1. Reads the existing `cache/voice-audition/voice-analysis.csv`
2. Finds any `.bin` files in `voices/` that don't have entries
3. Synthesises a test sentence for each missing voice using the ONNX model
4. Runs acoustic analysis (F0 via pyin, RMS, ZCR, spectral centroid) and appends rows

After running, restart Voice Lab (or click **Re-cache All Voices**) to rebuild the pitch axis with the extended data.

---

## Files

| File | Purpose |
|------|---------|
| `kokoro_voice_lab.py` | Main GUI application |
| `extend_voice_analysis.py` | Extend voice-analysis.csv to cover all .bin files |
| `synth_server.py` | Persistent synthesis server (model loaded once) |
| `infer.py` | Standalone synthesis script (also used as fallback) |
| `voice_lab_config.json` | Session config — auto-saved, contains your local paths |
| `voice_ratings.json` | Your voice ratings — created when you first save |
| `voice_match_mfcc.json` | Voice fingerprints built by Build Fingerprints |
| `logo.png` | The rat |

---

## Notes

- The tool saves your config (paths, slot settings, session state) to `voice_lab_config.json` automatically. This file is gitignored by default since it contains your local paths.
- `voice_ratings.json` is also gitignored — if you want to share your ratings with someone, commit it manually or send it directly.
- The `preview_cache/` folder holds pre-generated WAVs keyed to the test sentence. Delete it to force regeneration (e.g. after changing the test sentence). Or use **Re-cache All Voices** in the Config tab.
- Ratings use a 1–10 scale on each axis. Trait Match distance is Manhattan distance across all rated axes.

---

## Requirements

| Package | Required | Notes |
|---------|----------|-------|
| `onnxruntime` | Yes | Or `onnxruntime-gpu` for CUDA acceleration |
| `misaki[en]` | Yes | G2P phonemiser — converts text to phoneme tokens |
| `soundfile` | Yes | WAV read/write; also used to decode MP3/FLAC for reference audio playback |
| `numpy` | Yes | Voice bin arithmetic and tensor operations |
| `scikit-learn` | Recommended | Required for Bake slot pitch; gracefully disabled if absent |
| `Pillow` | Recommended | Logo transparency support; gracefully absent otherwise |
| `librosa` | Required for Voice Match | Feature extraction, F0 measurement, acoustic analysis; also needed for `extend_voice_analysis.py` |
| `scipy` | Recommended | Nelder-Mead refinement in Optimise Blend; skipped gracefully if absent |
| `faster-whisper` | Optional | Voice Match transcription (preferred — smaller/faster) |
| `openai-whisper` | Optional | Voice Match transcription (fallback if faster-whisper absent) |
| `ffmpeg` | Optional | Required for non-default pitch/speed when bake is off |

Python 3.10 or later.

---

## Acknowledgements

**[Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)** by [hexgrad](https://huggingface.co/hexgrad) — the TTS model this tool is built around. Kokoro is an open-weight, Apache 2.0 licensed text-to-speech model. All voice `.bin` files and the ONNX model are from this release.

**[misaki](https://github.com/hexgrad/misaki)** by hexgrad — the G2P (grapheme-to-phoneme) library used to convert input text into phoneme token sequences before synthesis. Handles both American and British English pronunciation.

**[ONNX Runtime](https://onnxruntime.ai/)** by Microsoft — cross-platform inference engine used to run the Kokoro model. The `onnxruntime-gpu` variant enables CUDA acceleration where available.

**[soundfile](https://python-soundfile.readthedocs.io/)** / **[libsndfile](https://libsndfile.github.io/libsndfile/)** — WAV I/O used for reading and writing synthesis output. soundfile is the Python wrapper; the underlying work is libsndfile by Erik de Castro Lopo.

**[NumPy](https://numpy.org/)** — voice bin loading, tensor arithmetic, and weight normalization.

**[scikit-learn](https://scikit-learn.org/)** — Ridge regression used to compute the pitch direction vector in voice embedding space for the Bake slot pitch feature.

**[librosa](https://librosa.org/)** — MFCC feature extraction, F0 estimation via pyin, and acoustic analysis used throughout the Voice Match pipeline.

**[Pillow](https://python-pillow.org/)** — used for logo transparency and image handling in the GUI watermark.

**[ffmpeg](https://ffmpeg.org/)** — optional post-processing for pitch shift and speed transforms applied to synthesized audio.

**[tkinter](https://docs.python.org/3/library/tkinter.html)** — Python's standard GUI toolkit, used for the entire interface.

---

*Built for [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) by hexgrad.*
