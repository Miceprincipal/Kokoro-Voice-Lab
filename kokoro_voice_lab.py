from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

try:
    import pygame
except Exception:
    pygame = None

try:
    import winsound as _winsound
except ImportError:
    _winsound = None


APP_TITLE = "Kokoro Voice Lab"
DEFAULT_TEST_SENTENCE = "The quick brown fox jumps over the lazy dog. This is a voice preview sample."
APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "voice_lab_config.json"
DEFAULT_RATINGS_PATH = APP_DIR / "voice_ratings.json"
DEFAULT_EXPORT_DIR = APP_DIR / "exports"

DEFAULT_TRAIT_LABELS = ["age", "authority", "clarity", "energy", "gender_pres", "pitch", "roughness", "warmth"]

TRAIT_DESC = {
    "age":         ("1 = child", "5 = elderly"),
    "authority":   ("1 = passive", "5 = commanding"),
    "clarity":     ("1 = mumbly", "5 = crisp"),
    "energy":      ("1 = drowsy", "5 = intense"),
    "gender_pres": ("1 = masc", "5 = fem"),
    "pitch":       ("1 = deep", "5 = high"),
    "roughness":   ("1 = smooth", "5 = gravelly"),
    "warmth":      ("1 = cold", "5 = warm"),
}


@dataclass
class VoiceBin:
    name: str
    path: Path
    size_bytes: int
    value_count: int
    dtype: str = "float32"


class EmbeddedAudioPlayer:
    def __init__(self) -> None:
        self.available = False
        self.current_path: Path | None = None
        self.duration_seconds = 0.0
        self._play_proc: subprocess.Popen | None = None
        if pygame is not None:
            try:
                # Match Kokoro's output rate — default 22050 Hz causes silent/garbled playback
                pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=512)
                self.available = True
            except Exception:
                pass

    def load(self, path: Path) -> None:
        self.current_path = path
        self.duration_seconds = 0.0
        if not self.available:
            # Get duration without pygame
            try:
                import soundfile as _sf  # noqa: PLC0415
                self.duration_seconds = float(_sf.info(str(path)).duration)
            except Exception:
                pass
            return
        self.stop()
        pygame.mixer.music.load(str(path))
        try:
            snd = pygame.mixer.Sound(str(path))
            self.duration_seconds = float(snd.get_length())
        except Exception:
            pass

    def _kill_proc(self) -> None:
        if self._play_proc is not None:
            try:
                self._play_proc.kill()
            except Exception:
                pass
            self._play_proc = None

    def play(self, start: float = 0.0) -> None:
        if self.available:
            pygame.mixer.music.play(start=max(0.0, start))
            return
        if not self.current_path or not self.current_path.exists():
            return
        self._kill_proc()
        p = str(self.current_path)
        # Spawn a fresh subprocess to play audio — completely isolated from tkinter's message loop.
        # winsound in a daemon thread can silently fail inside a GUI process; a subprocess avoids that.
        self._play_proc = subprocess.Popen(
            [sys.executable, "-c", f"import winsound; winsound.PlaySound({repr(p)}, winsound.SND_FILENAME)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

    def stop(self) -> None:
        if self.available:
            pygame.mixer.music.stop()
        self._kill_proc()

    def position_seconds(self) -> float:
        if not self.available:
            return 0.0
        pos_ms = pygame.mixer.music.get_pos()
        return 0.0 if pos_ms < 0 else pos_ms / 1000.0


class KokoroSynthAdapter:
    def __init__(self, command_template: str = "", ffmpeg_path: str = "ffmpeg", working_dir: str = "") -> None:
        self.command_template = command_template.strip()
        self.ffmpeg_path = ffmpeg_path.strip() or "ffmpeg"
        self.working_dir = working_dir.strip()

    def is_configured(self) -> bool:
        return bool(self.command_template)

    def synthesize(self, voice_path: Path, text: str, out_wav: Path) -> None:
        if not self.is_configured():
            raise RuntimeError("No synthesis command configured.")
        cmd = self.command_template.format(voice=str(voice_path), text=text, out=str(out_wav))
        subprocess.run(cmd, shell=True, check=True, cwd=self.working_dir or None)
        if not out_wav.exists():
            raise RuntimeError(f"Synthesis completed but no output file: {out_wav}")

    def transform_audio(self, src_wav: Path, dst_wav: Path, speed: float, pitch_semitones: float) -> None:
        filters: list[str] = []
        remaining = max(0.25, min(speed, 4.0))
        while remaining > 2.0:
            filters.append("atempo=2.0")
            remaining /= 2.0
        while remaining < 0.5:
            filters.append("atempo=0.5")
            remaining /= 0.5
        filters.append(f"atempo={remaining:.5f}")
        if abs(pitch_semitones) > 1e-6:
            ratio = 2 ** (pitch_semitones / 12.0)
            filters.insert(0, f"asetrate=24000*{ratio:.8f}")
            filters.insert(1, "aresample=24000")
        cmd = [self.ffmpeg_path, "-y", "-i", str(src_wav), "-af", ",".join(filters), str(dst_wav)]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not dst_wav.exists():
            raise RuntimeError(f"Audio transform failed: {dst_wav}")

    @staticmethod
    def is_identity(speed: float, pitch: float) -> bool:
        return abs(pitch) < 1e-6 and abs(speed - 1.0) < 0.005


class DirectAssetBackend:
    def __init__(self) -> None:
        self.root: Path | None = None
        self.model_path: Path | None = None
        self.config_path: Path | None = None
        self.tokenizer_path: Path | None = None
        self.voice_dir: Path | None = None

    def configure(self, root: Path) -> None:
        self.root = root
        self.model_path = root / "onnx" / "model_uint8.onnx"
        self.config_path = root / "config.json"
        self.tokenizer_path = root / "tokenizer.json"
        self.voice_dir = root / "voices"

    def diagnose(self) -> tuple[bool, str]:
        missing = [str(p) for p in [self.model_path, self.config_path, self.tokenizer_path, self.voice_dir]
                   if p is None or not p.exists()]
        if missing:
            return False, "Missing: " + ", ".join(missing)
        return True, "Asset bundle OK. Audible preview needs external synth mode."


class VoiceLibrary:
    def __init__(self) -> None:
        self.root: Path | None = None
        self.voices: list[VoiceBin] = []
        self._cache: dict[Path, np.ndarray] = {}

    def load_dir(self, folder: Path) -> list[VoiceBin]:
        bins = sorted(folder.rglob("*.bin"))
        found = []
        for p in bins:
            size = p.stat().st_size
            if size % 4 == 0:
                found.append(VoiceBin(name=p.stem, path=p, size_bytes=size, value_count=size // 4))
        self.root = folder
        self.voices = found
        self._cache.clear()
        return found

    def get(self, name: str) -> VoiceBin:
        for v in self.voices:
            if v.name == name:
                return v
        raise KeyError(name)

    def load_array(self, voice: VoiceBin) -> np.ndarray:
        if voice.path not in self._cache:
            self._cache[voice.path] = np.fromfile(voice.path, dtype=np.float32)
        return self._cache[voice.path]

    def mix(self, components: list[tuple[VoiceBin, float]], normalize: bool = True) -> np.ndarray:
        active = [(v, w) for v, w in components if w > 0]
        if not active:
            raise ValueError("No active voices.")
        lengths = {v.value_count for v, _ in active}
        if len(lengths) != 1:
            raise ValueError(f"Mismatched bin sizes: {sorted(lengths)}")
        weights = np.array([w for _, w in active], dtype=np.float32)
        if normalize:
            s = float(weights.sum())
            if s <= 0:
                raise ValueError("Weights sum to zero.")
            weights /= s
        out = np.zeros(next(iter(lengths)), dtype=np.float32)
        for i, (v, _) in enumerate(active):
            out += self.load_array(v) * weights[i]
        return out

    @staticmethod
    def save_bin(arr: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arr.astype(np.float32).tofile(path)


class RatingData:
    def __init__(self) -> None:
        self.path: Path = DEFAULT_RATINGS_PATH
        self.payload: dict[str, Any] = {"meta": {}, "voices": {}}

    def load(self, path: Path) -> None:
        self.payload = json.loads(path.read_text(encoding="utf-8"))
        self.path = path

    def save(self, path: Path | None = None) -> None:
        p = path or self.path
        p.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")
        self.path = p

    def get_voice_names(self) -> list[str]:
        return sorted(self.payload.get("voices", {}).keys())

    def get_traits(self, name: str) -> dict[str, int]:
        return dict(self.payload.get("voices", {}).get(name, {}).get("traits", {}))

    def set_traits(self, name: str, traits: dict[str, int]) -> None:
        self.payload.setdefault("voices", {}).setdefault(name, {})["traits"] = traits

    def get_notes(self, name: str) -> str:
        return self.payload.get("voices", {}).get(name, {}).get("notes", "")

    def set_notes(self, name: str, notes: str) -> None:
        self.payload.setdefault("voices", {}).setdefault(name, {})["notes"] = notes

    def is_rated(self, name: str) -> bool:
        return bool(self.get_traits(name))


PREVIEW_CACHE_DIR = APP_DIR / "preview_cache"
SYNTH_SERVER_SCRIPT = APP_DIR / "synth_server.py"


class PersistentSynthServer:
    """Keeps the ONNX model loaded in a subprocess — no cold-start per call."""

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self.providers: list[str] = []
        self.using_gpu = False

    def start(self) -> str:
        if self.is_running():
            return "Already running."
        self._proc = subprocess.Popen(
            [sys.executable, str(SYNTH_SERVER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(APP_DIR),
        )
        # Block until "ready" line arrives (model loaded)
        ready_line = self._proc.stdout.readline()
        data = json.loads(ready_line)
        self.providers = data.get("providers", [])
        self.using_gpu = any("CUDA" in p for p in self.providers)
        gpu_str = " [GPU]" if self.using_gpu else " [CPU]"
        return f"Server ready{gpu_str}"

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def synthesize(self, voice_path: Path, text: str, out_wav: Path) -> None:
        with self._lock:
            req = json.dumps({"voice": str(voice_path), "text": text, "out": str(out_wav)}) + "\n"
            self._proc.stdin.write(req)
            self._proc.stdin.flush()
            resp_line = self._proc.stdout.readline()
            if not resp_line:
                raise RuntimeError("Server closed unexpectedly.")
            resp = json.loads(resp_line)
            if "error" in resp:
                raise RuntimeError(resp["error"])

    def stop(self) -> None:
        if self._proc:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=3)
            except Exception:
                self._proc.kill()
            self._proc = None
        self.providers = []
        self.using_gpu = False


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

class VoiceLabApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1200x760")
        self.root.minsize(1000, 660)

        self.player = EmbeddedAudioPlayer()
        self.voice_lib = VoiceLibrary()
        self.ratings = RatingData()
        self.synth = KokoroSynthAdapter()
        self.direct_backend = DirectAssetBackend()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="kokoro_vl_"))
        self.preview_lock = threading.Lock()
        self.last_export_path: Path | None = None
        self.voice_dropdowns: list[ttk.Combobox] = []
        self._server = PersistentSynthServer()
        self._precache_stop = threading.Event()
        self._precache_thread: threading.Thread | None = None

        # ── shared vars ──────────────────────────────────────────────────────
        self.status_var        = tk.StringVar(value="Load a voice folder to begin.")
        self.position_var      = tk.StringVar(value="--:-- / --:--")
        self.summary_var       = tk.StringVar(value="")
        self.test_sentence_var = tk.StringVar(value=DEFAULT_TEST_SENTENCE)
        self.voice_dir_var     = tk.StringVar(value="")
        self.ratings_path_var  = tk.StringVar(value=str(DEFAULT_RATINGS_PATH))
        self.command_var       = tk.StringVar(value="")
        self.ffmpeg_var        = tk.StringVar(value="ffmpeg")
        self.working_dir_var   = tk.StringVar(value="")
        self.export_dir_var    = tk.StringVar(value=str(DEFAULT_EXPORT_DIR))
        self.export_name_var   = tk.StringVar(value="mixed_voice")
        self.normalize_var     = tk.BooleanVar(value=True)
        self.auto_preview_var  = tk.BooleanVar(value=False)
        self.preview_mode_var  = tk.StringVar(value="external_cmd")
        self.save_sidecar_var  = tk.BooleanVar(value=True)
        self.autosave_var      = tk.BooleanVar(value=True)

        # ── mixer slot vars ──────────────────────────────────────────────────
        self.active_vars = [tk.BooleanVar(value=True) for _ in range(3)]
        self.voice_vars  = [tk.StringVar() for _ in range(3)]
        self.weight_vars = [tk.DoubleVar(value=v) for v in (50.0, 30.0, 20.0)]
        self.pitch_vars  = [tk.DoubleVar(value=0.0) for _ in range(3)]
        self.speed_vars  = [tk.DoubleVar(value=1.0) for _ in range(3)]
        self.slot_status_vars = [tk.StringVar(value="") for _ in range(3)]

        # ── mix output controls (global, applied post-blend) ─────────────────
        self.mix_speed_var = tk.DoubleVar(value=1.0)
        self.mix_pitch_var = tk.DoubleVar(value=0.0)

        # ── ratings tab vars ─────────────────────────────────────────────────
        self.rating_trait_vars:  dict[str, tk.IntVar] = {}
        self.target_trait_vars:  dict[str, tk.IntVar] = {}
        self.ratings_sel_var     = tk.StringVar(value="")
        self.ratings_prog_var    = tk.StringVar(value="")
        self.server_status_var   = tk.StringVar(value="Server: not started")
        self.precache_var        = tk.StringVar(value="")
        self._ratings_listbox:   tk.Listbox | None = None
        self._ratings_notes:     tk.Text | None = None
        self._rating_trait_frame: ttk.Frame | None = None
        self._traits_inner:       ttk.Frame | None = None

        self._build_ui()
        self._load_config()
        self._bind_keys()
        self._bind_persistence()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick_playback()
        self._recompute_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # Top-level UI
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        nb = ttk.Notebook(self.root)
        nb.pack(fill="both", expand=True, padx=6, pady=(6, 0))

        t_ratings = ttk.Frame(nb, padding=8)
        t_mixer   = ttk.Frame(nb, padding=8)
        t_config  = ttk.Frame(nb, padding=8)
        nb.add(t_ratings, text="  Ratings  ")
        nb.add(t_mixer,   text="  Mixer  ")
        nb.add(t_config,  text="  Config  ")

        self._build_ratings_tab(t_ratings)
        self._build_mixer_tab(t_mixer)
        self._build_config_tab(t_config)

        bar = ttk.Frame(self.root, padding=(8, 2, 8, 4))
        bar.pack(fill="x")
        ttk.Label(bar, textvariable=self.status_var, anchor="w").pack(side="left", fill="x", expand=True)
        ttk.Label(bar, textvariable=self.position_var, anchor="e").pack(side="right")

        self._embed_logo()

    def _embed_logo(self) -> None:
        """Stamp the mouse logo as a semi-transparent watermark in the bottom-right corner."""
        # Find the logo — any PNG in APP_DIR that has 'pixil' or 'logo' in the name
        candidates = list(APP_DIR.glob("pixil*.png")) + list(APP_DIR.glob("logo*.png"))
        if not candidates:
            return
        logo_path = candidates[0]
        try:
            from PIL import Image, ImageTk
            img = Image.open(logo_path).convert("RGBA")

            # Strip white / near-white background to transparent
            pixels = img.load()
            w, h = img.size
            for y in range(h):
                for x in range(w):
                    r, g, b, a = pixels[x, y]
                    brightness = (r + g + b) / 3
                    if brightness > 230:
                        pixels[x, y] = (r, g, b, 0)          # fully transparent
                    elif brightness > 200:
                        fade = int(a * (1 - (brightness - 200) / 30))
                        pixels[x, y] = (r, g, b, fade)        # soft edge

            # Scale to a tasteful corner stamp size
            img = img.resize((130, 130), Image.LANCZOS)

            # Apply overall opacity (watermark strength)
            r_ch, g_ch, b_ch, a_ch = img.split()
            a_ch = a_ch.point(lambda v: int(v * 0.45))
            img = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_ch))

            self._logo_photo = ImageTk.PhotoImage(img)
            lbl = tk.Label(self.root, image=self._logo_photo, bg=self.root.cget("bg"), bd=0)
            lbl.place(relx=1.0, rely=1.0, anchor="se", x=-14, y=-8)

        except ImportError:
            # Pillow not installed — show at full opacity without alpha processing
            try:
                photo = tk.PhotoImage(file=str(logo_path))
                photo = photo.subsample(max(1, photo.width() // 110))
                self._logo_photo = photo
                lbl = tk.Label(self.root, image=self._logo_photo, bg=self.root.cget("bg"), bd=0)
                lbl.place(relx=1.0, rely=1.0, anchor="se", x=-14, y=-8)
            except Exception:
                pass
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Ratings tab
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ratings_tab(self, parent: ttk.Frame) -> None:
        # ── left: voice list ─────────────────────────────────────────────────
        left = ttk.Frame(parent)
        left.pack(side="left", fill="y", padx=(0, 8))

        ttk.Label(left, text="Voices", font=("Segoe UI", 9, "bold")).pack(anchor="w")
        lf = ttk.Frame(left)
        lf.pack(fill="both", expand=True)
        self._ratings_listbox = tk.Listbox(lf, width=18, exportselection=False,
                                            activestyle="dotbox", font=("Consolas", 9))
        sb = ttk.Scrollbar(lf, orient="vertical", command=self._ratings_listbox.yview)
        self._ratings_listbox.configure(yscrollcommand=sb.set)
        self._ratings_listbox.pack(side="left", fill="both", expand=True)
        sb.pack(side="left", fill="y")
        self._ratings_listbox.bind("<<ListboxSelect>>", self._on_ratings_select)

        ttk.Label(left, textvariable=self.ratings_prog_var,
                  font=("Segoe UI", 8), foreground="#666").pack(anchor="w", pady=(4, 0))

        # ── right: rating controls ───────────────────────────────────────────
        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)

        # Header: voice name + nav + action buttons
        hdr = ttk.Frame(right)
        hdr.pack(fill="x", pady=(0, 6))
        ttk.Label(hdr, textvariable=self.ratings_sel_var,
                  font=("Segoe UI", 13, "bold"), width=22, anchor="w").pack(side="left")
        ttk.Button(hdr, text="◀ Prev", width=7,
                   command=lambda: self._ratings_navigate(-1)).pack(side="left", padx=(8, 2))
        ttk.Button(hdr, text="Next ▶", width=7,
                   command=lambda: self._ratings_navigate(1)).pack(side="left", padx=(0, 12))
        ttk.Button(hdr, text="▶  Preview", width=12,
                   command=self._ratings_preview).pack(side="left", padx=(0, 4))
        ttk.Button(hdr, text="Stop", width=6,
                   command=self.stop_audio).pack(side="left", padx=(0, 12))
        ttk.Button(hdr, text="✓  Save Voice", width=14,
                   command=self._ratings_save_current).pack(side="left", padx=(0, 4))
        ttk.Button(hdr, text="Save All As…", width=12,
                   command=self._ratings_save_as).pack(side="left")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=(0, 8))

        # Trait grid: 2 columns × 4 traits
        trait_outer = ttk.LabelFrame(right, text="Trait Ratings", padding=(10, 6))
        trait_outer.pack(fill="x")
        self._rating_trait_frame = trait_outer
        self._build_rating_sliders(trait_outer)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=8)

        # Notes
        notes_hdr = ttk.Frame(right)
        notes_hdr.pack(fill="x")
        ttk.Label(notes_hdr, text="Notes", font=("Segoe UI", 9, "bold")).pack(side="left")
        ttk.Label(notes_hdr, text="  e.g. 'good villain, avoid children, strong American accent'",
                  foreground="#888", font=("Segoe UI", 8)).pack(side="left")
        nf = ttk.Frame(right)
        nf.pack(fill="x", pady=(2, 0))
        self._ratings_notes = tk.Text(nf, height=3, wrap="word",
                                       font=("Segoe UI", 10), relief="solid", borderwidth=1)
        ns = ttk.Scrollbar(nf, orient="vertical", command=self._ratings_notes.yview)
        self._ratings_notes.configure(yscrollcommand=ns.set)
        self._ratings_notes.pack(side="left", fill="x", expand=True)
        ns.pack(side="left", fill="y")

        # Test sentence + cache status
        tsf = ttk.Frame(right)
        tsf.pack(fill="x", pady=(8, 0))
        ttk.Label(tsf, text="Test sentence:", font=("Segoe UI", 9)).pack(side="left")
        ttk.Entry(tsf, textvariable=self.test_sentence_var).pack(side="left", fill="x",
                                                                   expand=True, padx=(6, 0))
        csf = ttk.Frame(right)
        csf.pack(fill="x", pady=(4, 0))
        ttk.Label(csf, textvariable=self.server_status_var,
                  font=("Segoe UI", 8), foreground="#0055aa").pack(side="left")
        ttk.Label(csf, text="  |  ", foreground="#ccc").pack(side="left")
        ttk.Label(csf, textvariable=self.precache_var,
                  font=("Segoe UI", 8), foreground="#558800").pack(side="left")

    def _build_rating_sliders(self, parent: ttk.Frame) -> None:
        for w in parent.winfo_children():
            w.destroy()
        self.rating_trait_vars.clear()
        labels = self._infer_trait_labels()

        # Split into 2 columns
        mid = len(labels) // 2 + len(labels) % 2
        col_a, col_b = labels[:mid], labels[mid:]

        left_col = ttk.Frame(parent)
        left_col.pack(side="left", fill="both", expand=True, padx=(0, 16))
        right_col = ttk.Frame(parent)
        right_col.pack(side="left", fill="both", expand=True)

        for col_frame, traits in ((left_col, col_a), (right_col, col_b)):
            for row, name in enumerate(traits):
                lo, hi = TRAIT_DESC.get(name, ("1=low", "5=high"))
                ttk.Label(col_frame, text=name, width=12, anchor="w",
                          font=("Segoe UI", 9, "bold")).grid(row=row, column=0, sticky="w", pady=3)
                ttk.Label(col_frame, text=lo, foreground="#888",
                          font=("Segoe UI", 8)).grid(row=row, column=1, sticky="e")
                var = tk.IntVar(value=3)
                self.rating_trait_vars[name] = var
                tk.Scale(col_frame, from_=1, to=5, orient="horizontal", resolution=1,
                         variable=var, length=160, showvalue=False).grid(row=row, column=2,
                                                                          sticky="ew", padx=4, pady=3)
                ttk.Label(col_frame, text=hi, foreground="#888",
                          font=("Segoe UI", 8)).grid(row=row, column=3, sticky="w")
                ttk.Spinbox(col_frame, from_=1, to=5, increment=1,
                            textvariable=var, width=4).grid(row=row, column=4,
                                                             sticky="w", padx=(4, 0))
            col_frame.columnconfigure(2, weight=1)

    def _on_ratings_select(self, _event=None) -> None:
        if not self._ratings_listbox:
            return
        sel = self._ratings_listbox.curselection()
        if not sel:
            return
        raw = self._ratings_listbox.get(sel[0])
        name = raw.lstrip("✓ ").strip()
        self.ratings_sel_var.set(name)
        traits = self.ratings.get_traits(name)
        for k, var in self.rating_trait_vars.items():
            var.set(int(traits.get(k, 3)))
        if self._ratings_notes:
            self._ratings_notes.delete("1.0", "end")
            self._ratings_notes.insert("1.0", self.ratings.get_notes(name))

    def _ratings_navigate(self, delta: int) -> None:
        """Auto-save current, move to next/prev voice."""
        if not self._ratings_listbox:
            return
        # Save current first
        name = self.ratings_sel_var.get().strip()
        if name:
            self._commit_current_rating(name)
        n = self._ratings_listbox.size()
        if n == 0:
            return
        sel = self._ratings_listbox.curselection()
        cur = sel[0] if sel else -1
        nxt = (cur + delta) % n
        self._ratings_listbox.selection_clear(0, "end")
        self._ratings_listbox.selection_set(nxt)
        self._ratings_listbox.see(nxt)
        self._on_ratings_select()

    def _commit_current_rating(self, name: str) -> None:
        traits = {k: int(v.get()) for k, v in self.rating_trait_vars.items()}
        notes = self._ratings_notes.get("1.0", "end").strip() if self._ratings_notes else ""
        self.ratings.set_traits(name, traits)
        self.ratings.set_notes(name, notes)

    def _ratings_save_current(self) -> None:
        name = self.ratings_sel_var.get().strip()
        if not name:
            messagebox.showerror(APP_TITLE, "Select a voice first.")
            return
        self._commit_current_rating(name)
        path = Path(self.ratings_path_var.get().strip() or str(DEFAULT_RATINGS_PATH))
        try:
            self.ratings.save(path)
            self.status_var.set(f"Saved: {name}  →  {path.name}")
            self._refresh_ratings_list()
        except Exception as e:
            messagebox.showerror(APP_TITLE, str(e))

    def _ratings_save_as(self) -> None:
        name = self.ratings_sel_var.get().strip()
        if name:
            self._commit_current_rating(name)
        path = filedialog.asksaveasfilename(
            title="Save ratings JSON", defaultextension=".json",
            filetypes=[("JSON", "*.json")], initialdir=str(APP_DIR),
            initialfile="voice_ratings.json",
        )
        if not path:
            return
        try:
            self.ratings.save(Path(path))
            self.ratings_path_var.set(path)
            self.status_var.set(f"Saved ratings → {path}")
        except Exception as e:
            messagebox.showerror(APP_TITLE, str(e))

    def _cache_wav(self, voice_name: str) -> Path:
        text = self.test_sentence_var.get().strip() or DEFAULT_TEST_SENTENCE
        h = hashlib.md5(text.encode()).hexdigest()[:8]
        return PREVIEW_CACHE_DIR / f"{voice_name}_{h}.wav"

    def _can_synthesize(self) -> bool:
        return self._server.is_running() or self.synth.is_configured()

    def _do_synthesize(self, voice_path: Path, text: str, out: Path) -> None:
        if self._server.is_running():
            self._server.synthesize(voice_path, text, out)
        elif self.synth.is_configured():
            self.synth.synthesize(voice_path, text, out)
        else:
            raise RuntimeError("No synthesis method available. Start server or configure command.")

    def _ratings_preview(self) -> None:
        name = self.ratings_sel_var.get().strip()
        if not name:
            messagebox.showerror(APP_TITLE, "Select a voice first.")
            return
        try:
            voice = self.voice_lib.get(name)
        except KeyError:
            messagebox.showerror(APP_TITLE, f"'{name}' not in loaded library.")
            return

        # Instant: use cached WAV if available
        cache = self._cache_wav(name)
        if cache.exists():
            self.player.load(cache)
            self.player.play()
            self.status_var.set(f"Playing {name}  (cached)")
            return

        if not self._can_synthesize():
            messagebox.showerror(APP_TITLE, "Start the server (Config tab) or configure a synth command.")
            return

        def task() -> None:
            PREVIEW_CACHE_DIR.mkdir(exist_ok=True)
            self._do_synthesize(voice.path, self.test_sentence_var.get() or DEFAULT_TEST_SENTENCE, cache)
            self.player.load(cache)
            self.player.play()
        self._run_bg(task, success=f"Playing {name}")

    def _precache_all(self) -> None:
        """Background: pre-generate WAVs for every voice using the current test sentence."""
        if not self._can_synthesize():
            self.root.after(0, lambda: self.precache_var.set("Start server first."))
            return
        PREVIEW_CACHE_DIR.mkdir(exist_ok=True)
        voices = list(self.voice_lib.voices)
        total = len(voices)
        self._precache_stop.clear()
        for i, v in enumerate(voices, 1):
            if self._precache_stop.is_set():
                break
            cache = self._cache_wav(v.name)
            if not cache.exists():
                try:
                    self._do_synthesize(v.path, self.test_sentence_var.get() or DEFAULT_TEST_SENTENCE, cache)
                except Exception:
                    pass
            n, t = i, total
            self.root.after(0, lambda n=n, t=t: self.precache_var.set(f"Pre-caching {n}/{t}…"))
        self.root.after(0, lambda: self.precache_var.set(f"All {total} voices cached"))
        self.root.after(0, self._refresh_ratings_list)

    def start_precache(self) -> None:
        if self._precache_thread and self._precache_thread.is_alive():
            return
        self._precache_thread = threading.Thread(target=self._precache_all, daemon=True)
        self._precache_thread.start()

    def _server_start(self) -> None:
        if not SYNTH_SERVER_SCRIPT.exists():
            messagebox.showerror(APP_TITLE, f"synth_server.py not found at {SYNTH_SERVER_SCRIPT}")
            return
        def task() -> None:
            msg = self._server.start()
            gpu = " [GPU]" if self._server.using_gpu else " [CPU only]"
            self.root.after(0, lambda: self.server_status_var.set(f"Server: running{gpu}"))
            self.root.after(0, lambda: self.status_var.set(msg + " — pre-caching in background…"))
            self.root.after(0, self.start_precache)
        threading.Thread(target=task, daemon=True).start()
        self.server_status_var.set("Server: starting…")

    def _server_stop(self) -> None:
        self._precache_stop.set()
        self._server.stop()
        self.server_status_var.set("Server: stopped")
        self.precache_var.set("")

    def _refresh_ratings_list(self) -> None:
        if not self._ratings_listbox:
            return
        cur_name = self.ratings_sel_var.get().strip()
        self._ratings_listbox.delete(0, "end")
        sel_idx = None
        for i, v in enumerate(self.voice_lib.voices):
            mark = "✓ " if self.ratings.is_rated(v.name) else "   "
            self._ratings_listbox.insert("end", f"{mark}{v.name}")
            if v.name == cur_name:
                sel_idx = i
        if sel_idx is not None:
            self._ratings_listbox.selection_set(sel_idx)
            self._ratings_listbox.see(sel_idx)
        rated = sum(1 for v in self.voice_lib.voices if self.ratings.is_rated(v.name))
        total = len(self.voice_lib.voices)
        self.ratings_prog_var.set(f"{rated} / {total} rated")

    def _infer_trait_labels(self) -> list[str]:
        keys: set[str] = set()
        for rec in self.ratings.payload.get("voices", {}).values():
            keys.update(rec.get("traits", {}).keys())
        return sorted(keys) if keys else DEFAULT_TRAIT_LABELS

    # ─────────────────────────────────────────────────────────────────────────
    # Mixer tab
    # ─────────────────────────────────────────────────────────────────────────

    def _build_mixer_tab(self, parent: ttk.Frame) -> None:
        # ── library bar ──────────────────────────────────────────────────────
        bar = ttk.LabelFrame(parent, text="Library", padding=(8, 4))
        bar.pack(fill="x", pady=(0, 8))
        ttk.Button(bar, text="Load Folder", command=self.select_voice_dir).grid(row=0, column=0)
        ttk.Entry(bar, textvariable=self.voice_dir_var, width=55).grid(row=0, column=1, padx=(6, 4))
        ttk.Button(bar, text="Reload", command=self.reload_voice_dir).grid(row=0, column=2)
        ttk.Button(bar, text="Load Ratings", command=self.select_ratings_json).grid(row=0, column=3, padx=(12, 0))
        ttk.Entry(bar, textvariable=self.ratings_path_var, width=28).grid(row=0, column=4, padx=(4, 0))

        # ── slots + mix panel ────────────────────────────────────────────────
        mid = ttk.Frame(parent)
        mid.pack(fill="both", expand=True)

        self.voice_dropdowns = []
        for i in range(3):
            self._build_slot(mid, i)

        self._build_mix_panel(mid)

        # ── trait match assist ───────────────────────────────────────────────
        ta = ttk.LabelFrame(parent, text="Trait Match Assist  (set target traits → suggest best voices)", padding=(8, 4))
        ta.pack(fill="x", pady=(8, 0))
        btn_row = ttk.Frame(ta)
        btn_row.pack(fill="x", pady=(0, 4))
        ttk.Button(btn_row, text="Suggest 3 Voices", command=self.suggest_from_traits).pack(side="left")
        ttk.Label(btn_row, text="  Set target trait values, then click Suggest:",
                  foreground="#666", font=("Segoe UI", 8)).pack(side="left")
        self._traits_inner = ttk.Frame(ta)
        self._traits_inner.pack(fill="x")
        self._build_trait_assist_sliders(self._traits_inner)

    def _build_slot(self, parent: ttk.Frame, idx: int) -> None:
        colors = ["#f0f4ff", "#f0fff4", "#fff8f0"]
        sf = ttk.LabelFrame(parent, text=f"  Slot {idx + 1}  ", padding=(8, 6))
        sf.pack(side="left", fill="both", expand=True, padx=(0, 6) if idx < 2 else 0)

        top = ttk.Frame(sf)
        top.pack(fill="x")
        ttk.Checkbutton(top, text="Active", variable=self.active_vars[idx],
                        command=self._on_mix_change).pack(side="left")
        combo = ttk.Combobox(sf, textvariable=self.voice_vars[idx], state="readonly")
        combo.pack(fill="x", pady=(4, 0))
        combo.bind("<<ComboboxSelected>>", lambda _e: self._on_mix_change())
        self.voice_dropdowns.append(combo)

        def _row(label: str, var: tk.Variable, lo: float, hi: float, res: float, length: int = 0) -> None:
            rf = ttk.Frame(sf)
            rf.pack(fill="x", pady=1)
            ttk.Label(rf, text=label, width=7, anchor="e").pack(side="left")
            sc = tk.Scale(rf, from_=lo, to=hi, orient="horizontal", resolution=res,
                          variable=var, showvalue=False)
            sc.pack(side="left", fill="x", expand=True, padx=4)
            ttk.Spinbox(rf, from_=lo, to=hi, increment=res, textvariable=var, width=6).pack(side="left")

        _row("Weight %", self.weight_vars[idx], 0, 100, 1)
        _row("Pitch st", self.pitch_vars[idx], -12, 12, 0.5)
        _row("Speed", self.speed_vars[idx], 0.25, 2.0, 0.01)

        btn_row = ttk.Frame(sf)
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row, text="▶ Preview Slot",
                   command=lambda i=idx: self.preview_slot(i)).pack(side="left")
        ttk.Label(btn_row, textvariable=self.slot_status_vars[idx],
                  font=("Segoe UI", 8), foreground="#555").pack(side="left", padx=(6, 0))

    def _build_mix_panel(self, parent: ttk.Frame) -> None:
        mf = ttk.LabelFrame(parent, text="  Mix Output  ", padding=(8, 6))
        mf.pack(side="left", fill="y", padx=(6, 0))

        ttk.Label(mf, text="Mode", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        ttk.Radiobutton(mf, text="External synth", variable=self.preview_mode_var,
                        value="external_cmd").pack(anchor="w")
        ttk.Radiobutton(mf, text="Asset check only", variable=self.preview_mode_var,
                        value="direct_assets").pack(anchor="w")

        ttk.Separator(mf, orient="horizontal").pack(fill="x", pady=6)

        # Global output speed + pitch (applied to the final blended synthesis)
        def _out_row(label: str, var: tk.Variable, lo: float, hi: float, res: float) -> None:
            rf = ttk.Frame(mf)
            rf.pack(fill="x", pady=1)
            ttk.Label(rf, text=label, width=9, anchor="e").pack(side="left")
            tk.Scale(rf, from_=lo, to=hi, orient="horizontal", resolution=res,
                     variable=var, showvalue=False).pack(side="left", fill="x", expand=True, padx=4)
            ttk.Spinbox(rf, from_=lo, to=hi, increment=res, textvariable=var, width=6).pack(side="left")

        ttk.Label(mf, text="Output", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        _out_row("Speed", self.mix_speed_var, 0.25, 2.0, 0.01)
        _out_row("Pitch st", self.mix_pitch_var, -12.0, 12.0, 0.5)
        ttk.Button(mf, text="Reset Output FX", command=self._reset_output_fx).pack(anchor="w", pady=(2, 0))

        ttk.Separator(mf, orient="horizontal").pack(fill="x", pady=6)
        ttk.Checkbutton(mf, text="Auto-preview on save", variable=self.auto_preview_var).pack(anchor="w")
        ttk.Checkbutton(mf, text="Sidecar preset JSON", variable=self.save_sidecar_var).pack(anchor="w")

        ttk.Separator(mf, orient="horizontal").pack(fill="x", pady=6)

        ef = ttk.Frame(mf)
        ef.pack(fill="x")
        ttk.Label(ef, text="Export dir", font=("Segoe UI", 8)).grid(row=0, column=0, sticky="w")
        ttk.Entry(ef, textvariable=self.export_dir_var, width=22).grid(row=0, column=1, padx=(4, 0))
        ttk.Button(ef, text="…", width=2, command=self.select_export_dir).grid(row=0, column=2, padx=(2, 0))
        ttk.Label(ef, text="Name", font=("Segoe UI", 8)).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(ef, textvariable=self.export_name_var, width=22).grid(row=1, column=1, padx=(4, 0), pady=(4, 0))

        ttk.Separator(mf, orient="horizontal").pack(fill="x", pady=6)

        for label, cmd in [
            ("▶  Preview Mix",      self.preview_mix),
            ("Export  .bin",        self.export_mix_bin_default),
            ("Save  .bin  As…",     self.save_mix_bin_as),
            ("Save Preset JSON",    self.save_mix_preset),
            ("Load Preset JSON",    self.load_mix_preset),
            ("Open Export Folder",  self.open_export_folder),
            ("■  Stop Audio",       self.stop_audio),
        ]:
            ttk.Button(mf, text=label, command=cmd).pack(fill="x", pady=2)

        ttk.Separator(mf, orient="horizontal").pack(fill="x", pady=6)
        ttk.Label(mf, textvariable=self.summary_var, font=("Segoe UI", 8),
                  wraplength=180, justify="left").pack(anchor="w")
        ttk.Label(mf, textvariable=self.position_var, font=("Segoe UI", 9)).pack(anchor="w", pady=(4, 0))

    def _build_trait_assist_sliders(self, parent: ttk.Frame) -> None:
        for w in parent.winfo_children():
            w.destroy()
        self.target_trait_vars.clear()
        labels = self._infer_trait_labels()
        # Horizontal row of sliders
        for i, name in enumerate(labels):
            col = ttk.Frame(parent)
            col.pack(side="left", padx=(0, 12))
            lo, hi = TRAIT_DESC.get(name, ("1", "5"))
            ttk.Label(col, text=name, font=("Segoe UI", 8, "bold"), anchor="center").pack()
            ttk.Label(col, text=lo, foreground="#888", font=("Segoe UI", 7)).pack()
            var = tk.IntVar(value=3)
            self.target_trait_vars[name] = var
            tk.Scale(col, from_=1, to=5, orient="vertical", resolution=1,
                     variable=var, length=60, showvalue=False).pack()
            ttk.Label(col, text=hi, foreground="#888", font=("Segoe UI", 7)).pack()
            ttk.Spinbox(col, from_=1, to=5, increment=1, textvariable=var, width=3).pack()

    # ─────────────────────────────────────────────────────────────────────────
    # Config tab
    # ─────────────────────────────────────────────────────────────────────────

    def _build_config_tab(self, parent: ttk.Frame) -> None:
        # ── Persistent server ─────────────────────────────────────────────────
        sv = ttk.LabelFrame(parent, text="Persistent Synth Server  (recommended — loads model once, no cold-start)", padding=10)
        sv.pack(fill="x", pady=(0, 10))
        sv_row = ttk.Frame(sv)
        sv_row.pack(fill="x")
        ttk.Button(sv_row, text="▶  Start Server", command=self._server_start).pack(side="left")
        ttk.Button(sv_row, text="■  Stop", command=self._server_stop).pack(side="left", padx=(6, 0))
        ttk.Button(sv_row, text="Pre-cache All Voices", command=self.start_precache).pack(side="left", padx=(12, 0))
        ttk.Label(sv_row, textvariable=self.server_status_var,
                  font=("Segoe UI", 9, "bold"), foreground="#0055aa").pack(side="left", padx=(12, 0))
        ttk.Label(sv_row, textvariable=self.precache_var,
                  foreground="#666", font=("Segoe UI", 8)).pack(side="left", padx=(8, 0))
        ttk.Label(sv, text="Uses GPU (CUDA) if onnxruntime-gpu is installed — otherwise CPU. "
                           "Start server → voices pre-generate in background → previews become instant clicks.",
                  foreground="#666", font=("Segoe UI", 8)).pack(anchor="w", pady=(4, 0))

        # ── Fallback command ──────────────────────────────────────────────────
        cf = ttk.LabelFrame(parent, text="Fallback: External Command  (used only if server is not running)", padding=10)
        cf.pack(fill="x")
        ttk.Label(cf, text="Command template  ({voice} {text} {out})",
                  font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        ttk.Entry(cf, textvariable=self.command_var, width=100).grid(row=0, column=1, sticky="ew", padx=(8, 0))
        ttk.Label(cf, text='Example: python infer.py --voice "{voice}" --text "{text}" --output "{out}"',
                  foreground="#666", font=("Segoe UI", 8)).grid(row=1, column=1, sticky="w", pady=(2, 0))
        ttk.Label(cf, text="Working dir").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(cf, textvariable=self.working_dir_var, width=100).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))
        ttk.Label(cf, text="ffmpeg path").grid(row=3, column=0, sticky="w")
        ttk.Entry(cf, textvariable=self.ffmpeg_var, width=40).grid(row=3, column=1, sticky="w", padx=(8, 0))
        ttk.Checkbutton(cf, text="Autosave config on change", variable=self.autosave_var).grid(row=4, column=1, sticky="w", pady=(8, 0))
        ttk.Button(cf, text="Save Config Now", command=self.save_config).grid(row=5, column=1, sticky="w", pady=(8, 0))
        cf.columnconfigure(1, weight=1)

        diag = ttk.LabelFrame(parent, text="Asset Diagnostics", padding=10)
        diag.pack(fill="x", pady=(12, 0))
        ttk.Button(diag, text="Check ONNX Assets", command=self.check_direct_assets).pack(anchor="w")

    # ─────────────────────────────────────────────────────────────────────────
    # Mixer logic
    # ─────────────────────────────────────────────────────────────────────────

    def _on_mix_change(self) -> None:
        self._recompute_summary()
        if self.autosave_var.get():
            self._safe_save_quiet()

    def _reset_output_fx(self) -> None:
        self.mix_speed_var.set(1.0)
        self.mix_pitch_var.set(0.0)

    def _recompute_summary(self) -> None:
        parts, total = [], 0.0
        for i in range(3):
            if not self.active_vars[i].get():
                continue
            name = self.voice_vars[i].get().strip() or f"slot{i+1}"
            w = float(self.weight_vars[i].get())
            total += w
            parts.append(f"{name}: {w:.0f}%")
        if not parts:
            self.summary_var.set("No active slots")
            return
        # Show effective percentages (always normalized for synthesis)
        if total > 0:
            pct_parts = []
            for i in range(3):
                if not self.active_vars[i].get():
                    continue
                name = self.voice_vars[i].get().strip() or f"slot{i+1}"
                w = float(self.weight_vars[i].get())
                pct_parts.append(f"{name}: {w/total*100:.0f}%")
            self.summary_var.set("\n".join(pct_parts) + f"\n(raw: {', '.join(p.split(': ')[1] for p in parts)})")
        else:
            self.summary_var.set("All weights zero")

    def _get_slot_voice(self, idx: int) -> VoiceBin | None:
        if not self.active_vars[idx].get():
            return None
        name = self.voice_vars[idx].get().strip()
        if not name:
            return None
        try:
            return self.voice_lib.get(name)
        except KeyError:
            return None

    def _synth_and_play(self, voice_path: Path, speed: float, pitch: float, wav_name: str) -> None:
        base = self.temp_dir / f"{wav_name}_base.wav"
        self._do_synthesize(voice_path, self.test_sentence_var.get() or DEFAULT_TEST_SENTENCE, base)
        if KokoroSynthAdapter.is_identity(speed, pitch):
            final = base
        else:
            final = self.temp_dir / f"{wav_name}_fx.wav"
            self.synth.transform_audio(base, final, speed, pitch)
        self.player.load(final)
        self.player.play()

    def preview_slot(self, idx: int) -> None:
        if not self._require_voices():
            return
        v = self._get_slot_voice(idx)
        if v is None:
            self.status_var.set(f"Slot {idx+1}: no active voice.")
            return
        if self.preview_mode_var.get() == "direct_assets":
            _, msg = self.direct_backend.diagnose()
            messagebox.showerror(APP_TITLE, msg + " Use external synth mode.")
            return
        if not self._can_synthesize():
            messagebox.showerror(APP_TITLE, "Start the server (Config tab) or configure a synth command.")
            return
        sp, pi = float(self.speed_vars[idx].get()), float(self.pitch_vars[idx].get())
        def task() -> None:
            self._synth_and_play(v.path, sp, pi, f"slot{idx}")
            self.root.after(0, lambda: self.slot_status_vars[idx].set(f"✓ {v.name}"))
        self._run_bg(task, success=f"Slot {idx+1}: {v.name}")

    def _mix_components(self) -> list[tuple[VoiceBin, float]]:
        return [(v, float(self.weight_vars[i].get()))
                for i in range(3)
                if (v := self._get_slot_voice(i)) is not None]

    def _build_mix_bin(self) -> tuple[np.ndarray, Path]:
        comps = self._mix_components()
        if not comps:
            raise RuntimeError("Select at least one active voice.")
        # Always normalize — mixed style tensors must be at expected scale for the ONNX model.
        # Unnormalized weights (e.g. 50+43=93×) produce out-of-distribution embeddings → silence.
        arr = self.voice_lib.mix(comps, normalize=True)
        p = self.temp_dir / "mix_preview.bin"
        VoiceLibrary.save_bin(arr, p)
        return arr, p

    def _weighted_fx(self) -> tuple[float, float]:
        """Return (pitch, speed) averaged by normalized active weights."""
        weights = np.array([float(self.weight_vars[i].get()) if self.active_vars[i].get() else 0.0
                             for i in range(3)], dtype=np.float32)
        if weights.sum() > 0:
            weights /= weights.sum()
        pitch = float(sum(float(self.pitch_vars[i].get()) * weights[i] for i in range(3)))
        speed = float(sum(float(self.speed_vars[i].get()) * weights[i] for i in range(3))) or 1.0
        return pitch, speed

    def preview_mix(self) -> None:
        if not self._require_voices():
            return
        if self.preview_mode_var.get() == "direct_assets":
            _, msg = self.direct_backend.diagnose()
            messagebox.showerror(APP_TITLE, msg + " Use external synth mode.")
            return
        if not self._can_synthesize():
            messagebox.showerror(APP_TITLE, "Start the server (Config tab) or configure a synth command.")
            return
        def task() -> None:
            _, mix_bin = self._build_mix_bin()
            speed = float(self.mix_speed_var.get())
            pitch = float(self.mix_pitch_var.get())
            self._synth_and_play(mix_bin, speed, pitch, "mix")
        self._run_bg(task, success="Mix preview playing")

    def _default_export_path(self) -> Path:
        d = Path(self.export_dir_var.get().strip() or DEFAULT_EXPORT_DIR)
        name = self.export_name_var.get().strip() or "mixed_voice"
        return d / (name if name.endswith(".bin") else name + ".bin")

    def _write_export(self, out: Path) -> None:
        # Export also normalizes — exported bins are meant to be used with Kokoro and must be
        # at the same scale as vendor voices. normalize_weights checkbox is respected but we
        # also force normalize so the exported file actually works.
        arr = self.voice_lib.mix(self._mix_components(), normalize=True)
        VoiceLibrary.save_bin(arr, out)
        if self.save_sidecar_var.get():
            out.with_suffix(".json").write_text(json.dumps(self._mix_preset_data(), indent=2))
        self.last_export_path = out
        if self.auto_preview_var.get():
            speed = float(self.mix_speed_var.get())
            pitch = float(self.mix_pitch_var.get())
            self._synth_and_play(out, speed, pitch, "mix_export")

    def export_mix_bin_default(self) -> None:
        if not self._require_voices():
            return
        out = self._default_export_path()
        self._run_bg(lambda: self._write_export(out), success=f"Exported → {out.name}")

    def save_mix_bin_as(self) -> None:
        if not self._require_voices():
            return
        path = filedialog.asksaveasfilename(title="Save .bin", defaultextension=".bin",
                                             filetypes=[("BIN", "*.bin")])
        if not path:
            return
        out = Path(path)
        self.export_dir_var.set(str(out.parent))
        self.export_name_var.set(out.stem)
        self._run_bg(lambda: self._write_export(out), success=f"Saved → {out.name}")

    def _mix_preset_data(self) -> dict[str, Any]:
        return {
            "voice_dir": self.voice_dir_var.get(),
            "ratings_path": self.ratings_path_var.get(),
            "test_sentence": self.test_sentence_var.get(),
            "mix_speed": float(self.mix_speed_var.get()),
            "mix_pitch": float(self.mix_pitch_var.get()),
            "slots": [{"active": bool(self.active_vars[i].get()), "voice": self.voice_vars[i].get(),
                       "weight": float(self.weight_vars[i].get()), "pitch": float(self.pitch_vars[i].get()),
                       "speed": float(self.speed_vars[i].get())} for i in range(3)],
            "traits": {k: int(v.get()) for k, v in self.target_trait_vars.items()},
        }

    def save_mix_preset(self) -> None:
        path = filedialog.asksaveasfilename(title="Save preset", defaultextension=".json",
                                             filetypes=[("JSON", "*.json")])
        if not path:
            return
        Path(path).write_text(json.dumps(self._mix_preset_data(), indent=2))
        self.status_var.set(f"Preset saved → {Path(path).name}")

    def load_mix_preset(self) -> None:
        path = filedialog.askopenfilename(title="Load preset", filetypes=[("JSON", "*.json")])
        if not path:
            return
        data = json.loads(Path(path).read_text())
        if data.get("voice_dir") and Path(data["voice_dir"]).exists():
            self._load_voice_dir(Path(data["voice_dir"]))
        if data.get("ratings_path") and Path(data["ratings_path"]).exists():
            self._load_ratings(Path(data["ratings_path"]))
        self.test_sentence_var.set(data.get("test_sentence", DEFAULT_TEST_SENTENCE))
        self.mix_speed_var.set(float(data.get("mix_speed", 1.0)))
        self.mix_pitch_var.set(float(data.get("mix_pitch", 0.0)))
        for i, slot in enumerate(data.get("slots", [])[:3]):
            self.active_vars[i].set(bool(slot.get("active", True)))
            self.voice_vars[i].set(slot.get("voice", ""))
            self.weight_vars[i].set(float(slot.get("weight", 0)))
            self.pitch_vars[i].set(float(slot.get("pitch", 0)))
            self.speed_vars[i].set(float(slot.get("speed", 1.0)))
        for k, v_val in data.get("traits", {}).items():
            if k in self.target_trait_vars:
                self.target_trait_vars[k].set(int(v_val))
        self.status_var.set(f"Loaded preset: {Path(path).name}")
        self._recompute_summary()

    def suggest_from_traits(self) -> None:
        if not self.ratings.payload.get("voices"):
            messagebox.showerror(APP_TITLE, "Load a ratings JSON first.")
            return
        target = {k: int(v.get()) for k, v in self.target_trait_vars.items()}
        scored = []
        for name in self.ratings.get_voice_names():
            traits = self.ratings.get_traits(name)
            if not traits:
                continue
            dist = sum(abs(int(traits.get(k, 3)) - tv) for k, tv in target.items())
            scored.append((dist, name))
        scored.sort()
        winners = [n for _, n in scored[:3]]
        for i in range(3):
            self.active_vars[i].set(i < len(winners))
            self.voice_vars[i].set(winners[i] if i < len(winners) else "")
            self.weight_vars[i].set(100.0 / len(winners) if winners and i < len(winners) else 0.0)
        self.status_var.set("Suggested: " + ", ".join(winners) if winners else "No rated voices matched.")
        self._recompute_summary()

    # ─────────────────────────────────────────────────────────────────────────
    # Library / config loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_voice_dir(self, root: Path) -> None:
        self.voice_dir_var.set(str(root))
        self.direct_backend.configure(root)
        vdir = root / "voices" if (root / "voices").exists() else root
        voices = self.voice_lib.load_dir(vdir)
        names = [v.name for v in voices]
        for combo in self.voice_dropdowns:
            combo["values"] = names
        for i in range(min(3, len(names))):
            if not self.voice_vars[i].get():
                self.voice_vars[i].set(names[i])
        self.status_var.set(f"Loaded {len(voices)} voices from {vdir.name}/")
        self._refresh_ratings_list()
        self._recompute_summary()

    def select_voice_dir(self) -> None:
        folder = filedialog.askdirectory(title="Select Kokoro asset root or voice folder")
        if folder:
            self._load_voice_dir(Path(folder))

    def reload_voice_dir(self) -> None:
        if t := self.voice_dir_var.get().strip():
            self._load_voice_dir(Path(t))

    def _load_ratings(self, path: Path) -> None:
        if not path.exists():
            return
        self.ratings.load(path)
        self.ratings_path_var.set(str(path))
        if self._traits_inner:
            self._build_trait_assist_sliders(self._traits_inner)
        if self._rating_trait_frame:
            self._build_rating_sliders(self._rating_trait_frame)
        self._refresh_ratings_list()
        self.status_var.set(f"Ratings loaded: {path.name}")

    def select_ratings_json(self) -> None:
        path = filedialog.askopenfilename(title="Load ratings JSON", filetypes=[("JSON", "*.json")])
        if path:
            self._load_ratings(Path(path))

    def reload_ratings_json(self) -> None:
        if t := self.ratings_path_var.get().strip():
            self._load_ratings(Path(t))

    def _load_config(self) -> None:
        if not DEFAULT_CONFIG_PATH.exists():
            return
        d = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.command_var.set(d.get("command_template", ""))
        self.ffmpeg_var.set(d.get("ffmpeg_path", "ffmpeg"))
        self.working_dir_var.set(d.get("working_dir", ""))
        self.voice_dir_var.set(d.get("voice_dir", ""))
        self.ratings_path_var.set(d.get("ratings_path", str(DEFAULT_RATINGS_PATH)))
        self.export_dir_var.set(d.get("export_dir", str(DEFAULT_EXPORT_DIR)))
        self.export_name_var.set(d.get("export_name", "mixed_voice"))
        self.test_sentence_var.set(d.get("test_sentence", DEFAULT_TEST_SENTENCE))
        self.preview_mode_var.set(d.get("preview_mode", "external_cmd"))
        self.normalize_var.set(bool(d.get("normalize_weights", True)))
        self.auto_preview_var.set(bool(d.get("auto_preview_mix", False)))
        self.save_sidecar_var.set(bool(d.get("save_sidecar", True)))
        self.mix_speed_var.set(float(d.get("mix_speed", 1.0)))
        self.mix_pitch_var.set(float(d.get("mix_pitch", 0.0)))
        self.autosave_var.set(bool(d.get("autosave_config", True)))
        for i, slot in enumerate(d.get("slots", [])[:3]):
            self.active_vars[i].set(bool(slot.get("active", True)))
            self.voice_vars[i].set(slot.get("voice", ""))
            self.weight_vars[i].set(float(slot.get("weight", 0)))
            self.pitch_vars[i].set(float(slot.get("pitch", 0)))
            self.speed_vars[i].set(float(slot.get("speed", 1.0)))
        self._sync_synth()

        # ── Auto-detect paths if config is blank or stale ───────────────────
        wd = self.working_dir_var.get().strip()
        if not wd or not Path(wd).exists():
            self.working_dir_var.set(str(APP_DIR))
            self._sync_synth()

        vd = self.voice_dir_var.get().strip()
        if not vd or not Path(vd).exists():
            if (APP_DIR / "voices").exists():
                self.voice_dir_var.set(str(APP_DIR))

        if t := self.voice_dir_var.get().strip():
            try:
                self._load_voice_dir(Path(t))
            except Exception as e:
                self.status_var.set(f"Voice dir error: {e}")
        if t := self.ratings_path_var.get().strip():
            try:
                self._load_ratings(Path(t))
            except Exception:
                pass

    def _collect_config(self) -> dict[str, Any]:
        return {
            "command_template": self.command_var.get().strip(),
            "ffmpeg_path": self.ffmpeg_var.get().strip() or "ffmpeg",
            "working_dir": self.working_dir_var.get().strip(),
            "voice_dir": self.voice_dir_var.get().strip(),
            "ratings_path": self.ratings_path_var.get().strip(),
            "export_dir": self.export_dir_var.get().strip(),
            "export_name": self.export_name_var.get().strip(),
            "test_sentence": self.test_sentence_var.get().strip(),
            "preview_mode": self.preview_mode_var.get(),
            "normalize_weights": bool(self.normalize_var.get()),
            "auto_preview_mix": bool(self.auto_preview_var.get()),
            "save_sidecar": bool(self.save_sidecar_var.get()),
            "autosave_config": bool(self.autosave_var.get()),
            "mix_speed": float(self.mix_speed_var.get()),
            "mix_pitch": float(self.mix_pitch_var.get()),
            "slots": [{"active": bool(self.active_vars[i].get()), "voice": self.voice_vars[i].get().strip(),
                       "weight": float(self.weight_vars[i].get()), "pitch": float(self.pitch_vars[i].get()),
                       "speed": float(self.speed_vars[i].get())} for i in range(3)],
            "last_export_path": str(self.last_export_path) if self.last_export_path else "",
        }

    def _sync_synth(self) -> None:
        self.synth.command_template = self.command_var.get().strip()
        self.synth.ffmpeg_path = self.ffmpeg_var.get().strip() or "ffmpeg"
        self.synth.working_dir = self.working_dir_var.get().strip()

    def save_config(self) -> None:
        self._sync_synth()
        DEFAULT_CONFIG_PATH.write_text(json.dumps(self._collect_config(), indent=2), encoding="utf-8")
        self.status_var.set(f"Config saved.")

    def _safe_save_quiet(self) -> None:
        try:
            self._sync_synth()
            DEFAULT_CONFIG_PATH.write_text(json.dumps(self._collect_config(), indent=2), encoding="utf-8")
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────────────

    def _require_voices(self) -> bool:
        if not self.voice_lib.voices:
            messagebox.showerror(APP_TITLE, "Load a voice folder first.")
            return False
        return True

    def select_export_dir(self) -> None:
        if folder := filedialog.askdirectory(title="Select export folder"):
            self.export_dir_var.set(folder)

    def open_export_folder(self) -> None:
        d = Path(self.export_dir_var.get().strip() or DEFAULT_EXPORT_DIR)
        d.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(d)])

    def stop_audio(self) -> None:
        self.player.stop()

    def check_direct_assets(self) -> None:
        t = self.voice_dir_var.get().strip()
        if not t:
            messagebox.showerror(APP_TITLE, "Load voice folder first.")
            return
        self.direct_backend.configure(Path(t))
        ok, msg = self.direct_backend.diagnose()
        (messagebox.showinfo if ok else messagebox.showerror)(APP_TITLE, msg)
        self.status_var.set(msg)

    def _run_bg(self, fn, success: str = "Done") -> None:
        if not self.preview_lock.acquire(blocking=False):
            self.status_var.set("Busy — wait for current operation to finish.")
            return
        self.status_var.set("Working…")
        def runner() -> None:
            try:
                fn()
                self.root.after(0, lambda: self.status_var.set(success))
            except Exception as e:
                msg = str(e)
                self.root.after(0, lambda m=msg: messagebox.showerror(APP_TITLE, m))
                self.root.after(0, lambda m=msg: self.status_var.set(f"Error: {m}"))
            finally:
                self.preview_lock.release()
        threading.Thread(target=runner, daemon=True).start()

    def _tick_playback(self) -> None:
        pos = self.player.position_seconds()
        dur = self.player.duration_seconds
        def fmt(s: float) -> str:
            t = max(0, int(s))
            return f"{t // 60:02}:{t % 60:02}"
        self.position_var.set(f"{fmt(pos)} / {fmt(dur)}")
        self.root.after(200, self._tick_playback)

    def _bind_keys(self) -> None:
        self.root.bind("<Control-s>", lambda _e: self.save_config())
        self.root.bind("<Escape>",    lambda _e: self.stop_audio())
        self.root.bind("<F5>",        lambda _e: self.reload_voice_dir())
        self.root.bind("<Right>",     lambda _e: self._ratings_navigate(1))
        self.root.bind("<Left>",      lambda _e: self._ratings_navigate(-1))

    def _bind_persistence(self) -> None:
        for var in [self.voice_dir_var, self.ratings_path_var, self.command_var,
                    self.ffmpeg_var, self.working_dir_var, self.export_dir_var,
                    self.export_name_var, self.test_sentence_var, self.preview_mode_var,
                    self.auto_preview_var, self.save_sidecar_var,
                    self.mix_speed_var, self.mix_pitch_var,
                    *self.active_vars, *self.voice_vars, *self.weight_vars,
                    *self.pitch_vars, *self.speed_vars]:
            try:
                var.trace_add("write", lambda *_: self._on_state_change())
            except Exception:
                pass

    def _on_state_change(self) -> None:
        self._recompute_summary()
        if self.autosave_var.get():
            self._safe_save_quiet()

    def _on_close(self) -> None:
        self._precache_stop.set()
        try:
            self._safe_save_quiet()
        except Exception:
            pass
        self.player.stop()
        self._server.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    for theme in ("vista", "clam", "default"):
        try:
            ttk.Style(root).theme_use(theme)
            break
        except Exception:
            pass
    VoiceLabApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
