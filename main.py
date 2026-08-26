# main.py
# Whisper STT by Devexus (https://devexus.net)
# Offline speech-to-text with a push-to-talk hotkey: hold CTRL + WINDOWS
# (CTRL + CMD on Mac) to record, release to transcribe -> clipboard / paste.
# Run `python main.py --help` for all flags.

# ---- BEGIN: NVIDIA DLL PATH FIX (Windows only) ----
import os, sys
import sysconfig
import warnings
from pathlib import Path

# Suppress warnings early (before imports that might emit them)
warnings.filterwarnings("ignore")

def _add_nvidia_bins_to_path():
    if sys.platform != "win32":
        return
    # Ask Python where site-packages is instead of guessing the venv layout
    # (the old sys.executable-relative guess broke for non-venv installs).
    # A PyInstaller build ships the wheels (if any) under _MEIPASS instead.
    sp = Path(getattr(sys, "_MEIPASS", None) or sysconfig.get_paths()["purelib"])
    cand = [
        sp / "nvidia" / "cudnn" / "bin",
        sp / "nvidia" / "cublas" / "bin",
        sp / "nvidia" / "cuda_runtime" / "bin",
        # Some wheels nest differently; add extra fallbacks if present
        sp / "nvidia" / "cupti" / "bin",
    ]
    for p in cand:
        if p.exists():
            os.add_dll_directory(str(p))
            os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

_add_nvidia_bins_to_path()
# ---- END: NVIDIA DLL PATH FIX ----

import argparse
import logging
import queue
import threading
import numpy as np
import sounddevice as sd
import keyboard
import pyperclip
import pyautogui
import sys
import time
from collections import deque
from PIL import Image, ImageDraw
import pystray
import webbrowser

from faster_whisper import WhisperModel
from platformdirs import user_data_dir, user_log_dir

__version__ = "1.0.2"
APP_NAME = "Whisper STT"
APP_AUTHOR = "Devexus"
APP_URL = "https://devexus.net"
EXE_NAME = "whisper-stt-devexus"

log = logging.getLogger("whisper_stt")

def attach_console():
    """Frozen windowed exe: reuse the parent terminal's console if there is one.

    PyInstaller builds with console=False give us no stdout/stderr at all
    (both are None). Attaching to the parent process means running the exe
    from cmd/PowerShell still shows output, while double-clicking it from
    Explorer stays silent. If nothing can be attached, point the streams at
    devnull so libraries that print (tqdm download bars) don't crash.
    """
    if sys.stdout is not None and sys.stderr is not None:
        return
    if sys.platform == "win32":
        import ctypes
        if ctypes.windll.kernel32.AttachConsole(-1):  # ATTACH_PARENT_PROCESS
            try:
                sys.stdout = open("CONOUT$", "w", encoding="utf-8", errors="replace")
                sys.stderr = sys.stdout
                return
            except OSError:
                pass
    devnull = open(os.devnull, "w")
    sys.stdout = sys.stdout or devnull
    sys.stderr = sys.stderr or devnull

def log_file_path() -> Path:
    return Path(user_log_dir("WhisperSTT", APP_AUTHOR)) / "whisper-stt.log"

def setup_logging(debug: bool):
    """INFO to stdout plus a rotating log file; --debug adds DEBUG and re-enables warnings."""
    from logging.handlers import RotatingFileHandler
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s %(levelname)s %(message)s" if debug else "%(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    try:
        path = log_file_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"))
        handlers.append(fh)
    except OSError as ex:
        print(f"Could not open log file: {ex}", file=sys.stderr)
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S", handlers=handlers)
    if debug:
        warnings.filterwarnings("default")
    # Keep --debug about this app, not library internals
    for noisy in ("PIL", "urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.INFO)

def cuda_runtime_libs_present():
    """Check the CUDA runtime DLLs CTranslate2 needs at inference time can be loaded.

    get_cuda_device_count() only asks the driver, so it says yes on any machine
    with an NVIDIA GPU even when cuBLAS/cuDNN aren't installed (e.g. the CPU-only
    exe). Those libs load lazily, so without this check the failure only shows
    up on the first transcription.
    """
    if sys.platform != "win32":
        return True  # Linux/mac: let CTranslate2 report its own errors
    import ctypes
    for name in ("cublas64_12.dll", "cudnn64_9.dll"):
        try:
            ctypes.WinDLL(name)
        except OSError:
            log.debug("CUDA runtime library %s not found", name)
            return False
    return True

def fatal(msg: str, exc_info=False):
    """Log a startup error and exit. With no console (windowed exe) also show a message box."""
    log.error("%s", msg, exc_info=exc_info)
    if sys.platform == "win32" and getattr(sys, "frozen", False):
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, f"{msg}\n\nLog: {log_file_path()}", APP_NAME, 0x10)  # MB_ICONERROR
    sys.exit(1)

def is_cuda_available():
    """Check if CUDA is available for faster-whisper (uses CTranslate2, not PyTorch)."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except (ImportError, RuntimeError, AttributeError):
        # If ctranslate2 check fails, assume CPU (faster-whisper will handle CUDA errors gracefully)
        return False
    return cuda_runtime_libs_present()

def validate_model_name(model_name: str):
    """Validate model name to prevent arbitrary file loading."""
    # Valid Whisper model names
    VALID_MODELS = {
        "tiny", "tiny.en",
        "base", "base.en",
        "small", "small.en",
        "medium", "medium.en",
        "large", "large-v1", "large-v2", "large-v3"
    }
    if model_name not in VALID_MODELS:
        raise ValueError(
            f"Invalid model name: '{model_name}'. "
            f"Valid models: {', '.join(sorted(VALID_MODELS))}"
        )

# ---------------- Defaults (overridable via CLI flags) ----------------
SAMPLE_RATE = 16000
CHANNELS = 1
DEFAULT_MIN_DURATION_SEC = 0.25   # --min-duration
DEFAULT_SILENCE_THRESH = 1e-4     # --silence-thresh
DEFAULT_BEAM_SIZE = 5             # --beam-size
# ---------------------------------------------------------------------


# --- Hotkey state tracking for "hold Ctrl + Windows" ---
_ctrl_down = False
_win_down = False
_key_state_lock = threading.Lock()  # Thread safety for key state
recording_q = queue.Queue()
is_recording = threading.Event()
stream = None
model = None
HISTORY_INLINE = 5   # newest transcripts shown directly in the tray menu
HISTORY_MAX = 25     # older ones go to the History submenu
transcript_history = deque(maxlen=HISTORY_MAX)
history_lock = threading.Lock()
tray_icon = None
keyboard_hook = None  # Store hook reference for cleanup
# Shared with the tray menu: current status text, resolved device and the
# argparse namespace (so menu toggles like output mode take effect live).
app_state = {"status": "Starting", "device": "cpu", "settings": None, "model_dir": None}
_recording_source = None  # "hotkey" | "menu": who started the current recording
shutdown_event = threading.Event()  # Set by any thread to request a clean exit
# Transcription jobs are handed off to a worker so the keyboard hook thread
# is never blocked by a multi-second Whisper run.
transcribe_q = queue.Queue()

def init_audio():
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS

def audio_callback(indata, frames, time_info, status):
    if status:
        log.debug("audio callback status: %s", status)
    if is_recording.is_set():
        recording_q.put(indata.copy())

def start_stream():
    global stream
    stream = sd.InputStream(callback=audio_callback,
                            samplerate=SAMPLE_RATE,
                            channels=CHANNELS,
                            dtype='float32')
    stream.start()

def stop_stream():
    global stream
    if stream:
        try:
            stream.stop()
            stream.close()
        except Exception:
            log.debug("closing audio stream failed", exc_info=True)
        stream = None

def drain_queue():
    chunks = []
    while True:
        try:
            chunks.append(recording_q.get_nowait())
        except queue.Empty:
            break
    if not chunks:
        return None
    audio = np.concatenate(chunks, axis=0)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32)

def trim_silence(audio, thresh=DEFAULT_SILENCE_THRESH):
    if audio is None or len(audio) == 0:
        return audio
    amp = np.abs(audio)
    if np.max(amp) < thresh:
        return np.array([], dtype=np.float32)
    idx = np.where(amp >= thresh)[0]
    start = max(idx[0] - int(0.02 * SAMPLE_RATE), 0)
    end = min(idx[-1] + int(0.02 * SAMPLE_RATE), len(audio))
    return audio[start:end]

MODEL_DIR_ENV = "WHISPER_STT_MODEL_DIR"

def resolve_model_dir(cli_value=None) -> Path:
    """Pick where Whisper models are stored/downloaded.

    Precedence:
      1. --model-dir
      2. $WHISPER_STT_MODEL_DIR
      3. ./hf_cache next to main.py, if it already exists (running from source)
      4. per-user data dir (platformdirs), e.g.
           Windows: %LOCALAPPDATA%/Devexus/WhisperSTT/models
           macOS:   ~/Library/Application Support/WhisperSTT/models
           Linux:   ~/.local/share/WhisperSTT/models

    A frozen exe (PyInstaller onefile) extracts to a temp dir, so anything
    derived from __file__ would be wiped on exit; that's why 3 only applies
    when the folder already exists and we're not frozen.
    """
    if cli_value:
        return Path(cli_value).expanduser()
    env = os.environ.get(MODEL_DIR_ENV)
    if env:
        return Path(env).expanduser()
    if not getattr(sys, "frozen", False):
        legacy = Path(__file__).resolve().parent / "hf_cache"
        if legacy.is_dir():
            return legacy
    return Path(user_data_dir("WhisperSTT", APP_AUTHOR)) / "models"

def make_model(device: str, model_name: str, cache_dir: Path):
    """
    device: 'cpu' | 'cuda'
    cache_dir: where models are downloaded to / loaded from
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    if device == "cuda":
        # Try CUDA first, fallback to CPU if it fails. Run a tiny warm-up
        # transcription: CUDA libs load lazily, so a missing DLL only shows up
        # here, not in the constructor.
        try:
            m = WhisperModel(model_name, device="cuda", compute_type="float16", download_root=str(cache_dir))
            list(m.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), beam_size=1)[0])
            return m
        except Exception:
            log.warning("CUDA failed, falling back to CPU", exc_info=log.isEnabledFor(logging.DEBUG))
            return WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(cache_dir))
    else:
        # CPU-friendly quantization
        return WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(cache_dir))

def transcribe_array(m, audio, language, beam_size=DEFAULT_BEAM_SIZE):
    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    segments, info = m.transcribe(
        audio,
        beam_size=beam_size,
        vad_filter=False,  # Disable since we're doing manual VAD
        language=language,
        condition_on_previous_text=False,  # Better for short clips
        temperature=0.0  # Deterministic
    )
    return "".join(seg.text for seg in segments).strip()

def output_text(text, mode: str, restore_clipboard: bool = False):
    if not text:
        return
    
    # Add to history
    with history_lock:
        transcript_history.append(text)
    if tray_icon:
        try:
            tray_icon.update_menu()
        except Exception:
            log.debug("tray menu update failed", exc_info=True)
    
    prev_clip = None
    try:
        prev_clip = pyperclip.paste()
    except Exception:
        log.debug("could not read clipboard", exc_info=True)
        prev_clip = None

    # Always copy transcript first so clipboard ends with it by default
    pyperclip.copy(text)

    if mode == 'paste':
        time.sleep(0.03)  # tiny delay helps some UIs
        pyautogui.hotkey('ctrl', 'v')
        if restore_clipboard and prev_clip is not None:
            time.sleep(0.05)  # give the target app time to read the clipboard
            pyperclip.copy(prev_clip)

def begin_recording(source="hotkey"):
    global _recording_source
    if is_recording.is_set():
        return
    drain_queue()           # clear any stale audio
    _recording_source = source
    is_recording.set()
    set_status("Recording...")

def end_recording(settings):
    """Stop recording and queue the captured audio for transcription.

    Runs on the keyboard hook thread, so it must return quickly: no
    transcription happens here, only a hand-off to the worker.
    """
    global _recording_source
    if not is_recording.is_set():
        return
    is_recording.clear()
    _recording_source = None
    set_status("Transcribing...")
    transcribe_q.put(settings)

def toggle_recording():
    """Tray menu action: start a recording, or stop the current one."""
    settings = app_state["settings"]
    if is_recording.is_set():
        log.info("Transcribing...")
        end_recording(settings)
    elif settings is not None:
        log.info("Recording...")
        begin_recording(source="menu")

def transcribe_worker():
    """Drain recordings and transcribe them one at a time."""
    while True:
        settings = transcribe_q.get()
        if settings is None:  # shutdown sentinel
            break
        min_samples = int(SAMPLE_RATE * settings.min_duration)
        try:
            time.sleep(0.05)        # let the audio callback flush its last block
            audio = drain_queue()
            if audio is None or len(audio) < min_samples:
                continue
            if not settings.no_trim:
                audio = trim_silence(audio, settings.silence_thresh)
                if audio is None or len(audio) < min_samples:
                    continue
            try:
                text = transcribe_array(model, audio, settings.lang, settings.beam_size)
            except Exception as ex:
                log.error("Transcribe error: %s", ex, exc_info=log.isEnabledFor(logging.DEBUG))
                continue
            output_text(text, settings.mode, settings.restore_clipboard)
        finally:
            log.debug("transcribe job done")
            if not is_recording.is_set():
                set_status("Ready")
            transcribe_q.task_done()

# --- Keyboard handling for hold combo ---
# On Mac, use "cmd" or "command" instead of "windows"
WIN_NAMES = {"windows", "win", "left windows", "right windows", "cmd", "command", "left cmd", "right cmd"}

def on_key_event(e, args):
    """Track ctrl + windows state and toggle recording accordingly."""
    global _ctrl_down, _win_down
    # Normalize name
    name = (e.name or "").lower()
    
    log.debug("key %s %s", name, e.event_type)
    
    # Thread-safe key state updates
    with _key_state_lock:
        if name in ("ctrl", "left ctrl", "right ctrl"):
            _ctrl_down = e.event_type == "down"
        elif name in WIN_NAMES:
            _win_down = e.event_type == "down"
        
        combo = _ctrl_down and _win_down
    
    # Recording control (outside lock to avoid holding it during I/O).
    # A recording started from the tray menu is only stopped from the menu,
    # otherwise any key release would end it.
    if combo and not is_recording.is_set():
        log.info("Recording...")
        begin_recording(source="hotkey")
    elif not combo and is_recording.is_set() and _recording_source == "hotkey":
        log.info("Transcribing...")
        end_recording(args)

def resource_path(rel: str) -> Path:
    """Path to a bundled asset, both from source and inside a PyInstaller build."""
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base / rel

def create_tray_icon():
    logo = resource_path("assets/logo.png")
    if logo.is_file():
        try:
            return Image.open(logo).convert("RGBA")
        except Exception:
            log.debug("could not load %s, using drawn icon", logo, exc_info=True)
    # Fallback: simple drawn microphone
    img = Image.new('RGB', (64, 64), color='white')
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 35, 44, 55], fill='black')
    draw.rectangle([28, 20, 36, 40], fill='black')
    draw.arc([15, 25, 49, 50], start=200, end=340, fill='black', width=3)
    draw.line([32, 50, 32, 58], fill='black', width=3)
    draw.line([25, 58, 39, 58], fill='black', width=3)
    return img

def recording_tray_icon(base: Image.Image) -> Image.Image:
    """Base icon with a red dot in the corner, shown while recording."""
    img = base.copy().convert("RGBA")
    w, h = img.size
    r = max(4, w // 3)
    draw = ImageDraw.Draw(img)
    draw.ellipse([w - r - 2, h - r - 2, w - 2, h - 2], fill=(220, 40, 40, 255), outline=(255, 255, 255, 255), width=max(1, w // 32))
    return img

_icons = {}  # "idle" / "recording" images, filled in by main()

def set_status(status: str):
    """Update status text, tray tooltip, icon and menu. Safe from any thread."""
    app_state["status"] = status
    if tray_icon is None:
        return
    try:
        tray_icon.title = f"{APP_NAME} - {status}"
        if _icons:
            tray_icon.icon = _icons["recording" if status.startswith("Recording") else "idle"]
        tray_icon.update_menu()
    except Exception:
        log.debug("tray update failed", exc_info=True)

def status_line() -> str:
    s = app_state["settings"]
    if s is None:
        return app_state["status"]
    return f"{app_state['status']}  |  {s.model} on {app_state['device']}  |  {s.mode}"

def copy_from_history(text):
    pyperclip.copy(text)

def copy_last_transcript():
    with history_lock:
        text = transcript_history[-1] if transcript_history else None
    if text:
        pyperclip.copy(text)

def open_path(p):
    """Open a file or folder with the OS default handler."""
    try:
        p = Path(p)
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.suffix:
                p.touch()
        if sys.platform == "win32":
            os.startfile(str(p))
        else:
            import subprocess
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(p)])
    except Exception:
        log.warning("could not open %s", p, exc_info=log.isEnabledFor(logging.DEBUG))

def open_tray_menu(icon):
    """Left click on Windows: show the same popup menu as right click.

    pystray only offers a 'default action' for left click, so we poke its
    Win32 backend with a synthetic right-click. Other platforms open the menu
    on left click by themselves.
    """
    if sys.platform != "win32":
        return
    try:
        from pystray._util import win32
        icon._on_notify(0, win32.WM_RBUTTONUP)
    except Exception:
        log.debug("could not open tray menu", exc_info=True)

def set_mode(mode: str):
    s = app_state["settings"]
    if s is not None:
        s.mode = mode
        log.info("Output mode: %s", mode)
        set_status(app_state["status"])

def menu_items():
    """Built fresh every time the menu opens (pystray calls this on update_menu)."""
    s = app_state["settings"]
    with history_lock:
        history = list(reversed(transcript_history))
    yield pystray.MenuItem("open", open_tray_menu, default=True, visible=False)
    yield pystray.MenuItem(status_line(), None, enabled=False)
    yield pystray.Menu.SEPARATOR
    yield pystray.MenuItem(lambda _: "Stop recording" if is_recording.is_set() else "Start recording",
                           lambda: toggle_recording())
    yield pystray.MenuItem("Copy last transcript", lambda: copy_last_transcript(), enabled=bool(history))
    yield pystray.Menu.SEPARATOR
    # Newest transcripts inline (click to copy), the rest in a submenu
    def history_item(t):
        return pystray.MenuItem((t[:50] + "...") if len(t) > 50 else t, lambda _, t=t: copy_from_history(t))
    if history:
        for t in history[:HISTORY_INLINE]:
            yield history_item(t)
    else:
        yield pystray.MenuItem("No transcripts yet", None, enabled=False)
    older = history[HISTORY_INLINE:]
    if older:
        yield pystray.MenuItem(f"Older ({len(older)})", pystray.Menu(*[history_item(t) for t in older]))
    yield pystray.Menu.SEPARATOR
    yield pystray.MenuItem("Output", pystray.Menu(
        pystray.MenuItem("Paste into active window", lambda: set_mode("paste"), radio=True,
                         checked=lambda _: s is not None and s.mode == "paste"),
        pystray.MenuItem("Clipboard only", lambda: set_mode("clipboard"), radio=True,
                         checked=lambda _: s is not None and s.mode == "clipboard"),
    ))
    yield pystray.MenuItem("Open log file", lambda: open_path(log_file_path()))
    yield pystray.MenuItem("Open models folder", lambda: open_path(app_state["model_dir"]),
                           enabled=app_state["model_dir"] is not None)
    yield pystray.Menu.SEPARATOR
    yield pystray.MenuItem(f"{APP_NAME} v{__version__} by {APP_AUTHOR}", lambda: webbrowser.open(APP_URL))
    yield pystray.MenuItem("Exit", lambda: stop_app())

def get_menu():
    return pystray.Menu(menu_items)

def stop_app():
    """Request a clean shutdown.

    Called from the tray thread. Raising SystemExit here would only kill the
    tray thread, so instead we signal the main thread, which owns cleanup.
    """
    shutdown_event.set()

def main():
    attach_console()
    parser = argparse.ArgumentParser(prog=EXE_NAME, description=f"{APP_NAME} v{__version__}")
    parser.add_argument("--model", default="small", help="Whisper model (tiny|base|small|medium|large-v3...)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage (default: auto-detect GPU)")
    parser.add_argument("--mode", choices=["paste", "clipboard"], default="paste",
                        help="Output mode (default: paste)")
    parser.add_argument("--lang", default=None, help='Force language like "en" or "hr" (default: auto)')
    parser.add_argument("--model-dir", default=None,
                        help=f"Where to store/load models (default: ${MODEL_DIR_ENV}, ./hf_cache if present, else per-user data dir)")
    parser.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION_SEC, metavar="SEC",
                        help=f"Ignore recordings shorter than this (default: {DEFAULT_MIN_DURATION_SEC})")
    parser.add_argument("--no-trim", action="store_true",
                        help="Don't trim leading/trailing silence before transcribing")
    parser.add_argument("--silence-thresh", type=float, default=DEFAULT_SILENCE_THRESH, metavar="AMP",
                        help=f"Amplitude below which audio counts as silence for trimming (default: {DEFAULT_SILENCE_THRESH})")
    parser.add_argument("--beam-size", type=int, default=DEFAULT_BEAM_SIZE, metavar="N",
                        help=f"Whisper beam size; higher is slower (default: {DEFAULT_BEAM_SIZE})")
    parser.add_argument("--restore-clipboard", action="store_true",
                        help="In paste mode, put the previous clipboard content back after pasting")
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__} - {APP_NAME} by {APP_AUTHOR} ({APP_URL})")
    parser.add_argument("--debug", action="store_true", help="Show warnings and debug output")
    args = parser.parse_args()
    if args.min_duration < 0 or args.silence_thresh < 0 or args.beam_size < 1:
        parser.error("--min-duration and --silence-thresh must be >= 0, --beam-size >= 1")
    
    setup_logging(args.debug)
    
    global tray_icon, keyboard_hook
    
    # Validate model name for security
    try:
        validate_model_name(args.model)
    except ValueError as e:
        fatal(str(e))

    # The `keyboard` package needs a global hook; on Linux/macOS that requires root.
    if sys.platform != "win32" and hasattr(os, "geteuid") and os.geteuid() != 0:
        fatal("Global keyboard hooks require root on Linux/macOS. "
              "Re-run with sudo (e.g. `sudo python main.py`).")

    # Resolve device
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if is_cuda_available() else "cpu"
        if device == "cpu":
            log.info("CUDA not available, using CPU")

    hotkey = "CTRL + CMD" if sys.platform == "darwin" else "CTRL + WINDOWS"
    log.info("%s v%s by %s (%s)", APP_NAME, __version__, APP_AUTHOR, APP_URL)
    log.info("- Hold %s to talk; release to transcribe.", hotkey)
    log.info("- Model: %s | Device: %s | Mode: %s | Lang: %s", args.model, device, args.mode, args.lang or "auto")
    model_dir = resolve_model_dir(args.model_dir)
    log.info("- Model dir: %s", model_dir)
    log.info("- Log file: %s", log_file_path())

    try:
        init_audio()
        start_stream()
    except Exception as ex:
        fatal(f"Audio error: {ex}", exc_info=args.debug)

    try:
        log.info("Loading Whisper model, first run may take a bit...")
        global model
        model = make_model(device, args.model, model_dir)
    except Exception as ex:
        stop_stream()
        fatal(f"Model load error: {ex}", exc_info=args.debug)

    # Worker that performs transcription off the keyboard hook thread
    worker = threading.Thread(target=transcribe_worker, name="transcribe", daemon=True)
    worker.start()

    # Low-level keyboard hook so we can detect both key down/up events
    # Store the hook reference for proper cleanup
    keyboard_hook = keyboard.hook(lambda e: on_key_event(e, args))

    # System tray icon. On macOS the Cocoa event loop must run on the main
    # thread, so pystray owns the main thread there and we block on keyboard
    # events from a helper thread instead. Elsewhere the tray runs in its own
    # thread and the main thread blocks.
    app_state.update(settings=args, device=device, model_dir=model_dir, status="Ready")
    icon_image = create_tray_icon()
    _icons.update(idle=icon_image, recording=recording_tray_icon(icon_image))
    tray_icon = pystray.Icon(EXE_NAME, icon_image, f"{APP_NAME} - Ready", menu=get_menu())

    log.info("Ready. Hold %s to record.", hotkey)

    def cleanup():
        transcribe_q.put(None)  # tell the worker to stop
        if keyboard_hook is not None:
            try:
                keyboard.unhook(keyboard_hook)
            except Exception:
                log.debug("unhooking keyboard failed", exc_info=True)
        stop_stream()
        log.info("Goodbye.")

    def stop_tray():
        try:
            tray_icon.stop()
        except Exception:
            log.debug("stopping tray failed", exc_info=True)

    if sys.platform == "darwin":
        # Tray owns the main thread. A helper waits for the shutdown request
        # (tray Exit sets shutdown_event) and stops the tray, which returns
        # control to main for cleanup.
        def _wait_for_shutdown():
            shutdown_event.wait()
            stop_tray()
        threading.Thread(target=_wait_for_shutdown, daemon=True).start()
        try:
            tray_icon.run()  # blocks until tray_icon.stop()
        except KeyboardInterrupt:
            pass
        finally:
            shutdown_event.set()
            cleanup()
    else:
        tray_thread = threading.Thread(target=tray_icon.run, daemon=False)
        tray_thread.start()
        try:
            # Block until the tray "Exit" item (or another thread) requests shutdown.
            # Keyboard events are delivered on the hook's own thread meanwhile.
            while not shutdown_event.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            stop_tray()
            cleanup()

if __name__ == "__main__":
    main()
