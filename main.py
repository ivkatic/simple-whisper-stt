# app.py
# Whisper Push-To-Talk
# Hold CTRL + WINDOWS (or CTRL + CMD on Mac) to record; release to transcribe -> clipboard (and paste if enabled).
# Flags:
#   --model <name>   e.g., tiny, base, small, medium, large-v3 (default: small)
#   --cpu            force CPU (default: auto-detect GPU)
#   --mode <paste|clipboard>  output behavior (default: paste)
#   --lang <code>    force language, e.g. "en", "hr" (default: auto)
# Example:
#   python app.py --model base        (auto-detect GPU)
#   python app.py --cpu --model small (force CPU)
#   python app.py --mode clipboard

# ---- BEGIN: NVIDIA DLL PATH FIX (Windows only) ----
import os, sys
import warnings
from pathlib import Path

# Suppress warnings early (before imports that might emit them)
warnings.filterwarnings("ignore")

def _add_nvidia_bins_to_path():
    if sys.platform != "win32":
        return
    # Find site-packages (works inside venv too)
    sp = Path(sys.executable).parent.parent / "Lib" / "site-packages"
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

from faster_whisper import WhisperModel

__version__ = "1.0.0"

def is_cuda_available():
    """Check if CUDA is available for faster-whisper (uses CTranslate2, not PyTorch)."""
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, RuntimeError, AttributeError):
        # If ctranslate2 check fails, assume CPU (faster-whisper will handle CUDA errors gracefully)
        return False

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

# ---------------- Defaults (can be overridden by CLI) ----------------
SAMPLE_RATE = 16000
CHANNELS = 1
KEEP_TRANSCRIPT_IN_CLIPBOARD = True
MIN_DURATION_SEC = 0.25
VAD_TRIM = True
SILENCE_THRESH = 1e-4
# ---------------------------------------------------------------------


# --- Hotkey state tracking for "hold Ctrl + Windows" ---
_ctrl_down = False
_win_down = False
_key_state_lock = threading.Lock()  # Thread safety for key state
recording_q = queue.Queue()
is_recording = threading.Event()
stream = None
model = None
transcript_history = deque(maxlen=5)
history_lock = threading.Lock()
tray_icon = None
keyboard_hook = None  # Store hook reference for cleanup
shutdown_event = threading.Event()  # Set by any thread to request a clean exit

def init_audio():
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS

def audio_callback(indata, frames, time_info, status):
    if status:
        # print(status, file=sys.stderr)
        pass
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
            pass
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

def trim_silence(audio, thresh=SILENCE_THRESH):
    if audio is None or len(audio) == 0:
        return audio
    amp = np.abs(audio)
    if np.max(amp) < thresh:
        return np.array([], dtype=np.float32)
    idx = np.where(amp >= thresh)[0]
    start = max(idx[0] - int(0.02 * SAMPLE_RATE), 0)
    end = min(idx[-1] + int(0.02 * SAMPLE_RATE), len(audio))
    return audio[start:end]

def make_model(device: str, model_name: str):
    """
    device: 'cpu' | 'cuda'
    """
    # Use local cache folder
    cache_dir = Path(__file__).parent / "hf_cache"
    cache_dir.mkdir(exist_ok=True)
    
    if device == "cuda":
        # Try CUDA first, fallback to CPU if it fails
        try:
            return WhisperModel(model_name, device="cuda", compute_type="float16", download_root=str(cache_dir))
        except Exception:
            print("[Warning] CUDA failed, falling back to CPU", file=sys.stderr)
            return WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(cache_dir))
    else:
        # CPU-friendly quantization
        return WhisperModel(model_name, device="cpu", compute_type="int8", download_root=str(cache_dir))

def transcribe_array(m, audio, language):
    # Normalize audio
    audio = audio / (np.max(np.abs(audio)) + 1e-8)
    
    segments, info = m.transcribe(
        audio,
        beam_size=10,  # Better accuracy
        vad_filter=False,  # Disable since we're doing manual VAD
        language=language,
        condition_on_previous_text=False,  # Better for short clips
        temperature=0.0  # Deterministic
    )
    return "".join(seg.text for seg in segments).strip()

def output_text(text, mode: str):
    if not text:
        return
    
    # Add to history
    with history_lock:
        transcript_history.append(text)
    if tray_icon:
        tray_icon.menu = get_menu()
    
    prev_clip = None
    try:
        prev_clip = pyperclip.paste()
    except Exception:
        prev_clip = None

    # Always copy transcript first so clipboard ends with it by default
    pyperclip.copy(text)

    if mode == 'paste':
        time.sleep(0.03)  # tiny delay helps some UIs
        pyautogui.hotkey('ctrl', 'v')
        if not KEEP_TRANSCRIPT_IN_CLIPBOARD and prev_clip is not None:
            pyperclip.copy(prev_clip)

def begin_recording():
    if is_recording.is_set():
        return
    drain_queue()           # clear any stale audio
    is_recording.set()

def end_recording_and_transcribe(lang, mode):
    if not is_recording.is_set():
        return
    is_recording.clear()
    time.sleep(0.05)        # let callback flush
    audio = drain_queue()
    if audio is None or len(audio) < int(SAMPLE_RATE * MIN_DURATION_SEC):
        return
    if VAD_TRIM:
        audio = trim_silence(audio)
        if audio is None or len(audio) < int(SAMPLE_RATE * MIN_DURATION_SEC):
            return
    try:
        text = transcribe_array(model, audio, lang)
    except Exception as ex:
        print(f"[Transcribe error] {ex}", file=sys.stderr)
        return
    output_text(text, mode)

# --- Keyboard handling for hold combo ---
# On Mac, use "cmd" or "command" instead of "windows"
WIN_NAMES = {"windows", "win", "left windows", "right windows", "cmd", "command", "left cmd", "right cmd"}

def on_key_event(e, args):
    """Track ctrl + windows state and toggle recording accordingly."""
    global _ctrl_down, _win_down
    # Normalize name
    name = (e.name or "").lower()
    
    # Debug: uncomment to see what keys are detected
    # print(f"Key: {name} ({e.event_type})", file=sys.stderr)
    
    # Thread-safe key state updates
    with _key_state_lock:
        if name in ("ctrl", "left ctrl", "right ctrl"):
            _ctrl_down = e.event_type == "down"
        elif name in WIN_NAMES:
            _win_down = e.event_type == "down"
        
        combo = _ctrl_down and _win_down
    
    # Recording control (outside lock to avoid holding it during I/O)
    if combo and not is_recording.is_set():
        # print("[Recording started]", file=sys.stderr)
        begin_recording()
    elif not combo and is_recording.is_set():
        print("[Recording stopped, transcribing...]", file=sys.stderr)
        end_recording_and_transcribe(args.lang, args.mode)

def create_tray_icon():
    # Create simple microphone icon
    img = Image.new('RGB', (64, 64), color='white')
    draw = ImageDraw.Draw(img)
    draw.ellipse([20, 35, 44, 55], fill='black')
    draw.rectangle([28, 20, 36, 40], fill='black')
    draw.arc([15, 25, 49, 50], start=200, end=340, fill='black', width=3)
    draw.line([32, 50, 32, 58], fill='black', width=3)
    draw.line([25, 58, 39, 58], fill='black', width=3)
    return img

def copy_from_history(text):
    pyperclip.copy(text)

def get_menu():
    with history_lock:
        items = []
        if transcript_history:
            for i, text in enumerate(reversed(transcript_history)):
                preview = (text[:50] + '...') if len(text) > 50 else text
                items.append(pystray.MenuItem(preview, lambda _, t=text: copy_from_history(t)))
            items.append(pystray.Menu.SEPARATOR)
        else:
            items.append(pystray.MenuItem('No history yet', lambda _: None, enabled=False))
            items.append(pystray.Menu.SEPARATOR)
        items.append(pystray.MenuItem('Exit', lambda _: stop_app()))
        return pystray.Menu(*items)

def stop_app():
    """Request a clean shutdown.

    Called from the tray thread. Raising SystemExit here would only kill the
    tray thread, so instead we signal the main thread, which owns cleanup.
    """
    shutdown_event.set()

def main():
    parser = argparse.ArgumentParser(description=f"Whisper Push-To-Talk v{__version__}")
    parser.add_argument("--model", default="small", help="Whisper model (tiny|base|small|medium|large-v3...)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU usage (default: auto-detect GPU)")
    parser.add_argument("--mode", choices=["paste", "clipboard"], default="paste",
                        help="Output mode (default: paste)")
    parser.add_argument("--lang", default=None, help='Force language like "en" or "hr" (default: auto)')
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--debug", action="store_true", help="Show warnings and debug output")
    args = parser.parse_args()
    
    # Re-enable warnings if debug mode is enabled
    if args.debug:
        warnings.filterwarnings("default")
    
    global tray_icon, keyboard_hook
    
    # Validate model name for security
    try:
        validate_model_name(args.model)
    except ValueError as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve device
    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if is_cuda_available() else "cpu"
        if device == "cpu":
            print("[Info] CUDA not available, using CPU")

    hotkey = "CTRL + CMD" if sys.platform == "darwin" else "CTRL + WINDOWS"
    print(f"Whisper PTT v{__version__} running.")
    print(f"- Hold {hotkey} to talk; release to transcribe.")
    print(f"- Model: {args.model} | Device: {device} | Mode: {args.mode} | Lang: {args.lang or 'auto'}")

    try:
        init_audio()
        start_stream()
    except Exception as ex:
        print(f"[Audio error] {ex}", file=sys.stderr)
        sys.exit(1)

    try:
        print("[Loading Whisper model] First run may take a bit...")
        global model
        model = make_model(device, args.model)
    except Exception as ex:
        print(f"[Model load error] {ex}", file=sys.stderr)
        stop_stream()
        sys.exit(1)

    # Low-level keyboard hook so we can detect both key down/up events
    # Store the hook reference for proper cleanup
    keyboard_hook = keyboard.hook(lambda e: on_key_event(e, args))

    # Create and run system tray icon in separate thread
    icon_image = create_tray_icon()
    tray_icon = pystray.Icon("whisper_ptt", icon_image, "Whisper PTT", menu=get_menu())
    tray_thread = threading.Thread(target=tray_icon.run, daemon=False)
    tray_thread.start()
    
    print(f"[Ready] Model loaded. Hold {hotkey} to record.")
    
    try:
        # Block until the tray "Exit" item (or another thread) requests shutdown.
        # Keyboard events are delivered on the hook's own thread meanwhile.
        while not shutdown_event.wait(0.5):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        # Proper resource cleanup
        if keyboard_hook is not None:
            try:
                keyboard.unhook(keyboard_hook)
            except Exception:
                pass
        if tray_icon:
            try:
                tray_icon.stop()
            except Exception:
                pass
        stop_stream()
        print("Goodbye.")

if __name__ == "__main__":
    main()
