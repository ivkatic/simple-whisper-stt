# Whisper STT

**by [Devexus](https://devexus.net)**

A simple offline speech-to-text application using faster-whisper. Runs completely locally. Hold **CTRL + WINDOWS** (or **CTRL + CMD** on Mac) to record, release to transcribe instantly. Text is automatically pasted where your cursor is (paste mode) or simply copied to clipboard (clipboard mode).

**Version:** 1.0.2

## Features

- Push-to-talk hotkey (hold to record, release to transcribe)
- GPU (CUDA) and CPU support with auto-detection
- Multiple Whisper models (tiny, base, small, medium, large-v3)
- Auto or manual language selection
- System tray integration with transcript history
- Two output modes:
  - **Paste mode** (default): Auto-paste transcribed text where your cursor is
  - **Clipboard mode**: Copy to clipboard only, no auto-paste
- **100% offline/local** - No internet connection required, all processing happens on your device

## Requirements

- Windows 10+ / macOS / Linux
- Python 3.9+ (tested on 3.13)
- CUDA-capable GPU (optional, for faster processing)
- Root/sudo access on Linux/Mac (the `keyboard` package installs a global hook; the app exits with an error if not root)
- On macOS the system tray runs on the main thread (Cocoa requirement); you may also need to grant Accessibility permission to your terminal for the hook and auto-paste to work

## Installation

**Optional: Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**For GPU support:**
Install CUDA toolkit (11.8 or 12.1+) and cuDNN. The `faster-whisper` package includes CTranslate2 which handles CUDA automatically - no PyTorch needed.

## Usage

Basic usage:
```bash
python main.py
# Defaults: model=small, device=auto-detect GPU, mode=paste, lang=auto-detect
```

With options:
```bash
python main.py --model medium           # Use medium model (auto-detect GPU)
python main.py --cpu --model small      # Force CPU with small model
python main.py --mode clipboard         # Clipboard only (no auto-paste)
python main.py --lang en                # Force English
```

## Command-Line Options

- `--model <name>` - Model size: tiny, base, small, medium, large-v3 (default: small)
- `--cpu` - Force CPU usage (default: auto-detect GPU, fallback to CPU)
- `--mode <paste|clipboard>` - Output behavior (default: paste)
- `--lang <code>` - Force language code like "en" or "hr" (default: auto-detect)
- `--model-dir <path>` - Where models are downloaded/loaded (see below)
- `--min-duration <sec>` - Ignore recordings shorter than this (default: 0.25)
- `--no-trim` - Don't trim leading/trailing silence before transcribing
- `--silence-thresh <amp>` - Amplitude treated as silence when trimming (default: 0.0001)
- `--beam-size <n>` - Whisper beam size, higher is slower (default: 5)
- `--restore-clipboard` - In paste mode, restore the previous clipboard content after pasting
- `--version` - Show version number and exit
- `--debug` - Debug logging (key events, tracebacks) and library warnings; off by default

The app automatically detects CUDA availability and falls back to CPU on non-NVIDIA systems.

## Building a standalone exe (Windows)

```bash
pip install -r requirements.txt "pyinstaller>=6,<7"
python build_tools/make_icon.py
python build_tools/make_version_info.py
python -m PyInstaller whisper-stt-devexus.spec
```

Output is `dist/whisper-stt-devexus/` (~250 MB, CPU only). Run `whisper-stt-devexus.exe` with the same flags as `main.py`. The exe has no console window: double-click it and only the tray icon appears; run it from cmd/PowerShell and the output shows up there. Startup errors pop a message box. Models are downloaded to the per-user data dir (see below). Set `WHISPER_STT_BUILD_CUDA=1` before building to bundle the CUDA libraries (adds ~1.6 GB).

Tagging `v*` runs the same build in GitHub Actions and attaches the zip to the release.

## Where models are stored

Models (~75 MB for `tiny` up to ~3 GB for `large-v3`) are downloaded once and cached. Location, in order of precedence:

1. `--model-dir <path>`
2. `WHISPER_STT_MODEL_DIR` environment variable
3. `hf_cache/` next to `main.py`, if that folder already exists (running from source)
4. Per-user data dir:
   - Windows: `%LOCALAPPDATA%\Devexus\WhisperSTT\models`
   - macOS: `~/Library/Application Support/WhisperSTT/models`
   - Linux: `~/.local/share/WhisperSTT/models`

The path in use is printed at startup.

## Logs

Everything printed to the console is also written to a rotating log file (1 MB x 4), useful for the exe which has no console:

- Windows: `%LOCALAPPDATA%\Devexus\WhisperSTT\Logs\whisper-stt.log`
- macOS: `~/Library/Logs/WhisperSTT/whisper-stt.log`
- Linux: `~/.local/state/WhisperSTT/log/whisper-stt.log`

`--debug` raises the level to DEBUG for both outputs. The path is printed at startup.

## How It Works

1. Hold **CTRL + WINDOWS** (or **CTRL + CMD** on Mac) to start recording
2. Speak into your microphone
3. Release keys to stop recording and transcribe
4. Text is automatically pasted or copied to clipboard
5. Access recent transcripts from the system tray icon

## Tray menu

Left or right click the tray icon (left click opens the menu on Windows; macOS/Linux use their own tray conventions). The tooltip and icon reflect the state: a red dot shows while recording.

- Status line: `Ready | small on cpu | paste` (Ready / Recording... / Transcribing...)
- **Start recording** / **Stop recording** - toggle a recording without holding the hotkey. A recording started here is only stopped here, the hotkey does not end it.
- **Copy last transcript** - puts the most recent transcript on the clipboard again
- Last 5 transcripts inline, click one to copy it; **Older** submenu holds up to 20 more
- **Output** - switch between pasting into the active window and clipboard only, takes effect immediately
- **Open log file** / **Open models folder**
- About (opens devexus.net) and **Exit**

## Changing the Hotkey

To customize the hotkey combination, edit these lines in `main.py`:

**Line 223:** Modify `WIN_NAMES` to your preferred key:
```python
WIN_NAMES = {"windows", "win", "left windows", "right windows", "cmd", "command", "left cmd", "right cmd"}
```

**Line 234-237:** Change the key names you want to track:
```python
if name in ("ctrl", "left ctrl", "right ctrl"):
    _ctrl_down = e.event_type == "down"
elif name in WIN_NAMES:
    _win_down = e.event_type == "down"
```

For example, to use **ALT + SPACE**:
- Change line 234 to check for "alt" or "left alt", "right alt"
- Change line 236 to check for "space"

## Platform-Specific Notes

- **Linux/Mac**: Run with `sudo python main.py` (keyboard hook requires elevated privileges)
- **Windows**: No special privileges needed
- **Mac**: System preferences may require allowing Terminal/Python to control the computer

