# PyInstaller spec. Build with:  python -m PyInstaller whisper-stt-devexus.spec
# Produces dist/whisper-stt-devexus/ (onedir). CPU-only: NVIDIA CUDA wheels are
# excluded (they add ~1.6 GB); set WHISPER_STT_BUILD_CUDA=1 to include them.
import os, re, sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_data_files

src = Path("main.py").read_text()
EXE_NAME = re.search(r'^EXE_NAME = "([^"]+)"', src, re.M).group(1)
CUDA = os.environ.get("WHISPER_STT_BUILD_CUDA") == "1"

datas, binaries, hiddenimports = [], [], []
for pkg in ["ctranslate2", "faster_whisper"]:
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h
datas += collect_data_files("sounddevice")  # portaudio dll lives in _sounddevice_data

excludes = ["tkinter", "matplotlib", "IPython", "notebook"]
if CUDA:
    for pkg in ["nvidia.cublas", "nvidia.cudnn", "nvidia.cuda_runtime"]:
        d, b, h = collect_all(pkg)
        datas += d; binaries += b; hiddenimports += h
else:
    excludes.append("nvidia")

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports + ["pystray._win32", "PIL._tkinter_finder"],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name=EXE_NAME,
    console=True,           # keeps the log output visible; tray still works
    icon="assets/icon.ico" if sys.platform == "win32" else None,
    version="build/version_info.txt" if sys.platform == "win32" else None,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name=EXE_NAME)
