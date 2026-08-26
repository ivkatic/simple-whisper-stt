"""Render assets/icon.ico from the tray icon drawn in main.py (run from repo root)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
src = Path("main.py").read_text().split('if __name__ == "__main__":')[0]
ns = {"__file__": str(Path("main.py").resolve())}
exec(compile(src, "main.py", "exec"), ns)
img = ns["create_tray_icon"]().resize((256, 256))
out = Path("assets/icon.ico")
out.parent.mkdir(exist_ok=True)
img.save(out, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("wrote", out)
