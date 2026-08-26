"""Generate build/version_info.txt for PyInstaller's Windows version resource from main.py constants."""
import re
from pathlib import Path
src = Path("main.py").read_text()
g = lambda k: re.search(rf'^{k} = "([^"]+)"', src, re.M).group(1)
ver, name, author, url, exe = g("__version__"), g("APP_NAME"), g("APP_AUTHOR"), g("APP_URL"), g("EXE_NAME")
parts = [int(x) for x in ver.split(".")] + [0] * 4
tup = tuple(parts[:4])
txt = f"""# UTF-8
VSVersionInfo(
  ffi=FixedFileInfo(filevers={tup}, prodvers={tup}, mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', '{author}'),
      StringStruct('FileDescription', '{name} - offline speech to text'),
      StringStruct('FileVersion', '{ver}'),
      StringStruct('InternalName', '{exe}'),
      StringStruct('LegalCopyright', '{author} ({url})'),
      StringStruct('OriginalFilename', '{exe}.exe'),
      StringStruct('ProductName', '{name}'),
      StringStruct('ProductVersion', '{ver}')])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
"""
out = Path("build/version_info.txt"); out.parent.mkdir(exist_ok=True); out.write_text(txt)
print("wrote", out)
