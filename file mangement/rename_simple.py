import re, os
from pathlib import Path

FOLDER = Path(r"D:\project\photo\json\m6_collected_images")
pat = re.compile(r"(Scene-\d+_frame_\d+)", re.IGNORECASE)

for p in FOLDER.iterdir():
    if not p.is_file(): 
        continue
    m = pat.search(p.stem)
    if not m:
        continue
    new = FOLDER / f"{m.group(1)}{p.suffix.lower()}"
    i = 1
    while new.exists():
        new = FOLDER / f"{m.group(1)}_{i}{p.suffix.lower()}"
        i += 1
    p.rename(new)

print("Done.")
