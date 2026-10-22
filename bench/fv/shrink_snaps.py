"""Shrink FV snapshot tarballs in place: keep only the members under app/ (the solution file), which is all the
regrader and plain_check read. Skips files modified in the last 2 minutes (jobs may still be writing).
  python bench/fv/shrink_snaps.py runs/jobs
"""
import gzip
import io
import os
import sys
import tarfile
import time
from pathlib import Path

saved = n = 0
for f in Path(sys.argv[1]).glob("*/*/agent/fv/snap/*.tar.gz"):
    if time.time() - f.stat().st_mtime < 120:
        continue
    before = f.stat().st_size
    try:
        with tarfile.open(fileobj=io.BytesIO(gzip.decompress(f.read_bytes()))) as t:
            keep = [(m, t.extractfile(m).read() if m.isfile() else None) for m in t.getmembers()
                    if m.name.lstrip("./").startswith("app")]
    except Exception as e:  # noqa: BLE001
        print("skip", f, e)
        continue
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for m, data in keep:
            t.addfile(m, io.BytesIO(data) if data is not None else None)
    tmp = f.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(buf.getvalue()))
    os.replace(tmp, f)
    saved += before - f.stat().st_size
    n += 1
print(f"shrunk {n} snapshots, freed {saved / 1e9:.2f} GB")
