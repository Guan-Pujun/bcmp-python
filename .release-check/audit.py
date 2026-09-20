"""Audit release archives against the exact committed candidate."""
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
import zipfile
source, dist = map(lambda p: Path(p).resolve(), sys.argv[1:3])
expected = {str(p.relative_to(source / "src")): p.read_bytes()
    for p in (source / "src/bcmp").rglob("*")
    if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")}
sensitive = re.compile(r"/Users/|BCMP_Dev|BCMP_PyPkg|migration_staging|PujunMacBookPro|[\\w.+-]+@163\\.com|-----BEGIN .*PRIVATE KEY-----")
records = []
for file in sorted(dist.iterdir()):
    if file.suffix == ".whl":
        with zipfile.ZipFile(file) as archive:
            entries = {n: archive.read(n) for n in archive.namelist() if not n.endswith("/")}
        payload = {n: b for n, b in entries.items() if n.startswith("bcmp/")}
        assert payload == expected, "Wheel package content differs from source"
        assert all(n.startswith(("bcmp/", "bcmp-1.0.0.dist-info/")) for n in entries)
        meta = entries["bcmp-1.0.0.dist-info/METADATA"]
    else:
        assert file.name == "bcmp-1.0.0.tar.gz", file
        with tarfile.open(file) as archive:
            entries = {m.name.split("/", 1)[1]: archive.extractfile(m).read()
                       for m in archive.getmembers() if m.isfile()}
        payload = {n.removeprefix("src/"): b for n, b in entries.items() if n.startswith("src/bcmp/")}
        assert payload == expected, "sdist package content differs from source"
        assert all(n.startswith(("src/bcmp/", "src/bcmp.egg-info/")) or
                   n in ("LICENSE", "README.md", "pyproject.toml", "MANIFEST.in", "PKG-INFO", "setup.cfg")
                   for n in entries), entries.keys()
        meta = entries["PKG-INFO"]
    assert b"Version: 1.0.0\n" in meta
    assert b"License-Expression: GPL-3.0-or-later" in meta
    for name, data in entries.items():
        assert not sensitive.search(data.decode("utf-8", errors="ignore")), name
        assert "DATASET.md" not in name
    records.append({"file": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "bytes": file.stat().st_size, "files": sorted(entries)})
print(json.dumps({"result": "passed", "archives": records}, indent=2))
