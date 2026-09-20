"""One-off installed-distribution verification; never part of the release."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument("candidate", type=Path)
p.add_argument("dist", type=Path)
p.add_argument("output", type=Path)
p.add_argument("--conda", action="store_true")
a = p.parse_args()
candidate, dist, output = a.candidate.resolve(), a.dist.resolve(), a.output.resolve()
output.mkdir(parents=True, exist_ok=True)
os.environ.pop("PYTHONPATH", None)
for key, folder in [("NUMBA_CACHE_DIR", "numba"), ("MPLCONFIGDIR", "mpl"), ("TMPDIR", "tmp")]:
    (output / folder).mkdir(exist_ok=True)
    os.environ[key] = str(output / folder)
def run(*args):
    print("+", *map(str, args), flush=True)
    subprocess.run(list(map(str, args)), cwd=output, check=True)
def pip(*args):
    run(sys.executable, "-m", "pip", *args)
def hashes(root):
    return {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(root.rglob("*")) if f.is_file()}
def committed_bytes(file):
    relative = file.relative_to(candidate).as_posix()
    return subprocess.check_output(["git", "-C", str(candidate), "show", f"HEAD:{relative}"])
golden = hashes(candidate / "tests/data")
wheel = next(dist.glob("*.whl"))
sdist = next(dist.glob("*.tar.gz"))
pip("install", *(["--no-deps"] if a.conda else []), wheel)
pip("check")
# Assert the import location and every shipped source/data file before testing.
import bcmp
installed = Path(bcmp.__file__).resolve().parent
assert installed.is_relative_to(Path(sys.prefix).resolve()), installed
assert not installed.is_relative_to(candidate), installed
assert metadata.version("bcmp") == "1.0.0"
for f in (candidate / "src/bcmp").rglob("*"):
    if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pyc", ".pyo"):
        original = committed_bytes(f)
        actual = (installed / f.relative_to(candidate / "src/bcmp")).read_bytes()
        if f.read_bytes() != original:
            assert f.read_bytes().replace(b"\r\n", b"\n") == original, f
            print("Git checkout newline conversion:", f.name, flush=True)
        assert original == actual, f
pip("install", "pytest")
pip("check")
record = {
    "candidate": subprocess.check_output(["git", "-C", str(candidate), "rev-parse", "HEAD"], text=True).strip(),
    "python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
    "prefix": sys.prefix, "bcmp_import": str(installed), "version": metadata.version("bcmp"),
    "dependencies": dict(sorted((d.metadata["Name"], d.version) for d in metadata.distributions())),
    "artifacts": hashes(dist), "golden_before": golden,
}
(output / "environment.json").write_text(json.dumps(record, indent=2) + "\n")
run(sys.executable, "-m", "pytest", candidate / "tests", "-v", "--import-mode=importlib",
    "-o", f"cache_dir={output / 'pytest-cache'}", "--basetemp", output / "pytest-temp",
    "--junitxml", output / "tests.xml")
assert golden == hashes(candidate / "tests/data"), "Fixtures changed"
# Build and reinstall from sdist, without replacing the validated runtime dependencies.
pip("install", "--force-reinstall", "--no-deps", sdist)
pip("check")
run(sys.executable, "-c",
    "from importlib.metadata import version; from bcmp import bcmp,bcmp_embedding; "
    "from bcmp.datasets import diabetic_kidney_lite; "
    "assert version('bcmp') == '1.0.0'; "
    "assert diabetic_kidney_lite().shape == (1500,27980); print('sdist installation verified')")
for f in (candidate / "src/bcmp").rglob("*"):
    if f.is_file() and "__pycache__" not in f.parts and f.suffix not in (".pyc", ".pyo"):
        assert committed_bytes(f) == (installed / f.relative_to(candidate / "src/bcmp")).read_bytes(), f
record["golden_after"] = hashes(candidate / "tests/data")
record["result"] = "passed"
(output / "environment.json").write_text(json.dumps(record, indent=2) + "\n")
print("PASS: wheel full suite, frozen golden, sdist install, dependency consistency", flush=True)
