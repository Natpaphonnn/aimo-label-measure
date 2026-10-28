#!/usr/bin/env python3
"""Build deterministic Codabench ZIPs (main + small) for a solution directory and print SHA-256.

Usage: python tools/build_submission.py submissions/<name> [--tag v1]
Writes dist/<name>-<tag>-main.zip and dist/<name>-<tag>-small.zip (small.txt added at root).
"""
import argparse, hashlib, sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = (2020, 1, 1, 0, 0, 0)

def entries(directory: Path) -> dict[str, bytes]:
    out = {}
    for p in sorted(directory.rglob("*")):
        rel = p.relative_to(directory)
        if p.is_file() and "__pycache__" not in rel.parts and not p.name.endswith((".pyc", ".pyo")) and not p.name.startswith("."):
            out[rel.as_posix()] = p.read_bytes()
    if "solution.py" not in out:
        sys.exit(f"{directory} has no solution.py at its root")
    return out

def archive(path: Path, items: dict[str, bytes]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED, compresslevel=9) as z:
        for name in sorted(items):
            info = ZipInfo(name, date_time=FIXED_TIME); info.compress_type = ZIP_DEFLATED; info.external_attr = 0o100644 << 16
            z.writestr(info, items[name])
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("solution", type=Path); ap.add_argument("--tag", default="v1")
    ap.add_argument("--include", nargs="*", default=[], help="package directories under src/ to embed at the ZIP root, e.g. src/runtime")
    a = ap.parse_args(); name = a.solution.name
    items = entries(a.solution)
    for inc in a.include:
        pkg = ROOT / inc
        for rel, data in entries(pkg).items() if (pkg / "solution.py").exists() else {}.items():
            pass
        for p in sorted(pkg.rglob("*")):
            r = p.relative_to(pkg)
            if p.is_file() and "__pycache__" not in r.parts and not p.name.endswith((".pyc", ".pyo")) and not p.name.startswith("."):
                items[f"{pkg.name}/{r.as_posix()}"] = p.read_bytes()
    items.pop("small.txt", None)
    main_zip = ROOT / "dist" / f"{name}-{a.tag}-main.zip"; small_zip = ROOT / "dist" / f"{name}-{a.tag}-small.zip"
    h1 = archive(main_zip, dict(items)); h2 = archive(small_zip, {**items, "small.txt": b""})
    print(f"{main_zip.relative_to(ROOT)}\t{main_zip.stat().st_size} bytes\tsha256={h1}")
    print(f"{small_zip.relative_to(ROOT)}\t{small_zip.stat().st_size} bytes\tsha256={h2}")

if __name__ == "__main__":
    main()
