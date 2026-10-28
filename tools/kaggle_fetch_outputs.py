#!/usr/bin/env python3
"""Resumable download of a Kaggle kernel's output files (the CLI breaks on multi-MB files over slow links).
Lists files via the Kaggle API (needs the kaggle package: run with the pipx kaggle interpreter), then fetches each URL with
HTTP Range requests, retrying and resuming until the size matches. Usage: <kaggle-python> tools/kaggle_fetch_outputs.py <slug> <dest_dir>"""
import json, pathlib, sys, time, urllib.request
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
except ImportError:
    sys.exit("[missing package] kaggle (pip install kaggle; only needed to download Kaggle kernel outputs, with your own API token)")

slug, dest = sys.argv[1], pathlib.Path(sys.argv[2]); dest.mkdir(parents=True, exist_ok=True)
api = KaggleApi(); api.authenticate()

def list_files():
    with api.build_kaggle_client() as kaggle:
        req = ApiListKernelSessionOutputRequest(); req.user_name = "<kaggle-user>"; req.kernel_slug = slug; req.page_size = 100
        resp = kaggle.kernels.kernels_api_client.list_kernel_session_output(req)
    files = [(f.file_name, f.url, getattr(f, "total_bytes", None)) for f in resp.files]
    return files, getattr(resp, "log", None)

def remote_size(url):
    """Content-Length from a 1-byte range request (Content-Range: bytes 0-0/TOTAL) or a plain GET header."""
    for attempt in range(10):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                cr = r.headers.get("Content-Range")
                if cr and "/" in cr: return int(cr.split("/")[-1])
                cl = r.headers.get("Content-Length")
                return int(cl) if cl and r.status == 200 else None
        except Exception:
            time.sleep(3)
    return None

def fetch(url, path, total):
    total = total or remote_size(url)
    if not total:
        raise RuntimeError(f"unknown size for {path.name}; refusing to trust a possibly truncated download")
    for attempt in range(200):
        have = path.stat().st_size if path.exists() else 0
        if total and have >= total: return
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            with urllib.request.urlopen(req, timeout=120) as r, path.open("ab" if have else "wb") as f:
                if have and r.status != 206:  # server ignored Range: restart
                    f.seek(0); f.truncate()
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    f.write(b)
            if not total or path.stat().st_size >= total: return
        except Exception as e:
            print(f"  retry {attempt+1} {path.name} at {path.stat().st_size if path.exists() else 0} bytes: {type(e).__name__}", flush=True); time.sleep(5)
    raise RuntimeError(f"gave up on {path.name}")

files, log = list_files()
print(f"{len(files)} output files:", [(n, s) for n, _, s in files], flush=True)
if log:
    (dest / f"{slug}.log").write_text(log if isinstance(log, str) else json.dumps(log)); print("saved log", flush=True)
for name, url, size in files:
    p = dest / pathlib.Path(name).name; t = time.time()
    if not (p.suffix in (".npz", ".jsonl", ".log")): continue   # skip the packed source files echoed back by the kernel
    if size and p.exists() and p.stat().st_size == size: print(f"ok {p.name} (cached)", flush=True); continue
    fetch(url, p, size)
    if p.suffix == ".npz":
        import zipfile; assert zipfile.is_zipfile(p), f"{p.name} is not a valid zip"
    print(f"done {p.name} {p.stat().st_size} bytes in {time.time()-t:.0f}s", flush=True)
print("ALL DONE")
