#!/usr/bin/env python3
"""Parallel, resumable HTTP range downloader (stdlib only). Splits each file into chunks fetched concurrently,
keeps finished chunks on disk, and concatenates at the end. Use for large HF files on slow single-stream links.
Usage: python tools/pdl.py --dest data/models/Qwen3.5-4B --repo Qwen/Qwen3.5-4B --rev <sha> --files a.safetensors b.json ... [--streams 12] [--chunk-mb 32]"""
import argparse, concurrent.futures as cf, os, pathlib, sys, threading, time, urllib.request

def head(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("Content-Length", 0)), r.url

class Resolver:
    """The HF resolve URL redirects to a signed CDN URL that expires (about 1 h). Re-resolve on errors and every 20 min;
    before 2026-10-16 the URL was resolved once and every request failed after expiry (silent stall)."""
    def __init__(self, url):
        self.url = url; self.lock = threading.Lock(); self.refresh(force=True)
    def refresh(self, force=False):
        with self.lock:
            if force or time.time() - self.t > 1200:
                self.size, self.final = head(self.url); self.t = time.time()
        return self.final

def fetch_range(res, start, end, path, tries=200):
    want = end - start + 1
    for t in range(tries):
        have = path.stat().st_size if path.exists() else 0
        if have > want:                                   # corrupt part (should not happen): restart it
            path.unlink(); have = 0
        if have == want:
            return have
        try:
            req = urllib.request.Request(res.refresh(), headers={"Range": f"bytes={start+have}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:                       # server ignored the range: never append a full body
                    raise IOError(f"status {r.status}")
                with path.open("ab") as f:
                    while True:
                        b = r.read(1 << 20)
                        if not b: break
                        f.write(b)
        except Exception as e:
            if getattr(e, "code", None) in (401, 403, 404, 410):
                try: res.refresh(force=True)
                except Exception: pass
            time.sleep(min(60, 3 * (t + 1)))
    raise RuntimeError(f"failed {path}")

def download(url, dest: pathlib.Path, streams, chunk):
    if dest.exists() and not dest.with_suffix(dest.suffix + ".parts").exists():
        print(f"skip {dest.name} (exists)", flush=True); return
    try:
        res = Resolver(url); size = res.size
    except Exception as e:
        print(f"SKIP {dest.name}: {e}", flush=True); return
    parts = dest.with_suffix(dest.suffix + ".parts"); parts.mkdir(parents=True, exist_ok=True)
    ranges = [(s, min(s + chunk, size) - 1) for s in range(0, size, chunk)]
    t0 = time.time(); done_bytes = [0]
    def job(i):
        s, e = ranges[i]; return fetch_range(res, s, e, parts / f"{i:05d}")
    print(f"{dest.name}: {size/1e9:.2f} GB in {len(ranges)} chunks, {streams} streams", flush=True)
    with cf.ThreadPoolExecutor(streams) as ex:
        futs = {ex.submit(job, i): i for i in range(len(ranges))}; n = 0
        for fu in cf.as_completed(futs):
            fu.result(); n += 1
            if n % max(1, len(ranges) // 20) == 0:
                got = sum((parts / f"{i:05d}").stat().st_size for i in range(len(ranges)) if (parts / f"{i:05d}").exists())
                print(f"  {dest.name}: {got/size*100:5.1f}% {got/1e6/(time.time()-t0):.2f} MB/s", flush=True)
    bad = [i for i, (s, e) in enumerate(ranges) if not (parts / f"{i:05d}").exists() or (parts / f"{i:05d}").stat().st_size != e - s + 1]
    assert not bad, f"incomplete chunks {bad[:10]}"
    with dest.open("wb") as out:
        for i in range(len(ranges)):
            out.write((parts / f"{i:05d}").read_bytes())
    assert dest.stat().st_size == size, "size mismatch"
    for p in parts.iterdir(): p.unlink()
    parts.rmdir(); print(f"done {dest.name} {size/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min", flush=True)

ap = argparse.ArgumentParser(); ap.add_argument("--repo", required=True); ap.add_argument("--rev", default="main"); ap.add_argument("--dest", type=pathlib.Path, required=True)
ap.add_argument("--files", nargs="+", required=True); ap.add_argument("--streams", type=int, default=12); ap.add_argument("--chunk-mb", type=int, default=32)
a = ap.parse_args(); a.dest.mkdir(parents=True, exist_ok=True)
for f in a.files:
    download(f"https://huggingface.co/{a.repo}/resolve/{a.rev}/{f}", a.dest / f, a.streams, a.chunk_mb << 20)
print("ALL DONE", flush=True)
