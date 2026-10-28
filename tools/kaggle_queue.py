#!/usr/bin/env python3
"""Push rendered Kaggle kernels (runs/kaggle/<slug>/) in order, at most --slots at a time, each only after its kernel sources
have COMPLETED, not before --start, and never once the summed --hours estimates would pass --cap. Polls every 5 min; finished
kernels' outputs are downloaded to runs/kaggle/<slug>/output. Log lines go to stdout (run it under nohup).

Usage: python tools/kaggle_queue.py --start "2026-10-24 07:10" --cap 24 \
          kx-tt-skywork:2.8 kx-tt-deepseek:2.8 kx-states-skywork-amc:1.5 ...   (slug:estimated session hours)
--external slug ... counts kernels pushed by hand against the slots. Stop: create runs/STOP_QUEUE (running kernels are left alone).
"""
import argparse, datetime as dt, json, pathlib, re, subprocess, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
KAGGLE = pathlib.Path.home() / "Library/Application Support/pipx/venvs/kaggle/bin/kaggle"
KG = str(KAGGLE) if KAGGLE.exists() else "kaggle"
USER = "<kaggle-user>"


def status(slug: str) -> str:
    r = subprocess.run([KG, "kernels", "status", f"{USER}/{slug}"], capture_output=True, text=True)
    m = re.search(r"Status\.([A-Z_]+)", r.stdout + r.stderr); return m.group(1) if m else "UNKNOWN"


def log(*a):
    print(dt.datetime.now().strftime("%m-%d %H:%M"), *a, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("jobs", nargs="+"); ap.add_argument("--start", default=None)
    ap.add_argument("--cap", type=float, required=True, help="max summed estimated hours this queue may push")
    ap.add_argument("--slots", type=int, default=2); ap.add_argument("--timeout", type=int, default=21600)
    ap.add_argument("--external", nargs="*", default=[], help="kernels pushed outside this queue that occupy slots while RUNNING/QUEUED")
    a = ap.parse_args()
    start = dt.datetime.strptime(a.start, "%Y-%m-%d %H:%M") if a.start else dt.datetime.now()
    todo = [(j.split(":")[0], float(j.split(":")[1])) for j in a.jobs]; pushed, done, spent = [], set(), 0.0
    while todo or any(s not in done for s in pushed):
        if (ROOT / "runs/STOP_QUEUE").exists(): log("STOP_QUEUE present; exiting"); return
        running = [x for x in a.external if status(x) in ("RUNNING", "QUEUED")]
        for s in pushed:
            if s in done: continue
            st = status(s)
            if st in ("RUNNING", "QUEUED", "UNKNOWN"): running.append(s); continue
            done.add(s); log(f"{s} finished: {st}")
            out = ROOT / "runs/kaggle" / s / "output"; out.mkdir(parents=True, exist_ok=True)
            r = subprocess.run([KG, "kernels", "output", f"{USER}/{s}", "-p", str(out)], capture_output=True, text=True); log((r.stdout + r.stderr).strip()[-200:])
        if dt.datetime.now() >= start and todo and len(running) < a.slots:
            slug, h = todo[0]; meta = json.loads((ROOT / "runs/kaggle" / slug / "kernel-metadata.json").read_text())
            srcs = [x.split("/")[-1] for x in meta.get("kernel_sources", [])]; blocked = [x for x in srcs if status(x) != "COMPLETE"]
            if spent + h > a.cap:
                log(f"cap: {slug} ({h} h) would bring the queue to {spent + h:.1f} > {a.cap} h; stopping pushes"); todo = []
            elif blocked:
                log(f"{slug} waits for {blocked}")
            else:
                r = subprocess.run([KG, "kernels", "push", "-p", str(ROOT / "runs/kaggle" / slug), "-t", str(a.timeout)], capture_output=True, text=True)
                log(f"push {slug} (est {h} h, queue total {spent + h:.1f} h):", (r.stdout + r.stderr).strip()[-160:])
                if "successfully" in (r.stdout + r.stderr).lower(): pushed.append(slug); spent += h; todo.pop(0); continue
        time.sleep(300)
    log(f"queue finished; pushed {pushed}, estimated {spent:.1f} h")


if __name__ == "__main__":
    main()
