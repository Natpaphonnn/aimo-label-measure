#!/usr/bin/env python3
"""Fetch the public leaderboard feed (website source) and print the top entries plus any rows matching a name.
Usage: python tools/check_board.py --name <codabench-user> [--top 8]"""
import argparse, json, urllib.request, collections
ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True, help="Codabench user name to highlight"); ap.add_argument("--top", type=int, default=8); a = ap.parse_args()
d = json.load(urllib.request.urlopen("https://vps.josefkuchar.com/leaderboards", timeout=30))
print("updated:", d["last_updated_at"])
for track, n in [("main", 21), ("small", 14)]:
    rows = d["leaderboards"][track]; hist = collections.Counter(round(r["accuracy"], 2) for r in rows)
    print(f"\n{track.upper()} ({len(rows)} participants): " + ", ".join(f"{k:.2f}x{v}" for k, v in sorted(hist.items(), reverse=True)))
    for r in rows[: a.top]: print(f"  {r['accuracy']:.2f} ({round(r['accuracy']*n)}/{n}) {r['owner_name']:20s} {r['submission_time'][:16]} id={r['submission_id']}")
    mine = [r for r in rows if a.name.lower() in r["owner_name"].lower()]
    print("  matching", repr(a.name) + ":", [(r["owner_name"], r["accuracy"], r["submission_id"]) for r in mine] or "none yet")
