"""Parse the server stderr logs of validation runs (runs/val_logs/<id>.log): one dict per are_robust call. Run directly to
compare the wb/bb/hybrid predictions case by case."""
import ast, re, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from datapaths import need  # noqa: E402
import os  # noqa: E402
D = pathlib.Path(os.environ.get("AIMO_VAL_LOGS", pathlib.Path(__file__).resolve().parents[1] / "runs/val_logs"))
def parse(f):
    need(D / f)
    cases, cur = [], None
    for line in (D / f).read_text(errors="replace").replace("\r", "\n").splitlines():
        m = re.search(r"call \d+: (\S+) effort=(\S+) n=(\d+) prior=([\d.]+) allot=(\d+)s arm=(\S+) model=(\S+)", line)
        if m: cur = {"model": m[1].split("/")[-1], "effort": m[2], "n": int(m[3]), "arm": m[6], "loaded": m[7]}; continue
        m = re.search(r"case: .* gen ([\d.]+)s steps (\d+) cut (\d+) k (\d+) agree_top ([\d.]+) finished (\d+)/(\d+)", line)
        if m and cur is not None: cur.update(gen=float(m[1]), steps=int(m[2]), cut=int(m[3]), k=int(m[4]), agree=float(m[5])); continue
        m = re.search(r"two-stage: (\{.*\})", line)
        if m and cur is not None: cur["ts"] = ast.literal_eval(m[1]); continue
        m = re.search(r"done .*: (\d+)/(\d+) robust in ([\d.]+)s", line)
        if m and cur is not None: cur.update(robust=int(m[1]), sec=float(m[3])); cases.append(cur); cur = None
        if "falling back" in line or "Traceback" in line or "timeout" in line.lower(): print("!!", f, line[:200])
    return cases
if __name__ == "__main__":
    runs = {"main": {"wb": "064c9648.log", "bb": "a883f9ed.log", "hybrid": "55f8f640.log"},
            "small": {"wb": "ef31bdf0.log", "bb": "053a6aa5.log", "hybrid": "b78892ec.log"}}
    for track, rs in runs.items():
        P = {a: parse(f) for a, f in rs.items()}
        print(f"\n### {track}: cases", {a: len(v) for a, v in P.items()})
        n = len(P["wb"]); diff = 0
        print("i model effort | k cut agree(wb/bb/hy) | arm_used wb/bb/hy | p wb/bb/hy | robust wb/bb/hy | sec")
        for i in range(n):
            r = {a: P[a][i] for a in P}
            assert len({(x["model"], x["effort"]) for x in r.values()}) == 1
            rob = [r[a]["robust"] for a in ("wb", "bb", "hybrid")]; diff += len(set(rob)) > 1
            g = lambda a, k: r[a].get("ts", {}).get(k, "-")
            print(i + 1, r["wb"]["model"], r["wb"]["effort"], "|", r["wb"].get("k"), r["wb"].get("cut"), [r[a].get("agree") for a in ("wb","bb","hybrid")], "|",
                  g("wb","arm_used"), g("bb","arm_used"), g("hybrid","arm_used"), "|", g("wb","p_robust"), g("bb","p_robust"), g("hybrid","p_robust"), "|", rob, "|", [r[a].get("sec") for a in ("wb","bb","hybrid")])
        print("cases where predictions differ:", diff, "| robust counts", {a: sum(x["robust"] for x in v) for a, v in P.items()})
        print("missing two-stage line:", {a: sum("ts" not in x for x in v) for a, v in P.items()})
