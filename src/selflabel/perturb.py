"""Template perturbation families (answer-preserving by construction), deterministic per (problem_id, family, seed).

Adapted from byeolki/aimo-interp src/aimo_interp/perturb.py (MIT) with changes:
  rename   - person names and single-letter point/variable labels are renamed consistently; letters are only renamed
             when they occur as standalone labels in math spans and the mapping avoids letters already present;
             LaTeX commands are never touched. Files a 'skipped' reason when nothing safe to rename exists.
  typos    - adjacent-letter swap or an inserted space in 2-3 prose words of >= 6 letters, never inside math or numbers.
  distract - inserts one sentence with an irrelevant numeric fact in the organizers' style (a plausible aside that does
             not change any condition), drawn from a fixed list.
"""
from __future__ import annotations
import random, re

PERSON_NAMES = ("Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Mallory", "Oscar", "Peggy",
                "Trent", "Victor", "Walter", "Sybil", "Rupert", "Nadia", "Tomas", "Yuki", "Priya", "Kofi", "Lena", "Mateo", "Sana",
                "Omar", "Ines", "Hugo", "Mira", "Anya", "Leo", "Zara", "Ravi", "Nora", "Felix")
DISTRACTORS = (
    "(For reference, a related warm-up problem from the same set had the answer 36, which is not needed here.)",
    "As a side note, the problem was selected from a list of 120 candidates and rated 3.5 out of 5 for difficulty.",
    "(Incidentally, the same configuration appears in a textbook chapter spanning pages 214 to 231.)",
    "A minor remark: an earlier draft used the number 17 in a different role, but that version was discarded.",
    "(The statement is unchanged from the original; a diagram, if drawn to scale, would fit in a 10 cm by 10 cm box.)",
    "Note that a committee of 5 people checked this statement, each spending about 20 minutes on it.",
)
MATH_SPAN = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])", re.DOTALL)
LABEL_LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # skip I and O (ambiguous)
FAMILIES = ("rename", "typos", "distract")

def _split(text: str) -> list[str]:
    return MATH_SPAN.split(text)   # odd indices are math spans

def rename(text: str, rng: random.Random) -> tuple[str, str | None]:
    parts = _split(text)
    names = [w for i in range(0, len(parts), 2) for w in re.findall(r"\b[A-Z][a-z]{2,}\b", parts[i]) if w in PERSON_NAMES]
    note = []
    if names:
        pool = [n for n in PERSON_NAMES if n not in names]; rng.shuffle(pool)
        mapping = {n: pool.pop() for n in dict.fromkeys(names)}
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(r"\b[A-Z][a-z]{2,}\b", lambda m: mapping.get(m.group(0), m.group(0)), parts[i])
        note.append("names " + ",".join(f"{a}>{b}" for a, b in mapping.items()))
    # capital-letter label runs inside math spans (e.g. $ABC$, $AB = 20$, $\\angle BAD$), not after a backslash;
    # every letter of a run is a label, so $ABC$ -> three labels mapped individually and consistently
    RUN = r"(?<![A-Za-z\\])([A-Z]+)(?![a-z])"
    letters = sorted({ch for i in range(1, len(parts), 2) for run in re.findall(RUN, parts[i]) for ch in run} & set(LABEL_LETTERS))
    if letters and len(letters) <= 10:
        free = [ch for ch in LABEL_LETTERS if ch not in letters and ch not in "".join(parts)]
        rng.shuffle(free)
        if len(free) >= len(letters):
            lm = dict(zip(letters, free[: len(letters)]))
            sub = lambda m: "".join(lm.get(c, c) for c in m.group(1))
            for i in range(1, len(parts), 2):
                parts[i] = re.sub(RUN, sub, parts[i])
            CUES = r"(?:point|points|vertex|vertices|center|centre|line|lines|segment|segments|triangle|circle|angle|side|sides|midpoint|at|from|to|and|or|through|with|of|between)"
            for i in range(0, len(parts), 2):   # plain-text references: multi-letter runs, or single letters after a cue word (never the article "A")
                parts[i] = re.sub(r"(?<![A-Za-z\\$])([A-Z]{2,4})(?![A-Za-z])", lambda m: sub(m) if all(c in lm for c in m.group(1)) else m.group(1), parts[i])
                parts[i] = re.sub(r"(?i:\b" + CUES + r")\s+([A-Z])(?![A-Za-z])", lambda m: m.group(0)[: -1] + lm.get(m.group(1), m.group(1)), parts[i])
            note.append("labels " + ",".join(f"{a}>{b}" for a, b in lm.items()))
    out = "".join(parts)
    return (out, "; ".join(note)) if note else (text, None)

def typos(text: str, rng: random.Random, n_words: int = 3) -> tuple[str, str | None]:
    parts = _split(text); cands = []
    for i in range(0, len(parts), 2):
        for m in re.finditer(r"\b[a-z]{6,}\b", parts[i]):
            cands.append((i, m.start(), m.end()))
    if not cands:
        return text, None
    rng.shuffle(cands); chosen = sorted(cands[:n_words], reverse=True); changed = []
    for i, s, e in chosen:
        w = parts[i][s:e]
        if rng.random() < 0.5:
            k = rng.randrange(1, len(w) - 2); nw = w[:k] + w[k + 1] + w[k] + w[k + 2:]
        else:
            k = rng.randrange(2, len(w) - 2); nw = w[:k] + " " + w[k:]
        parts[i] = parts[i][:s] + nw + parts[i][e:]; changed.append(f"{w}>{nw}")
    return "".join(parts), "typos " + ",".join(changed)

def distract(text: str, rng: random.Random) -> tuple[str, str | None]:
    sentence = rng.choice(DISTRACTORS)
    # insert after the first sentence that ends outside a math span, else prepend
    parts = _split(text); idx = None
    for i in range(0, len(parts), 2):
        m = re.search(r"[.!?](\s+)(?=[A-Z$\\])", parts[i])
        if m: idx = (i, m.end(1)); break
    if idx is None:
        return f"{sentence} {text.strip()}", "distract prepended"
    i, pos = idx; parts[i] = parts[i][:pos] + sentence + " " + parts[i][pos:]
    return "".join(parts), "distract inserted after first sentence"

def make_variant(text: str, family: str, seed: int) -> tuple[str, str | None]:
    rng = random.Random(seed)
    return {"rename": rename, "typos": typos, "distract": distract}[family](text, rng)
