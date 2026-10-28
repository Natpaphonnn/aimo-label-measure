"""Parse a final answer from a model response under the organizers' solve prompt.

Priority: the LAST line of the form ``ANSWER: <x>`` after the end of thinking; fallback: the last
``\\boxed{...}``; fallback: a trailing "final answer is X". Returns an int, the string "NaN" for
no-solution claims, or None when nothing parsable is found (scored incorrect, as in the FAQ).
"""
from __future__ import annotations
import re

THINK_END = re.compile(r"</think>", re.I)
ANSWER_LINE = re.compile(r"ANSWER\s*[:：]\s*(.+)", re.I)
BOXED = re.compile(r"\\boxed\s*\{")
NAN_WORDS = re.compile(r"^(nan|none|no\s+solution|no\s+such|does\s+not\s+exist|undefined|impossible|not\s+possible|no\s+valid|no\s+answer)\b", re.I)
FINAL_IS = re.compile(r"final\s+answer\s+is[:\s]*\$?\\?(?:boxed\{)?\s*(-?[\d,]+)", re.I)

def _after_thinking(text: str) -> str:
    parts = THINK_END.split(text)
    return parts[-1] if len(parts) > 1 else text

def _last_boxed(text: str) -> str | None:
    starts = [m.end() for m in BOXED.finditer(text)]
    if not starts:
        return None
    s, depth = starts[-1], 1
    for i in range(s, len(text)):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: return text[s:i]
    return None

def normalize(raw: str) -> int | str | None:
    s = raw.strip().strip("*").strip()
    s = s.split("\n")[0].strip()
    s = re.sub(r"^\$+|\$+$", "", s).strip()
    if NAN_WORDS.match(s):
        return "NaN"
    b = _last_boxed(s) if "\\boxed" in s else None
    if b is not None:
        s = b
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[,!; ]|\^\\circ|\\%|%|\\\$|\$|\s", "", s)
    s = s.replace("{", "").replace("}", "").replace(",", "")
    m = re.fullmatch(r"\\?d?frac(-?\d+)(\d+)", s)
    s = re.sub(r"\\(?:d)?frac\(?(-?\d+)\)?\(?1\)?$", r"\1", s)
    if NAN_WORDS.match(s):
        return "NaN"
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d+(?:\.0+)?", s):
        return int(float(s))
    return None

def parse_answer(response: str) -> int | str | None:
    tail = _after_thinking(response)
    lines = ANSWER_LINE.findall(tail)
    if lines:
        v = normalize(lines[-1])
        if v is not None:
            return v
    b = _last_boxed(tail)
    if b is not None:
        v = normalize(b)
        if v is not None:
            return v
    m = FINAL_IS.findall(tail)
    if m:
        return normalize(m[-1])
    # Boxed answers inside thinking never count (FAQ: no parsable final answer = incorrect).
    if THINK_END.search(response):
        return None
    b = _last_boxed(response)
    return normalize(b) if b is not None else None

def is_correct(pred: int | str | None, gold: int | str | None) -> bool | None:
    if gold is None:   # statements without a published answer (train-main-v2): correctness unknown
        return None
    if pred is None:
        return False
    if isinstance(gold, str) and gold == "NaN":
        return pred == "NaN"
    return isinstance(pred, int) and pred == int(gold)
