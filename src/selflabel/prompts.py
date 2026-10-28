"""Prompts used for self-labelling. SOLVE_SYSTEM is the organizers' solve prompt
(aimo-interp/baselines, prompts/solve.txt, Apache-2.0), reproduced verbatim."""

SOLVE_SYSTEM = (
    "You are a precise math problem solver. Solve the given problem step by step, then output your final answer on the last line in the exact format:\n"
    "ANSWER: <your answer>\n"
    "This result should be an integer value or NaN if the given problem does not have a solution."
)

# Forced-answer continuation when a sample hits the token cap (documented deviation from the
# organizers' 100k-token regime; both forced and unforced correctness are stored).
FORCE_SUFFIX = "\n</think>\n\nTime is up. Based on the reasoning so far, the final answer in the required format is:\nANSWER:"
FORCE_TOKENS = 16


def chat_messages(problem: str) -> list[dict]:
    return [{"role": "system", "content": SOLVE_SYSTEM}, {"role": "user", "content": problem}]
