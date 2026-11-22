# Timestamp of the revision pre-registration (disclosure, 2026-11-23)

`PREREG_REVISION_2026-11-22.md` (sha256 7e514c9987b1f606cf9e93156857f0745d94435f00744dbb2b680a12761bda7b) was committed in the
author's private repository as commit `868ffe7` on 2026-11-22 13:47 UTC. The copy of `commit_history.txt` in this repository's
first revision snapshot (commit b07e404) lists that commit; the current copy lists its re-authored version `0350aac`.
The revision analyses ran after it: the log of the first nested run (`analysis/wb_nested.py`) was created at 13:48 UTC. The logs of
the truncated-probe run, the label decomposition and the finished/truncated stage-1 analysis were created between 15:49 and
16:33 UTC on the same day.

On 2026-11-23 the two unpushed commits (the pre-registration and the revision) had to be re-authored with the GitHub no-reply
address, because GitHub rejects pushes that expose a private e-mail address. `git commit --amend --reset-author` also resets the
author date. As a result, the private repository on GitHub shows the pre-registration as commit `0350aac` dated 2026-11-22 18:42
UTC, which is after the runs. Two facts are unchanged:

- The content is identical. Both commits have the same tree (`6e6ac195837c6eb98ef186ef623f9fbef853a8b3`), and the
  pre-registration file has the same sha256 in both.
- The original commit object `868ffe7` is kept locally (tag `prereg-original`). It cannot be pushed, because its author e-mail
  is the private one.

Git dates are set by the author's machine, so neither date is independent proof. The pre-registration is a record of intent
written before the analyses, and every result it covers is reported in the paper whatever its sign. No already-pushed history
was rewritten.
