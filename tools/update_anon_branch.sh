#!/usr/bin/env bash
# Anonymized mirror: branch `anon-review` holds the unzipped supplementary ZIP (one commit per update, author "Anonymous",
# UTC dates), for anonymous.4open.science. The working tree and the current branch are not touched.
# Usage: python tools/build_supplementary.py && bash tools/update_anon_branch.sh && git push origin anon-review
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
unzip -q dist/openreview-supplementary-code.zip -d "$TMP"
export GIT_INDEX_FILE="$TMP/index" TZ=UTC
export GIT_AUTHOR_NAME=Anonymous GIT_AUTHOR_EMAIL=anonymous@example.invalid
export GIT_COMMITTER_NAME=Anonymous GIT_COMMITTER_EMAIL=anonymous@example.invalid
git --work-tree="$TMP/aimo-interp-code" add -A
TREE=$(git write-tree)
PARENT=$(git rev-parse -q --verify refs/heads/anon-review || true)
if [ -n "$PARENT" ] && [ "$(git rev-parse "$PARENT^{tree}")" = "$TREE" ]; then echo "anon-review unchanged"; exit 0; fi
COMMIT=$(git commit-tree "$TREE" ${PARENT:+-p "$PARENT"} -m "Anonymized code for review")
git update-ref refs/heads/anon-review "$COMMIT"
echo "anon-review -> $(git rev-parse --short "$COMMIT") ($(git ls-tree -r --name-only "$COMMIT" | wc -l | tr -d ' ') files)"
