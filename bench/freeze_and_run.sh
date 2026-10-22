#!/usr/bin/env bash
# Freeze → build sealed FreshCP-TEST → run preregistered experiments. Never prints statements/tests.
#   bench/freeze_and_run.sh <freeze-date YYYY-MM-DD>
set -euo pipefail
cd "$(dirname "$0")/.."
FREEZE=$1
. scripts/env.sh
git diff --quiet && git diff --cached --quiet || { echo "commit everything before freezing"; exit 1; }
SHA=$(shasum -a 256 bench/PREREG.md | cut -d' ' -f1)
git tag -a v0.9.0-frozen -m "preregistration frozen; PREREG.md sha256=$SHA; FreshCP-TEST window 2026-07-01..$FREEZE"
echo "frozen: PREREG sha256=$SHA at $(git rev-parse HEAD)"
# 1) sealed test set (Modal) — only aggregate stats come back
FRESHCP_FREEZE=$FREEZE .venv/bin/python bench/freshcp/pipeline.py scrape TEST
FRESHCP_FREEZE=$FREEZE .venv/bin/python bench/freshcp/pipeline.py build TEST
.venv/bin/python bench/freshcp/pipeline.py stats TEST
mkdir -p sealed/test && (cd sealed/test && modal volume get sorryless-sealed /TEST/tasks . --force >/dev/null)
# 2) sealed suites (specs see only statement + public examples)
.venv/bin/python bench/suites.py submit sealed/test/tasks test
.venv/bin/python bench/suites.py collect test
echo "TEST ready: $(ls sealed/test/tasks | wc -l) tasks"
