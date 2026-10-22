#!/usr/bin/env bash
# FV study launcher (Millennium Harness FV v1.0, docs/research/win-plan-synthesis.md). Usage:
#   bench/run_fv.sh <tasks_dir> <tag> [oc|cc|both]        (FV_DRY_RUN=1 prints the harbor commands instead)
# oc:   OpenCode bench.agents_fv:FVBranch (A, B and C from one trajectory per task)
# cc:   Claude Code bench.agents_fv:ClaudeCodeFV (same-trajectory A/B)
# both: both, launched concurrently.
# Guard: bench/spend.py runs first. A non-zero exit, or a printed total at or over its cap, refuses the launch.
# Tasks come from bench/adapters/vericoding_fv.py (non-root agent, API-only egress, harnesses baked into the image).
set -euo pipefail
cd "$(dirname "$0")/.."
TASKS=$(cd "${1:?tasks_dir}" && pwd); TAG=${2:?tag}; WHICH=${3:-oc}; N=${FV_CONCURRENCY:-16}
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
NT=$(find "$TASKS" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
echo "planned: $NT tasks from $TASKS, arms: $WHICH"

SPEND=$($PY bench/spend.py --study fv --modal-cap "${FV_MODAL_CAP:-20}") || {
  echo "$SPEND"; echo "refused: bench/spend.py exited non-zero"; exit 1; }
echo "$SPEND"
if echo "$SPEND" | grep -Eo '\([0-9.]+%\)' | tr -d '()%' | awk '$1 >= 100 {over=1} END {exit !over}'; then
  echo "refused: spend at or over the cap"; exit 1
fi

. scripts/env.sh
unset OPENROUTER_API_KEY ANTHROPIC_API_KEY_2
OCFG=$(python3 -c "import json;print(json.dumps(json.load(open('bench/jobs/opencode_cfg.json'))))")
ENVF=(-e modal --ek app_name=millennium-harness-harbor --ek sandbox_timeout_secs=6000 --ek sandbox_idle_timeout_secs=600
      --cpus limit -n "$N" -r 2 -o runs/jobs -y -q)
FV=(--ak max_rounds=2 --ak "tasks_dir=$TASKS")
OC=(-a bench.agents_fv:FVBranch -m anthropic/claude-haiku-5-5 --ak version=1.18.35 --ak variant=medium
    --ak disable_web_search=true --ak "opencode_config=$OCFG" "${FV[@]}" --ae OPENCODE_DISABLE_MODELS_FETCH=1)
CC=(-a bench.agents_fv:ClaudeCodeFV -m claude-haiku-5-5 --ak version=2.1.296 --ak reasoning_effort=medium
    --ak disable_web_search=true "${FV[@]}"
    --ae CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000 --ae CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000
    --ae ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-5-5 --ae CLAUDE_CODE_SUBAGENT_MODEL=claude-haiku-5-5
    --ae DISABLE_AUTOUPDATER=1)
export PYTHONPATH=.

launch() {  # job name, harbor args...
  local name=$1; shift
  if [ -n "${FV_DRY_RUN:-}" ]; then echo "harbor run $* --job-name $name"; return; fi
  harbor run "$@" --job-name "$name" > "runs/$name.log" 2>&1 &
}
[ "$WHICH" = cc ] || launch "$TAG-oc" -p "$TASKS" "${OC[@]}" "${ENVF[@]}"
[ "$WHICH" = oc ] || launch "$TAG-cc" -p "$TASKS" "${CC[@]}" "${ENVF[@]}"
wait
echo "done: $TAG ($WHICH)"
