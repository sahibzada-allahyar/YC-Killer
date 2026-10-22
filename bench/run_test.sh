#!/usr/bin/env bash
# Preregistered FreshCP-TEST runs (executed unchanged after the freeze). Usage:
#   bench/run_test.sh <tasks_dir> <suite_dir> <tag> [kA kB kC]
# Arms run concurrently (interleaved in time); seeds = Harbor attempts (-k). Never edit after the freeze tag.
set -euo pipefail
cd "$(dirname "$0")/.."
TASKS=$1; SUITES=$2; TAG=$3; KA=${4:-3}; KB=${5:-3}; KC=${6:-2}
. scripts/env.sh
unset OPENROUTER_API_KEY ANTHROPIC_API_KEY_2
OC=$(python3 -c "import json;print(json.dumps(json.load(open('bench/jobs/opencode_cfg.json'))))")
ENVF=(-e modal --ek app_name=millennium-harness-harbor --ek sandbox_timeout_secs=3600 --ek sandbox_idle_timeout_secs=600
      --cpus limit --agent-timeout-multiplier 2.5 -n 24 -r 2 -o runs/jobs -y -q)
OCA=(-a bench.agents:OpenCodeGated -m anthropic/claude-haiku-5-5 --ak version=1.18.35 --ak variant=medium
     --ak disable_web_search=true --ak "opencode_config=$OC" --ak max_rounds=2 --ak "suite_dir=$PWD/$SUITES")
export PYTHONPATH=.
harbor run -p "$TASKS" "${OCA[@]}" --ak gate=none       -k "$KA" "${ENVF[@]}" --job-name "$TAG-A" > "runs/$TAG-A.log" 2>&1 &
harbor run -p "$TASKS" "${OCA[@]}" --ak gate=millennium -k "$KB" "${ENVF[@]}" --job-name "$TAG-B" > "runs/$TAG-B.log" 2>&1 &
harbor run -p "$TASKS" "${OCA[@]}" --ak gate=selfstress -k "$KC" "${ENVF[@]}" --job-name "$TAG-C" > "runs/$TAG-C.log" 2>&1 &
wait
echo "done: $TAG A/B/C"
