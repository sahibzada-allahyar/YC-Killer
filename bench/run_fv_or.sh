#!/usr/bin/env bash
# FV probe with a non-Anthropic model via OpenRouter (cross-model). Usage: bench/run_fv_or.sh <tasks_dir> <tag> <openrouter_model>
# Tasks must allow-list openrouter.ai (sed the task.toml); same FVBranch harness, same caps.
set -euo pipefail
cd "$(dirname "$0")/.."
TASKS=$(cd "${1:?tasks}" && pwd); TAG=${2:?tag}; MODEL=${3:?model}; N=${FV_CONCURRENCY:-16}
. scripts/env.sh
unset ANTHROPIC_API_KEY ANTHROPIC_API_KEY_2
MID=${MODEL#openrouter/}
OCFG=$(python3 -c "import json,sys;print(json.dumps({'small_model':'openrouter/'+sys.argv[1],'share':'disabled','autoupdate':False,'provider':{'openrouter':{'models':{sys.argv[1]:{'limit':{'context':1000000,'input':100000,'output':64000}}}}}}))" "$MID")
export PYTHONPATH=.
harbor run -p "$TASKS" -a bench.agents_fv:FVBranch -m "openrouter/$MID" --ak version=1.18.35 --ak disable_web_search=true \
  --ak "opencode_config=$OCFG" --ak max_rounds=2 --ak "tasks_dir=$TASKS" --ae OPENCODE_DISABLE_MODELS_FETCH=1 \
  --ae "OPENROUTER_API_KEY=$OPENROUTER_API_KEY" \
  -e modal --ek app_name=millennium-harness-harbor --ek sandbox_timeout_secs=6000 --ek sandbox_idle_timeout_secs=600 \
  --cpus limit -n "$N" -r 2 -o runs/jobs -y -q --job-name "$TAG" > "runs/$TAG.log" 2>&1
echo "done: $TAG"
