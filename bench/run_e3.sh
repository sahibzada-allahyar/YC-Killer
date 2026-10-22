#!/usr/bin/env bash
# Preregistered cross-harness reference rows (E3), executed unchanged after the freeze. Usage:
#   bench/run_e3.sh <tasks_dir> <suite_dir> <tag>
# Same model (Haiku 5.5, effort medium), ~100K input cap, web off, 1 run each, launched concurrently.
set -euo pipefail
cd "$(dirname "$0")/.."
TASKS=$1; SUITES=$2; TAG=$3
. scripts/env.sh
ENVF=(-e modal --ek app_name=millennium-harness-harbor --ek sandbox_timeout_secs=3600 --ek sandbox_idle_timeout_secs=600
      --cpus limit --agent-timeout-multiplier 2.5 -n 16 -r 2 -o runs/jobs -y -q)
export PYTHONPATH=.
CC=(-a bench.agents:ClaudeCodeGated -m claude-haiku-5-5 --ak version=2.1.296 --ak reasoning_effort=medium
    --ak "disallowed_tools=WebSearch,WebFetch" --ak max_rounds=2 --ak "suite_dir=$PWD/$SUITES"
    --ae CLAUDE_CODE_AUTO_COMPACT_WINDOW=100000 --ae CLAUDE_CODE_MAX_OUTPUT_TOKENS=64000
    --ae ANTHROPIC_DEFAULT_HAIKU_MODEL=claude-haiku-5-5 --ae CLAUDE_CODE_SUBAGENT_MODEL=claude-haiku-5-5)
OCL='{"models":{"providers":{"anthropic":{"models":[{"id":"claude-haiku-5-5","name":"Claude Haiku 5.5","contextTokens":100000,"maxTokens":64000}]}}}}'
(
  unset OPENROUTER_API_KEY ANTHROPIC_API_KEY_2
  harbor run -p "$TASKS" "${CC[@]}" --ak gate=none      "${ENVF[@]}" --job-name "$TAG-cc"    > "runs/$TAG-cc.log" 2>&1 &
  harbor run -p "$TASKS" "${CC[@]}" --ak gate=millennium "${ENVF[@]}" --job-name "$TAG-cc-sl" > "runs/$TAG-cc-sl.log" 2>&1 &
  harbor run -p "$TASKS" -a bench.agents:HermesCapped -m anthropic/claude-haiku-5-5 --ak toolsets=terminal,file \
    "${ENVF[@]}" --job-name "$TAG-hermes" > "runs/$TAG-hermes.log" 2>&1 &
  harbor run -p "$TASKS" -a bench.agents:OpenClaw24 -m anthropic/claude-haiku-5-5 --ak version=2026.9.9 \
    --ak thinking=medium --ak disable_web_search=true --ak "openclaw_config=$OCL" "${ENVF[@]}" \
    --job-name "$TAG-openclaw" > "runs/$TAG-openclaw.log" 2>&1 &
  wait
) &
(
  export OPENAI_API_KEY="$OPENROUTER_API_KEY" OPENAI_BASE_URL=https://openrouter.ai/api/v1
  unset OPENROUTER_API_KEY ANTHROPIC_API_KEY ANTHROPIC_API_KEY_2
  harbor run -p "$TASKS" -a bench.agents:CodexBridged -m openai/anthropic/claude-haiku-5.5 --ak version=0.162.1 \
    --ak reasoning_effort=medium --ak disable_web_search=true \
    --ak 'config={"model_context_window":100000,"model_auto_compact_token_limit":85000,"model_max_output_tokens":64000}' \
    "${ENVF[@]}" --job-name "$TAG-codex" > "runs/$TAG-codex.log" 2>&1
) &
wait
echo "done: $TAG E3"
