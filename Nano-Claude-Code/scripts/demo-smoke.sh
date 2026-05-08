#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -n "${NANO_CLAUDE_CODE_BIN:-}" ]]; then
  CLI_CMD=("$NANO_CLAUDE_CODE_BIN")
else
  CLI_CMD=(cargo run --quiet --)
fi

run_cli() {
  "${CLI_CMD[@]}" "$@"
}

json_path() {
  local payload="$1"
  local path="$2"
  PAYLOAD="$payload" PATH_EXPR="$path" python3 - <<'PY'
import json
import os

value = json.loads(os.environ["PAYLOAD"])
for part in os.environ["PATH_EXPR"].split("."):
    value = value[part]
print(value)
PY
}

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

export CLAUDE_CONFIG_DIR="$tmpdir/config"
export NANO_CLAUDE_FIXED_TIME_MS=1700000000123

workspace="$tmpdir/workspace"
mkdir -p "$workspace"

if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  live_api_key="$ANTHROPIC_API_KEY"
  setup_json="$(printf '%s\n' "$live_api_key" | run_cli setup --json)"
  unset ANTHROPIC_API_KEY

  git -C "$workspace" init -q
  git -C "$workspace" config user.email smoke@example.com
  git -C "$workspace" config user.name smoke

  cat >"$workspace/app.txt" <<'EOF'
status=old
EOF
  cat >"$workspace/NOTES.md" <<'EOF'
TODO
EOF
  cat >"$workspace/verify.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
grep -qx 'status=new' app.txt
EOF
  chmod +x "$workspace/verify.sh"
  git -C "$workspace" add app.txt NOTES.md verify.sh
  git -C "$workspace" commit -qm "init"

  first_prompt=$'Inspect this repo. Use the edit tool to change app.txt from status=old to status=new. Then run ./verify.sh with bash. Then spawn a general-purpose agent with prompt "Summarize verify.sh in one sentence." Wait for it. Final reply: short.'
  resume_prompt=$'Resume this session. Use the edit tool to change NOTES.md from TODO to RESUMED. Final reply: include the exact line from app.txt.'

  chat_json="$(printf '%s\n' "$first_prompt" | run_cli chat --json --cwd "$workspace" --shell /bin/sh)"
  session_id="$(json_path "$chat_json" sessionId)"
  resume_json="$(printf '%s\n' "$resume_prompt" | run_cli chat --json --resume "$session_id" --shell /bin/sh)"
  transcript_json="$(run_cli transcript --json "$session_id")"

  CHAT_JSON="$chat_json" \
  RESUME_JSON="$resume_json" \
  SETUP_JSON="$setup_json" \
  TRANSCRIPT_JSON="$transcript_json" \
  WORKSPACE="$workspace" \
  CONFIG_DIR="$CLAUDE_CONFIG_DIR" \
  FIRST_PROMPT="$first_prompt" \
  RESUME_PROMPT="$resume_prompt" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

setup = json.loads(os.environ["SETUP_JSON"])
assert setup["configured"] is True, setup
assert setup["saved"] is True, setup
assert setup["source"] == "config", setup

chat = json.loads(os.environ["CHAT_JSON"])
assert chat["turns"] == 1, chat
assert len(chat["assistantTexts"]) == 1, chat
assert chat["assistantTexts"][0].strip(), chat

resume = json.loads(os.environ["RESUME_JSON"])
assert resume["turns"] == 1, resume
assert len(resume["assistantTexts"]) == 1, resume
assert "status=new" in resume["assistantTexts"][0], resume

workspace = Path(os.environ["WORKSPACE"])
assert workspace.joinpath("app.txt").read_text() == "status=new\n"
assert workspace.joinpath("NOTES.md").read_text() == "RESUMED\n"

agents_dir = Path(os.environ["CONFIG_DIR"]) / "agents"
agent_files = sorted(agents_dir.glob("*.json"))
assert agent_files, list(agents_dir.glob("*"))
agent_record = json.loads(agent_files[-1].read_text())
assert agent_record["status"] == "completed", agent_record
assert agent_record["response"].strip(), agent_record

transcript = json.loads(os.environ["TRANSCRIPT_JSON"])
assert "Inspect this repo." in transcript["transcript"], transcript
assert "Resume this session." in transcript["transcript"], transcript
PY

  echo "demo-smoke: ok live session=$session_id"
  exit 0
fi

chat_json="$(printf 'hello\n/echo smoke\n' | run_cli chat --json --cwd "$workspace" --shell /bin/sh)"
session_id="$(json_path "$chat_json" sessionId)"

resume_json="$(printf '/tools\n' | run_cli chat --json --resume "$session_id" --shell /bin/sh)"
route_json="$(run_cli mcp route --json mcp__bash__exec)"
bash_json="$(run_cli mcp call --json --cwd "$workspace" --shell /bin/sh mcp__bash__exec --input '{"command":"printf routed"}')"
edit_json="$(run_cli mcp call --json --cwd "$workspace" mcp__file__edit --input '{"filePath":"demo.txt","oldString":"","newString":"draft\n","replaceAll":false}')"
agent_json="$(run_cli mcp call --json --cwd "$workspace" mcp__agent__spawn --input '{"agentType":"general-purpose","prompt":"smoke agent"}')"
agent_id="$(json_path "$agent_json" result.record.agentId)"
agent_wait_json="$(run_cli agent wait --json "$agent_id")"
session_info_json="$(run_cli mcp call --json --cwd "$workspace" mcp__session__info --input "{\"sessionId\":\"$session_id\"}")"
transcript_json="$(run_cli mcp call --json --cwd "$workspace" mcp__session__transcript --input "{\"sessionId\":\"$session_id\"}")"

CHAT_JSON="$chat_json" \
RESUME_JSON="$resume_json" \
ROUTE_JSON="$route_json" \
BASH_JSON="$bash_json" \
EDIT_JSON="$edit_json" \
AGENT_JSON="$agent_json" \
AGENT_WAIT_JSON="$agent_wait_json" \
SESSION_INFO_JSON="$session_info_json" \
TRANSCRIPT_JSON="$transcript_json" \
WORKSPACE="$workspace" \
python3 - <<'PY'
import json
import os
from pathlib import Path

chat = json.loads(os.environ["CHAT_JSON"])
assert chat["assistantTexts"] == ["mock:hello", "smoke"], chat

resume = json.loads(os.environ["RESUME_JSON"])
assert resume["assistantTexts"] == ["bash\nedit\necho"], resume

route = json.loads(os.environ["ROUTE_JSON"])
assert route["route"] == "bash.exec", route

bash = json.loads(os.environ["BASH_JSON"])
assert bash["result"]["kind"] == "bashExec", bash
assert bash["result"]["report"]["stdout"] == "routed", bash

edit = json.loads(os.environ["EDIT_JSON"])
assert edit["result"]["kind"] == "fileEdit", edit
assert Path(os.environ["WORKSPACE"], "demo.txt").read_text() == "draft\n"

agent = json.loads(os.environ["AGENT_JSON"])
assert agent["result"]["kind"] == "agentSpawn", agent
assert agent["result"]["record"]["status"] == "completed", agent

agent_wait = json.loads(os.environ["AGENT_WAIT_JSON"])
assert agent_wait["status"] == "completed", agent_wait

session_info = json.loads(os.environ["SESSION_INFO_JSON"])
assert session_info["result"]["kind"] == "sessionInfo", session_info
assert session_info["result"]["info"]["sessionId"] == chat["sessionId"], session_info

transcript = json.loads(os.environ["TRANSCRIPT_JSON"])
assert transcript["result"]["kind"] == "sessionTranscript", transcript
assert "hello" in transcript["result"]["scan"]["transcript"], transcript
PY

echo "demo-smoke: ok session=$session_id agent=$agent_id"
