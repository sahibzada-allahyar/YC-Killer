# Source me: loads secrets from the research-projects .env without echoing them.
# Maps ANTHROPIC_API_KEY_2 -> ANTHROPIC_API_KEY. Never commit secrets.
_SL_ENV="${MILLENNIUM_ENV_FILE:-${SORRYLESS_ENV_FILE:-./.env}}"
set -a; . "$_SL_ENV"; set +a
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$ANTHROPIC_API_KEY_2}"
unset MODAL_API_KEY
export MODAL_ENVIRONMENT=research
