use std::collections::{BTreeMap, VecDeque};
use std::env;
use std::path::Path;

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::anthropic::RequestMessage;
use crate::config::load_anthropic_api_key;
use crate::provider::{LiveCompletionInput, LiveToolUse, execute_live_completion};
use crate::session::now_ms;
use crate::tools::{
    BuiltInToolExecutor, MainLoopToolExecutor, ScriptedToolExecutor, ScriptedToolResult, ToolCall,
    ToolExecutor, ToolResult, ToolUse, main_loop_tool_definitions, maybe_extract_bash_command,
    maybe_extract_edit_command, maybe_run_echo, parse_inline_edit_command,
};

pub const DEFAULT_MODEL: &str = "mock-claude";
pub const LIVE_DEFAULT_MODEL: &str = "claude-sonnet-4-6";

const LIVE_MAX_OUTPUT_TOKENS: u64 = 1_024;
const LIVE_MAX_TURNS: usize = 8;
const USER_AGENT: &str = "nano-claude-code/0.1.0";
const MAX_OUTPUT_TOKENS_ENV: &str = "CLAUDE_CODE_MAX_OUTPUT_TOKENS";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ConversationMessage {
    pub role: MessageRole,
    pub content: String,
}

pub struct RunOptions<'a> {
    pub prompt: &'a str,
    pub history: &'a [ConversationMessage],
    pub api_history: Option<&'a [RequestMessage]>,
    pub cwd: &'a Path,
    pub shell: &'a str,
    pub env_vars: &'a BTreeMap<String, String>,
    pub model: &'a str,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RunOutcome {
    pub assistant_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<ToolUse>,
    #[serde(skip_serializing)]
    pub serialized_outcome: SerializedRunOutcome,
    #[serde(skip_serializing)]
    pub api_messages: Option<Vec<RequestMessage>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SerializedRunOutcome {
    pub assistant_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<ToolUse>,
    pub stop_reason: StopReason,
    pub turn_count: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub messages: Vec<LoopMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Completed,
    MaxTurns,
    ModelError,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct LoopMessage {
    pub role: MessageRole,
    pub content: Vec<LoopContent>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub is_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    Assistant,
    User,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LoopContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: String,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default, skip_serializing_if = "is_false")]
        is_error: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AssistantAction {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub assistant_text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_uses: Vec<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub retry_error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fatal_error: Option<String>,
}

impl AssistantAction {
    fn validate(&self) -> Result<()> {
        if self.retry_error.is_some() && self.fatal_error.is_some() {
            bail!("assistant action cannot set both retryError and fatalError");
        }
        Ok(())
    }

    fn to_message(&self) -> Option<LoopMessage> {
        let mut content = Vec::new();
        if !self.assistant_text.is_empty() {
            content.push(LoopContent::Text {
                text: self.assistant_text.clone(),
            });
        }
        content.extend(self.tool_uses.iter().map(|tool_use| LoopContent::ToolUse {
            id: tool_use.id.clone(),
            name: tool_use.tool.clone(),
            input: tool_use.input.clone(),
        }));

        if content.is_empty() {
            return None;
        }

        Some(LoopMessage {
            role: MessageRole::Assistant,
            content,
            is_error: false,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ScriptedRunRequest {
    #[serde(default)]
    pub max_turns: Option<usize>,
    pub actions: Vec<AssistantAction>,
    #[serde(default)]
    pub tool_results: Vec<ScriptedToolResult>,
}

pub struct LoopState<'a> {
    pub turn_count: usize,
    pub last_tool_results: &'a [ToolResult],
}

pub trait AssistantPlanner {
    fn next_action(&mut self, state: &LoopState<'_>) -> Result<AssistantAction>;

    fn api_messages(&self) -> Option<Vec<RequestMessage>> {
        None
    }
}

struct LoopExecutionResult {
    serialized_outcome: SerializedRunOutcome,
    api_messages: Option<Vec<RequestMessage>>,
}

pub fn effective_model(requested_model: &str) -> String {
    if has_live_api_key() {
        return resolve_requested_model(requested_model);
    }

    let trimmed = requested_model.trim();
    if trimmed.is_empty() {
        DEFAULT_MODEL.to_string()
    } else {
        trimmed.to_string()
    }
}

pub fn has_live_api_key() -> bool {
    anthropic_api_key().is_some()
}

pub fn resolve_requested_model(requested_model: &str) -> String {
    let trimmed = requested_model.trim();
    if !trimmed.is_empty() && trimmed != DEFAULT_MODEL {
        return trimmed.to_string();
    }

    env::var("ANTHROPIC_MODEL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| LIVE_DEFAULT_MODEL.to_string())
}

pub fn run(
    prompt: &str,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<RunOutcome> {
    run_with_options(RunOptions {
        prompt,
        history: &[],
        api_history: None,
        cwd,
        shell,
        env_vars,
        model: DEFAULT_MODEL,
    })
}

pub fn run_with_options(options: RunOptions<'_>) -> Result<RunOutcome> {
    let execution = run_serialized_with_options_internal(options)?;
    Ok(RunOutcome {
        assistant_text: execution.serialized_outcome.assistant_text.clone(),
        tool_use: execution.serialized_outcome.tool_use.clone(),
        serialized_outcome: execution.serialized_outcome,
        api_messages: execution.api_messages,
    })
}

pub fn run_with_history(
    prompt: &str,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
    requested_model: &str,
    history: &[ConversationMessage],
) -> Result<RunOutcome> {
    run_with_options(RunOptions {
        prompt,
        history,
        api_history: None,
        cwd,
        shell,
        env_vars,
        model: requested_model,
    })
}

pub fn run_serialized(
    prompt: &str,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<SerializedRunOutcome> {
    run_serialized_with_options(RunOptions {
        prompt,
        history: &[],
        api_history: None,
        cwd,
        shell,
        env_vars,
        model: DEFAULT_MODEL,
    })
}

pub fn run_serialized_with_options(options: RunOptions<'_>) -> Result<SerializedRunOutcome> {
    Ok(run_serialized_with_options_internal(options)?.serialized_outcome)
}

fn run_serialized_with_options_internal(options: RunOptions<'_>) -> Result<LoopExecutionResult> {
    run_serialized_with_history_internal(
        options.prompt,
        options.cwd,
        options.shell,
        options.env_vars,
        options.model,
        options.history,
        options.api_history,
    )
}

pub fn run_serialized_with_history(
    prompt: &str,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
    requested_model: &str,
    history: &[ConversationMessage],
) -> Result<SerializedRunOutcome> {
    Ok(run_serialized_with_history_internal(
        prompt,
        cwd,
        shell,
        env_vars,
        requested_model,
        history,
        None,
    )?
    .serialized_outcome)
}

fn run_serialized_with_history_internal(
    prompt: &str,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
    requested_model: &str,
    history: &[ConversationMessage],
    api_history: Option<&[RequestMessage]>,
) -> Result<LoopExecutionResult> {
    if should_use_prompt_planner(prompt) {
        let mut planner = PromptPlanner::new(prompt)?;
        let mut tools = BuiltInToolExecutor::new(cwd, shell, env_vars);
        return execute_tool_loop_internal(&mut planner, &mut tools, None);
    }

    let resolved_model = effective_model(requested_model);
    if resolved_model == DEFAULT_MODEL {
        let mut planner = PromptPlanner::new(prompt)?;
        let mut tools = BuiltInToolExecutor::new(cwd, shell, env_vars);
        return execute_tool_loop_internal(&mut planner, &mut tools, None);
    }

    let mut planner =
        LiveAnthropicPlanner::new(prompt, history, api_history, &resolved_model, cwd, shell);
    let mut tools = MainLoopToolExecutor::with_model(cwd, shell, env_vars, &resolved_model);
    execute_tool_loop_internal(&mut planner, &mut tools, Some(LIVE_MAX_TURNS))
}

pub fn execute_scripted_run(request: ScriptedRunRequest) -> Result<SerializedRunOutcome> {
    let mut planner = ScriptedAssistantPlanner::new(request.actions);
    let mut tools = ScriptedToolExecutor::new(request.tool_results);
    execute_tool_loop(&mut planner, &mut tools, request.max_turns)
}

pub fn execute_tool_loop<P: AssistantPlanner, T: ToolExecutor>(
    planner: &mut P,
    tools: &mut T,
    max_turns: Option<usize>,
) -> Result<SerializedRunOutcome> {
    Ok(execute_tool_loop_internal(planner, tools, max_turns)?.serialized_outcome)
}

fn execute_tool_loop_internal<P: AssistantPlanner, T: ToolExecutor>(
    planner: &mut P,
    tools: &mut T,
    max_turns: Option<usize>,
) -> Result<LoopExecutionResult> {
    let mut messages = Vec::new();
    let mut last_tool_results = Vec::new();
    let mut turn_count = 0_usize;
    let mut assistant_text = String::new();
    let mut first_tool_use = None;

    loop {
        let state = LoopState {
            turn_count,
            last_tool_results: &last_tool_results,
        };
        let action = planner.next_action(&state)?;
        action.validate()?;

        if let Some(tool_use) = action.tool_uses.first()
            && first_tool_use.is_none()
        {
            first_tool_use = Some(tool_use.summary());
        }

        if let Some(message) = action.to_message() {
            if !message.is_error && !action.assistant_text.is_empty() {
                assistant_text = action.assistant_text.clone();
            }
            messages.push(message);
        }

        if let Some(error) = action.retry_error.as_deref() {
            append_missing_tool_results(&mut messages, &action.tool_uses, error);
            last_tool_results.clear();
            continue;
        }

        if let Some(error) = action.fatal_error.as_deref() {
            append_missing_tool_results(&mut messages, &action.tool_uses, error);
            messages.push(assistant_error_message(error));
            return Ok(LoopExecutionResult {
                serialized_outcome: SerializedRunOutcome {
                    assistant_text: if assistant_text.is_empty() {
                        error.to_string()
                    } else {
                        assistant_text
                    },
                    tool_use: first_tool_use,
                    stop_reason: StopReason::ModelError,
                    turn_count,
                    error: Some(error.to_string()),
                    messages,
                },
                api_messages: planner.api_messages(),
            });
        }

        if action.tool_uses.is_empty() {
            return Ok(LoopExecutionResult {
                serialized_outcome: SerializedRunOutcome {
                    assistant_text,
                    tool_use: first_tool_use,
                    stop_reason: StopReason::Completed,
                    turn_count,
                    error: None,
                    messages,
                },
                api_messages: planner.api_messages(),
            });
        }

        let mut executed_results = Vec::new();
        for tool_use in &action.tool_uses {
            let result = tools.execute(tool_use)?;
            messages.push(tool_result_message(&result));
            executed_results.push(result);
        }
        last_tool_results = executed_results;

        let next_turn_count = turn_count + 1;
        if let Some(limit) = max_turns
            && next_turn_count > limit
        {
            return Ok(LoopExecutionResult {
                serialized_outcome: SerializedRunOutcome {
                    assistant_text,
                    tool_use: first_tool_use,
                    stop_reason: StopReason::MaxTurns,
                    turn_count: next_turn_count,
                    error: None,
                    messages,
                },
                api_messages: planner.api_messages(),
            });
        }

        turn_count = next_turn_count;
    }
}

struct ScriptedAssistantPlanner {
    actions: VecDeque<AssistantAction>,
}

impl ScriptedAssistantPlanner {
    fn new(actions: Vec<AssistantAction>) -> Self {
        Self {
            actions: actions.into(),
        }
    }
}

impl AssistantPlanner for ScriptedAssistantPlanner {
    fn next_action(&mut self, _state: &LoopState<'_>) -> Result<AssistantAction> {
        self.actions
            .pop_front()
            .ok_or_else(|| anyhow!("assistant script exhausted before loop completed"))
    }
}

enum PromptMode {
    Direct { assistant_text: String },
    Tool { tool_call: ToolCall },
}

struct PromptPlanner {
    mode: PromptMode,
    stage: usize,
}

impl PromptPlanner {
    fn new(prompt: &str) -> Result<Self> {
        if let Some((tool_use, _assistant_text)) = maybe_run_echo(prompt) {
            return Ok(Self {
                mode: PromptMode::Tool {
                    tool_call: ToolCall {
                        id: "toolu_1".to_string(),
                        tool: tool_use.tool,
                        input: tool_use.input,
                    },
                },
                stage: 0,
            });
        }

        if let Some(command) = maybe_extract_bash_command(prompt) {
            return Ok(Self {
                mode: PromptMode::Tool {
                    tool_call: ToolCall {
                        id: "toolu_1".to_string(),
                        tool: "bash".to_string(),
                        input: command.to_string(),
                    },
                },
                stage: 0,
            });
        }

        if let Some(command) = maybe_extract_edit_command(prompt) {
            let _parsed = parse_inline_edit_command(command)?;
            return Ok(Self {
                mode: PromptMode::Tool {
                    tool_call: ToolCall {
                        id: "toolu_1".to_string(),
                        tool: "edit".to_string(),
                        input: command.to_string(),
                    },
                },
                stage: 0,
            });
        }

        let assistant_text = if prompt == "/tools" {
            "bash\nedit\necho".to_string()
        } else {
            format!("mock:{prompt}")
        };

        Ok(Self {
            mode: PromptMode::Direct { assistant_text },
            stage: 0,
        })
    }
}

impl AssistantPlanner for PromptPlanner {
    fn next_action(&mut self, state: &LoopState<'_>) -> Result<AssistantAction> {
        match (&self.mode, self.stage) {
            (PromptMode::Direct { assistant_text }, 0) => {
                self.stage = 1;
                Ok(AssistantAction {
                    assistant_text: assistant_text.clone(),
                    tool_uses: Vec::new(),
                    retry_error: None,
                    fatal_error: None,
                })
            }
            (PromptMode::Tool { tool_call }, 0) => {
                self.stage = 1;
                Ok(AssistantAction {
                    assistant_text: String::new(),
                    tool_uses: vec![tool_call.clone()],
                    retry_error: None,
                    fatal_error: None,
                })
            }
            (PromptMode::Tool { .. }, 1) => {
                let result = state
                    .last_tool_results
                    .first()
                    .ok_or_else(|| anyhow!("missing tool result for prompt planner"))?;
                self.stage = 2;
                Ok(AssistantAction {
                    assistant_text: result.content.clone(),
                    tool_uses: Vec::new(),
                    retry_error: None,
                    fatal_error: None,
                })
            }
            _ => bail!("prompt planner exhausted before loop completed"),
        }
    }
}

struct LiveAnthropicPlanner {
    session_id: String,
    model: String,
    system: Vec<String>,
    tools: Vec<Value>,
    messages: Vec<RequestMessage>,
    awaiting_tool_results: bool,
}

impl LiveAnthropicPlanner {
    fn new(
        prompt: &str,
        history: &[ConversationMessage],
        api_history: Option<&[RequestMessage]>,
        model: &str,
        cwd: &Path,
        shell: &str,
    ) -> Self {
        let mut messages = api_history
            .map(|messages| messages.to_vec())
            .unwrap_or_else(|| {
                history
                    .iter()
                    .filter_map(history_message_to_request_message)
                    .collect::<Vec<_>>()
            });
        messages.push(text_request_message("user", prompt));

        Self {
            session_id: format!("nano-claude-code-{}", now_ms()),
            model: model.to_string(),
            system: vec![main_loop_system_prompt(cwd, shell)],
            tools: main_loop_tool_definitions(),
            messages,
            awaiting_tool_results: false,
        }
    }
}

impl AssistantPlanner for LiveAnthropicPlanner {
    fn next_action(&mut self, state: &LoopState<'_>) -> Result<AssistantAction> {
        if self.awaiting_tool_results {
            if state.last_tool_results.is_empty() {
                bail!("missing tool results for live planner");
            }
            self.messages
                .push(tool_results_request_message(state.last_tool_results));
            self.awaiting_tool_results = false;
        }

        let api_key = anthropic_api_key()
            .ok_or_else(|| anyhow!("Anthropic API key is required for live Anthropic requests"))?;
        let output = execute_live_completion(&LiveCompletionInput {
            session_id: self.session_id.clone(),
            user_agent: USER_AGENT.to_string(),
            api_key,
            model: self.model.clone(),
            base_url: anthropic_base_url(),
            messages: self.messages.clone(),
            system: self.system.clone(),
            tools: self.tools.clone(),
            max_tokens: Some(live_max_output_tokens()),
        })?;

        if let Some(message) = assistant_request_message_from_live_output(&output) {
            self.messages.push(message);
        }

        let action = assistant_action_from_live_output(&output)?;
        self.awaiting_tool_results = !action.tool_uses.is_empty();
        Ok(action)
    }

    fn api_messages(&self) -> Option<Vec<RequestMessage>> {
        Some(self.messages.clone())
    }
}

fn should_use_prompt_planner(prompt: &str) -> bool {
    maybe_run_echo(prompt).is_some()
        || maybe_extract_bash_command(prompt).is_some()
        || maybe_extract_edit_command(prompt).is_some()
        || prompt == "/tools"
}

fn anthropic_api_key() -> Option<String> {
    load_anthropic_api_key().map(|entry| entry.value)
}

fn anthropic_base_url() -> String {
    env::var("ANTHROPIC_BASE_URL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "https://api.anthropic.com".to_string())
}

fn live_max_output_tokens() -> u64 {
    env::var(MAX_OUTPUT_TOKENS_ENV)
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(LIVE_MAX_OUTPUT_TOKENS)
}

fn main_loop_system_prompt(cwd: &Path, shell: &str) -> String {
    format!(
        "You are nano-claude-code working in the repository at {cwd}. \
The shell is {shell}. \
Use bash to inspect code and run tests or linters. \
Use read_file, grep_search, and glob_search for repository inspection when they are more direct than shell quoting. \
Use edit and write_file for direct file changes instead of shell-based rewrites. \
Use spawn_agent only for bounded side tasks. Use wait_agent to collect the first result, send_agent for a bounded follow-up when needed, and close_agent once the helper is no longer needed. \
Do not claim tool actions you did not take. Keep progress updates short and concrete.",
        cwd = cwd.display()
    )
}

fn history_message_to_request_message(message: &ConversationMessage) -> Option<RequestMessage> {
    if message.content.trim().is_empty() {
        return None;
    }

    let role = match message.role {
        MessageRole::User => "user",
        MessageRole::Assistant => "assistant",
    };

    Some(text_request_message(role, &message.content))
}

fn text_request_message(role: &str, text: &str) -> RequestMessage {
    RequestMessage {
        role: role.to_string(),
        content: json!([
            {
                "type": "text",
                "text": text
            }
        ]),
    }
}

fn tool_results_request_message(results: &[ToolResult]) -> RequestMessage {
    let mut content = Vec::with_capacity(results.len());
    for result in results {
        let mut block = json!({
            "type": "tool_result",
            "tool_use_id": result.tool_use_id,
            "content": result.content,
        });
        if result.is_error
            && let Some(object) = block.as_object_mut()
        {
            object.insert("is_error".to_string(), Value::Bool(true));
        }
        content.push(block);
    }

    RequestMessage {
        role: "user".to_string(),
        content: Value::Array(content),
    }
}

fn assistant_request_message_from_live_output(
    output: &crate::provider::LiveCompletionOutput,
) -> Option<RequestMessage> {
    if output.content.is_empty() {
        return None;
    }

    Some(RequestMessage {
        role: "assistant".to_string(),
        content: Value::Array(output.content.clone()),
    })
}

fn assistant_action_from_live_output(
    output: &crate::provider::LiveCompletionOutput,
) -> Result<AssistantAction> {
    let tool_uses = output
        .tool_uses
        .iter()
        .map(tool_call_from_live_tool_use)
        .collect::<Result<Vec<_>>>()?;

    Ok(AssistantAction {
        assistant_text: output.assistant_text.clone(),
        tool_uses,
        retry_error: None,
        fatal_error: output.error.as_ref().map(|error| error.content.clone()),
    })
}

fn tool_call_from_live_tool_use(tool_use: &LiveToolUse) -> Result<ToolCall> {
    Ok(ToolCall {
        id: tool_use.id.clone(),
        tool: normalize_tool_name(&tool_use.name).to_string(),
        input: serialize_tool_input(&tool_use.input)?,
    })
}

fn normalize_tool_name(name: &str) -> &str {
    match name {
        "agent_spawn" => "spawn_agent",
        "agent_wait" => "wait_agent",
        "agent_send" => "send_agent",
        "agent_close" => "close_agent",
        other => other,
    }
}

fn serialize_tool_input(value: &Value) -> Result<String> {
    match value {
        Value::String(text) => Ok(text.clone()),
        other => Ok(serde_json::to_string(other)?),
    }
}

fn append_missing_tool_results(
    messages: &mut Vec<LoopMessage>,
    tool_uses: &[ToolCall],
    error: &str,
) {
    for tool_use in tool_uses {
        messages.push(LoopMessage {
            role: MessageRole::User,
            content: vec![LoopContent::ToolResult {
                tool_use_id: tool_use.id.clone(),
                content: error.to_string(),
                is_error: true,
            }],
            is_error: false,
        });
    }
}

fn tool_result_message(result: &ToolResult) -> LoopMessage {
    LoopMessage {
        role: MessageRole::User,
        content: vec![LoopContent::ToolResult {
            tool_use_id: result.tool_use_id.clone(),
            content: result.content.clone(),
            is_error: result.is_error,
        }],
        is_error: false,
    }
}

fn assistant_error_message(error: &str) -> LoopMessage {
    LoopMessage {
        role: MessageRole::Assistant,
        content: vec![LoopContent::Text {
            text: error.to_string(),
        }],
        is_error: true,
    }
}

fn is_false(value: &bool) -> bool {
    !*value
}
