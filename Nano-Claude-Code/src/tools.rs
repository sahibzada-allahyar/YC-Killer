use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::agent::{
    built_in_agents, close_agent, send_agent, spawn_agent_with_options, wait_agent,
};
use crate::core::DEFAULT_MODEL;
use crate::edit::{perform_edit, perform_edit_tool};
use crate::file_tools::{expand_path, read_text_file, write_file};
use crate::search::{GlobSearchArgs, GrepOutputMode, GrepSearchArgs, glob_search, grep_search};
use crate::shell::execute as execute_shell;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ToolUse {
    pub tool: String,
    pub input: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub id: String,
    pub tool: String,
    pub input: String,
}

impl ToolCall {
    pub fn summary(&self) -> ToolUse {
        ToolUse {
            tool: self.tool.clone(),
            input: self.input.clone(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ToolResult {
    pub tool_use_id: String,
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
}

impl ToolResult {
    pub fn success(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: false,
        }
    }

    pub fn error(tool_use_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ScriptedToolResult {
    pub tool_use_id: String,
    pub content: String,
    #[serde(default)]
    pub is_error: bool,
}

pub trait ToolExecutor {
    fn execute(&mut self, tool_call: &ToolCall) -> Result<ToolResult>;
}

pub struct BuiltInToolExecutor<'a> {
    cwd: &'a Path,
    shell: &'a str,
    env_vars: &'a BTreeMap<String, String>,
    model: &'a str,
}

impl<'a> BuiltInToolExecutor<'a> {
    pub fn new(cwd: &'a Path, shell: &'a str, env_vars: &'a BTreeMap<String, String>) -> Self {
        Self::with_model(cwd, shell, env_vars, DEFAULT_MODEL)
    }

    pub fn with_model(
        cwd: &'a Path,
        shell: &'a str,
        env_vars: &'a BTreeMap<String, String>,
        model: &'a str,
    ) -> Self {
        Self {
            cwd,
            shell,
            env_vars,
            model,
        }
    }
}

impl ToolExecutor for BuiltInToolExecutor<'_> {
    fn execute(&mut self, tool_call: &ToolCall) -> Result<ToolResult> {
        match tool_call.tool.as_str() {
            "echo" => execute_echo(tool_call),
            "bash" => execute_bash(tool_call, self.cwd, self.shell, self.env_vars),
            "edit" => execute_edit(tool_call, self.cwd),
            "read_file" => execute_read_file(tool_call, self.cwd),
            "write_file" => execute_write_file(tool_call, self.cwd),
            "grep_search" => execute_grep_search(tool_call, self.cwd),
            "glob_search" => execute_glob_search(tool_call, self.cwd),
            "spawn_agent" | "agent_spawn" => {
                execute_spawn_agent(tool_call, self.cwd, self.shell, self.model, self.env_vars)
            }
            "wait_agent" => execute_wait_agent(tool_call),
            "send_agent" | "agent_send" => execute_send_agent(tool_call),
            "close_agent" | "agent_close" => execute_close_agent(tool_call),
            other => Ok(ToolResult::error(
                tool_call.id.clone(),
                format!("Error: No such tool available: {other}"),
            )),
        }
    }
}

pub struct MainLoopToolExecutor<'a> {
    inner: BuiltInToolExecutor<'a>,
}

impl<'a> MainLoopToolExecutor<'a> {
    pub fn new(cwd: &'a Path, shell: &'a str, env_vars: &'a BTreeMap<String, String>) -> Self {
        Self::with_model(cwd, shell, env_vars, DEFAULT_MODEL)
    }

    pub fn with_model(
        cwd: &'a Path,
        shell: &'a str,
        env_vars: &'a BTreeMap<String, String>,
        model: &'a str,
    ) -> Self {
        Self {
            inner: BuiltInToolExecutor::with_model(cwd, shell, env_vars, model),
        }
    }
}

impl ToolExecutor for MainLoopToolExecutor<'_> {
    fn execute(&mut self, tool_call: &ToolCall) -> Result<ToolResult> {
        match self.inner.execute(tool_call) {
            Ok(result) => Ok(result),
            Err(error) => Ok(ToolResult::error(
                tool_call.id.clone(),
                format!("{error:#}"),
            )),
        }
    }
}

pub fn main_loop_tool_definitions() -> Vec<Value> {
    let agent_types: Vec<String> = built_in_agents()
        .into_iter()
        .map(|agent| agent.agent_type)
        .collect();

    vec![
        json!({
            "name": "bash",
            "description": "Run a shell command in the current repository. Use this to inspect code and run tests or linters. Do not use it for direct file edits when the edit tool is sufficient.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The exact shell command to run."
                    }
                },
                "required": ["command"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "edit",
            "description": "Edit a single file by replacing exact text. Prefer this over bash for direct source changes.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "Path to the file relative to the current repository root."
                    },
                    "oldString": {
                        "type": "string",
                        "description": "The exact text to replace."
                    },
                    "newString": {
                        "type": "string",
                        "description": "The replacement text."
                    },
                    "replaceAll": {
                        "type": "boolean",
                        "description": "Replace every exact occurrence when true."
                    }
                },
                "required": ["filePath", "oldString", "newString"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "read_file",
            "description": "Read a text file from the repository. Supports optional 1-based line offsets and limits.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "Path to the file relative to the current repository root."
                    },
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional 1-based starting line."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional number of lines to read."
                    }
                },
                "required": ["filePath"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "write_file",
            "description": "Write the full contents of a text file, creating it if needed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "filePath": {
                        "type": "string",
                        "description": "Path to the file relative to the current repository root."
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete replacement file contents."
                    }
                },
                "required": ["filePath", "content"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "grep_search",
            "description": "Search file contents with ripgrep. Use this to find symbols, strings, and matching files.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string" },
                    "glob": { "type": "string" },
                    "outputMode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"]
                    },
                    "contextBefore": { "type": "integer", "minimum": 0 },
                    "contextAfter": { "type": "integer", "minimum": 0 },
                    "context": { "type": "integer", "minimum": 0 },
                    "showLineNumbers": { "type": "boolean" },
                    "caseInsensitive": { "type": "boolean" },
                    "fileType": { "type": "string" },
                    "headLimit": { "type": "integer", "minimum": 0 },
                    "offset": { "type": "integer", "minimum": 0 },
                    "multiline": { "type": "boolean" }
                },
                "required": ["pattern"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "glob_search",
            "description": "List files matching a glob pattern in the repository.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string" },
                    "limit": { "type": "integer", "minimum": 0 },
                    "offset": { "type": "integer", "minimum": 0 }
                },
                "required": ["pattern"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "spawn_agent",
            "description": "Spawn a bounded helper agent for a side task, then wait for it before reporting its result.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agentType": {
                        "type": "string",
                        "enum": agent_types,
                        "description": "The helper agent type to run."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "A concise bounded task for the helper agent."
                    }
                },
                "required": ["agentType", "prompt"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "wait_agent",
            "description": "Wait for a previously spawned helper agent and retrieve its final response.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agentId": {
                        "type": "string",
                        "description": "The helper agent id returned by spawn_agent."
                    }
                },
                "required": ["agentId"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "send_agent",
            "description": "Send a follow-up prompt to a previously spawned helper agent and retrieve its updated response.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agentId": {
                        "type": "string",
                        "description": "The helper agent id returned by spawn_agent."
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The follow-up task or question for the helper agent."
                    }
                },
                "required": ["agentId", "prompt"],
                "additionalProperties": false
            }
        }),
        json!({
            "name": "close_agent",
            "description": "Mark a helper agent as closed when no more follow-up work is needed.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "agentId": {
                        "type": "string",
                        "description": "The helper agent id to close."
                    }
                },
                "required": ["agentId"],
                "additionalProperties": false
            }
        }),
    ]
}

pub struct ScriptedToolExecutor {
    results: BTreeMap<String, ScriptedToolResult>,
}

impl ScriptedToolExecutor {
    pub fn new(results: Vec<ScriptedToolResult>) -> Self {
        let results = results
            .into_iter()
            .map(|result| (result.tool_use_id.clone(), result))
            .collect();
        Self { results }
    }
}

impl ToolExecutor for ScriptedToolExecutor {
    fn execute(&mut self, tool_call: &ToolCall) -> Result<ToolResult> {
        let Some(result) = self.results.remove(&tool_call.id) else {
            return Ok(ToolResult::error(
                tool_call.id.clone(),
                format!("Error: Missing scripted tool result for {}", tool_call.id),
            ));
        };

        Ok(ToolResult {
            tool_use_id: result.tool_use_id,
            content: result.content,
            is_error: result.is_error,
        })
    }
}

fn execute_echo(tool_call: &ToolCall) -> Result<ToolResult> {
    Ok(ToolResult::success(
        tool_call.id.clone(),
        decode_echo_input(&tool_call.input),
    ))
}

fn execute_bash(
    tool_call: &ToolCall,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<ToolResult> {
    let command = decode_bash_input(&tool_call.input)?;
    let report = execute_shell(&command, cwd, shell, env_vars)?;
    let structured = parse_tool_input_object(&tool_call.input).is_some();
    let content = if structured {
        serde_json::to_string_pretty(&report)?
    } else if !report.stdout.is_empty() {
        report.stdout
    } else if !report.stderr.is_empty() {
        report.stderr
    } else {
        format!("command exited with {}", report.exit_code)
    };
    Ok(ToolResult::success(tool_call.id.clone(), content))
}

fn execute_edit(tool_call: &ToolCall, cwd: &Path) -> Result<ToolResult> {
    let args = decode_edit_input(&tool_call.input)?;
    let file_path = resolve_tool_path(cwd, &args.file_path)?;
    if parse_tool_input_object(&tool_call.input).is_some() {
        let report = perform_edit_tool(
            &file_path,
            &args.old_string,
            &args.new_string,
            args.replace_all,
        )?;
        return Ok(ToolResult::success(
            tool_call.id.clone(),
            serde_json::to_string_pretty(&report)?,
        ));
    }

    let _report = perform_edit(
        &file_path,
        &args.old_string,
        &args.new_string,
        args.replace_all,
    )?;
    Ok(ToolResult::success(
        tool_call.id.clone(),
        format!("Edited {}", args.file_path),
    ))
}

fn execute_read_file(tool_call: &ToolCall, cwd: &Path) -> Result<ToolResult> {
    let args = decode_read_file_input(&tool_call.input)?;
    let file_path = resolve_tool_path(cwd, &args.file_path)?;
    let report = read_text_file(&file_path, args.offset.unwrap_or(1), args.limit)?;
    Ok(ToolResult::success(
        tool_call.id.clone(),
        serde_json::to_string_pretty(&report)?,
    ))
}

fn execute_write_file(tool_call: &ToolCall, cwd: &Path) -> Result<ToolResult> {
    let args = decode_write_file_input(&tool_call.input)?;
    let file_path = resolve_tool_path(cwd, &args.file_path)?;
    let report = write_file(&file_path, &args.content)?;
    Ok(ToolResult::success(
        tool_call.id.clone(),
        serde_json::to_string_pretty(&report)?,
    ))
}

fn execute_grep_search(tool_call: &ToolCall, cwd: &Path) -> Result<ToolResult> {
    let args = decode_grep_search_input(&tool_call.input, cwd)?;
    let report = grep_search(args)?;
    Ok(ToolResult::success(
        tool_call.id.clone(),
        serde_json::to_string_pretty(&report)?,
    ))
}

fn execute_glob_search(tool_call: &ToolCall, cwd: &Path) -> Result<ToolResult> {
    let args = decode_glob_search_input(&tool_call.input, cwd)?;
    let report = glob_search(args)?;
    Ok(ToolResult::success(
        tool_call.id.clone(),
        serde_json::to_string_pretty(&report)?,
    ))
}

fn execute_spawn_agent(
    tool_call: &ToolCall,
    cwd: &Path,
    shell: &str,
    model: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<ToolResult> {
    let (agent_type, prompt) = decode_spawn_agent_input(&tool_call.input)?;
    let record = spawn_agent_with_options(&agent_type, cwd, &prompt, shell, model, env_vars)?;
    Ok(ToolResult::success(
        tool_call.id.clone(),
        serde_json::to_string(&record)?,
    ))
}

fn execute_wait_agent(tool_call: &ToolCall) -> Result<ToolResult> {
    let agent_id = decode_wait_agent_input(&tool_call.input)?;
    let record = wait_agent(&agent_id)?.ok_or_else(|| anyhow!("unknown agent: {agent_id}"))?;
    Ok(ToolResult::success(
        tool_call.id.clone(),
        serde_json::to_string(&record)?,
    ))
}

fn execute_send_agent(tool_call: &ToolCall) -> Result<ToolResult> {
    let (agent_id, prompt) = decode_send_agent_input(&tool_call.input)?;
    let record =
        send_agent(&agent_id, &prompt)?.ok_or_else(|| anyhow!("unknown agent: {agent_id}"))?;
    Ok(ToolResult::success(
        tool_call.id.clone(),
        serde_json::to_string(&record)?,
    ))
}

fn execute_close_agent(tool_call: &ToolCall) -> Result<ToolResult> {
    let agent_id = decode_close_agent_input(&tool_call.input)?;
    let record = close_agent(&agent_id)?.ok_or_else(|| anyhow!("unknown agent: {agent_id}"))?;
    Ok(ToolResult::success(
        tool_call.id.clone(),
        serde_json::to_string(&record)?,
    ))
}

pub fn maybe_run_echo(prompt: &str) -> Option<(ToolUse, String)> {
    let rest = prompt.strip_prefix("/echo ")?;
    Some((
        ToolUse {
            tool: "echo".to_string(),
            input: rest.to_string(),
        },
        rest.to_string(),
    ))
}

pub fn maybe_extract_bash_command(prompt: &str) -> Option<&str> {
    prompt.strip_prefix("/bash ")
}

pub fn maybe_extract_edit_command(prompt: &str) -> Option<&str> {
    prompt.strip_prefix("/edit ")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InlineEditCommand {
    pub file_path: String,
    pub old_string: String,
    pub new_string: String,
    pub replace_all: bool,
}

pub fn parse_inline_edit_command(input: &str) -> Result<InlineEditCommand> {
    let tokens = split_command_words(input)?;
    let mut file_path = None;
    let mut old_string = None;
    let mut new_string = None;
    let mut replace_all = false;
    let mut idx = 0;

    while idx < tokens.len() {
        let token = &tokens[idx];
        match token.as_str() {
            "--file" => {
                let Some(value) = tokens.get(idx + 1) else {
                    bail!("missing value for --file");
                };
                file_path = Some(value.clone());
                idx += 2;
            }
            "--old" => {
                let Some(value) = tokens.get(idx + 1) else {
                    bail!("missing value for --old");
                };
                old_string = Some(value.clone());
                idx += 2;
            }
            "--new" => {
                let Some(value) = tokens.get(idx + 1) else {
                    bail!("missing value for --new");
                };
                new_string = Some(value.clone());
                idx += 2;
            }
            "--replace-all" => {
                replace_all = true;
                idx += 1;
            }
            _ => {
                if let Some(value) = token.strip_prefix("--file=") {
                    file_path = Some(value.to_string());
                    idx += 1;
                } else if let Some(value) = token.strip_prefix("--old=") {
                    old_string = Some(value.to_string());
                    idx += 1;
                } else if let Some(value) = token.strip_prefix("--new=") {
                    new_string = Some(value.to_string());
                    idx += 1;
                } else {
                    bail!("unknown edit flag: {token}");
                }
            }
        }
    }

    Ok(InlineEditCommand {
        file_path: file_path.ok_or_else(|| anyhow::anyhow!("edit requires --file"))?,
        old_string: old_string.ok_or_else(|| anyhow::anyhow!("edit requires --old"))?,
        new_string: new_string.ok_or_else(|| anyhow::anyhow!("edit requires --new"))?,
        replace_all,
    })
}

pub fn split_command_words(input: &str) -> Result<Vec<String>> {
    let chars: Vec<char> = input.chars().collect();
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut idx = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut token_started = false;

    while idx < chars.len() {
        let ch = chars[idx];
        if in_single {
            if ch == '\'' {
                in_single = false;
            } else {
                current.push(ch);
            }
            idx += 1;
            continue;
        }

        if in_double {
            if ch == '"' {
                in_double = false;
                idx += 1;
                continue;
            }
            if ch == '\\' {
                idx += 1;
                let Some(next) = chars.get(idx) else {
                    bail!("unterminated escape sequence");
                };
                current.push(*next);
                idx += 1;
                token_started = true;
                continue;
            }
            current.push(ch);
            idx += 1;
            token_started = true;
            continue;
        }

        match ch {
            '\'' => {
                in_single = true;
                token_started = true;
                idx += 1;
            }
            '"' => {
                in_double = true;
                token_started = true;
                idx += 1;
            }
            '\\' => {
                idx += 1;
                let Some(next) = chars.get(idx) else {
                    bail!("unterminated escape sequence");
                };
                current.push(*next);
                idx += 1;
                token_started = true;
            }
            ch if ch.is_whitespace() => {
                if token_started {
                    tokens.push(std::mem::take(&mut current));
                    token_started = false;
                }
                idx += 1;
            }
            _ => {
                current.push(ch);
                token_started = true;
                idx += 1;
            }
        }
    }

    if in_single || in_double {
        bail!("unterminated quoted string");
    }

    if token_started {
        tokens.push(current);
    }

    Ok(tokens)
}

fn decode_echo_input(raw: &str) -> String {
    parse_tool_input_object(raw)
        .and_then(|value| {
            value
                .get("text")
                .and_then(Value::as_str)
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| raw.to_string())
}

fn decode_bash_input(raw: &str) -> Result<String> {
    if let Some(value) = parse_tool_input_object(raw) {
        return value
            .get("command")
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .ok_or_else(|| anyhow::anyhow!("bash tool input requires command"));
    }

    Ok(raw.to_string())
}

fn decode_edit_input(raw: &str) -> Result<InlineEditCommand> {
    let Some(value) = parse_tool_input_object(raw) else {
        return parse_inline_edit_command(raw);
    };

    Ok(InlineEditCommand {
        file_path: required_string_field(&value, "filePath")?,
        old_string: required_string_field(&value, "oldString")?,
        new_string: required_string_field(&value, "newString")?,
        replace_all: value
            .get("replaceAll")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    })
}

fn decode_spawn_agent_input(raw: &str) -> Result<(String, String)> {
    let value = parse_tool_input_object(raw)
        .ok_or_else(|| anyhow::anyhow!("spawn_agent tool input must be a JSON object"))?;
    let agent_type = normalize_agent_type(&required_string_field(&value, "agentType")?);
    let prompt = required_string_field(&value, "prompt")?;
    Ok((agent_type, prompt))
}

fn decode_wait_agent_input(raw: &str) -> Result<String> {
    let value = parse_tool_input_object(raw)
        .ok_or_else(|| anyhow!("wait_agent tool input must be a JSON object"))?;
    required_string_field(&value, "agentId")
}

fn decode_send_agent_input(raw: &str) -> Result<(String, String)> {
    let value = parse_tool_input_object(raw)
        .ok_or_else(|| anyhow!("send_agent tool input must be a JSON object"))?;
    let agent_id = required_string_field(&value, "agentId")?;
    let prompt = required_string_field(&value, "prompt")?;
    Ok((agent_id, prompt))
}

fn decode_close_agent_input(raw: &str) -> Result<String> {
    let value = parse_tool_input_object(raw)
        .ok_or_else(|| anyhow!("close_agent tool input must be a JSON object"))?;
    required_string_field(&value, "agentId")
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ReadFileInput {
    file_path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct WriteFileInput {
    file_path: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GrepSearchInput {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    glob: Option<String>,
    #[serde(default)]
    output_mode: Option<String>,
    #[serde(default)]
    context_before: Option<usize>,
    #[serde(default)]
    context_after: Option<usize>,
    #[serde(default)]
    context: Option<usize>,
    #[serde(default)]
    show_line_numbers: Option<bool>,
    #[serde(default)]
    case_insensitive: bool,
    #[serde(default)]
    file_type: Option<String>,
    #[serde(default)]
    head_limit: Option<usize>,
    #[serde(default)]
    offset: usize,
    #[serde(default)]
    multiline: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GlobSearchInput {
    pattern: String,
    #[serde(default)]
    path: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
    #[serde(default)]
    offset: usize,
}

fn decode_read_file_input(raw: &str) -> Result<ReadFileInput> {
    let value = parse_tool_input_object(raw)
        .ok_or_else(|| anyhow!("read_file tool input must be a JSON object"))?;
    Ok(serde_json::from_value(value)?)
}

fn decode_write_file_input(raw: &str) -> Result<WriteFileInput> {
    let value = parse_tool_input_object(raw)
        .ok_or_else(|| anyhow!("write_file tool input must be a JSON object"))?;
    Ok(serde_json::from_value(value)?)
}

fn decode_grep_search_input(raw: &str, cwd: &Path) -> Result<GrepSearchArgs> {
    let value = parse_tool_input_object(raw)
        .ok_or_else(|| anyhow!("grep_search tool input must be a JSON object"))?;
    let input: GrepSearchInput = serde_json::from_value(value)?;
    Ok(GrepSearchArgs {
        cwd: cwd.to_path_buf(),
        pattern: input.pattern,
        path: input.path,
        glob: input.glob,
        output_mode: input
            .output_mode
            .as_deref()
            .map(parse_grep_output_mode)
            .transpose()?,
        context_before: input.context_before,
        context_after: input.context_after,
        context: input.context,
        show_line_numbers: input.show_line_numbers,
        case_insensitive: input.case_insensitive,
        file_type: input.file_type,
        head_limit: input.head_limit,
        offset: input.offset,
        multiline: input.multiline,
    })
}

fn decode_glob_search_input(raw: &str, cwd: &Path) -> Result<GlobSearchArgs> {
    let value = parse_tool_input_object(raw)
        .ok_or_else(|| anyhow!("glob_search tool input must be a JSON object"))?;
    let input: GlobSearchInput = serde_json::from_value(value)?;
    Ok(GlobSearchArgs {
        cwd: cwd.to_path_buf(),
        pattern: input.pattern,
        path: input.path,
        limit: input.limit.unwrap_or(100),
        offset: input.offset,
    })
}

fn parse_grep_output_mode(raw: &str) -> Result<GrepOutputMode> {
    match raw {
        "content" => Ok(GrepOutputMode::Content),
        "files_with_matches" => Ok(GrepOutputMode::FilesWithMatches),
        "count" => Ok(GrepOutputMode::Count),
        other => bail!("unsupported grep_search outputMode: {other}"),
    }
}

fn required_string_field(value: &Value, key: &str) -> Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .ok_or_else(|| anyhow::anyhow!("{key} is required"))
}

fn parse_tool_input_object(raw: &str) -> Option<Value> {
    let trimmed = raw.trim();
    if !(trimmed.starts_with('{') && trimmed.ends_with('}')) {
        return None;
    }
    serde_json::from_str(trimmed).ok()
}

fn normalize_agent_type(raw: &str) -> String {
    match raw {
        "explore" | "Explore" => "Explore".to_string(),
        "plan" | "Plan" => "Plan".to_string(),
        "verification" => "verification".to_string(),
        "general" | "general-purpose" => "general-purpose".to_string(),
        other => other.to_string(),
    }
}

fn resolve_tool_path(cwd: &Path, raw_path: &str) -> Result<PathBuf> {
    Ok(PathBuf::from(expand_path(raw_path, Some(cwd))?))
}
