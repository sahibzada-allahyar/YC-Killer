use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde::Serialize;
use serde_json::Value;

use crate::agent::{
    AgentRecord, AgentTypeInfo, built_in_agents, close_agent, list_agent_records, send_agent,
    spawn_agent_with_options, wait_agent,
};
use crate::core::DEFAULT_MODEL;
use crate::edit::{EditReport, perform_edit};
use crate::session::{
    ListSessionsOptions, SessionInfo, SessionTranscriptScan, get_session_info,
    get_session_transcript_scan, list_sessions,
};
use crate::shell::{ExecReport, execute as execute_shell};

const CLAUDEAI_SERVER_PREFIX: &str = "claude.ai ";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct McpInfo {
    pub server_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct McpRouteReport {
    pub raw_tool_name: String,
    pub server_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
    pub route: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum McpCallResult {
    BashExec { report: ExecReport },
    FileEdit { report: EditReport },
    AgentTypes { types: Vec<AgentTypeInfo> },
    AgentSpawn { record: AgentRecord },
    AgentWait { record: Option<AgentRecord> },
    AgentSend { record: Option<AgentRecord> },
    AgentClose { record: Option<AgentRecord> },
    AgentPs { records: Vec<AgentRecord> },
    SessionInfo { info: Option<SessionInfo> },
    SessionTranscript { scan: Option<SessionTranscriptScan> },
    SessionList { sessions: Vec<SessionInfo> },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct McpCallReport {
    pub raw_tool_name: String,
    pub server_name: String,
    pub tool_name: String,
    pub route: String,
    pub result: McpCallResult,
}

pub fn normalize_name_for_mcp(name: &str) -> String {
    let mut normalized: String = name
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect();

    if name.starts_with(CLAUDEAI_SERVER_PREFIX) {
        let mut collapsed = String::with_capacity(normalized.len());
        let mut last_was_underscore = false;
        for ch in normalized.chars() {
            if ch == '_' {
                if !last_was_underscore {
                    collapsed.push(ch);
                }
                last_was_underscore = true;
            } else {
                collapsed.push(ch);
                last_was_underscore = false;
            }
        }
        normalized = collapsed.trim_matches('_').to_string();
    }

    normalized
}

pub fn mcp_info_from_string(tool_string: &str) -> Option<McpInfo> {
    let mut parts = tool_string.split("__");
    let mcp_part = parts.next()?;
    let server_name = parts.next()?;
    if mcp_part != "mcp" || server_name.is_empty() {
        return None;
    }

    let tool_name_parts: Vec<&str> = parts.collect();
    let tool_name = if tool_name_parts.is_empty() {
        None
    } else {
        Some(tool_name_parts.join("__"))
    };

    Some(McpInfo {
        server_name: server_name.to_string(),
        tool_name,
    })
}

pub fn get_mcp_prefix(server_name: &str) -> String {
    format!("mcp__{}__", normalize_name_for_mcp(server_name))
}

pub fn build_mcp_tool_name(server_name: &str, tool_name: &str) -> String {
    format!(
        "{}{}",
        get_mcp_prefix(server_name),
        normalize_name_for_mcp(tool_name)
    )
}

pub fn route_mcp_tool(raw_tool_name: &str) -> Result<McpRouteReport> {
    let info = mcp_info_from_string(raw_tool_name)
        .ok_or_else(|| anyhow!("invalid mcp tool name: {raw_tool_name}"))?;

    Ok(McpRouteReport {
        raw_tool_name: raw_tool_name.to_string(),
        server_name: info.server_name.clone(),
        tool_name: info.tool_name.clone(),
        route: resolve_demo_route(&info.server_name, info.tool_name.as_deref()).to_string(),
    })
}

pub fn call_mcp_tool(
    raw_tool_name: &str,
    input_text: Option<&str>,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<McpCallReport> {
    let route = route_mcp_tool(raw_tool_name)?;
    let tool_name = route
        .tool_name
        .clone()
        .ok_or_else(|| anyhow!("mcp tool name is missing a tool segment: {raw_tool_name}"))?;
    let input = parse_input(input_text)?;

    let result = match route.route.as_str() {
        "bash.exec" => McpCallResult::BashExec {
            report: call_bash_exec(&input, cwd, shell, env_vars)?,
        },
        "file.edit" => McpCallResult::FileEdit {
            report: call_file_edit(&input, cwd)?,
        },
        "agent.types" => McpCallResult::AgentTypes {
            types: built_in_agents(),
        },
        "agent.spawn" => McpCallResult::AgentSpawn {
            record: call_agent_spawn(&input, cwd, shell, env_vars)?,
        },
        "agent.wait" => McpCallResult::AgentWait {
            record: call_agent_wait(&input)?,
        },
        "agent.send" => McpCallResult::AgentSend {
            record: call_agent_send(&input)?,
        },
        "agent.close" => McpCallResult::AgentClose {
            record: call_agent_close(&input)?,
        },
        "agent.ps" => McpCallResult::AgentPs {
            records: list_agent_records()?,
        },
        "session.info" => McpCallResult::SessionInfo {
            info: call_session_info(&input, cwd)?,
        },
        "session.transcript" => McpCallResult::SessionTranscript {
            scan: call_session_transcript(&input, cwd)?,
        },
        "session.list" => McpCallResult::SessionList {
            sessions: call_session_list(&input, cwd)?,
        },
        "unsupported" => bail!("unsupported demo mcp route: {raw_tool_name}"),
        other => bail!("unknown demo mcp route: {other}"),
    };

    Ok(McpCallReport {
        raw_tool_name: route.raw_tool_name,
        server_name: route.server_name,
        tool_name,
        route: route.route,
        result,
    })
}

fn resolve_demo_route(server_name: &str, tool_name: Option<&str>) -> &'static str {
    match (server_name, tool_name) {
        ("bash", Some("exec")) => "bash.exec",
        ("file", Some("edit")) => "file.edit",
        ("agent", Some("types")) => "agent.types",
        ("agent", Some("spawn")) => "agent.spawn",
        ("agent", Some("wait")) => "agent.wait",
        ("agent", Some("send")) => "agent.send",
        ("agent", Some("close")) => "agent.close",
        ("agent", Some("ps")) => "agent.ps",
        ("session", Some("info")) => "session.info",
        ("session", Some("transcript")) => "session.transcript",
        ("session", Some("list")) => "session.list",
        _ => "unsupported",
    }
}

fn parse_input(input_text: Option<&str>) -> Result<Value> {
    let Some(input_text) = input_text else {
        return Ok(Value::Null);
    };
    if input_text.trim().is_empty() {
        return Ok(Value::Null);
    }
    Ok(serde_json::from_str(input_text)?)
}

fn call_bash_exec(
    input: &Value,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<ExecReport> {
    let command = required_string(input, "command")?;
    let exec_cwd = optional_path(input, "cwd").unwrap_or_else(|| cwd.to_path_buf());
    let exec_shell = optional_string(input, "shell").unwrap_or(shell);
    let mut merged_env = env_vars.clone();
    merged_env.extend(read_env_map(input, "env")?);
    execute_shell(command, &exec_cwd, exec_shell, &merged_env)
}

fn call_file_edit(input: &Value, cwd: &Path) -> Result<EditReport> {
    let file_path = required_string(input, "filePath")?;
    let old_string = required_string(input, "oldString")?;
    let new_string = required_string(input, "newString")?;
    let replace_all = input
        .get("replaceAll")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let file_path = resolve_against_cwd(cwd, file_path);
    perform_edit(&file_path, old_string, new_string, replace_all)
}

fn call_agent_spawn(
    input: &Value,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<AgentRecord> {
    let agent_type = required_string(input, "agentType")?;
    let prompt = required_string(input, "prompt")?;
    let agent_cwd = optional_path(input, "cwd").unwrap_or_else(|| cwd.to_path_buf());
    spawn_agent_with_options(
        agent_type,
        &agent_cwd,
        prompt,
        shell,
        DEFAULT_MODEL,
        env_vars,
    )
}

fn call_agent_wait(input: &Value) -> Result<Option<AgentRecord>> {
    wait_agent(required_string(input, "agentId")?)
}

fn call_agent_send(input: &Value) -> Result<Option<AgentRecord>> {
    send_agent(
        required_string(input, "agentId")?,
        required_string(input, "prompt")?,
    )
}

fn call_agent_close(input: &Value) -> Result<Option<AgentRecord>> {
    close_agent(required_string(input, "agentId")?)
}

fn call_session_info(input: &Value, cwd: &Path) -> Result<Option<SessionInfo>> {
    get_session_info(
        required_string(input, "sessionId")?,
        Some(session_dir(input, cwd).as_path()),
    )
}

fn call_session_transcript(input: &Value, cwd: &Path) -> Result<Option<SessionTranscriptScan>> {
    get_session_transcript_scan(
        required_string(input, "sessionId")?,
        Some(session_dir(input, cwd).as_path()),
    )
}

fn call_session_list(input: &Value, cwd: &Path) -> Result<Vec<SessionInfo>> {
    let all = input.get("all").and_then(Value::as_bool).unwrap_or(false);
    let limit = optional_usize(input, "limit")?;
    let offset = optional_usize(input, "offset")?.unwrap_or(0);
    let dir = if all {
        None
    } else {
        Some(session_dir(input, cwd))
    };

    list_sessions(ListSessionsOptions { dir, limit, offset })
}

fn session_dir(input: &Value, cwd: &Path) -> PathBuf {
    optional_path(input, "cwd").unwrap_or_else(|| cwd.to_path_buf())
}

fn read_env_map(input: &Value, key: &str) -> Result<BTreeMap<String, String>> {
    let Some(value) = input.get(key) else {
        return Ok(BTreeMap::new());
    };
    let Some(object) = value.as_object() else {
        bail!("mcp input field `{key}` must be an object");
    };

    let mut env = BTreeMap::new();
    for (env_key, env_value) in object {
        let Some(env_value) = env_value.as_str() else {
            bail!("mcp env values must be strings");
        };
        env.insert(env_key.clone(), env_value.to_string());
    }
    Ok(env)
}

fn optional_usize(input: &Value, key: &str) -> Result<Option<usize>> {
    let Some(value) = input.get(key) else {
        return Ok(None);
    };
    let Some(value) = value.as_u64() else {
        bail!("mcp input field `{key}` must be a non-negative integer");
    };
    Ok(Some(value as usize))
}

fn required_string<'a>(input: &'a Value, key: &str) -> Result<&'a str> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing string field `{key}` in mcp input"))
}

fn optional_string<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input.get(key).and_then(Value::as_str)
}

fn optional_path(input: &Value, key: &str) -> Option<PathBuf> {
    optional_string(input, key).map(PathBuf::from)
}

fn resolve_against_cwd(cwd: &Path, raw_path: &str) -> PathBuf {
    let candidate = PathBuf::from(raw_path);
    if candidate.is_absolute() {
        candidate
    } else {
        cwd.join(candidate)
    }
}
