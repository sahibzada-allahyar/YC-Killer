use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::anthropic::RequestMessage;
use crate::config::{get_claude_config_home_dir, normalize_nfc};
use crate::core::{
    ConversationMessage, DEFAULT_MODEL, MessageRole, RunOptions, effective_model, run_with_options,
};
use crate::session::{deterministic_session_id, now_ms};
use crate::shell::default_shell;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentTypeInfo {
    pub agent_type: String,
    pub when_to_use: String,
    pub background: bool,
    pub one_shot: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct AgentRecord {
    pub agent_id: String,
    pub agent_type: String,
    pub cwd: String,
    pub prompt: String,
    pub status: String,
    pub response: String,
    pub created_at: i64,
    pub background: bool,
    pub one_shot: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shell: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentState {
    #[serde(default)]
    api_messages: Vec<RequestMessage>,
}

pub fn built_in_agents() -> Vec<AgentTypeInfo> {
    vec![
        AgentTypeInfo {
            agent_type: "general-purpose".to_string(),
            when_to_use: "General-purpose agent for researching complex questions, searching for code, and executing multi-step tasks.".to_string(),
            background: false,
            one_shot: false,
        },
        AgentTypeInfo {
            agent_type: "Explore".to_string(),
            when_to_use: "Fast agent specialized for exploring codebases.".to_string(),
            background: false,
            one_shot: true,
        },
        AgentTypeInfo {
            agent_type: "Plan".to_string(),
            when_to_use: "Software architect agent for designing implementation plans.".to_string(),
            background: false,
            one_shot: true,
        },
        AgentTypeInfo {
            agent_type: "verification".to_string(),
            when_to_use: "Use this agent to verify that implementation work is correct before reporting completion.".to_string(),
            background: true,
            one_shot: false,
        },
    ]
}

pub fn spawn_agent(agent_type: &str, cwd: &Path, prompt: &str) -> Result<AgentRecord> {
    let definition = built_in_agents()
        .into_iter()
        .find(|agent| agent.agent_type == agent_type)
        .ok_or_else(|| anyhow!("unknown agent type: {agent_type}"))?;

    let cwd = normalize_nfc(&cwd.to_string_lossy());
    let created_at = now_ms();
    let agent_id = deterministic_session_id(&cwd, prompt, agent_type, created_at);
    let record = AgentRecord {
        agent_id: agent_id.clone(),
        agent_type: definition.agent_type,
        cwd,
        prompt: prompt.to_string(),
        status: "completed".to_string(),
        response: mock_agent_response(agent_type, prompt),
        created_at,
        background: definition.background,
        one_shot: definition.one_shot,
        model: None,
        shell: None,
    };

    write_agent_record(&record)?;
    Ok(record)
}

pub fn spawn_agent_with_options(
    agent_type: &str,
    cwd: &Path,
    prompt: &str,
    shell: &str,
    model: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<AgentRecord> {
    let definition = built_in_agents()
        .into_iter()
        .find(|agent| agent.agent_type == agent_type)
        .ok_or_else(|| anyhow!("unknown agent type: {agent_type}"))?;

    let cwd = normalize_nfc(&cwd.to_string_lossy());
    let created_at = now_ms();
    let agent_id = deterministic_session_id(&cwd, prompt, agent_type, created_at);

    let resolved_model = effective_model(model);
    let (response, resolved_model, resolved_shell, api_messages) =
        if resolved_model != DEFAULT_MODEL {
            let outcome = run_with_options(RunOptions {
                prompt,
                history: &[],
                api_history: None,
                cwd: Path::new(&cwd),
                shell,
                env_vars,
                model: &resolved_model,
            })?;
            (
                outcome.assistant_text,
                Some(resolved_model),
                Some(shell.to_string()),
                outcome.api_messages,
            )
        } else {
            (mock_agent_response(agent_type, prompt), None, None, None)
        };

    let record = AgentRecord {
        agent_id: agent_id.clone(),
        agent_type: definition.agent_type,
        cwd,
        prompt: prompt.to_string(),
        status: "completed".to_string(),
        response,
        created_at,
        background: definition.background,
        one_shot: definition.one_shot,
        model: resolved_model,
        shell: resolved_shell,
    };

    write_agent_record(&record)?;
    if let Some(messages) = api_messages.as_ref() {
        write_agent_state(&record.agent_id, messages)?;
    }
    Ok(record)
}

pub fn wait_agent(agent_id: &str) -> Result<Option<AgentRecord>> {
    read_agent_record(agent_id)
}

pub fn send_agent(agent_id: &str, prompt: &str) -> Result<Option<AgentRecord>> {
    let Some(mut record) = read_agent_record(agent_id)? else {
        return Ok(None);
    };
    if record.status == "closed" {
        return Err(anyhow!("agent is closed: {agent_id}"));
    }
    if record.one_shot {
        return Err(anyhow!(
            "agent type does not accept follow-up input: {}",
            record.agent_type
        ));
    }

    let previous_prompt = record.prompt.clone();
    let previous_response = record.response.clone();
    record.prompt = prompt.to_string();
    if record.model.is_some() {
        let cwd = PathBuf::from(&record.cwd);
        let shell = record
            .shell
            .as_deref()
            .map(ToString::to_string)
            .unwrap_or_else(default_shell);
        let model = record.model.clone().unwrap_or_default();
        let api_history = load_agent_api_history(agent_id)?;
        let mut history = Vec::new();
        if !previous_prompt.trim().is_empty() {
            history.push(ConversationMessage {
                role: MessageRole::User,
                content: previous_prompt,
            });
        }
        if !previous_response.trim().is_empty() {
            history.push(ConversationMessage {
                role: MessageRole::Assistant,
                content: previous_response,
            });
        }
        let env_vars = BTreeMap::new();
        let outcome = run_with_options(RunOptions {
            prompt,
            history: &history,
            api_history: api_history.as_deref(),
            cwd: &cwd,
            shell: &shell,
            env_vars: &env_vars,
            model: &model,
        })?;
        record.response = outcome.assistant_text;
        if let Some(messages) = outcome.api_messages.as_ref() {
            write_agent_state(agent_id, messages)?;
        }
    } else {
        record.response = mock_agent_response(&record.agent_type, prompt);
    }
    record.status = "completed".to_string();
    write_agent_record(&record)?;
    Ok(Some(record))
}

pub fn close_agent(agent_id: &str) -> Result<Option<AgentRecord>> {
    let Some(mut record) = read_agent_record(agent_id)? else {
        return Ok(None);
    };
    record.status = "closed".to_string();
    write_agent_record(&record)?;
    Ok(Some(record))
}

pub fn list_agent_records() -> Result<Vec<AgentRecord>> {
    let mut records = Vec::new();
    let Ok(entries) = fs::read_dir(agents_dir()) else {
        return Ok(records);
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
            continue;
        }
        let contents = fs::read(&path)?;
        let record: AgentRecord = serde_json::from_slice(&contents)?;
        records.push(record);
    }

    records.sort_by(|left, right| {
        right
            .created_at
            .cmp(&left.created_at)
            .then_with(|| left.agent_id.cmp(&right.agent_id))
    });
    Ok(records)
}

pub fn agents_dir() -> PathBuf {
    get_claude_config_home_dir().join("agents")
}

fn agent_file_path(agent_id: &str) -> PathBuf {
    agents_dir().join(format!("{agent_id}.json"))
}

fn read_agent_record(agent_id: &str) -> Result<Option<AgentRecord>> {
    let path = agent_file_path(agent_id);
    if !path.is_file() {
        return Ok(None);
    }
    let contents = fs::read(&path)?;
    let record: AgentRecord = serde_json::from_slice(&contents)?;
    Ok(Some(record))
}

fn write_agent_record(record: &AgentRecord) -> Result<()> {
    fs::create_dir_all(agents_dir())?;
    let mut payload = serde_json::to_vec_pretty(record)?;
    payload.push(b'\n');
    fs::write(agent_file_path(&record.agent_id), payload)?;
    Ok(())
}

fn load_agent_api_history(agent_id: &str) -> Result<Option<Vec<RequestMessage>>> {
    let path = agent_state_file_path(agent_id);
    if !path.is_file() {
        return Ok(None);
    }

    let contents = fs::read(&path)?;
    let state: AgentState = serde_json::from_slice(&contents)?;
    if state.api_messages.is_empty() {
        return Ok(None);
    }
    Ok(Some(state.api_messages))
}

fn write_agent_state(agent_id: &str, messages: &[RequestMessage]) -> Result<()> {
    fs::create_dir_all(agents_dir())?;
    let mut payload = serde_json::to_vec_pretty(&AgentState {
        api_messages: messages.to_vec(),
    })?;
    payload.push(b'\n');
    fs::write(agent_state_file_path(agent_id), payload)?;
    Ok(())
}

fn agent_state_file_path(agent_id: &str) -> PathBuf {
    agents_dir().join(format!("{agent_id}.live-state"))
}

fn mock_agent_response(agent_type: &str, prompt: &str) -> String {
    match agent_type {
        "Explore" => format!("Explore summary: {prompt}"),
        "Plan" => format!("Plan summary: {prompt}"),
        "verification" => format!("Verification summary: {prompt}\nVERDICT: PASS"),
        "general-purpose" => format!("General-purpose summary: {prompt}"),
        other => {
            let _ = other;
            String::new()
        }
    }
}
