use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use filetime::{FileTime, set_file_mtime};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha1::{Digest, Sha1};
use time::OffsetDateTime;
use time::format_description::well_known::Rfc3339;
use time::macros::format_description;

use crate::anthropic::RequestMessage;
use crate::config::{get_claude_config_home_dir, normalize_nfc};
use crate::core::{ConversationMessage, MessageRole, RunOutcome};

pub const MAX_SANITIZED_LENGTH: usize = 200;
const LITE_READ_BUF_SIZE: usize = 65_536;
const READ_BATCH_SIZE: usize = 32;
pub const SKIP_PRECOMPACT_THRESHOLD: u64 = 5 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct RunReport {
    pub cwd: String,
    pub model: String,
    pub session_id: String,
    pub session_path: String,
    pub assistant_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_use: Option<crate::tools::ToolUse>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SessionInfo {
    pub session_id: String,
    pub summary: String,
    pub last_modified: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tag: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<i64>,
}

#[derive(Debug, Clone)]
pub struct PersistedRun {
    pub report: RunReport,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ChatReport {
    pub cwd: String,
    pub model: String,
    pub session_id: String,
    pub session_path: String,
    pub turns: usize,
    pub assistant_texts: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_assistant_text: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct ListSessionsOptions {
    pub dir: Option<PathBuf>,
    pub limit: Option<usize>,
    pub offset: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ResolvedSessionFile {
    pub file_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_path: Option<String>,
    pub file_size: u64,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptScan {
    pub boundary_start_offset: usize,
    pub has_preserved_segment: bool,
    pub transcript: String,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct SessionTranscriptScan {
    pub session_id: String,
    pub file_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project_path: Option<String>,
    pub file_size: u64,
    pub boundary_start_offset: usize,
    pub has_preserved_segment: bool,
    pub transcript: String,
}

#[derive(Debug, Clone)]
pub struct SessionHandle {
    pub session_id: String,
    pub session_path: PathBuf,
    pub cwd: String,
    pub model: String,
}

#[derive(Debug, Clone)]
struct LiteSessionFile {
    mtime: i64,
    size: u64,
    head: String,
    tail: String,
}

#[derive(Debug, Clone)]
struct Candidate {
    session_id: String,
    file_path: PathBuf,
    mtime: i64,
    project_path: Option<String>,
}

#[derive(Serialize)]
struct SessionStartLine<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    timestamp: &'a str,
    #[serde(rename = "sessionId")]
    session_id: &'a str,
    cwd: &'a str,
    model: &'a str,
}

#[derive(Serialize)]
struct Message<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct MessageLine<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    timestamp: &'a str,
    message: Message<'a>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LiveSessionState {
    messages: Vec<RequestMessage>,
}

pub fn projects_dir() -> PathBuf {
    get_claude_config_home_dir().join("projects")
}

pub fn sanitize_path(name: &str) -> String {
    let sanitized: String = name
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '-' })
        .collect();

    if sanitized.len() <= MAX_SANITIZED_LENGTH {
        return sanitized;
    }

    let hash = to_base36_u64(djb2_hash(name).unsigned_abs() as u64);
    format!("{}-{hash}", &sanitized[..MAX_SANITIZED_LENGTH])
}

pub fn project_dir_for(cwd: &Path) -> PathBuf {
    let normalized = canonicalize_path(cwd);
    projects_dir().join(sanitize_path(&normalized))
}

pub fn get_project_dir(project_dir: &str) -> PathBuf {
    projects_dir().join(sanitize_path(project_dir))
}

pub fn now_ms() -> i64 {
    if let Ok(value) = std::env::var("NANO_CLAUDE_FIXED_TIME_MS")
        && let Ok(parsed) = value.parse::<i64>()
    {
        return parsed;
    }

    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64
}

pub fn deterministic_session_id(cwd: &str, prompt: &str, model: &str, now_ms: i64) -> String {
    let mut hasher = Sha1::new();
    hasher.update(cwd.as_bytes());
    hasher.update([0]);
    hasher.update(prompt.as_bytes());
    hasher.update([0]);
    hasher.update(model.as_bytes());
    hasher.update([0]);
    hasher.update(now_ms.to_string().as_bytes());
    let digest = hasher.finalize();

    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest[..16]);
    bytes[6] = (bytes[6] & 0x0f) | 0x50;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;

    format!(
        "{:08x}-{:04x}-{:04x}-{:04x}-{:012x}",
        u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        u16::from_be_bytes([bytes[4], bytes[5]]),
        u16::from_be_bytes([bytes[6], bytes[7]]),
        u16::from_be_bytes([bytes[8], bytes[9]]),
        u64::from_be_bytes([
            0, 0, bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
        ]),
    )
}

pub fn write_session(
    cwd: &Path,
    model: &str,
    prompt: &str,
    outcome: &RunOutcome,
) -> Result<PersistedRun> {
    let now_ms = now_ms();
    let handle = create_session_with_now(cwd, model, prompt, now_ms)?;
    append_turn(&handle, prompt, outcome, now_ms)?;

    Ok(PersistedRun {
        report: RunReport {
            cwd: handle.cwd.clone(),
            model: handle.model.clone(),
            session_id: handle.session_id.clone(),
            session_path: handle.session_path.to_string_lossy().to_string(),
            assistant_text: outcome.assistant_text.clone(),
            tool_use: outcome.tool_use.clone(),
        },
    })
}

pub fn create_session(cwd: &Path, model: &str, seed_prompt: &str) -> Result<SessionHandle> {
    create_session_with_now(cwd, model, seed_prompt, now_ms())
}

pub fn load_session_handle(session_id: &str, dir: Option<&Path>) -> Result<Option<SessionHandle>> {
    let Some(resolved) = resolve_session_file_path(session_id, dir) else {
        return Ok(None);
    };
    let file_path = PathBuf::from(&resolved.file_path);
    let header = read_session_header(&file_path);
    let cwd = header
        .as_ref()
        .and_then(|header| header.cwd.clone())
        .or_else(|| dir.map(canonicalize_path))
        .unwrap_or_default();
    let model = header
        .and_then(|header| header.model)
        .unwrap_or_else(|| crate::core::DEFAULT_MODEL.to_string());

    Ok(Some(SessionHandle {
        session_id: session_id.to_string(),
        session_path: file_path,
        cwd,
        model,
    }))
}

pub fn append_turn(
    handle: &SessionHandle,
    prompt: &str,
    outcome: &RunOutcome,
    turn_ms: i64,
) -> Result<()> {
    let timestamp = iso_timestamp(turn_ms);
    let payload = format!(
        "{}\n{}\n",
        serde_json::to_string(&MessageLine {
            kind: "user",
            timestamp: &timestamp,
            message: Message {
                role: "user",
                content: prompt,
            },
        })?,
        serde_json::to_string(&MessageLine {
            kind: "assistant",
            timestamp: &timestamp,
            message: Message {
                role: "assistant",
                content: &outcome.assistant_text,
            },
        })?,
    );

    let mut file = OpenOptions::new()
        .append(true)
        .open(&handle.session_path)
        .with_context(|| format!("opening {}", handle.session_path.display()))?;
    file.write_all(payload.as_bytes())
        .with_context(|| format!("writing {}", handle.session_path.display()))?;
    set_session_mtime(&handle.session_path, turn_ms)?;

    if let Some(messages) = outcome.api_messages.as_ref() {
        write_live_session_state(&handle.session_path, messages)?;
    }

    Ok(())
}

pub fn load_conversation_history(session_path: &Path) -> Result<Vec<ConversationMessage>> {
    let content = fs::read_to_string(session_path)
        .with_context(|| format!("reading {}", session_path.display()))?;
    let mut messages = Vec::new();

    for line in content.lines() {
        if line.trim().is_empty() {
            continue;
        }

        let Ok(entry) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        let kind = entry.get("type").and_then(Value::as_str);
        if !matches!(kind, Some("user" | "assistant")) {
            continue;
        }

        let Some(message) = entry.get("message") else {
            continue;
        };
        let role = match message.get("role").and_then(Value::as_str) {
            Some("user") => MessageRole::User,
            Some("assistant") => MessageRole::Assistant,
            _ => continue,
        };
        let content = extract_message_content(message);
        if content.trim().is_empty() {
            continue;
        }
        messages.push(ConversationMessage { role, content });
    }

    Ok(messages)
}

pub fn load_live_request_history(session_path: &Path) -> Result<Option<Vec<RequestMessage>>> {
    let state_path = live_state_path(session_path);
    if !state_path.is_file() {
        return Ok(None);
    }

    let contents =
        fs::read(&state_path).with_context(|| format!("reading {}", state_path.display()))?;
    let state: LiveSessionState = serde_json::from_slice(&contents)
        .with_context(|| format!("parsing {}", state_path.display()))?;
    Ok(Some(state.messages))
}

pub fn build_chat_report(handle: &SessionHandle, assistant_texts: Vec<String>) -> ChatReport {
    ChatReport {
        cwd: handle.cwd.clone(),
        model: handle.model.clone(),
        session_id: handle.session_id.clone(),
        session_path: handle.session_path.to_string_lossy().to_string(),
        turns: assistant_texts.len(),
        last_assistant_text: assistant_texts.last().cloned(),
        assistant_texts,
    }
}

pub fn list_sessions(options: ListSessionsOptions) -> Result<Vec<SessionInfo>> {
    let do_stat = matches!(options.limit, Some(limit) if limit > 0) || options.offset > 0;
    let candidates = if let Some(dir) = options.dir.as_deref() {
        gather_project_candidates(dir, do_stat)
    } else {
        gather_all_candidates(do_stat)
    };

    if !do_stat {
        return Ok(read_all_and_sort(candidates));
    }

    Ok(apply_sort_and_limit(
        candidates,
        options.limit,
        options.offset,
    ))
}

pub fn get_session_info(session_id: &str, dir: Option<&Path>) -> Result<Option<SessionInfo>> {
    let Some(resolved) = resolve_session_file_path(session_id, dir) else {
        return Ok(None);
    };

    let lite = read_session_lite(Path::new(&resolved.file_path));
    Ok(lite.and_then(|lite| {
        parse_session_info_from_lite(session_id, &lite, resolved.project_path.as_deref())
    }))
}

pub fn resolve_session_file_path(
    session_id: &str,
    dir: Option<&Path>,
) -> Option<ResolvedSessionFile> {
    let file_name = format!("{session_id}.jsonl");

    if dir.is_some() && !validate_uuid(session_id) {
        return None;
    }

    if let Some(dir) = dir {
        let canonical = canonicalize_path(dir);
        if let Some(project_dir) = find_project_dir(&canonical) {
            let file_path = project_dir.join(&file_name);
            if let Some(file_size) = existing_non_empty_file_size(&file_path) {
                return Some(ResolvedSessionFile {
                    file_path: file_path.to_string_lossy().to_string(),
                    project_path: Some(canonical),
                    file_size,
                });
            }
        }
        return None;
    }

    let Ok(entries) = fs::read_dir(projects_dir()) else {
        return None;
    };
    for entry in entries.flatten() {
        let file_path = entry.path().join(&file_name);
        if let Some(file_size) = existing_non_empty_file_size(&file_path) {
            return Some(ResolvedSessionFile {
                file_path: file_path.to_string_lossy().to_string(),
                project_path: None,
                file_size,
            });
        }
    }

    None
}

pub fn get_session_transcript_scan(
    session_id: &str,
    dir: Option<&Path>,
) -> Result<Option<SessionTranscriptScan>> {
    let Some(resolved) = resolve_session_file_path(session_id, dir) else {
        return Ok(None);
    };
    let scan = read_transcript_for_load(Path::new(&resolved.file_path))?;
    Ok(Some(SessionTranscriptScan {
        session_id: session_id.to_string(),
        file_path: resolved.file_path,
        project_path: resolved.project_path,
        file_size: resolved.file_size,
        boundary_start_offset: scan.boundary_start_offset,
        has_preserved_segment: scan.has_preserved_segment,
        transcript: scan.transcript,
    }))
}

pub fn djb2_hash(input: &str) -> i32 {
    let mut hash = 0_i32;
    for value in input.chars() {
        hash = ((hash << 5).wrapping_sub(hash)).wrapping_add(value as i32);
    }
    hash
}

pub fn extract_json_string_field(text: &str, key: &str) -> Option<String> {
    for pattern in [format!("\"{key}\":\""), format!("\"{key}\": \"")] {
        if let Some(value) = extract_with_pattern(text, &pattern) {
            return Some(value);
        }
    }

    None
}

pub fn extract_last_json_string_field(text: &str, key: &str) -> Option<String> {
    let mut last_value = None;
    for pattern in [format!("\"{key}\":\""), format!("\"{key}\": \"")] {
        let mut search_from = 0;
        while let Some(idx) = text[search_from..].find(&pattern) {
            let value_start = search_from + idx + pattern.len();
            if let Some((value, end)) = scan_json_string(text, value_start) {
                last_value = Some(value);
                search_from = end;
            } else {
                break;
            }
        }
    }

    last_value
}

pub fn extract_first_prompt_from_head(head: &str) -> String {
    let mut command_fallback = String::new();

    for line in head.lines() {
        if !line.contains("\"type\":\"user\"") && !line.contains("\"type\": \"user\"") {
            continue;
        }
        if line.contains("\"tool_result\"") {
            continue;
        }
        if line.contains("\"isMeta\":true") || line.contains("\"isMeta\": true") {
            continue;
        }
        if line.contains("\"isCompactSummary\":true") || line.contains("\"isCompactSummary\": true")
        {
            continue;
        }

        let Ok(entry) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if entry.get("type").and_then(Value::as_str) != Some("user") {
            continue;
        }

        let Some(message) = entry.get("message") else {
            continue;
        };
        let mut texts = Vec::new();
        match message.get("content") {
            Some(Value::String(value)) => texts.push(value.clone()),
            Some(Value::Array(blocks)) => {
                for block in blocks {
                    if block.get("type").and_then(Value::as_str) == Some("text")
                        && let Some(text) = block.get("text").and_then(Value::as_str)
                    {
                        texts.push(text.to_string());
                    }
                }
            }
            _ => {}
        }

        for raw in texts {
            let mut result = raw.replace('\n', " ").trim().to_string();
            if result.is_empty() {
                continue;
            }

            if let Some(command_name) = extract_tag(&result, "command-name") {
                if command_fallback.is_empty() {
                    command_fallback = command_name;
                }
                continue;
            }

            if let Some(bash_input) = extract_tag(&result, "bash-input") {
                return format!("! {}", bash_input.trim());
            }

            if should_skip_first_prompt(&result) {
                continue;
            }

            if result.chars().count() > 200 {
                result = truncate_chars(&result, 200);
                result.push('…');
            }

            return result;
        }
    }

    command_fallback
}

fn canonicalize_path(dir: &Path) -> String {
    match fs::canonicalize(dir) {
        Ok(path) => normalize_nfc(&path.to_string_lossy()),
        Err(_) => normalize_nfc(&dir.to_string_lossy()),
    }
}

fn create_session_with_now(
    cwd: &Path,
    model: &str,
    seed_prompt: &str,
    created_at_ms: i64,
) -> Result<SessionHandle> {
    let cwd_str = normalize_nfc(&cwd.to_string_lossy());
    let session_id = deterministic_session_id(&cwd_str, seed_prompt, model, created_at_ms);
    let project_dir = project_dir_for(cwd);
    fs::create_dir_all(&project_dir)
        .with_context(|| format!("creating {}", project_dir.display()))?;
    let session_path = project_dir.join(format!("{session_id}.jsonl"));
    let timestamp = iso_timestamp(created_at_ms);
    let payload = format!(
        "{}\n",
        serde_json::to_string(&SessionStartLine {
            kind: "system",
            timestamp: &timestamp,
            session_id: &session_id,
            cwd: &cwd_str,
            model,
        })?
    );
    fs::write(&session_path, payload)
        .with_context(|| format!("writing {}", session_path.display()))?;
    set_session_mtime(&session_path, created_at_ms)?;

    Ok(SessionHandle {
        session_id,
        session_path,
        cwd: cwd_str,
        model: model.to_string(),
    })
}

fn write_live_session_state(session_path: &Path, messages: &[RequestMessage]) -> Result<()> {
    let state = LiveSessionState {
        messages: messages.to_vec(),
    };
    let mut payload = serde_json::to_vec_pretty(&state)?;
    payload.push(b'\n');
    fs::write(live_state_path(session_path), payload)
        .with_context(|| format!("writing {}", live_state_path(session_path).display()))
}

fn live_state_path(session_path: &Path) -> PathBuf {
    let mut path = session_path.to_path_buf();
    let base_name = session_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("session.jsonl");
    path.set_file_name(format!("{base_name}.live.json"));
    path
}

fn set_session_mtime(session_path: &Path, now_ms: i64) -> Result<()> {
    set_file_mtime(
        session_path,
        FileTime::from_unix_time(now_ms / 1_000, ((now_ms % 1_000) * 1_000_000) as u32),
    )
    .with_context(|| format!("setting mtime for {}", session_path.display()))
}

#[derive(Debug, Clone)]
struct SessionHeader {
    cwd: Option<String>,
    model: Option<String>,
}

fn read_session_header(file_path: &Path) -> Option<SessionHeader> {
    let lite = read_session_lite(file_path)?;
    Some(SessionHeader {
        cwd: extract_json_string_field(&lite.head, "cwd"),
        model: extract_json_string_field(&lite.head, "model"),
    })
}

fn find_project_dir(project_path: &str) -> Option<PathBuf> {
    let exact = get_project_dir(project_path);
    if exact.is_dir() {
        return Some(exact);
    }

    let sanitized = sanitize_path(project_path);
    if sanitized.len() <= MAX_SANITIZED_LENGTH {
        return None;
    }

    let prefix = &sanitized[..MAX_SANITIZED_LENGTH];
    let Ok(entries) = fs::read_dir(projects_dir()) else {
        return None;
    };
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if file_type.is_dir() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with(&format!("{prefix}-")) {
                return Some(entry.path());
            }
        }
    }

    None
}

fn gather_project_candidates(dir: &Path, do_stat: bool) -> Vec<Candidate> {
    let canonical_dir = canonicalize_path(dir);
    let Some(project_dir) = find_project_dir(&canonical_dir) else {
        return Vec::new();
    };
    list_candidates(&project_dir, do_stat, Some(canonical_dir))
}

fn gather_all_candidates(do_stat: bool) -> Vec<Candidate> {
    let Ok(entries) = fs::read_dir(projects_dir()) else {
        return Vec::new();
    };

    let mut all = Vec::new();
    for entry in entries.flatten() {
        let Ok(file_type) = entry.file_type() else {
            continue;
        };
        if !file_type.is_dir() {
            continue;
        }
        all.extend(list_candidates(&entry.path(), do_stat, None));
    }
    all
}

fn list_candidates(
    project_dir: &Path,
    do_stat: bool,
    project_path: Option<String>,
) -> Vec<Candidate> {
    let Ok(entries) = fs::read_dir(project_dir) else {
        return Vec::new();
    };

    let mut results = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|value| value.to_str()) != Some("jsonl") {
            continue;
        }

        let Some(session_id) = path
            .file_stem()
            .and_then(|value| value.to_str())
            .map(ToString::to_string)
        else {
            continue;
        };
        if !validate_uuid(&session_id) {
            continue;
        }

        let mtime = if do_stat {
            let Ok(metadata) = fs::metadata(&path) else {
                continue;
            };
            modified_ms(&metadata).unwrap_or(0)
        } else {
            0
        };

        results.push(Candidate {
            session_id,
            file_path: path,
            mtime,
            project_path: project_path.clone(),
        });
    }

    results
}

fn apply_sort_and_limit(
    mut candidates: Vec<Candidate>,
    limit: Option<usize>,
    offset: usize,
) -> Vec<SessionInfo> {
    candidates.sort_by(compare_desc);

    let want = match limit {
        Some(0) | None => usize::MAX,
        Some(limit) => limit,
    };
    let mut sessions = Vec::new();
    let mut skipped = 0;
    let mut seen = HashSet::new();
    let mut index = 0;

    while index < candidates.len() && sessions.len() < want {
        let batch_end = (index + READ_BATCH_SIZE).min(candidates.len());
        let batch = &candidates[index..batch_end];
        let results: Vec<Option<SessionInfo>> = batch.iter().map(read_candidate).collect();
        index = batch_end;

        for result in results.into_iter().flatten() {
            if !seen.insert(result.session_id.clone()) {
                continue;
            }
            if skipped < offset {
                skipped += 1;
                continue;
            }
            sessions.push(result);
            if sessions.len() >= want {
                break;
            }
        }
    }

    sessions
}

fn read_all_and_sort(candidates: Vec<Candidate>) -> Vec<SessionInfo> {
    let mut by_id = HashMap::new();
    for session in candidates.iter().filter_map(read_candidate) {
        let existing = by_id.get(&session.session_id);
        if existing
            .map(|existing: &SessionInfo| session.last_modified > existing.last_modified)
            .unwrap_or(true)
        {
            by_id.insert(session.session_id.clone(), session);
        }
    }

    let mut sessions: Vec<SessionInfo> = by_id.into_values().collect();
    sessions.sort_by(|a, b| {
        b.last_modified
            .cmp(&a.last_modified)
            .then_with(|| b.session_id.cmp(&a.session_id))
    });
    sessions
}

fn compare_desc(a: &Candidate, b: &Candidate) -> std::cmp::Ordering {
    b.mtime
        .cmp(&a.mtime)
        .then_with(|| b.session_id.cmp(&a.session_id))
}

fn read_candidate(candidate: &Candidate) -> Option<SessionInfo> {
    let lite = read_session_lite(&candidate.file_path)?;
    let mut info = parse_session_info_from_lite(
        &candidate.session_id,
        &lite,
        candidate.project_path.as_deref(),
    )?;
    if candidate.mtime != 0 {
        info.last_modified = candidate.mtime;
    }
    Some(info)
}

fn read_session_lite(file_path: &Path) -> Option<LiteSessionFile> {
    let mut file = File::open(file_path).ok()?;
    let metadata = file.metadata().ok()?;
    let mut buf = vec![0_u8; LITE_READ_BUF_SIZE];
    let head_bytes = file.read(&mut buf).ok()?;
    if head_bytes == 0 {
        return None;
    }

    let head = String::from_utf8_lossy(&buf[..head_bytes]).to_string();
    let tail_offset = metadata.len().saturating_sub(LITE_READ_BUF_SIZE as u64);
    let tail = if tail_offset > 0 {
        file.seek(SeekFrom::Start(tail_offset)).ok()?;
        let tail_bytes = file.read(&mut buf).ok()?;
        String::from_utf8_lossy(&buf[..tail_bytes]).to_string()
    } else {
        head.clone()
    };

    Some(LiteSessionFile {
        mtime: modified_ms(&metadata).unwrap_or(0),
        size: metadata.len(),
        head,
        tail,
    })
}

fn parse_session_info_from_lite(
    session_id: &str,
    lite: &LiteSessionFile,
    project_path: Option<&str>,
) -> Option<SessionInfo> {
    let first_line = lite.head.lines().next().unwrap_or_default();
    if first_line.contains("\"isSidechain\":true") || first_line.contains("\"isSidechain\": true") {
        return None;
    }

    let custom_title = extract_last_json_string_field(&lite.tail, "customTitle")
        .or_else(|| extract_last_json_string_field(&lite.head, "customTitle"))
        .or_else(|| extract_last_json_string_field(&lite.tail, "aiTitle"))
        .or_else(|| extract_last_json_string_field(&lite.head, "aiTitle"));
    let first_prompt = {
        let value = extract_first_prompt_from_head(&lite.head);
        if value.is_empty() { None } else { Some(value) }
    };
    let created_at = extract_json_string_field(&lite.head, "timestamp").and_then(|timestamp| {
        OffsetDateTime::parse(&timestamp, &Rfc3339)
            .ok()
            .map(|value| (value.unix_timestamp_nanos() / 1_000_000) as i64)
    });
    let summary = custom_title
        .clone()
        .or_else(|| extract_last_json_string_field(&lite.tail, "lastPrompt"))
        .or_else(|| extract_last_json_string_field(&lite.tail, "summary"))
        .or_else(|| first_prompt.clone())?;
    let git_branch = extract_last_json_string_field(&lite.tail, "gitBranch")
        .or_else(|| extract_json_string_field(&lite.head, "gitBranch"));
    let session_cwd = extract_json_string_field(&lite.head, "cwd")
        .or_else(|| project_path.map(ToString::to_string));
    let tag = lite
        .tail
        .split('\n')
        .rev()
        .find(|line| line.starts_with("{\"type\":\"tag\""))
        .and_then(|line| extract_last_json_string_field(line, "tag"));

    Some(SessionInfo {
        session_id: session_id.to_string(),
        summary,
        last_modified: lite.mtime,
        file_size: Some(lite.size),
        custom_title,
        first_prompt,
        git_branch,
        cwd: session_cwd,
        tag,
        created_at,
    })
}

fn extract_message_content(message: &Value) -> String {
    match message.get("content") {
        Some(Value::String(text)) => text.to_string(),
        Some(Value::Array(blocks)) => {
            let mut text = String::new();
            for block in blocks {
                if block.get("type").and_then(Value::as_str) == Some("text")
                    && let Some(value) = block.get("text").and_then(Value::as_str)
                {
                    text.push_str(value);
                }
            }
            text
        }
        _ => String::new(),
    }
}

fn modified_ms(metadata: &fs::Metadata) -> Option<i64> {
    metadata
        .modified()
        .ok()
        .and_then(|value| value.duration_since(UNIX_EPOCH).ok())
        .map(|value| value.as_millis() as i64)
}

fn existing_non_empty_file_size(file_path: &Path) -> Option<u64> {
    fs::metadata(file_path)
        .ok()
        .map(|metadata| metadata.len())
        .filter(|size| *size > 0)
}

pub fn read_transcript_for_load(file_path: &Path) -> Result<TranscriptScan> {
    let buf = fs::read(file_path).with_context(|| format!("reading {}", file_path.display()))?;
    let mut out = Vec::with_capacity(buf.len());
    let mut boundary_start_offset = 0_usize;
    let mut has_preserved_segment = false;
    let mut last_snapshot: Option<Vec<u8>> = None;
    let mut offset = 0_usize;

    for line in split_lines_inclusive(&buf) {
        if has_prefix(line, br#"{"type":"attribution-snapshot""#) {
            last_snapshot = Some(line.to_vec());
            offset += line.len();
            continue;
        }

        if line
            .windows(br#""compact_boundary""#.len())
            .any(|window| window == br#""compact_boundary""#)
        {
            match parse_boundary_line(strip_trailing_lf(line)) {
                Some(true) => {
                    has_preserved_segment = true;
                }
                Some(false) => {
                    out.clear();
                    boundary_start_offset = offset;
                    has_preserved_segment = false;
                    last_snapshot = None;
                }
                None => {}
            }
        }

        out.extend_from_slice(line);
        offset += line.len();
    }

    if let Some(last_snapshot) = last_snapshot {
        if !out.is_empty() && out.last().copied() != Some(b'\n') {
            out.push(b'\n');
        }
        out.extend_from_slice(&last_snapshot);
    }

    Ok(TranscriptScan {
        boundary_start_offset,
        has_preserved_segment,
        transcript: String::from_utf8_lossy(&out).to_string(),
    })
}

fn split_lines_inclusive(buf: &[u8]) -> Vec<&[u8]> {
    if buf.is_empty() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let mut start = 0;
    for (idx, byte) in buf.iter().enumerate() {
        if *byte == b'\n' {
            lines.push(&buf[start..=idx]);
            start = idx + 1;
        }
    }
    if start < buf.len() {
        lines.push(&buf[start..]);
    }
    lines
}

fn has_prefix(value: &[u8], prefix: &[u8]) -> bool {
    value.len() >= prefix.len() && &value[..prefix.len()] == prefix
}

fn strip_trailing_lf(line: &[u8]) -> &[u8] {
    if line.last().copied() == Some(b'\n') {
        &line[..line.len() - 1]
    } else {
        line
    }
}

fn parse_boundary_line(line: &[u8]) -> Option<bool> {
    let parsed = serde_json::from_slice::<Value>(line).ok()?;
    if parsed.get("type").and_then(Value::as_str) != Some("system")
        || parsed.get("subtype").and_then(Value::as_str) != Some("compact_boundary")
    {
        return None;
    }
    Some(
        parsed
            .get("compactMetadata")
            .and_then(|value| value.get("preservedSegment"))
            .map(Value::is_object)
            .unwrap_or(false),
    )
}

fn validate_uuid(candidate: &str) -> bool {
    let bytes = candidate.as_bytes();
    if bytes.len() != 36 {
        return false;
    }

    for (idx, byte) in bytes.iter().enumerate() {
        let is_dash = matches!(idx, 8 | 13 | 18 | 23);
        if is_dash {
            if *byte != b'-' {
                return false;
            }
            continue;
        }

        if !byte.is_ascii_hexdigit() {
            return false;
        }
    }

    true
}

fn extract_with_pattern(text: &str, pattern: &str) -> Option<String> {
    let idx = text.find(pattern)?;
    let value_start = idx + pattern.len();
    scan_json_string(text, value_start).map(|(value, _)| value)
}

fn scan_json_string(text: &str, value_start: usize) -> Option<(String, usize)> {
    let bytes = text.as_bytes();
    let mut idx = value_start;
    while idx < bytes.len() {
        if bytes[idx] == b'\\' {
            idx += 2;
            continue;
        }
        if bytes[idx] == b'"' {
            let raw = &text[value_start..idx];
            let value = if raw.contains('\\') {
                serde_json::from_str::<String>(&format!("\"{raw}\""))
                    .unwrap_or_else(|_| raw.to_string())
            } else {
                raw.to_string()
            };
            return Some((value, idx + 1));
        }
        idx += 1;
    }

    None
}

fn extract_tag(input: &str, tag: &str) -> Option<String> {
    let start_tag = format!("<{tag}>");
    let end_tag = format!("</{tag}>");
    let start = input.find(&start_tag)? + start_tag.len();
    let end = input[start..].find(&end_tag)? + start;
    Some(input[start..end].to_string())
}

fn should_skip_first_prompt(value: &str) -> bool {
    let trimmed = value.trim_start();
    if trimmed.starts_with("[Request interrupted by user") {
        return true;
    }

    if let Some(rest) = trimmed.strip_prefix('<') {
        let mut chars = rest.chars();
        if let Some(first) = chars.next()
            && first.is_ascii_lowercase()
        {
            match chars.next() {
                Some(ch)
                    if ch.is_ascii_alphanumeric()
                        || ch == '-'
                        || ch.is_whitespace()
                        || ch == '>' =>
                {
                    return true;
                }
                Some(_) | None => {}
            }
        }
    }

    false
}

fn truncate_chars(value: &str, count: usize) -> String {
    value
        .chars()
        .take(count)
        .collect::<String>()
        .trim()
        .to_string()
}

fn iso_timestamp(now_ms: i64) -> String {
    const ISO_8601_MILLIS: &[time::format_description::FormatItem<'static>] =
        format_description!("[year]-[month]-[day]T[hour]:[minute]:[second].[subsecond digits:3]Z");

    OffsetDateTime::from_unix_timestamp_nanos(now_ms as i128 * 1_000_000)
        .unwrap_or(OffsetDateTime::UNIX_EPOCH)
        .format(ISO_8601_MILLIS)
        .unwrap_or_else(|_| "1970-01-01T00:00:00.000Z".to_string())
}

fn to_base36_u64(mut value: u64) -> String {
    if value == 0 {
        return "0".to_string();
    }

    let mut chars = Vec::new();
    while value > 0 {
        let digit = (value % 36) as u8;
        chars.push(match digit {
            0..=9 => (b'0' + digit) as char,
            _ => (b'a' + (digit - 10)) as char,
        });
        value /= 36;
    }
    chars.iter().rev().collect()
}
