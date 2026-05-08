use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

pub const API_ERROR_MESSAGE_PREFIX: &str = "API Error";
pub const API_TIMEOUT_ERROR_MESSAGE: &str = "Request timed out";
pub const PROMPT_TOO_LONG_ERROR_MESSAGE: &str = "Prompt is too long";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RequestBuildInput {
    pub session_id: String,
    pub user_agent: String,
    #[serde(default)]
    pub custom_headers: Option<String>,
    #[serde(default)]
    pub container_id: Option<String>,
    #[serde(default)]
    pub remote_session_id: Option<String>,
    #[serde(default)]
    pub client_app: Option<String>,
    #[serde(default)]
    pub authorization: Option<String>,
    #[serde(default)]
    pub additional_protection: bool,
    pub body: RequestBodyInput,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct RequestBodyInput {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<RequestMessage>,
    #[serde(default)]
    pub system: Vec<String>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub betas: Vec<String>,
    #[serde(default)]
    pub extra_body: Option<Value>,
    #[serde(default)]
    pub tool_choice: Option<Value>,
    #[serde(default)]
    pub metadata: Option<Value>,
    #[serde(default)]
    pub thinking: Option<Value>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub output_config: Option<Value>,
    pub max_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestMessage {
    pub role: String,
    pub content: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BuiltRequest {
    pub headers: BTreeMap<String, String>,
    pub body: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ServerToolUseUsage {
    pub web_search_requests: u64,
    pub web_fetch_requests: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CacheCreationUsage {
    pub ephemeral_1h_input_tokens: u64,
    pub ephemeral_5m_input_tokens: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Usage {
    pub input_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub output_tokens: u64,
    pub server_tool_use: ServerToolUseUsage,
    pub service_tier: String,
    pub cache_creation: CacheCreationUsage,
    pub inference_geo: String,
    pub iterations: Vec<Value>,
    pub speed: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct AssistantOutput {
    pub content: Vec<Value>,
    #[serde(default, rename = "stopReason")]
    pub stop_reason: Option<String>,
    pub usage: Usage,
    #[serde(default, rename = "requestId", skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ErrorOutput {
    pub content: String,
    pub error: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_error: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_details: Option<String>,
    pub is_api_error_message: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamEmission {
    Assistant {
        message: AssistantOutput,
    },
    Error {
        message: ErrorOutput,
    },
    StreamEvent {
        event: Value,
        #[serde(default, rename = "ttftMs", skip_serializing_if = "Option::is_none")]
        ttft_ms: Option<i64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StreamRunOutput {
    pub emissions: Vec<StreamEmission>,
}

pub fn empty_usage() -> Usage {
    Usage {
        input_tokens: 0,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
        output_tokens: 0,
        server_tool_use: ServerToolUseUsage {
            web_search_requests: 0,
            web_fetch_requests: 0,
        },
        service_tier: "standard".to_string(),
        cache_creation: CacheCreationUsage {
            ephemeral_1h_input_tokens: 0,
            ephemeral_5m_input_tokens: 0,
        },
        inference_geo: String::new(),
        iterations: Vec::new(),
        speed: "standard".to_string(),
    }
}

pub fn create_error_output(
    content: impl Into<String>,
    error: impl Into<String>,
    api_error: Option<&str>,
    error_details: Option<String>,
) -> ErrorOutput {
    ErrorOutput {
        content: content.into(),
        error: error.into(),
        api_error: api_error.map(ToString::to_string),
        error_details,
        is_api_error_message: true,
    }
}

pub fn safe_parse_json(input: &str) -> Option<Value> {
    if input.is_empty() {
        return None;
    }
    serde_json::from_str(input).ok()
}

pub fn extract_json_string_field(text: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{key}\"");
    let start = text.find(&pattern)?;
    let after_key = &text[start + pattern.len()..];
    let colon = after_key.find(':')?;
    let mut chars = after_key[colon + 1..].chars().peekable();
    while matches!(chars.peek(), Some(ch) if ch.is_whitespace()) {
        chars.next();
    }
    if chars.next()? != '"' {
        return None;
    }

    let mut value = String::new();
    let mut escaped = false;
    for ch in chars {
        if escaped {
            value.push(match ch {
                '"' => '"',
                '\\' => '\\',
                '/' => '/',
                'b' => '\u{0008}',
                'f' => '\u{000c}',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                other => other,
            });
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == '"' {
            return Some(value);
        }
        value.push(ch);
    }

    None
}

pub fn update_usage(mut usage: Usage, part_usage: Option<&Value>) -> Usage {
    let Some(part_usage) = part_usage else {
        return usage;
    };

    if let Some(value) = positive_u64(part_usage.get("input_tokens")) {
        usage.input_tokens = value;
    }
    if let Some(value) = positive_u64(part_usage.get("cache_creation_input_tokens")) {
        usage.cache_creation_input_tokens = value;
    }
    if let Some(value) = positive_u64(part_usage.get("cache_read_input_tokens")) {
        usage.cache_read_input_tokens = value;
    }
    if let Some(value) = part_usage.get("output_tokens").and_then(Value::as_u64) {
        usage.output_tokens = value;
    }

    if let Some(server_tool_use) = part_usage.get("server_tool_use").and_then(Value::as_object) {
        if let Some(value) = server_tool_use
            .get("web_search_requests")
            .and_then(Value::as_u64)
        {
            usage.server_tool_use.web_search_requests = value;
        }
        if let Some(value) = server_tool_use
            .get("web_fetch_requests")
            .and_then(Value::as_u64)
        {
            usage.server_tool_use.web_fetch_requests = value;
        }
    }

    if let Some(cache_creation) = part_usage.get("cache_creation").and_then(Value::as_object) {
        if let Some(value) = cache_creation
            .get("ephemeral_1h_input_tokens")
            .and_then(Value::as_u64)
        {
            usage.cache_creation.ephemeral_1h_input_tokens = value;
        }
        if let Some(value) = cache_creation
            .get("ephemeral_5m_input_tokens")
            .and_then(Value::as_u64)
        {
            usage.cache_creation.ephemeral_5m_input_tokens = value;
        }
    }

    if let Some(service_tier) = part_usage.get("service_tier").and_then(Value::as_str)
        && !service_tier.is_empty()
    {
        usage.service_tier = service_tier.to_string();
    }
    if let Some(inference_geo) = part_usage.get("inference_geo").and_then(Value::as_str) {
        usage.inference_geo = inference_geo.to_string();
    }
    if let Some(iterations) = part_usage.get("iterations").and_then(Value::as_array) {
        usage.iterations = iterations.clone();
    }
    if let Some(speed) = part_usage.get("speed").and_then(Value::as_str)
        && !speed.is_empty()
    {
        usage.speed = speed.to_string();
    }

    usage
}

pub fn normalize_content_block(block: &Value) -> anyhow::Result<Value> {
    let Some(kind) = block.get("type").and_then(Value::as_str) else {
        return Ok(block.clone());
    };

    match kind {
        "tool_use" => {
            let mut normalized = object_clone(block)?;
            let input = normalized.remove("input").unwrap_or(Value::Null);
            let normalized_input = match input {
                Value::String(text) => safe_parse_json(&text).unwrap_or_else(|| json!({})),
                Value::Object(_) => input,
                Value::Null => json!({}),
                _ => anyhow::bail!("Tool use input must be a string or object"),
            };
            normalized.insert("input".to_string(), normalized_input);
            Ok(Value::Object(normalized))
        }
        "server_tool_use" => {
            let mut normalized = object_clone(block)?;
            let input = normalized.remove("input").unwrap_or(Value::Null);
            let normalized_input = match input {
                Value::String(text) => safe_parse_json(&text).unwrap_or_else(|| json!({})),
                Value::Object(_) => input,
                Value::Null => json!({}),
                other => other,
            };
            normalized.insert("input".to_string(), normalized_input);
            Ok(Value::Object(normalized))
        }
        _ => Ok(block.clone()),
    }
}

pub fn build_request(input: &RequestBuildInput) -> BuiltRequest {
    let mut headers = BTreeMap::from([
        ("x-app".to_string(), "cli".to_string()),
        ("User-Agent".to_string(), input.user_agent.clone()),
        (
            "X-Claude-Code-Session-Id".to_string(),
            input.session_id.clone(),
        ),
    ]);

    for (name, value) in parse_custom_headers(input.custom_headers.as_deref()) {
        headers.insert(name, value);
    }
    if let Some(container_id) = &input.container_id {
        headers.insert(
            "x-claude-remote-container-id".to_string(),
            container_id.clone(),
        );
    }
    if let Some(remote_session_id) = &input.remote_session_id {
        headers.insert(
            "x-claude-remote-session-id".to_string(),
            remote_session_id.clone(),
        );
    }
    if let Some(client_app) = &input.client_app {
        headers.insert("x-client-app".to_string(), client_app.clone());
    }
    if input.additional_protection {
        headers.insert(
            "x-anthropic-additional-protection".to_string(),
            "true".to_string(),
        );
    }
    if let Some(authorization) = &input.authorization {
        headers.insert("Authorization".to_string(), authorization.clone());
    }

    let mut body = match input.body.extra_body.as_ref().and_then(Value::as_object) {
        Some(extra) => extra.clone(),
        None => Map::new(),
    };

    merge_beta_headers(&mut body, &input.body.betas);
    merge_output_config(&mut body, input.body.output_config.as_ref());

    body.insert("model".to_string(), Value::String(input.body.model.clone()));
    body.insert(
        "messages".to_string(),
        Value::Array(
            input
                .body
                .messages
                .iter()
                .map(|message| json!({ "role": message.role, "content": message.content }))
                .collect(),
        ),
    );
    body.insert("max_tokens".to_string(), Value::from(input.body.max_tokens));
    body.insert("stream".to_string(), Value::Bool(true));

    if !input.body.system.is_empty() {
        body.insert(
            "system".to_string(),
            Value::Array(
                input
                    .body
                    .system
                    .iter()
                    .map(|text| json!({ "type": "text", "text": text }))
                    .collect(),
            ),
        );
    }
    if !input.body.tools.is_empty() {
        body.insert("tools".to_string(), Value::Array(input.body.tools.clone()));
    }
    if let Some(tool_choice) = &input.body.tool_choice {
        body.insert("tool_choice".to_string(), tool_choice.clone());
    }
    if let Some(metadata) = &input.body.metadata {
        body.insert("metadata".to_string(), metadata.clone());
    }
    if let Some(thinking) = &input.body.thinking {
        body.insert("thinking".to_string(), thinking.clone());
    }
    if let Some(temperature) = input.body.temperature {
        body.insert("temperature".to_string(), Value::from(temperature));
    }

    BuiltRequest {
        headers,
        body: Value::Object(body),
    }
}

fn positive_u64(value: Option<&Value>) -> Option<u64> {
    match value.and_then(Value::as_u64) {
        Some(0) | None => None,
        Some(value) => Some(value),
    }
}

fn parse_custom_headers(raw: Option<&str>) -> Vec<(String, String)> {
    let Some(raw) = raw else {
        return Vec::new();
    };

    raw.lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                return None;
            }
            let (name, value) = trimmed.split_once(':')?;
            let name = name.trim();
            if name.is_empty() {
                return None;
            }
            Some((name.to_string(), value.trim().to_string()))
        })
        .collect()
}

fn merge_beta_headers(body: &mut Map<String, Value>, betas: &[String]) {
    if betas.is_empty() {
        return;
    }

    match body.get_mut("anthropic_beta") {
        Some(Value::Array(existing)) => {
            for beta in betas {
                if !existing
                    .iter()
                    .any(|value| value.as_str() == Some(beta.as_str()))
                {
                    existing.push(Value::String(beta.clone()));
                }
            }
        }
        _ => {
            body.insert(
                "anthropic_beta".to_string(),
                Value::Array(betas.iter().cloned().map(Value::String).collect()),
            );
        }
    }
}

fn merge_output_config(body: &mut Map<String, Value>, output_config: Option<&Value>) {
    let Some(Value::Object(request_output_config)) = output_config else {
        return;
    };

    let mut merged = match body.get("output_config").and_then(Value::as_object) {
        Some(existing) => existing.clone(),
        None => Map::new(),
    };
    for (key, value) in request_output_config {
        merged.insert(key.clone(), value.clone());
    }
    body.insert("output_config".to_string(), Value::Object(merged));
}

fn object_clone(value: &Value) -> anyhow::Result<Map<String, Value>> {
    value
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("expected object"))
}
