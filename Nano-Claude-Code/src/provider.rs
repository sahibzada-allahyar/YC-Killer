use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use reqwest::blocking::Client;
use reqwest::header::{ACCEPT, CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::anthropic::{
    API_ERROR_MESSAGE_PREFIX, API_TIMEOUT_ERROR_MESSAGE, AssistantOutput, ErrorOutput,
    PROMPT_TOO_LONG_ERROR_MESSAGE, RequestBodyInput, RequestBuildInput, RequestMessage,
    StreamEmission, StreamRunOutput, build_request, create_error_output, empty_usage,
    extract_json_string_field, normalize_content_block, update_usage,
};
use crate::message_protocol::parse_sse_events;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct StreamProcessInput {
    pub sse: String,
    pub model: String,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub non_interactive: bool,
    #[serde(default)]
    pub max_output_tokens: Option<u64>,
    #[serde(default)]
    pub ttft_ms: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ProviderErrorInput {
    pub kind: String,
    pub model: String,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub status: Option<u16>,
    #[serde(default)]
    pub non_interactive: bool,
    #[serde(default = "default_api_provider")]
    pub api_provider: String,
    #[serde(default)]
    pub external_api_key: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LiveCompletionInput {
    pub session_id: String,
    pub user_agent: String,
    pub api_key: String,
    pub model: String,
    #[serde(default = "default_base_url")]
    pub base_url: String,
    #[serde(default)]
    pub messages: Vec<RequestMessage>,
    #[serde(default)]
    pub system: Vec<String>,
    #[serde(default)]
    pub tools: Vec<Value>,
    #[serde(default)]
    pub max_tokens: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LiveToolUse {
    pub id: String,
    pub name: String,
    pub input: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct LiveCompletionOutput {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub content: Vec<Value>,
    pub assistant_text: String,
    #[serde(default)]
    pub tool_uses: Vec<LiveToolUse>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<ErrorOutput>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
}

pub fn process_sse_stream(input: &StreamProcessInput) -> Result<StreamRunOutput> {
    let events = parse_sse_events(&input.sse)?;
    let mut emissions = Vec::new();
    let mut partial_message_seen = false;
    let mut usage = empty_usage();
    let mut stop_reason: Option<String> = None;
    let mut content_blocks: Vec<Option<Value>> = Vec::new();
    let mut last_assistant_emission: Option<usize> = None;
    let mut emitted_assistants = 0usize;

    for event in events {
        match event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "message_start" => {
                partial_message_seen = true;
                usage = update_usage(
                    usage,
                    event
                        .get("message")
                        .and_then(|message| message.get("usage")),
                );
            }
            "content_block_start" => {
                let index = event_index(&event)?;
                let content_block = event
                    .get("content_block")
                    .ok_or_else(|| anyhow!("missing content_block"))?;
                let initialized = initialize_content_block(content_block)?;
                ensure_slot(&mut content_blocks, index);
                content_blocks[index] = Some(initialized);
            }
            "content_block_delta" => {
                let index = event_index(&event)?;
                let delta = event.get("delta").ok_or_else(|| anyhow!("missing delta"))?;
                let content_block = content_blocks
                    .get_mut(index)
                    .and_then(Option::as_mut)
                    .ok_or_else(|| anyhow!("Content block not found"))?;
                apply_delta(content_block, delta)?;
            }
            "content_block_stop" => {
                let index = event_index(&event)?;
                if !partial_message_seen {
                    bail!("Message not found");
                }
                let content_block = content_blocks
                    .get(index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| anyhow!("Content block not found"))?;
                let normalized = normalize_content_block(content_block)?;
                emissions.push(StreamEmission::Assistant {
                    message: AssistantOutput {
                        content: vec![normalized],
                        stop_reason: None,
                        usage: usage.clone(),
                        request_id: input.request_id.clone(),
                    },
                });
                last_assistant_emission = Some(emissions.len() - 1);
                emitted_assistants += 1;
            }
            "message_delta" => {
                usage = update_usage(usage, event.get("usage"));
                stop_reason = event
                    .get("delta")
                    .and_then(|delta| delta.get("stop_reason"))
                    .and_then(value_as_optional_string);

                if let Some(index) = last_assistant_emission
                    && let Some(StreamEmission::Assistant { message }) = emissions.get_mut(index)
                {
                    message.usage = usage.clone();
                    message.stop_reason = stop_reason.clone();
                }

                if let Some(stop_reason) = stop_reason.as_deref() {
                    if stop_reason == "refusal" {
                        emissions.push(StreamEmission::Error {
                            message: refusal_error(&input.model, input.non_interactive),
                        });
                    }
                    if stop_reason == "max_tokens" {
                        let max_output_tokens = input.max_output_tokens.unwrap_or_default();
                        emissions.push(StreamEmission::Error {
                            message: create_error_output(
                                format!(
                                    "{API_ERROR_MESSAGE_PREFIX}: Claude's response exceeded the {max_output_tokens} output token maximum. To configure this behavior, set the CLAUDE_CODE_MAX_OUTPUT_TOKENS environment variable."
                                ),
                                "max_output_tokens",
                                Some("max_output_tokens"),
                                None,
                            ),
                        });
                    }
                    if stop_reason == "model_context_window_exceeded" {
                        emissions.push(StreamEmission::Error {
                            message: create_error_output(
                                format!(
                                    "{API_ERROR_MESSAGE_PREFIX}: The model has reached its context window limit."
                                ),
                                "max_output_tokens",
                                Some("max_output_tokens"),
                                None,
                            ),
                        });
                    }
                }
            }
            "message_stop" | "ping" | "error" => {}
            _ => {}
        }

        emissions.push(StreamEmission::StreamEvent {
            event: event.clone(),
            ttft_ms: if event.get("type").and_then(Value::as_str) == Some("message_start") {
                Some(input.ttft_ms.unwrap_or(0))
            } else {
                None
            },
        });
    }

    if !partial_message_seen || (emitted_assistants == 0 && stop_reason.is_none()) {
        bail!("Stream ended without receiving any events");
    }

    Ok(StreamRunOutput { emissions })
}

pub fn classify_error(input: &ProviderErrorInput) -> ErrorOutput {
    let lower = input.message.to_lowercase();
    if input.kind == "connection_timeout"
        || (input.kind == "connection_error" && lower.contains("timeout"))
    {
        return create_error_output(API_TIMEOUT_ERROR_MESSAGE, "unknown", None, None);
    }

    if input
        .message
        .contains("Extra usage is required for long context")
        && input.status == Some(429)
    {
        let hint = if input.non_interactive {
            "enable extra usage at claude.ai/settings/usage, or use --model to switch to standard context"
        } else {
            "run /extra-usage to enable, or /model to switch to standard context"
        };
        return create_error_output(
            format!("{API_ERROR_MESSAGE_PREFIX}: Extra usage is required for 1M context · {hint}"),
            "rate_limit",
            None,
            None,
        );
    }

    if input.status == Some(429) {
        let stripped = input
            .message
            .strip_prefix("429 ")
            .unwrap_or(input.message.as_str());
        let detail = extract_json_string_field(stripped, "message")
            .filter(|detail| !detail.is_empty())
            .unwrap_or_else(|| stripped.to_string());
        return create_error_output(
            format!(
                "{API_ERROR_MESSAGE_PREFIX}: Request rejected (429) · {}",
                if detail.is_empty() {
                    "this may be a temporary capacity issue — check status.anthropic.com"
                } else {
                    detail.as_str()
                }
            ),
            "rate_limit",
            None,
            None,
        );
    }

    if lower.contains("prompt is too long") {
        return create_error_output(
            PROMPT_TOO_LONG_ERROR_MESSAGE,
            "invalid_request",
            None,
            Some(input.message.clone()),
        );
    }

    if lower.contains("x-api-key") {
        let content = if input.external_api_key {
            "Invalid API key · Fix external API key"
        } else {
            "Not logged in · Please run /login"
        };
        return create_error_output(content, "authentication_failed", None, None);
    }

    if matches!(input.status, Some(401 | 403)) {
        let content = if input.non_interactive {
            format!(
                "Failed to authenticate. {API_ERROR_MESSAGE_PREFIX}: {}",
                input.message
            )
        } else {
            format!(
                "Please run /login · {API_ERROR_MESSAGE_PREFIX}: {}",
                input.message
            )
        };
        return create_error_output(content, "authentication_failed", None, None);
    }

    if input.status == Some(404) {
        let switch_cmd = if input.non_interactive {
            "--model"
        } else {
            "/model"
        };
        if input.api_provider != "firstParty" {
            return create_error_output(
                format!(
                    "The model {} is not available on your {} deployment. Try {} to pick a different model.",
                    input.model, input.api_provider, switch_cmd
                ),
                "invalid_request",
                None,
                None,
            );
        }
        return create_error_output(
            format!(
                "There's an issue with the selected model ({}). It may not exist or you may not have access to it. Run {} to pick a different model.",
                input.model, switch_cmd
            ),
            "invalid_request",
            None,
            None,
        );
    }

    if input.kind == "api_error" || input.kind == "error" || input.kind == "connection_error" {
        return create_error_output(
            format!("{API_ERROR_MESSAGE_PREFIX}: {}", input.message),
            "unknown",
            None,
            None,
        );
    }

    create_error_output(API_ERROR_MESSAGE_PREFIX, "unknown", None, None)
}

pub fn execute_live_completion(input: &LiveCompletionInput) -> Result<LiveCompletionOutput> {
    let request = build_request(&RequestBuildInput {
        session_id: input.session_id.clone(),
        user_agent: input.user_agent.clone(),
        custom_headers: std::env::var("ANTHROPIC_CUSTOM_HEADERS").ok(),
        container_id: None,
        remote_session_id: None,
        client_app: None,
        authorization: None,
        additional_protection: false,
        body: RequestBodyInput {
            model: input.model.clone(),
            messages: input.messages.clone(),
            system: input.system.clone(),
            tools: input.tools.clone(),
            betas: Vec::new(),
            extra_body: None,
            tool_choice: None,
            metadata: None,
            thinking: None,
            temperature: Some(0.0),
            output_config: None,
            max_tokens: input.max_tokens.unwrap_or(4096),
        },
    });
    let client = Client::builder()
        .timeout(Duration::from_secs(120))
        .build()
        .context("building Anthropic HTTP client")?;
    let response = client
        .post(messages_url(&input.base_url))
        .headers(build_http_headers(&request.headers, &input.api_key)?)
        .json(&request.body)
        .send();

    let response = match response {
        Ok(response) => response,
        Err(error) => {
            let kind = if error.is_timeout() {
                "connection_timeout"
            } else {
                "connection_error"
            };
            let classified = classify_error(&ProviderErrorInput {
                kind: kind.to_string(),
                model: input.model.clone(),
                message: error.to_string(),
                status: error.status().map(|status| status.as_u16()),
                non_interactive: false,
                api_provider: default_api_provider(),
                external_api_key: true,
            });
            return Ok(LiveCompletionOutput {
                content: Vec::new(),
                assistant_text: String::new(),
                tool_uses: Vec::new(),
                error: Some(classified),
                request_id: None,
            });
        }
    };

    let request_id = extract_request_id(response.headers());
    let status = response.status();
    let body = response.text().context("reading Anthropic response body")?;
    if !status.is_success() {
        let classified = classify_error(&ProviderErrorInput {
            kind: "api_error".to_string(),
            model: input.model.clone(),
            message: parse_error_message(&body),
            status: Some(status.as_u16()),
            non_interactive: false,
            api_provider: default_api_provider(),
            external_api_key: true,
        });
        return Ok(LiveCompletionOutput {
            content: Vec::new(),
            assistant_text: String::new(),
            tool_uses: Vec::new(),
            error: Some(classified),
            request_id,
        });
    }

    let stream = process_sse_stream(&StreamProcessInput {
        sse: body,
        model: input.model.clone(),
        request_id: request_id.clone(),
        non_interactive: false,
        max_output_tokens: input.max_tokens,
        ttft_ms: None,
    })?;
    let (content, assistant_text, tool_uses, error) = collect_live_completion(&stream)?;
    Ok(LiveCompletionOutput {
        content,
        assistant_text,
        tool_uses,
        error,
        request_id,
    })
}

fn default_api_provider() -> String {
    "firstParty".to_string()
}

fn default_base_url() -> String {
    "https://api.anthropic.com".to_string()
}

fn initialize_content_block(content_block: &Value) -> Result<Value> {
    let mut content_block = as_object(content_block)?;
    match content_block
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "tool_use" | "server_tool_use" => {
            content_block.insert("input".to_string(), Value::String(String::new()));
        }
        "text" => {
            content_block.insert("text".to_string(), Value::String(String::new()));
        }
        "thinking" => {
            content_block.insert("thinking".to_string(), Value::String(String::new()));
            content_block.insert("signature".to_string(), Value::String(String::new()));
        }
        _ => {}
    }
    Ok(Value::Object(content_block))
}

fn apply_delta(content_block: &mut Value, delta: &Value) -> Result<()> {
    let block = content_block
        .as_object_mut()
        .ok_or_else(|| anyhow!("expected content block object"))?;
    let delta_type = delta
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing delta type"))?;
    let block_type = block
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("missing content block type"))?;

    match delta_type {
        "citations_delta" => {}
        "input_json_delta" => {
            if block_type != "tool_use" && block_type != "server_tool_use" {
                bail!("Content block is not a input_json block");
            }
            let partial_json = delta
                .get("partial_json")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing partial_json"))?;
            append_string_field(block, "input", partial_json)?;
        }
        "text_delta" => {
            if block_type != "text" {
                bail!("Content block is not a text block");
            }
            let text = delta
                .get("text")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing text delta"))?;
            append_string_field(block, "text", text)?;
        }
        "thinking_delta" => {
            if block_type != "thinking" {
                bail!("Content block is not a thinking block");
            }
            let thinking = delta
                .get("thinking")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing thinking delta"))?;
            append_string_field(block, "thinking", thinking)?;
        }
        "signature_delta" => {
            if block_type != "thinking" && block_type != "connector_text" {
                bail!("Content block is not a thinking block");
            }
            let signature = delta
                .get("signature")
                .and_then(Value::as_str)
                .ok_or_else(|| anyhow!("missing signature delta"))?;
            block.insert(
                "signature".to_string(),
                Value::String(signature.to_string()),
            );
        }
        _ => {}
    }

    Ok(())
}

fn refusal_error(model: &str, non_interactive: bool) -> ErrorOutput {
    let base_message = if non_interactive {
        format!(
            "{API_ERROR_MESSAGE_PREFIX}: Claude Code is unable to respond to this request, which appears to violate our Usage Policy (https://www.anthropic.com/legal/aup). Try rephrasing the request or attempting a different approach."
        )
    } else {
        format!(
            "{API_ERROR_MESSAGE_PREFIX}: Claude Code is unable to respond to this request, which appears to violate our Usage Policy (https://www.anthropic.com/legal/aup). Please double press esc to edit your last message or start a new session for Claude Code to assist with a different task."
        )
    };
    let model_suggestion = if model != crate::core::LIVE_DEFAULT_MODEL {
        &format!(
            " If you are seeing this refusal repeatedly, try running /model {} to switch models.",
            crate::core::LIVE_DEFAULT_MODEL
        )
    } else {
        ""
    };
    create_error_output(
        format!("{base_message}{model_suggestion}"),
        "invalid_request",
        None,
        None,
    )
}

fn append_string_field(block: &mut Map<String, Value>, field: &str, suffix: &str) -> Result<()> {
    let current = block
        .get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("Content block input is not a string"))?
        .to_string();
    block.insert(
        field.to_string(),
        Value::String(format!("{current}{suffix}")),
    );
    Ok(())
}

fn event_index(event: &Value) -> Result<usize> {
    event
        .get("index")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| anyhow!("missing event index"))
}

fn ensure_slot<T>(values: &mut Vec<Option<T>>, index: usize) {
    if values.len() <= index {
        values.resize_with(index + 1, || None);
    }
}

fn as_object(value: &Value) -> Result<Map<String, Value>> {
    value
        .as_object()
        .cloned()
        .ok_or_else(|| anyhow!("expected object"))
}

fn value_as_optional_string(value: &Value) -> Option<String> {
    if value.is_null() {
        return None;
    }
    value.as_str().map(ToString::to_string)
}

fn messages_url(base_url: &str) -> String {
    format!("{}/v1/messages", base_url.trim_end_matches('/'))
}

fn build_http_headers(
    request_headers: &std::collections::BTreeMap<String, String>,
    api_key: &str,
) -> Result<HeaderMap> {
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("x-api-key"),
        HeaderValue::from_str(api_key).context("invalid x-api-key header")?,
    );
    headers.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static("2023-06-01"),
    );
    headers.insert(ACCEPT, HeaderValue::from_static("text/event-stream"));
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    for (name, value) in request_headers {
        let name = HeaderName::from_bytes(name.as_bytes())
            .with_context(|| format!("invalid request header name: {name}"))?;
        let value = HeaderValue::from_str(value)
            .with_context(|| format!("invalid request header value for {name}"))?;
        headers.insert(name, value);
    }

    Ok(headers)
}

fn extract_request_id(headers: &HeaderMap) -> Option<String> {
    ["request-id", "anthropic-request-id"]
        .into_iter()
        .find_map(|name| headers.get(name))
        .and_then(|value| value.to_str().ok())
        .map(ToString::to_string)
}

fn parse_error_message(body: &str) -> String {
    extract_json_string_field(body, "message").unwrap_or_else(|| body.to_string())
}

fn collect_live_completion(
    stream: &StreamRunOutput,
) -> Result<(Vec<Value>, String, Vec<LiveToolUse>, Option<ErrorOutput>)> {
    let mut content = Vec::new();
    let mut assistant_text = String::new();
    let mut tool_uses = Vec::new();
    let mut first_error = None;

    for emission in &stream.emissions {
        match emission {
            StreamEmission::Assistant { message } => {
                for block in &message.content {
                    content.push(block.clone());
                    match block
                        .get("type")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                    {
                        "text" => {
                            if let Some(text) = block.get("text").and_then(Value::as_str) {
                                assistant_text.push_str(text);
                            }
                        }
                        "tool_use" => {
                            tool_uses.push(LiveToolUse {
                                id: block
                                    .get("id")
                                    .and_then(Value::as_str)
                                    .ok_or_else(|| anyhow!("missing tool use id"))?
                                    .to_string(),
                                name: block
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .ok_or_else(|| anyhow!("missing tool use name"))?
                                    .to_string(),
                                input: block.get("input").cloned().unwrap_or(Value::Null),
                            });
                        }
                        _ => {}
                    }
                }
            }
            StreamEmission::Error { message } => {
                if first_error.is_none() {
                    first_error = Some(message.clone());
                }
            }
            StreamEmission::StreamEvent { .. } => {}
        }
    }

    Ok((content, assistant_text, tool_uses, first_error))
}
