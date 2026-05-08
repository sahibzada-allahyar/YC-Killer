use anyhow::{Result, anyhow};
use serde_json::Value;

pub fn parse_sse_events(input: &str) -> Result<Vec<Value>> {
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");
    let mut events = Vec::new();
    let mut current_event = String::new();
    let mut data_lines: Vec<String> = Vec::new();

    for line in normalized.split('\n') {
        if line.is_empty() {
            flush_event(&mut events, &mut current_event, &mut data_lines)?;
            continue;
        }
        if line.starts_with(':') {
            continue;
        }

        let (field, raw_value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };

        match field {
            "event" => current_event = raw_value.to_string(),
            "data" => data_lines.push(raw_value.to_string()),
            _ => {}
        }
    }

    flush_event(&mut events, &mut current_event, &mut data_lines)?;
    Ok(events)
}

fn flush_event(
    events: &mut Vec<Value>,
    current_event: &mut String,
    data_lines: &mut Vec<String>,
) -> Result<()> {
    if data_lines.is_empty() && current_event.is_empty() {
        return Ok(());
    }

    let payload = data_lines.join("\n");
    let event_name = current_event.clone();
    current_event.clear();
    data_lines.clear();

    if payload.is_empty() || payload == "[DONE]" {
        return Ok(());
    }

    let mut event: Value =
        serde_json::from_str(&payload).map_err(|err| anyhow!("invalid SSE JSON payload: {err}"))?;
    if event.get("type").is_none() {
        let object = match event.as_object_mut() {
            Some(object) => object,
            None => return Err(anyhow!("SSE payload must decode to an object")),
        };
        if !event_name.is_empty() {
            object.insert("type".to_string(), Value::String(event_name));
        }
    }

    events.push(event);
    Ok(())
}
