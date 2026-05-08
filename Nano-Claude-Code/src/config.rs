use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use unicode_normalization::UnicodeNormalization;

const ANTHROPIC_API_KEY_FILE: &str = "anthropic_api_key";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicApiKeySource {
    Env,
    Config,
}

impl AnthropicApiKeySource {
    pub fn label(self) -> &'static str {
        match self {
            Self::Env => "ANTHROPIC_API_KEY",
            Self::Config => "config",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicApiKey {
    pub value: String,
    pub source: AnthropicApiKeySource,
}

pub fn normalize_nfc(input: &str) -> String {
    input.nfc().collect()
}

pub fn get_claude_config_home_dir() -> PathBuf {
    if let Ok(dir) = env::var("CLAUDE_CONFIG_DIR") {
        return PathBuf::from(normalize_nfc(&dir));
    }

    let home = env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(normalize_nfc(&format!("{home}/.claude")))
}

pub fn is_env_truthy(value: Option<&str>) -> bool {
    let Some(value) = value else {
        return false;
    };

    let normalized = value.trim().to_ascii_lowercase();
    matches!(normalized.as_str(), "1" | "true" | "yes" | "on")
}

pub fn parse_env_vars(raw_env_args: &[String]) -> Result<BTreeMap<String, String>, String> {
    let mut parsed = BTreeMap::new();

    for env_str in raw_env_args {
        let mut parts = env_str.split('=');
        let key = parts.next().unwrap_or_default();
        let value_parts: Vec<&str> = parts.collect();
        if key.is_empty() || value_parts.is_empty() {
            return Err(format!(
                "Invalid environment variable format: {env_str}, environment variables should be added as: -e KEY1=value1 -e KEY2=value2"
            ));
        }

        parsed.insert(key.to_string(), value_parts.join("="));
    }

    Ok(parsed)
}

pub fn anthropic_api_key_path() -> PathBuf {
    get_claude_config_home_dir().join(ANTHROPIC_API_KEY_FILE)
}

pub fn load_anthropic_api_key() -> Option<AnthropicApiKey> {
    if let Some(value) = trim_non_empty(env::var("ANTHROPIC_API_KEY").ok().as_deref()) {
        return Some(AnthropicApiKey {
            value,
            source: AnthropicApiKeySource::Env,
        });
    }

    load_saved_anthropic_api_key().map(|value| AnthropicApiKey {
        value,
        source: AnthropicApiKeySource::Config,
    })
}

pub fn save_anthropic_api_key(api_key: &str) -> io::Result<PathBuf> {
    let normalized = trim_non_empty(Some(api_key))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "Anthropic API key is empty"))?;
    let path = anthropic_api_key_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, format!("{normalized}\n"))?;
    #[cfg(unix)]
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    Ok(path)
}

fn load_saved_anthropic_api_key() -> Option<String> {
    let path = anthropic_api_key_path();
    let raw = fs::read_to_string(path).ok()?;
    trim_non_empty(Some(&raw))
}

fn trim_non_empty(value: Option<&str>) -> Option<String> {
    let trimmed = value?.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}
