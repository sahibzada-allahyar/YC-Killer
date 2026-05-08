use std::collections::HashSet;
use std::env;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::sandbox::{SandboxSettings, should_use_sandbox};

pub const BASH_TOOL_NAME: &str = "Bash";

const DEFAULT_ASK_REASON: &str = "This command requires approval";
const SANDBOX_AUTO_ALLOW_REASON: &str =
    "Auto-allowed with sandbox (autoAllowBashIfSandboxed enabled)";
const DENIAL_WORKAROUND_GUIDANCE: &str = concat!(
    "IMPORTANT: You *may* attempt to accomplish this action using other tools that might naturally be used to accomplish this goal, ",
    "e.g. using head instead of cat. But you *should not* attempt to work around this denial in malicious ways, ",
    "e.g. do not use your ability to run tests to execute non-test actions. ",
    "You should only try to work around this restriction in reasonable ways that do not attempt to bypass the intent behind this denial. ",
    "If you believe this capability is essential to complete the user's request, STOP and explain to the user ",
    "what you were trying to do and why you need this permission. Let the user decide how to proceed."
);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MatchMode {
    Exact,
    Prefix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum PermissionBehavior {
    Allow,
    Ask,
    Deny,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PermissionMode {
    Default,
    AcceptEdits,
    BypassPermissions,
    DontAsk,
    Plan,
}

impl Default for PermissionMode {
    fn default() -> Self {
        Self::Default
    }
}

impl PermissionMode {
    pub fn parse(value: Option<&str>) -> Self {
        match value.unwrap_or("default") {
            "acceptEdits" => Self::AcceptEdits,
            "bypassPermissions" => Self::BypassPermissions,
            "dontAsk" => Self::DontAsk,
            "plan" => Self::Plan,
            _ => Self::Default,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::AcceptEdits => "acceptEdits",
            Self::BypassPermissions => "bypassPermissions",
            Self::DontAsk => "dontAsk",
            Self::Plan => "plan",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PermissionRuleValue {
    pub tool_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule_content: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShellPermissionRule {
    Exact { command: String },
    Prefix { prefix: String },
    Wildcard { pattern: String },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum DecisionReason {
    Rule { rule: String },
    Mode { mode: String },
    Other { reason: String },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ExecPermissionDecision {
    pub behavior: PermissionBehavior,
    pub should_use_sandbox: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<DecisionReason>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct NormalizedExecError {
    pub kind: String,
    pub behavior: PermissionBehavior,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase", default)]
pub struct ExecPolicyConfig {
    pub mode: Option<String>,
    pub allow: Vec<String>,
    pub deny: Vec<String>,
    pub ask: Vec<String>,
    pub sandbox: SandboxSettings,
    pub dangerously_disable_sandbox: bool,
}

#[derive(Debug, Clone)]
struct ParsedRule {
    original: String,
    shell_rule: ShellPermissionRule,
}

#[derive(Debug, Default)]
struct ParsedPolicyRules {
    tool_wide_allow: Option<String>,
    tool_wide_deny: Option<String>,
    tool_wide_ask: Option<String>,
    allow: Vec<ParsedRule>,
    deny: Vec<ParsedRule>,
    ask: Vec<ParsedRule>,
}

impl ParsedPolicyRules {
    fn from_policy(policy: &ExecPolicyConfig) -> Self {
        let mut parsed = Self::default();
        populate_rules(
            &mut parsed.tool_wide_allow,
            &mut parsed.allow,
            &policy.allow,
        );
        populate_rules(&mut parsed.tool_wide_deny, &mut parsed.deny, &policy.deny);
        populate_rules(&mut parsed.tool_wide_ask, &mut parsed.ask, &policy.ask);
        parsed
    }
}

fn populate_rules(
    tool_wide_slot: &mut Option<String>,
    content_rules: &mut Vec<ParsedRule>,
    raw_rules: &[String],
) {
    for raw_rule in raw_rules {
        let parsed = permission_rule_value_from_string(raw_rule);
        if parsed.tool_name != BASH_TOOL_NAME {
            continue;
        }

        if let Some(content) = parsed.rule_content {
            content_rules.push(ParsedRule {
                original: raw_rule.clone(),
                shell_rule: parse_permission_rule(&content),
            });
            continue;
        }

        if tool_wide_slot.is_none() {
            *tool_wide_slot = Some(raw_rule.clone());
        }
    }
}

pub fn policy_from_env() -> Result<Option<ExecPolicyConfig>> {
    let Ok(raw) = env::var("NANO_CLAUDE_EXEC_POLICY") else {
        return Ok(None);
    };
    let parsed = serde_json::from_str::<ExecPolicyConfig>(&raw)
        .with_context(|| "parsing NANO_CLAUDE_EXEC_POLICY")?;
    Ok(Some(parsed))
}

pub fn normalize_blocked_exec_error(
    decision: &ExecPermissionDecision,
) -> Option<NormalizedExecError> {
    match decision.behavior {
        PermissionBehavior::Allow => None,
        PermissionBehavior::Ask | PermissionBehavior::Deny => Some(NormalizedExecError {
            kind: "PermissionDenied".to_string(),
            behavior: decision.behavior,
            message: decision.message.clone().unwrap_or_default(),
        }),
    }
}

pub fn evaluate_exec_command(command: &str, policy: &ExecPolicyConfig) -> ExecPermissionDecision {
    let command = command.trim();
    let mode = PermissionMode::parse(policy.mode.as_deref());
    let should_use_sandbox =
        should_use_sandbox(command, policy.dangerously_disable_sandbox, &policy.sandbox);
    let parsed_rules = ParsedPolicyRules::from_policy(policy);

    if mode == PermissionMode::BypassPermissions {
        return allow_with_reason(
            should_use_sandbox,
            DecisionReason::Mode {
                mode: mode.as_str().to_string(),
            },
        );
    }

    if let Some(rule) = parsed_rules.tool_wide_deny.as_ref() {
        return deny_with_rule_message(
            should_use_sandbox,
            format!("Permission to use {BASH_TOOL_NAME} has been denied."),
            rule.clone(),
        );
    }

    let can_sandbox_auto_allow =
        policy.sandbox.enabled && policy.sandbox.auto_allow_bash_if_sandboxed && should_use_sandbox;

    if let Some(rule) = parsed_rules.tool_wide_ask.as_ref() {
        if !can_sandbox_auto_allow {
            return apply_mode_to_ask(
                ask_with_rule(
                    should_use_sandbox,
                    default_permission_request_message(BASH_TOOL_NAME),
                    rule.clone(),
                ),
                mode,
            );
        }
    }

    if can_sandbox_auto_allow {
        if let Some(rule) =
            first_matching_rule(command, &parsed_rules.deny, MatchMode::Prefix, true, true)
        {
            return deny_with_rule_message(
                should_use_sandbox,
                command_rule_denied_message(command),
                rule.original.clone(),
            );
        }

        let subcommands = split_compound_commands(command);
        if subcommands.len() > 1 {
            let mut first_ask: Option<String> = None;
            for subcommand in &subcommands {
                if let Some(rule) = first_matching_rule(
                    subcommand,
                    &parsed_rules.deny,
                    MatchMode::Prefix,
                    true,
                    true,
                ) {
                    return deny_with_rule_message(
                        should_use_sandbox,
                        command_rule_denied_message(command),
                        rule.original.clone(),
                    );
                }
                if first_ask.is_none() {
                    first_ask = first_matching_rule(
                        subcommand,
                        &parsed_rules.ask,
                        MatchMode::Prefix,
                        true,
                        true,
                    )
                    .map(|rule| rule.original.clone());
                }
            }
            if let Some(rule) = first_ask {
                return apply_mode_to_ask(
                    ask_with_rule(
                        should_use_sandbox,
                        default_permission_request_message(BASH_TOOL_NAME),
                        rule,
                    ),
                    mode,
                );
            }
        }

        if let Some(rule) =
            first_matching_rule(command, &parsed_rules.ask, MatchMode::Prefix, true, true)
        {
            return apply_mode_to_ask(
                ask_with_rule(
                    should_use_sandbox,
                    default_permission_request_message(BASH_TOOL_NAME),
                    rule.original.clone(),
                ),
                mode,
            );
        }

        return allow_with_reason(
            should_use_sandbox,
            DecisionReason::Other {
                reason: SANDBOX_AUTO_ALLOW_REASON.to_string(),
            },
        );
    }

    if let Some(rule) =
        first_matching_rule(command, &parsed_rules.deny, MatchMode::Exact, true, true)
    {
        return deny_with_rule_message(
            should_use_sandbox,
            command_rule_denied_message(command),
            rule.original.clone(),
        );
    }

    if let Some(rule) =
        first_matching_rule(command, &parsed_rules.ask, MatchMode::Exact, true, true)
    {
        return apply_mode_to_ask(
            ask_with_rule(
                should_use_sandbox,
                default_permission_request_message(BASH_TOOL_NAME),
                rule.original.clone(),
            ),
            mode,
        );
    }

    let exact_allow =
        first_matching_rule(command, &parsed_rules.allow, MatchMode::Exact, false, true);

    if let Some(rule) =
        first_matching_rule(command, &parsed_rules.deny, MatchMode::Prefix, true, true)
    {
        return deny_with_rule_message(
            should_use_sandbox,
            command_rule_denied_message(command),
            rule.original.clone(),
        );
    }

    if let Some(rule) =
        first_matching_rule(command, &parsed_rules.ask, MatchMode::Prefix, true, true)
    {
        return apply_mode_to_ask(
            ask_with_rule(
                should_use_sandbox,
                default_permission_request_message(BASH_TOOL_NAME),
                rule.original.clone(),
            ),
            mode,
        );
    }

    if let Some(rule) = exact_allow {
        return allow_with_reason(
            should_use_sandbox,
            DecisionReason::Rule {
                rule: rule.original.clone(),
            },
        );
    }

    if let Some(rule) = first_matching_rule(
        command,
        &parsed_rules.allow,
        MatchMode::Prefix,
        false,
        false,
    ) {
        return allow_with_reason(
            should_use_sandbox,
            DecisionReason::Rule {
                rule: rule.original.clone(),
            },
        );
    }

    if let Some(rule) = parsed_rules.tool_wide_allow.as_ref() {
        return allow_with_reason(
            should_use_sandbox,
            DecisionReason::Rule { rule: rule.clone() },
        );
    }

    apply_mode_to_ask(
        ask_with_reason(
            should_use_sandbox,
            DEFAULT_ASK_REASON.to_string(),
            DecisionReason::Other {
                reason: DEFAULT_ASK_REASON.to_string(),
            },
        ),
        mode,
    )
}

fn allow_with_reason(should_use_sandbox: bool, reason: DecisionReason) -> ExecPermissionDecision {
    ExecPermissionDecision {
        behavior: PermissionBehavior::Allow,
        should_use_sandbox,
        message: None,
        reason: Some(reason),
    }
}

fn ask_with_rule(
    should_use_sandbox: bool,
    message: String,
    rule: String,
) -> ExecPermissionDecision {
    ExecPermissionDecision {
        behavior: PermissionBehavior::Ask,
        should_use_sandbox,
        message: Some(message),
        reason: Some(DecisionReason::Rule { rule }),
    }
}

fn ask_with_reason(
    should_use_sandbox: bool,
    message: String,
    reason: DecisionReason,
) -> ExecPermissionDecision {
    ExecPermissionDecision {
        behavior: PermissionBehavior::Ask,
        should_use_sandbox,
        message: Some(message),
        reason: Some(reason),
    }
}

fn deny_with_rule_message(
    should_use_sandbox: bool,
    message: String,
    rule: String,
) -> ExecPermissionDecision {
    ExecPermissionDecision {
        behavior: PermissionBehavior::Deny,
        should_use_sandbox,
        message: Some(message),
        reason: Some(DecisionReason::Rule { rule }),
    }
}

fn apply_mode_to_ask(
    decision: ExecPermissionDecision,
    mode: PermissionMode,
) -> ExecPermissionDecision {
    if decision.behavior != PermissionBehavior::Ask || mode != PermissionMode::DontAsk {
        return decision;
    }

    ExecPermissionDecision {
        behavior: PermissionBehavior::Deny,
        should_use_sandbox: decision.should_use_sandbox,
        message: Some(format!(
            "Permission to use {BASH_TOOL_NAME} has been denied because Claude Code is running in don't ask mode. {DENIAL_WORKAROUND_GUIDANCE}"
        )),
        reason: Some(DecisionReason::Mode {
            mode: mode.as_str().to_string(),
        }),
    }
}

fn default_permission_request_message(tool_name: &str) -> String {
    format!("Claude requested permissions to use {tool_name}, but you haven't granted it yet.")
}

fn command_rule_denied_message(command: &str) -> String {
    format!("Permission to use {BASH_TOOL_NAME} with command {command} has been denied.")
}

fn first_matching_rule<'a>(
    command: &str,
    rules: &'a [ParsedRule],
    match_mode: MatchMode,
    strip_all_env_vars: bool,
    skip_compound_check: bool,
) -> Option<&'a ParsedRule> {
    let commands_to_try = build_commands_to_try(command, match_mode, strip_all_env_vars);

    let compound_checks = if match_mode == MatchMode::Prefix && !skip_compound_check {
        Some(
            commands_to_try
                .iter()
                .map(|candidate| {
                    (
                        candidate.clone(),
                        split_compound_commands(candidate).len() > 1,
                    )
                })
                .collect::<std::collections::HashMap<_, _>>(),
        )
    } else {
        None
    };

    rules.iter().find(|rule| {
        commands_to_try
            .iter()
            .any(|candidate| match &rule.shell_rule {
                ShellPermissionRule::Exact { command } => command == candidate,
                ShellPermissionRule::Prefix { prefix } => match match_mode {
                    MatchMode::Exact => prefix == candidate,
                    MatchMode::Prefix => {
                        if compound_checks
                            .as_ref()
                            .and_then(|values| values.get(candidate))
                            .copied()
                            .unwrap_or(false)
                        {
                            return false;
                        }
                        candidate == prefix
                            || candidate.starts_with(&format!("{prefix} "))
                            || candidate == &format!("xargs {prefix}")
                            || candidate.starts_with(&format!("xargs {prefix} "))
                    }
                },
                ShellPermissionRule::Wildcard { pattern } => {
                    if match_mode == MatchMode::Exact {
                        return false;
                    }
                    if compound_checks
                        .as_ref()
                        .and_then(|values| values.get(candidate))
                        .copied()
                        .unwrap_or(false)
                    {
                        return false;
                    }
                    match_wildcard_pattern(pattern, candidate, false)
                }
            })
    })
}

fn build_commands_to_try(
    command: &str,
    match_mode: MatchMode,
    strip_all_env_vars: bool,
) -> Vec<String> {
    let command = command.trim();
    let command_without_redirections = strip_output_redirections(command);
    let mut candidates = match match_mode {
        MatchMode::Exact => vec![command.to_string(), command_without_redirections],
        MatchMode::Prefix => vec![command_without_redirections],
    };

    let mut commands_to_try = Vec::new();
    for candidate in candidates.drain(..) {
        let stripped = strip_safe_wrappers(&candidate);
        commands_to_try.push(candidate.clone());
        if stripped != candidate {
            commands_to_try.push(stripped);
        }
    }

    if !strip_all_env_vars {
        return commands_to_try;
    }

    let mut seen = commands_to_try.iter().cloned().collect::<HashSet<_>>();
    let mut start_idx = 0;
    while start_idx < commands_to_try.len() {
        let end_idx = commands_to_try.len();
        for idx in start_idx..end_idx {
            let candidate = commands_to_try[idx].clone();
            let env_stripped = strip_all_leading_env_vars(&candidate, None);
            if seen.insert(env_stripped.clone()) {
                commands_to_try.push(env_stripped);
            }
            let wrapper_stripped = strip_safe_wrappers(&candidate);
            if seen.insert(wrapper_stripped.clone()) {
                commands_to_try.push(wrapper_stripped);
            }
        }
        start_idx = end_idx;
    }

    commands_to_try
}

pub fn permission_rule_extract_prefix(permission_rule: &str) -> Option<String> {
    permission_rule
        .strip_suffix(":*")
        .filter(|prefix| !prefix.is_empty())
        .map(ToString::to_string)
}

pub fn has_wildcards(pattern: &str) -> bool {
    if pattern.ends_with(":*") {
        return false;
    }

    let chars: Vec<char> = pattern.chars().collect();
    for (idx, ch) in chars.iter().enumerate() {
        if *ch != '*' {
            continue;
        }

        let mut backslash_count = 0;
        let mut cursor = idx;
        while cursor > 0 && chars[cursor - 1] == '\\' {
            backslash_count += 1;
            cursor -= 1;
        }
        if backslash_count % 2 == 0 {
            return true;
        }
    }
    false
}

pub fn parse_permission_rule(permission_rule: &str) -> ShellPermissionRule {
    if let Some(prefix) = permission_rule_extract_prefix(permission_rule) {
        return ShellPermissionRule::Prefix { prefix };
    }
    if has_wildcards(permission_rule) {
        return ShellPermissionRule::Wildcard {
            pattern: permission_rule.to_string(),
        };
    }
    ShellPermissionRule::Exact {
        command: permission_rule.to_string(),
    }
}

pub fn match_wildcard_pattern(pattern: &str, command: &str, case_insensitive: bool) -> bool {
    const ESCAPED_STAR_PLACEHOLDER: &str = "__ESCAPED_STAR__";
    const ESCAPED_BACKSLASH_PLACEHOLDER: &str = "__ESCAPED_BACKSLASH__";

    let trimmed_pattern = pattern.trim();
    let mut processed = String::new();
    let chars: Vec<char> = trimmed_pattern.chars().collect();
    let mut idx = 0;

    while idx < chars.len() {
        let ch = chars[idx];
        if ch == '\\' && idx + 1 < chars.len() {
            match chars[idx + 1] {
                '*' => {
                    processed.push_str(ESCAPED_STAR_PLACEHOLDER);
                    idx += 2;
                    continue;
                }
                '\\' => {
                    processed.push_str(ESCAPED_BACKSLASH_PLACEHOLDER);
                    idx += 2;
                    continue;
                }
                _ => {}
            }
        }

        processed.push(ch);
        idx += 1;
    }

    let mut regex_pattern = regex::escape(&processed).replace("\\*", ".*");
    regex_pattern = regex_pattern.replace(ESCAPED_STAR_PLACEHOLDER, "\\*");
    regex_pattern = regex_pattern.replace(ESCAPED_BACKSLASH_PLACEHOLDER, "\\\\");

    let unescaped_star_count = processed.chars().filter(|ch| *ch == '*').count();
    if regex_pattern.ends_with(" .*") && unescaped_star_count == 1 {
        regex_pattern.truncate(regex_pattern.len() - 3);
        regex_pattern.push_str("( .*)?");
    }

    let prefix = if case_insensitive { "(?si)" } else { "(?s)" };
    let regex =
        Regex::new(&format!("{prefix}^{regex_pattern}$")).expect("wildcard regex should compile");
    regex.is_match(command)
}

pub fn permission_rule_value_from_string(rule_string: &str) -> PermissionRuleValue {
    let open_paren_index = find_first_unescaped_char(rule_string, '(');
    if open_paren_index.is_none() {
        return PermissionRuleValue {
            tool_name: rule_string.to_string(),
            rule_content: None,
        };
    }

    let open_paren_index = open_paren_index.expect("open paren index");
    let close_paren_index = find_last_unescaped_char(rule_string, ')');
    if close_paren_index.is_none()
        || close_paren_index.expect("close paren index") <= open_paren_index
    {
        return PermissionRuleValue {
            tool_name: rule_string.to_string(),
            rule_content: None,
        };
    }

    let close_paren_index = close_paren_index.expect("close paren index");
    if close_paren_index != rule_string.len() - 1 {
        return PermissionRuleValue {
            tool_name: rule_string.to_string(),
            rule_content: None,
        };
    }

    let tool_name = &rule_string[..open_paren_index];
    if tool_name.is_empty() {
        return PermissionRuleValue {
            tool_name: rule_string.to_string(),
            rule_content: None,
        };
    }

    let raw_content = &rule_string[(open_paren_index + 1)..close_paren_index];
    if raw_content.is_empty() || raw_content == "*" {
        return PermissionRuleValue {
            tool_name: tool_name.to_string(),
            rule_content: None,
        };
    }

    PermissionRuleValue {
        tool_name: tool_name.to_string(),
        rule_content: Some(unescape_rule_content(raw_content)),
    }
}

fn find_first_unescaped_char(value: &str, needle: char) -> Option<usize> {
    let indices = value.char_indices().collect::<Vec<_>>();
    for (position, (byte_idx, ch)) in indices.iter().enumerate() {
        if *ch != needle {
            continue;
        }

        let mut backslash_count = 0;
        let mut cursor = position;
        while cursor > 0 && indices[cursor - 1].1 == '\\' {
            backslash_count += 1;
            cursor -= 1;
        }

        if backslash_count % 2 == 0 {
            return Some(*byte_idx);
        }
    }
    None
}

fn find_last_unescaped_char(value: &str, needle: char) -> Option<usize> {
    let indices = value.char_indices().collect::<Vec<_>>();
    for position in (0..indices.len()).rev() {
        if indices[position].1 != needle {
            continue;
        }

        let mut backslash_count = 0;
        let mut cursor = position;
        while cursor > 0 && indices[cursor - 1].1 == '\\' {
            backslash_count += 1;
            cursor -= 1;
        }

        if backslash_count % 2 == 0 {
            return Some(indices[position].0);
        }
    }
    None
}

fn unescape_rule_content(content: &str) -> String {
    content
        .replace("\\(", "(")
        .replace("\\)", ")")
        .replace("\\\\", "\\")
}

pub fn split_compound_commands(command: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = command.chars().collect();
    let mut idx = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut escape = false;

    while idx < chars.len() {
        let ch = chars[idx];

        if escape {
            current.push(ch);
            escape = false;
            idx += 1;
            continue;
        }

        if ch == '\\' {
            escape = true;
            current.push(ch);
            idx += 1;
            continue;
        }

        if ch == '\'' && !in_double {
            in_single = !in_single;
            current.push(ch);
            idx += 1;
            continue;
        }

        if ch == '"' && !in_single {
            in_double = !in_double;
            current.push(ch);
            idx += 1;
            continue;
        }

        if !in_single && !in_double {
            let is_double_operator = idx + 1 < chars.len()
                && ((ch == '&' && chars[idx + 1] == '&') || (ch == '|' && chars[idx + 1] == '|'));
            if is_double_operator {
                push_trimmed_segment(&mut result, &mut current);
                idx += 2;
                continue;
            }

            if ch == ';' || ch == '|' || ch == '\n' {
                push_trimmed_segment(&mut result, &mut current);
                idx += 1;
                continue;
            }
        }

        current.push(ch);
        idx += 1;
    }

    push_trimmed_segment(&mut result, &mut current);
    result
}

fn push_trimmed_segment(result: &mut Vec<String>, current: &mut String) {
    let trimmed = current.trim();
    if !trimmed.is_empty() {
        result.push(trimmed.to_string());
    }
    current.clear();
}

pub fn strip_safe_wrappers(command: &str) -> String {
    let mut stripped = command.to_string();
    let mut previous = String::new();

    while stripped != previous {
        previous = stripped.clone();
        stripped = strip_comment_lines(&stripped);
        if let Some(captures) = safe_env_var_regex().captures(&stripped) {
            if let Some(var_name) = captures.get(1).map(|value| value.as_str()) {
                if is_safe_env_var(var_name) {
                    if let Some(full_match) = captures.get(0) {
                        stripped = stripped[full_match.end()..].to_string();
                    }
                }
            }
        }
    }

    previous.clear();
    while stripped != previous {
        previous = stripped.clone();
        stripped = strip_comment_lines(&stripped);
        for pattern in safe_wrapper_patterns() {
            stripped = pattern.replace(&stripped, "").to_string();
        }
    }

    stripped.trim().to_string()
}

pub fn strip_all_leading_env_vars(command: &str, blocklist: Option<&Regex>) -> String {
    let mut stripped = command.to_string();
    let mut previous = String::new();

    while stripped != previous {
        previous = stripped.clone();
        stripped = strip_comment_lines(&stripped);

        let Some(captures) = all_env_var_regex().captures(&stripped) else {
            continue;
        };

        if let Some(var_name) = captures.get(1).map(|value| value.as_str()) {
            if blocklist
                .map(|pattern| pattern.is_match(var_name))
                .unwrap_or(false)
            {
                break;
            }
        }

        if let Some(full_match) = captures.get(0) {
            stripped = stripped[full_match.end()..].to_string();
        }
    }

    stripped.trim().to_string()
}

fn strip_comment_lines(command: &str) -> String {
    let non_comment_lines = command
        .lines()
        .filter_map(|line| {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                None
            } else {
                Some(line)
            }
        })
        .collect::<Vec<_>>();

    if non_comment_lines.is_empty() {
        return command.to_string();
    }

    non_comment_lines.join("\n")
}

fn strip_output_redirections(command: &str) -> String {
    let tokens = tokenize_command(command);
    let mut kept = Vec::new();
    let mut idx = 0;

    while idx < tokens.len() {
        let token = &tokens[idx];
        if token == ">" || token == ">>" || token == ">&" {
            let next = tokens.get(idx + 1);
            if token == ">&" {
                if next
                    .map(|value| matches!(value.as_str(), "0" | "1" | "2"))
                    .unwrap_or(false)
                {
                    idx += 2;
                    continue;
                }
            } else if let Some(target) = next {
                if is_static_redirect_target(target)
                    || matches!(target.as_str(), "&0" | "&1" | "&2")
                {
                    idx += 2;
                    continue;
                }
            }
        }

        kept.push(token.clone());
        idx += 1;
    }

    kept.join(" ").trim().to_string()
}

fn tokenize_command(command: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = command.chars().collect();
    let mut idx = 0;
    let mut in_single = false;
    let mut in_double = false;
    let mut escape = false;

    while idx < chars.len() {
        let ch = chars[idx];
        if escape {
            current.push(ch);
            escape = false;
            idx += 1;
            continue;
        }

        if ch == '\\' {
            current.push(ch);
            escape = true;
            idx += 1;
            continue;
        }

        if ch == '\'' && !in_double {
            in_single = !in_single;
            current.push(ch);
            idx += 1;
            continue;
        }

        if ch == '"' && !in_single {
            in_double = !in_double;
            current.push(ch);
            idx += 1;
            continue;
        }

        if !in_single && !in_double {
            if ch.is_whitespace() {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                idx += 1;
                continue;
            }

            if ch == '>' {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                if idx + 1 < chars.len() && chars[idx + 1] == '>' {
                    tokens.push(">>".to_string());
                    idx += 2;
                    continue;
                }
                if idx + 1 < chars.len() && chars[idx + 1] == '&' {
                    tokens.push(">&".to_string());
                    idx += 2;
                    continue;
                }
                tokens.push(">".to_string());
                idx += 1;
                continue;
            }
        }

        current.push(ch);
        idx += 1;
    }

    if !current.is_empty() {
        tokens.push(current);
    }

    tokens
}

fn is_static_redirect_target(target: &str) -> bool {
    !target.is_empty()
        && !target
            .chars()
            .any(|ch| matches!(ch, ' ' | '\t' | '\n' | '\r' | '\'' | '"'))
        && !target.starts_with('#')
        && !target.starts_with('!')
        && !target.starts_with('=')
        && !target.contains('$')
        && !target.contains('`')
        && !target.contains('*')
        && !target.contains('?')
        && !target.contains('[')
        && !target.contains('{')
        && !target.contains('~')
        && !target.contains('(')
        && !target.contains('<')
        && !target.starts_with('&')
}

fn is_safe_env_var(var_name: &str) -> bool {
    matches!(
        var_name,
        "GOEXPERIMENT"
            | "GOOS"
            | "GOARCH"
            | "CGO_ENABLED"
            | "GO111MODULE"
            | "RUST_BACKTRACE"
            | "RUST_LOG"
            | "NODE_ENV"
            | "PYTHONUNBUFFERED"
            | "PYTHONDONTWRITEBYTECODE"
            | "PYTEST_DISABLE_PLUGIN_AUTOLOAD"
            | "PYTEST_DEBUG"
            | "ANTHROPIC_API_KEY"
            | "LANG"
            | "LANGUAGE"
            | "LC_ALL"
            | "LC_CTYPE"
            | "LC_TIME"
            | "CHARSET"
            | "TERM"
            | "COLORTERM"
            | "NO_COLOR"
            | "FORCE_COLOR"
            | "TZ"
            | "LS_COLORS"
            | "LSCOLORS"
            | "GREP_COLOR"
            | "GREP_COLORS"
            | "GCC_COLORS"
            | "TIME_STYLE"
            | "BLOCK_SIZE"
            | "BLOCKSIZE"
    )
}

pub fn binary_hijack_vars() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"^(LD_|DYLD_|PATH$)").expect("binary hijack regex"))
}

fn safe_env_var_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^([A-Za-z_][A-Za-z0-9_]*)=([A-Za-z0-9_./:-]+)[ \t]+").expect("safe env regex")
    })
}

fn all_env_var_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r#"^([A-Za-z_][A-Za-z0-9_]*(?:\[[^\]]*\])?)\+?=(?:'[^'\n\r]*'|"(?:\\.|[^"$`\\\n\r])*"|\\.|[^ \t\n\r$`;|&()<>\\'"])*[ \t]+"#,
        )
        .expect("all env regex")
    })
}

fn safe_wrapper_patterns() -> &'static [Regex] {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    PATTERNS.get_or_init(|| {
        vec![
            Regex::new(
                r"^timeout[ \t]+(?:(?:--(?:foreground|preserve-status|verbose)|--(?:kill-after|signal)=[A-Za-z0-9_.+-]+|--(?:kill-after|signal)[ \t]+[A-Za-z0-9_.+-]+|-v|-[ks][ \t]+[A-Za-z0-9_.+-]+|-[ks][A-Za-z0-9_.+-]+)[ \t]+)*(?:--[ \t]+)?\d+(?:\.\d+)?[smhd]?[ \t]+",
            )
            .expect("timeout wrapper regex"),
            Regex::new(r"^time[ \t]+(?:--[ \t]+)?").expect("time wrapper regex"),
            Regex::new(r"^nice(?:[ \t]+-n[ \t]+-?\d+|[ \t]+-\d+)?[ \t]+(?:--[ \t]+)?")
                .expect("nice wrapper regex"),
            Regex::new(r"^stdbuf(?:[ \t]+-[ioe][LN0-9]+)+[ \t]+(?:--[ \t]+)?")
                .expect("stdbuf wrapper regex"),
            Regex::new(r"^nohup[ \t]+(?:--[ \t]+)?").expect("nohup wrapper regex"),
        ]
    })
}
