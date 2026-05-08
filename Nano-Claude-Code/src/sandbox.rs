use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::permissions::{
    ShellPermissionRule, binary_hijack_vars, match_wildcard_pattern, parse_permission_rule,
    split_compound_commands, strip_all_leading_env_vars, strip_safe_wrappers,
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase", default)]
pub struct SandboxSettings {
    pub enabled: bool,
    pub auto_allow_bash_if_sandboxed: bool,
    pub allow_unsandboxed_commands: bool,
    pub excluded_commands: Vec<String>,
}

impl Default for SandboxSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            auto_allow_bash_if_sandboxed: true,
            allow_unsandboxed_commands: true,
            excluded_commands: Vec::new(),
        }
    }
}

pub fn should_use_sandbox(
    command: &str,
    dangerously_disable_sandbox: bool,
    sandbox: &SandboxSettings,
) -> bool {
    if !sandbox.enabled {
        return false;
    }

    if dangerously_disable_sandbox && sandbox.allow_unsandboxed_commands {
        return false;
    }

    if command.trim().is_empty() {
        return false;
    }

    if contains_excluded_command(command, &sandbox.excluded_commands) {
        return false;
    }

    true
}

fn contains_excluded_command(command: &str, excluded_commands: &[String]) -> bool {
    if excluded_commands.is_empty() {
        return false;
    }

    for subcommand in split_compound_commands(command) {
        let trimmed = subcommand.trim();
        let mut candidates = vec![trimmed.to_string()];
        let mut seen = candidates.iter().cloned().collect::<HashSet<_>>();
        let mut start_idx = 0;

        while start_idx < candidates.len() {
            let end_idx = candidates.len();
            for idx in start_idx..end_idx {
                let candidate = candidates[idx].clone();
                let env_stripped =
                    strip_all_leading_env_vars(&candidate, Some(binary_hijack_vars()));
                if seen.insert(env_stripped.clone()) {
                    candidates.push(env_stripped);
                }

                let wrapper_stripped = strip_safe_wrappers(&candidate);
                if seen.insert(wrapper_stripped.clone()) {
                    candidates.push(wrapper_stripped);
                }
            }
            start_idx = end_idx;
        }

        for pattern in excluded_commands {
            let rule = parse_permission_rule(pattern);
            if candidates
                .iter()
                .any(|candidate| matches_rule(&rule, candidate))
            {
                return true;
            }
        }
    }

    false
}

fn matches_rule(rule: &ShellPermissionRule, command: &str) -> bool {
    match rule {
        ShellPermissionRule::Exact { command: expected } => expected == command,
        ShellPermissionRule::Prefix { prefix } => {
            command == prefix || command.starts_with(&format!("{prefix} "))
        }
        ShellPermissionRule::Wildcard { pattern } => {
            match_wildcard_pattern(pattern, command, false)
        }
    }
}
