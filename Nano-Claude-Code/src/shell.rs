use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use serde::Serialize;

use crate::config::normalize_nfc;
use crate::permissions::{
    ExecPermissionDecision, ExecPolicyConfig, NormalizedExecError, evaluate_exec_command,
    normalize_blocked_exec_error, policy_from_env,
};
use crate::process::{CommandResult, ManagedProcess, pre_spawn_failure};
use crate::task::{BackgroundTask, TaskSnapshot};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ExecReport {
    pub cwd: String,
    pub shell: String,
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

pub struct ShellRuntime {
    original_cwd: PathBuf,
    current_cwd: PathBuf,
    shell: String,
    env: BTreeMap<String, String>,
    tasks: BTreeMap<String, BackgroundTask>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct PolicyExecReport {
    pub decision: ExecPermissionDecision,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub report: Option<ExecReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<NormalizedExecError>,
}

pub fn default_shell() -> String {
    std::env::var("SHELL")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| "/bin/sh".to_string())
}

impl ShellRuntime {
    pub fn new(initial_cwd: &Path, shell: &str, env: &BTreeMap<String, String>) -> Self {
        let physical = normalize_path(initial_cwd);
        Self {
            original_cwd: physical.clone(),
            current_cwd: physical,
            shell: shell.to_string(),
            env: env.clone(),
            tasks: BTreeMap::new(),
        }
    }

    pub fn cwd(&self) -> String {
        self.current_cwd.to_string_lossy().to_string()
    }

    pub fn foreground(
        &mut self,
        command: &str,
        env: &BTreeMap<String, String>,
    ) -> Result<CommandResult> {
        let mut process = self.spawn_process(command, env)?;
        let result = process.wait()?;
        if result.code == 0 {
            if let Some(new_cwd) = process.take_cwd_after_success()? {
                self.current_cwd = PathBuf::from(normalize_nfc(&new_cwd));
            }
        } else {
            process.discard_cwd_capture();
        }
        Ok(result)
    }

    pub fn start_background(
        &mut self,
        task_id: &str,
        command: &str,
        description: &str,
        env: &BTreeMap<String, String>,
    ) -> Result<TaskSnapshot> {
        if self.tasks.contains_key(task_id) {
            return Err(anyhow!("task already exists: {task_id}"));
        }

        let process = self.spawn_process(command, env)?;
        let mut task = BackgroundTask::start(task_id, description, command, process);
        let snapshot = task.status()?;
        self.tasks.insert(task_id.to_string(), task);
        Ok(snapshot)
    }

    pub fn status(&mut self, task_id: &str) -> Result<TaskSnapshot> {
        self.tasks
            .get_mut(task_id)
            .ok_or_else(|| anyhow!("unknown task: {task_id}"))?
            .status()
    }

    pub fn wait(&mut self, task_id: &str) -> Result<(TaskSnapshot, CommandResult)> {
        self.tasks
            .get_mut(task_id)
            .ok_or_else(|| anyhow!("unknown task: {task_id}"))?
            .wait()
    }

    pub fn cancel(&mut self, task_id: &str) -> Result<TaskSnapshot> {
        self.tasks
            .get_mut(task_id)
            .ok_or_else(|| anyhow!("unknown task: {task_id}"))?
            .cancel()
    }

    pub fn snapshot_tasks(&mut self) -> Result<BTreeMap<String, TaskSnapshot>> {
        let task_ids: Vec<String> = self.tasks.keys().cloned().collect();
        let mut snapshots = BTreeMap::new();
        for task_id in task_ids {
            snapshots.insert(task_id.clone(), self.status(&task_id)?);
        }
        Ok(snapshots)
    }

    pub fn resolve_path(&self, path: &str) -> PathBuf {
        let candidate = PathBuf::from(path);
        if candidate.is_absolute() {
            candidate
        } else {
            self.current_cwd.join(candidate)
        }
    }

    fn spawn_process(
        &mut self,
        command: &str,
        env: &BTreeMap<String, String>,
    ) -> Result<ManagedProcess> {
        let cwd = match self.ensure_working_cwd() {
            Some(cwd) => cwd,
            None => {
                let message = format!(
                    "Working directory \"{}\" no longer exists. Please restart Claude from an existing directory.",
                    self.current_cwd.display()
                );
                return Ok(pre_spawn_failure(
                    command,
                    &self.current_cwd,
                    &self.shell,
                    message,
                ));
            }
        };

        ManagedProcess::spawn(command, &cwd, &self.shell, &merge_env(&self.env, env))
    }

    fn ensure_working_cwd(&mut self) -> Option<PathBuf> {
        if let Some(current) = existing_physical_path(&self.current_cwd) {
            self.current_cwd = current.clone();
            return Some(current);
        }

        let recovered = existing_physical_path(&self.original_cwd)?;
        self.current_cwd = recovered.clone();
        Some(recovered)
    }
}

pub fn execute(
    command: &str,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<ExecReport> {
    if let Some(policy) = policy_from_env()? {
        let outcome = execute_with_policy(command, cwd, shell, env_vars, &policy)?;
        if let Some(report) = outcome.report {
            return Ok(report);
        }
        if let Some(error) = outcome.error {
            return Err(anyhow!(error.message));
        }
        return Err(anyhow!("Permission to use Bash has been denied."));
    }

    execute_unchecked(command, cwd, shell, env_vars)
}

pub fn execute_with_policy(
    command: &str,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
    policy: &ExecPolicyConfig,
) -> Result<PolicyExecReport> {
    let decision = evaluate_exec_command(command, policy);
    if let Some(error) = normalize_blocked_exec_error(&decision) {
        return Ok(PolicyExecReport {
            decision,
            report: None,
            error: Some(error),
        });
    }

    let report = execute_unchecked(command, cwd, shell, env_vars)?;
    Ok(PolicyExecReport {
        decision,
        report: Some(report),
        error: None,
    })
}

fn execute_unchecked(
    command: &str,
    cwd: &Path,
    shell: &str,
    env_vars: &BTreeMap<String, String>,
) -> Result<ExecReport> {
    let mut runtime = ShellRuntime::new(cwd, shell, &BTreeMap::new());
    let result = runtime.foreground(command, env_vars)?;
    Ok(ExecReport {
        cwd: normalize_nfc(&cwd.to_string_lossy()),
        shell: result.shell,
        command: result.command,
        stdout: result.stdout,
        stderr: result.stderr,
        exit_code: result.code,
    })
}

fn normalize_path(path: &Path) -> PathBuf {
    existing_physical_path(path)
        .unwrap_or_else(|| PathBuf::from(normalize_nfc(&path.to_string_lossy())))
}

fn existing_physical_path(path: &Path) -> Option<PathBuf> {
    fs::canonicalize(path)
        .ok()
        .map(|value| PathBuf::from(normalize_nfc(&value.to_string_lossy())))
}

fn merge_env(
    base: &BTreeMap<String, String>,
    overrides: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut merged = base.clone();
    merged.extend(
        overrides
            .iter()
            .map(|(key, value)| (key.clone(), value.clone())),
    );
    merged
}
