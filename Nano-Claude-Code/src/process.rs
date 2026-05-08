use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use serde::Serialize;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

static TEMP_FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

const KILLED_EXIT_CODE: i32 = 137;
const SPAWN_FAILURE_EXIT_CODE: i32 = 126;

const INTERNAL_COMMAND_ENV: &str = "NANO_CLAUDE_CODE_INTERNAL_COMMAND_PAYLOAD";
const INTERNAL_CWD_FILE_ENV: &str = "NANO_CLAUDE_CODE_INTERNAL_CWD_FILE";
const SHELL_WRAPPER: &str = "eval \"$NANO_CLAUDE_CODE_INTERNAL_COMMAND_PAYLOAD\"; code=$?; if [ \"$code\" -eq 0 ]; then pwd -P > \"$NANO_CLAUDE_CODE_INTERNAL_CWD_FILE\"; fi; exit \"$code\"";

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct CommandResult {
    pub cwd: String,
    pub shell: String,
    pub command: String,
    pub stdout: String,
    pub stderr: String,
    pub code: i32,
    pub interrupted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub background_task_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pre_spawn_error: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessState {
    Running,
    Backgrounded,
    Completed,
    Killed,
}

pub struct ManagedProcess {
    child: Option<Child>,
    cwd: PathBuf,
    shell: String,
    command: String,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
    cwd_capture_path: Option<PathBuf>,
    state: ProcessState,
    background_task_id: Option<String>,
    forced_exit_code: Option<i32>,
    completed: Option<CommandResult>,
}

impl ManagedProcess {
    pub fn spawn(
        command: &str,
        cwd: &Path,
        shell: &str,
        env_vars: &BTreeMap<String, String>,
    ) -> Result<Self> {
        let stdout_path = unique_temp_path("stdout");
        let stderr_path = unique_temp_path("stderr");
        let cwd_capture_path = unique_temp_path("cwd");

        let stdout = create_output_file(&stdout_path)?;
        let stderr = create_output_file(&stderr_path)?;

        let mut spawn_command = Command::new(shell);
        spawn_command.arg("-lc").arg(SHELL_WRAPPER);
        spawn_command.current_dir(cwd);
        spawn_command.stdin(Stdio::null());
        spawn_command.stdout(Stdio::from(stdout));
        spawn_command.stderr(Stdio::from(stderr));
        #[cfg(unix)]
        spawn_command.process_group(0);

        for (key, value) in std::env::vars() {
            spawn_command.env(key, value);
        }
        for (key, value) in env_vars {
            spawn_command.env(key, value);
        }
        spawn_command.env(INTERNAL_COMMAND_ENV, command);
        spawn_command.env(
            INTERNAL_CWD_FILE_ENV,
            cwd_capture_path.to_string_lossy().to_string(),
        );

        match spawn_command.spawn() {
            Ok(child) => Ok(Self {
                child: Some(child),
                cwd: cwd.to_path_buf(),
                shell: shell.to_string(),
                command: command.to_string(),
                stdout_path,
                stderr_path,
                cwd_capture_path: Some(cwd_capture_path),
                state: ProcessState::Running,
                background_task_id: None,
                forced_exit_code: None,
                completed: None,
            }),
            Err(error) => {
                cleanup_temp_file(&stdout_path);
                cleanup_temp_file(&stderr_path);
                cleanup_temp_file(&cwd_capture_path);
                Ok(Self::immediate(CommandResult {
                    cwd: cwd.to_string_lossy().to_string(),
                    shell: shell.to_string(),
                    command: command.to_string(),
                    stdout: String::new(),
                    stderr: error.to_string(),
                    code: SPAWN_FAILURE_EXIT_CODE,
                    interrupted: false,
                    background_task_id: None,
                    pre_spawn_error: None,
                }))
            }
        }
    }

    pub fn immediate(result: CommandResult) -> Self {
        Self {
            child: None,
            cwd: PathBuf::from(&result.cwd),
            shell: result.shell.clone(),
            command: result.command.clone(),
            stdout_path: PathBuf::new(),
            stderr_path: PathBuf::new(),
            cwd_capture_path: None,
            state: ProcessState::Completed,
            background_task_id: result.background_task_id.clone(),
            forced_exit_code: None,
            completed: Some(result),
        }
    }

    pub fn background(&mut self, task_id: &str) -> bool {
        if self.state != ProcessState::Running {
            return false;
        }
        self.state = ProcessState::Backgrounded;
        self.background_task_id = Some(task_id.to_string());
        true
    }

    pub fn kill(&mut self) {
        if self.completed.is_some() || self.state == ProcessState::Killed {
            return;
        }

        self.state = ProcessState::Killed;
        self.forced_exit_code = Some(KILLED_EXIT_CODE);

        if let Some(child) = self.child.as_mut() {
            #[cfg(unix)]
            {
                let _ = kill_process_group(child.id());
            }
            let _ = child.kill();
        }
    }

    pub fn poll(&mut self) -> Result<Option<CommandResult>> {
        if let Some(result) = &self.completed {
            return Ok(Some(result.clone()));
        }

        let Some(child) = self.child.as_mut() else {
            return Ok(None);
        };

        let Some(status) = child.try_wait().context("polling child process")? else {
            return Ok(None);
        };

        self.finalize(status).map(Some)
    }

    pub fn wait(&mut self) -> Result<CommandResult> {
        if let Some(result) = &self.completed {
            return Ok(result.clone());
        }

        let status = self
            .child
            .as_mut()
            .expect("managed child process")
            .wait()
            .context("waiting for child process")?;
        self.finalize(status)
    }

    pub fn take_cwd_after_success(&mut self) -> Result<Option<String>> {
        let Some(path) = self.cwd_capture_path.take() else {
            return Ok(None);
        };

        let contents = match fs::read_to_string(&path) {
            Ok(contents) => {
                let trimmed = contents.trim().to_string();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            }
            Err(_) => None,
        };
        cleanup_temp_file(&path);
        Ok(contents)
    }

    pub fn discard_cwd_capture(&mut self) {
        if let Some(path) = self.cwd_capture_path.take() {
            cleanup_temp_file(&path);
        }
    }

    fn finalize(&mut self, status: ExitStatus) -> Result<CommandResult> {
        let code = self
            .forced_exit_code
            .unwrap_or_else(|| status.code().unwrap_or(1));
        let stdout = read_output_file(&self.stdout_path)?;
        let stderr = read_output_file(&self.stderr_path)?;

        cleanup_temp_file(&self.stdout_path);
        cleanup_temp_file(&self.stderr_path);
        self.child = None;

        if matches!(
            self.state,
            ProcessState::Running | ProcessState::Backgrounded
        ) {
            self.state = ProcessState::Completed;
        }

        let result = CommandResult {
            cwd: self.cwd.to_string_lossy().to_string(),
            shell: self.shell.clone(),
            command: self.command.clone(),
            stdout,
            stderr,
            code,
            interrupted: code == KILLED_EXIT_CODE,
            background_task_id: self.background_task_id.clone(),
            pre_spawn_error: None,
        };
        self.completed = Some(result.clone());
        Ok(result)
    }
}

pub fn pre_spawn_failure(
    command: &str,
    cwd: &Path,
    shell: &str,
    message: impl Into<String>,
) -> ManagedProcess {
    let message = message.into();
    ManagedProcess::immediate(CommandResult {
        cwd: cwd.to_string_lossy().to_string(),
        shell: shell.to_string(),
        command: command.to_string(),
        stdout: String::new(),
        stderr: message.clone(),
        code: 1,
        interrupted: false,
        background_task_id: None,
        pre_spawn_error: Some(message),
    })
}

fn create_output_file(path: &Path) -> Result<File> {
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))
}

fn read_output_file(path: &Path) -> Result<String> {
    match fs::read(path) {
        Ok(bytes) => Ok(String::from_utf8_lossy(&bytes).into_owned()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(String::new()),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

fn unique_temp_path(kind: &str) -> PathBuf {
    let counter = TEMP_FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "nano-claude-code-{kind}-{}-{counter}",
        std::process::id()
    ))
}

fn cleanup_temp_file(path: &Path) {
    if path.as_os_str().is_empty() {
        return;
    }
    let _ = fs::remove_file(path);
}

#[cfg(unix)]
fn kill_process_group(pid: u32) -> Result<()> {
    Command::new("kill")
        .arg("-KILL")
        .arg(format!("-{pid}"))
        .status()
        .with_context(|| format!("killing process group {pid}"))?;
    Ok(())
}
