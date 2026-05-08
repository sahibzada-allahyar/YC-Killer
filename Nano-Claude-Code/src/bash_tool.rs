use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::thread;
use std::time::Duration;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::process::CommandResult;
use crate::shell::ShellRuntime;
use crate::task::TaskSnapshot;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BashScenario {
    pub initial_cwd: String,
    pub shell: String,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    pub steps: Vec<BashStep>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", tag = "op")]
pub enum BashStep {
    Foreground {
        command: String,
        #[serde(default)]
        env: BTreeMap<String, String>,
    },
    StartBackground {
        task: String,
        command: String,
        #[serde(default)]
        description: Option<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
    },
    Status {
        task: String,
    },
    Wait {
        task: String,
    },
    Cancel {
        task: String,
    },
    Cwd,
    ReadFile {
        path: String,
    },
    Exists {
        path: String,
    },
    Sleep {
        ms: u64,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct BashScenarioReport {
    pub steps: Vec<StepOutcome>,
    pub final_cwd: String,
    pub tasks: BTreeMap<String, TaskSnapshot>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct StepOutcome {
    pub op: String,
    pub cwd: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exec: Option<CommandResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<TaskSnapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exists: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub fn run_scenario(input: &BashScenario) -> Result<BashScenarioReport> {
    let mut runtime = ShellRuntime::new(Path::new(&input.initial_cwd), &input.shell, &input.env);
    let mut steps = Vec::with_capacity(input.steps.len());

    for step in &input.steps {
        steps.push(run_step(&mut runtime, step));
    }

    Ok(BashScenarioReport {
        steps,
        final_cwd: runtime.cwd(),
        tasks: runtime.snapshot_tasks()?,
    })
}

fn run_step(runtime: &mut ShellRuntime, step: &BashStep) -> StepOutcome {
    match step {
        BashStep::Foreground { command, env } => match runtime.foreground(command, env) {
            Ok(exec) => StepOutcome {
                op: "foreground".to_string(),
                cwd: runtime.cwd(),
                exec: Some(exec),
                task: None,
                exists: None,
                content: None,
                error: None,
            },
            Err(error) => error_outcome("foreground", runtime.cwd(), error.to_string()),
        },
        BashStep::StartBackground {
            task,
            command,
            description,
            env,
        } => match runtime.start_background(
            task,
            command,
            description.as_deref().unwrap_or(command),
            env,
        ) {
            Ok(snapshot) => StepOutcome {
                op: "startBackground".to_string(),
                cwd: runtime.cwd(),
                exec: None,
                task: Some(snapshot),
                exists: None,
                content: None,
                error: None,
            },
            Err(error) => error_outcome("startBackground", runtime.cwd(), error.to_string()),
        },
        BashStep::Status { task } => match runtime.status(task) {
            Ok(snapshot) => StepOutcome {
                op: "status".to_string(),
                cwd: runtime.cwd(),
                exec: None,
                task: Some(snapshot),
                exists: None,
                content: None,
                error: None,
            },
            Err(error) => error_outcome("status", runtime.cwd(), error.to_string()),
        },
        BashStep::Wait { task } => match runtime.wait(task) {
            Ok((snapshot, exec)) => StepOutcome {
                op: "wait".to_string(),
                cwd: runtime.cwd(),
                exec: Some(exec),
                task: Some(snapshot),
                exists: None,
                content: None,
                error: None,
            },
            Err(error) => error_outcome("wait", runtime.cwd(), error.to_string()),
        },
        BashStep::Cancel { task } => match runtime.cancel(task) {
            Ok(snapshot) => StepOutcome {
                op: "cancel".to_string(),
                cwd: runtime.cwd(),
                exec: None,
                task: Some(snapshot),
                exists: None,
                content: None,
                error: None,
            },
            Err(error) => error_outcome("cancel", runtime.cwd(), error.to_string()),
        },
        BashStep::Cwd => StepOutcome {
            op: "cwd".to_string(),
            cwd: runtime.cwd(),
            exec: None,
            task: None,
            exists: None,
            content: None,
            error: None,
        },
        BashStep::ReadFile { path } => {
            let resolved = runtime.resolve_path(path);
            match fs::read(&resolved) {
                Ok(bytes) => StepOutcome {
                    op: "readFile".to_string(),
                    cwd: runtime.cwd(),
                    exec: None,
                    task: None,
                    exists: Some(true),
                    content: Some(String::from_utf8_lossy(&bytes).into_owned()),
                    error: None,
                },
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => StepOutcome {
                    op: "readFile".to_string(),
                    cwd: runtime.cwd(),
                    exec: None,
                    task: None,
                    exists: Some(false),
                    content: None,
                    error: None,
                },
                Err(error) => error_outcome("readFile", runtime.cwd(), error.to_string()),
            }
        }
        BashStep::Exists { path } => StepOutcome {
            op: "exists".to_string(),
            cwd: runtime.cwd(),
            exec: None,
            task: None,
            exists: Some(runtime.resolve_path(path).exists()),
            content: None,
            error: None,
        },
        BashStep::Sleep { ms } => {
            thread::sleep(Duration::from_millis(*ms));
            StepOutcome {
                op: "sleep".to_string(),
                cwd: runtime.cwd(),
                exec: None,
                task: None,
                exists: None,
                content: None,
                error: None,
            }
        }
    }
}

fn error_outcome(op: &str, cwd: String, error: String) -> StepOutcome {
    StepOutcome {
        op: op.to_string(),
        cwd,
        exec: None,
        task: None,
        exists: None,
        content: None,
        error: Some(error),
    }
}
