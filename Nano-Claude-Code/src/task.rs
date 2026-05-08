use anyhow::Result;
use serde::Serialize;

use crate::process::{CommandResult, ManagedProcess};

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TaskResult {
    pub code: i32,
    pub interrupted: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TaskSnapshot {
    pub task_id: String,
    pub description: String,
    pub command: String,
    pub status: String,
    pub is_backgrounded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<TaskResult>,
}

pub struct BackgroundTask {
    snapshot: TaskSnapshot,
    process: ManagedProcess,
}

impl BackgroundTask {
    pub fn start(
        task_id: &str,
        description: &str,
        command: &str,
        mut process: ManagedProcess,
    ) -> Self {
        let _ = process.background(task_id);
        Self {
            snapshot: TaskSnapshot {
                task_id: task_id.to_string(),
                description: description.to_string(),
                command: command.to_string(),
                status: "running".to_string(),
                is_backgrounded: true,
                result: None,
            },
            process,
        }
    }

    pub fn status(&mut self) -> Result<TaskSnapshot> {
        self.poll()?;
        Ok(self.snapshot.clone())
    }

    pub fn wait(&mut self) -> Result<(TaskSnapshot, CommandResult)> {
        let result = self.process.wait()?;
        self.process.discard_cwd_capture();
        if self.snapshot.status != "killed" {
            self.snapshot.status = if result.code == 0 {
                "completed".to_string()
            } else {
                "failed".to_string()
            };
            self.snapshot.result = Some(TaskResult {
                code: result.code,
                interrupted: result.interrupted,
            });
        }
        Ok((self.snapshot.clone(), result))
    }

    pub fn cancel(&mut self) -> Result<TaskSnapshot> {
        self.process.kill();
        self.snapshot.status = "killed".to_string();
        self.snapshot.result = None;
        Ok(self.snapshot.clone())
    }

    fn poll(&mut self) -> Result<()> {
        let Some(result) = self.process.poll()? else {
            return Ok(());
        };

        self.process.discard_cwd_capture();
        if self.snapshot.status == "killed" {
            return Ok(());
        }

        self.snapshot.status = if result.code == 0 {
            "completed".to_string()
        } else {
            "failed".to_string()
        };
        self.snapshot.result = Some(TaskResult {
            code: result.code,
            interrupted: result.interrupted,
        });
        Ok(())
    }
}
