use std::collections::BTreeMap;
use std::env;
use std::fmt::{Display, Formatter};
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, IsTerminal, Read, Write, stdin, stdout};
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::agent::{
    AgentRecord, AgentTypeInfo, built_in_agents, close_agent, list_agent_records, send_agent,
    spawn_agent_with_options, wait_agent,
};
use crate::anthropic::RequestMessage;
use crate::config::{
    AnthropicApiKeySource, anthropic_api_key_path, load_anthropic_api_key, parse_env_vars,
    save_anthropic_api_key,
};
use crate::core::{
    ConversationMessage, DEFAULT_MODEL, MessageRole, RunOptions, effective_model, run_with_options,
};
use crate::edit::{EditReport, perform_edit};
use crate::mcp::{McpCallReport, McpCallResult, McpRouteReport, call_mcp_tool, route_mcp_tool};
use crate::session::{
    ChatReport, ListSessionsOptions, RunReport, SessionHandle, append_turn, build_chat_report,
    get_session_info, get_session_transcript_scan, list_sessions, load_conversation_history,
    load_live_request_history, load_session_handle, now_ms, write_session,
};
use crate::shell::{ExecReport, default_shell, execute as execute_shell};

#[derive(Debug)]
pub struct CliError {
    message: String,
    exit_code: i32,
}

impl CliError {
    fn usage(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            exit_code: 2,
        }
    }

    fn internal(error: anyhow::Error) -> Self {
        Self {
            message: error.to_string(),
            exit_code: 1,
        }
    }

    pub fn exit_code(&self) -> i32 {
        self.exit_code
    }
}

impl Display for CliError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for CliError {}

#[derive(Debug)]
struct RunArgs {
    cwd: PathBuf,
    shell: String,
    model: String,
    prompt: String,
    env_vars: BTreeMap<String, String>,
    json: bool,
}

#[derive(Debug)]
struct ChatArgs {
    cwd: Option<PathBuf>,
    shell: String,
    model: String,
    resume_session_id: Option<String>,
    env_vars: BTreeMap<String, String>,
    json: bool,
}

#[derive(Debug)]
struct PsArgs {
    dir: Option<PathBuf>,
    limit: Option<usize>,
    offset: usize,
    json: bool,
}

#[derive(Debug)]
struct InfoArgs {
    dir: Option<PathBuf>,
    session_id: String,
    json: bool,
}

#[derive(Debug)]
struct TranscriptArgs {
    dir: Option<PathBuf>,
    session_id: String,
    json: bool,
}

#[derive(Debug)]
struct ExecArgs {
    cwd: PathBuf,
    shell: String,
    command: String,
    env_vars: BTreeMap<String, String>,
    json: bool,
}

#[derive(Debug)]
struct EditArgs {
    file_path: PathBuf,
    old_string: String,
    new_string: String,
    replace_all: bool,
    json: bool,
}

#[derive(Debug)]
struct AgentSpawnArgs {
    cwd: PathBuf,
    shell: String,
    model: String,
    agent_type: String,
    prompt: String,
    env_vars: BTreeMap<String, String>,
    json: bool,
}

#[derive(Debug)]
struct AgentWaitArgs {
    agent_id: String,
    json: bool,
}

#[derive(Debug)]
struct AgentSendArgs {
    agent_id: String,
    prompt: String,
    json: bool,
}

#[derive(Debug)]
struct AgentCloseArgs {
    agent_id: String,
    json: bool,
}

#[derive(Debug)]
struct McpRouteArgs {
    tool_name: String,
    json: bool,
}

#[derive(Debug)]
struct McpCallArgs {
    cwd: PathBuf,
    shell: String,
    tool_name: String,
    input: Option<String>,
    env_vars: BTreeMap<String, String>,
    json: bool,
}

#[derive(Debug)]
struct SetupArgs {
    force: bool,
    json: bool,
}

#[derive(Debug)]
struct StartupArgs {
    cwd: Option<PathBuf>,
    shell: String,
    model: String,
    resume_session_id: Option<String>,
    continue_latest: bool,
    env_vars: BTreeMap<String, String>,
    print_mode: bool,
    prompt: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct SetupReport {
    configured: bool,
    saved: bool,
    source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    path: Option<String>,
}

struct ChatSessionState {
    handle: Option<SessionHandle>,
    cwd: PathBuf,
    model: String,
    history: Vec<ConversationMessage>,
    api_history: Option<Vec<RequestMessage>>,
    assistant_texts: Vec<String>,
    next_turn_ms: i64,
}

enum InteractiveCommandResult {
    Continue,
    HandledLocally,
}

pub fn run(argv: Vec<String>) -> Result<(), CliError> {
    maybe_forward_to_c_compiler(&argv).map_err(CliError::internal)?;

    let args = argv.get(1..).unwrap_or(&[]);

    if args.len() == 1 && matches!(args[0].as_str(), "--version" | "-v" | "-V") {
        println!("0.1.0 (nano-claude-code)");
        return Ok(());
    }

    let first = args.first().map(String::as_str);
    let handled_as_subcommand = matches!(
        first,
        Some(
            "setup"
                | "run"
                | "chat"
                | "ps"
                | "info"
                | "transcript"
                | "exec"
                | "edit"
                | "agent"
                | "mcp"
        )
    );

    if !handled_as_subcommand {
        let startup = parse_startup_args(args)?;
        return execute_startup(startup);
    }

    match first {
        Some("setup") => {
            let cmd = parse_setup_args(&args[1..])?;
            let json = cmd.json;
            let report = execute_setup(cmd).map_err(CliError::internal)?;
            print_setup(report, json)
        }
        Some("run") => {
            let cmd = parse_run_args(&args[1..])?;
            let json = cmd.json;
            let report = execute_run(cmd).map_err(CliError::internal)?;
            print_run(report, json)
        }
        Some("chat") => {
            let cmd = parse_chat_args(&args[1..])?;
            if should_use_interactive_chat(&cmd) {
                execute_interactive_chat(cmd, None)?;
                return Ok(());
            }
            if cmd.json && stdin().is_terminal() {
                return Err(CliError::usage(
                    "interactive chat does not support --json; omit --json or pipe prompts on stdin",
                ));
            }
            let json = cmd.json;
            let report = execute_chat(cmd)?;
            print_chat(report, json)
        }
        Some("ps") => {
            let cmd = parse_ps_args(&args[1..])?;
            let sessions = list_sessions(ListSessionsOptions {
                dir: cmd.dir,
                limit: cmd.limit,
                offset: cmd.offset,
            })
            .map_err(CliError::internal)?;
            if cmd.json {
                println!(
                    "{}",
                    serde_json::to_string(&sessions)
                        .map_err(|err| CliError::internal(err.into()))?
                );
            } else {
                for session in sessions {
                    println!("{}\t{}", session.session_id, session.summary);
                }
            }
            Ok(())
        }
        Some("info") => {
            let cmd = parse_info_args(&args[1..])?;
            let info = get_session_info(&cmd.session_id, cmd.dir.as_deref())
                .map_err(CliError::internal)?;
            if cmd.json {
                println!(
                    "{}",
                    serde_json::to_string(&info).map_err(|err| CliError::internal(err.into()))?
                );
            } else if let Some(info) = info {
                println!("{}", info.summary);
            }
            Ok(())
        }
        Some("transcript") => {
            let cmd = parse_transcript_args(&args[1..])?;
            let scan = get_session_transcript_scan(&cmd.session_id, cmd.dir.as_deref())
                .map_err(CliError::internal)?;
            if cmd.json {
                println!(
                    "{}",
                    serde_json::to_string(&scan).map_err(|err| CliError::internal(err.into()))?
                );
            } else if let Some(scan) = scan {
                print!("{}", scan.transcript);
            }
            Ok(())
        }
        Some("exec") => {
            let cmd = parse_exec_args(&args[1..])?;
            let json = cmd.json;
            let report = execute_exec(cmd).map_err(CliError::internal)?;
            print_exec(report, json)
        }
        Some("edit") => {
            let cmd = parse_edit_args(&args[1..])?;
            let json = cmd.json;
            let report = execute_edit(cmd).map_err(CliError::internal)?;
            print_edit(report, json)
        }
        Some("agent") => run_agent_command(&args[1..]),
        Some("mcp") => run_mcp_command(&args[1..]),
        Some(other) => Err(CliError::usage(format!("unknown command: {other}"))),
        None => Err(CliError::usage("expected a command")),
    }
}

fn parse_startup_args(args: &[String]) -> Result<StartupArgs, CliError> {
    let cwd = parse_string_flag("--cwd", args)?.map(PathBuf::from);
    let shell = eager_parse_cli_flag("--shell", args).unwrap_or_else(default_shell);
    let model = eager_parse_cli_flag("--model", args).unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let json = args.iter().any(|arg| arg == "--json");
    if json {
        return Err(CliError::usage(
            "top-level startup does not support --json; use explicit subcommands instead",
        ));
    }

    let resume_session_id = parse_resume_flag(args)?;
    let continue_latest = args.iter().any(|arg| arg == "-c" || arg == "--continue");
    if continue_latest && resume_session_id.is_some() {
        return Err(CliError::usage(
            "cannot combine -c/--continue with -r/--resume",
        ));
    }

    let print_mode = args.iter().any(|arg| arg == "-p" || arg == "--print");
    let env_values = collect_env_values(args)?;
    let env_vars = parse_env_vars(&env_values).map_err(CliError::usage)?;
    let positionals = collect_startup_positionals(args)?;
    let prompt = if positionals.is_empty() {
        None
    } else {
        Some(positionals.join(" ").trim().to_string())
    };

    Ok(StartupArgs {
        cwd,
        shell,
        model,
        resume_session_id,
        continue_latest,
        env_vars,
        print_mode,
        prompt,
    })
}

fn execute_startup(mut args: StartupArgs) -> Result<(), CliError> {
    if args.continue_latest {
        args.resume_session_id = Some(resolve_latest_session_id(args.cwd.as_deref())?);
    }

    if args.print_mode {
        let prompt = args
            .prompt
            .ok_or_else(|| CliError::usage("-p/--print requires a prompt"))?;
        let cwd = args
            .cwd
            .or_else(|| std::env::current_dir().ok())
            .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
        let report = execute_run(RunArgs {
            cwd,
            shell: args.shell,
            model: args.model,
            prompt,
            env_vars: args.env_vars,
            json: false,
        })
        .map_err(CliError::internal)?;
        return print_run(report, false);
    }

    let chat_args = ChatArgs {
        cwd: args.cwd,
        shell: args.shell,
        model: args.model,
        resume_session_id: args.resume_session_id,
        env_vars: args.env_vars,
        json: false,
    };

    if should_use_interactive_chat(&chat_args) {
        return execute_interactive_chat(chat_args, args.prompt);
    }

    if let Some(prompt) = args.prompt {
        let cwd = chat_args
            .cwd
            .clone()
            .or_else(|| std::env::current_dir().ok())
            .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
        let report = execute_run(RunArgs {
            cwd,
            shell: chat_args.shell,
            model: chat_args.model,
            prompt,
            env_vars: chat_args.env_vars,
            json: false,
        })
        .map_err(CliError::internal)?;
        return print_run(report, false);
    }

    let report = execute_chat(chat_args)?;
    print_chat(report, false)
}

fn resolve_latest_session_id(dir: Option<&Path>) -> Result<String, CliError> {
    let cwd = dir
        .map(Path::to_path_buf)
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
    let sessions = list_sessions(ListSessionsOptions {
        dir: Some(cwd),
        limit: Some(1),
        offset: 0,
    })
    .map_err(CliError::internal)?;
    let Some(session) = sessions.into_iter().next() else {
        return Err(CliError::usage("no existing sessions to continue"));
    };
    Ok(session.session_id)
}

fn maybe_forward_to_c_compiler(argv: &[String]) -> Result<()> {
    let args = argv.get(1..).unwrap_or(&[]);
    if !looks_like_c_compiler_invocation(args) {
        return Ok(());
    }

    let forwarded_args = normalize_compiler_args(args);
    let current_exe = env::current_exe().ok();
    let compiler = find_real_c_compiler(current_exe.as_deref())
        .ok_or_else(|| anyhow::anyhow!("unable to locate a real C compiler for CC passthrough"))?;

    let status = Command::new(&compiler)
        .args(&forwarded_args)
        .env_remove("CC")
        .status()
        .with_context(|| format!("forwarding compiler invocation to {compiler}"))?;
    std::process::exit(status.code().unwrap_or(1));
}

fn looks_like_c_compiler_invocation(args: &[String]) -> bool {
    if args.is_empty() {
        return false;
    }
    if !args[0].starts_with('-') {
        return false;
    }

    args.iter().any(|arg| {
        arg.starts_with("-O")
            || arg.starts_with("-I")
            || arg.starts_with("-D")
            || arg.starts_with("-U")
            || arg.starts_with("-f")
            || arg.starts_with("-W")
            || arg.starts_with("-m")
            || arg.starts_with("-std=")
            || arg.starts_with("-arch")
            || arg.starts_with("-isysroot")
            || arg.starts_with("-Fo")
            || arg.starts_with("/Fo")
            || arg.starts_with("/wd")
            || arg.starts_with("/Zc:")
            || matches!(
                arg.as_str(),
                "-nologo" | "-Brepro" | "-Z7" | "-W4" | "/Wall"
            )
            || arg == "-E"
            || arg == "-S"
            || arg.ends_with(".c")
            || arg.ends_with(".cc")
            || arg.ends_with(".cpp")
            || arg.ends_with(".cxx")
            || arg.ends_with(".s")
            || arg.ends_with(".S")
            || arg.ends_with(".o")
    })
}

fn normalize_compiler_args(args: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        let arg = &args[idx];
        match arg.as_str() {
            "-nologo" | "-Brepro" => {}
            "-Z7" => normalized.push("-g".to_string()),
            "-W4" | "/Wall" => normalized.push("-Wall".to_string()),
            "/Gy" => normalized.push("-ffunction-sections".to_string()),
            "/Zc:wchar_t" | "/Zc:forScope" | "/Zc:inline" => {}
            value if value.starts_with("/wd") => {}
            value if value.starts_with("-Fo") || value.starts_with("/Fo") => {
                let output = value[3..].to_string();
                if !output.is_empty() {
                    normalized.push("-o".to_string());
                    normalized.push(output);
                } else if let Some(next) = args.get(idx + 1) {
                    normalized.push("-o".to_string());
                    normalized.push(next.clone());
                    idx += 1;
                }
            }
            "--" => {
                normalized.extend(args[idx + 1..].iter().cloned());
                break;
            }
            _ => normalized.push(arg.clone()),
        }
        idx += 1;
    }
    normalized
}

fn find_real_c_compiler(current_exe: Option<&Path>) -> Option<String> {
    let mut candidates = Vec::new();
    if let Ok(value) = env::var("HOST_CC") {
        candidates.push(value);
    }
    candidates.push("cc".to_string());
    candidates.push("clang".to_string());

    for candidate in candidates {
        let trimmed = candidate.trim();
        if trimmed.is_empty() {
            continue;
        }
        if current_exe.is_some_and(|exe| Path::new(trimmed) == exe) {
            continue;
        }
        return Some(trimmed.to_string());
    }

    None
}

pub fn eager_parse_cli_flag(flag_name: &str, argv: &[String]) -> Option<String> {
    for (idx, arg) in argv.iter().enumerate() {
        if let Some(rest) = arg.strip_prefix(&format!("{flag_name}=")) {
            return Some(rest.to_string());
        }

        if arg == flag_name {
            return argv.get(idx + 1).cloned();
        }
    }

    None
}

pub fn extract_args_after_double_dash(
    command_or_value: &str,
    args: &[String],
) -> (String, Vec<String>) {
    if command_or_value == "--" && !args.is_empty() {
        return (args[0].clone(), args[1..].to_vec());
    }

    (command_or_value.to_string(), args.to_vec())
}

fn parse_run_args(args: &[String]) -> Result<RunArgs, CliError> {
    let cwd = eager_parse_cli_flag("--cwd", args)
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
    let shell = eager_parse_cli_flag("--shell", args).unwrap_or_else(default_shell);
    let model = eager_parse_cli_flag("--model", args).unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let json = args.iter().any(|arg| arg == "--json");
    let env_values = collect_env_values(args)?;
    let env_vars = parse_env_vars(&env_values).map_err(CliError::usage)?;
    let positionals = collect_positionals(args)?;
    let Some(first) = positionals.first() else {
        return Err(CliError::usage("run requires a prompt"));
    };
    let (command, rest) = extract_args_after_double_dash(first, &positionals[1..]);
    let mut prompt_parts = vec![command];
    prompt_parts.extend(rest);
    let prompt = prompt_parts.join(" ").trim().to_string();
    if prompt.is_empty() {
        return Err(CliError::usage("run requires a prompt"));
    }

    Ok(RunArgs {
        cwd,
        shell,
        model,
        prompt,
        env_vars,
        json,
    })
}

fn parse_chat_args(args: &[String]) -> Result<ChatArgs, CliError> {
    let cwd = parse_string_flag("--cwd", args)?.map(PathBuf::from);
    let shell = eager_parse_cli_flag("--shell", args).unwrap_or_else(default_shell);
    let model = eager_parse_cli_flag("--model", args).unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let resume_session_id = parse_string_flag("--resume", args)?;
    let json = args.iter().any(|arg| arg == "--json");
    let env_values = collect_env_values(args)?;
    let env_vars = parse_env_vars(&env_values).map_err(CliError::usage)?;
    let positionals = collect_chat_positionals(args)?;
    if !positionals.is_empty() {
        return Err(CliError::usage(
            "chat does not accept prompt arguments; pipe prompts on stdin",
        ));
    }

    Ok(ChatArgs {
        cwd,
        shell,
        model,
        resume_session_id,
        env_vars,
        json,
    })
}

fn parse_ps_args(args: &[String]) -> Result<PsArgs, CliError> {
    let all = args.iter().any(|arg| arg == "--all");
    let dir = if all {
        None
    } else {
        Some(
            eager_parse_cli_flag("--cwd", args)
                .map(PathBuf::from)
                .or_else(|| std::env::current_dir().ok())
                .ok_or_else(|| CliError::usage("could not resolve current directory"))?,
        )
    };
    let limit = parse_usize_flag("--limit", args)?;
    let offset = parse_usize_flag("--offset", args)?.unwrap_or(0);
    let json = args.iter().any(|arg| arg == "--json");
    Ok(PsArgs {
        dir,
        limit,
        offset,
        json,
    })
}

fn parse_info_args(args: &[String]) -> Result<InfoArgs, CliError> {
    let dir = eager_parse_cli_flag("--cwd", args).map(PathBuf::from);
    let positionals = collect_positionals(args)?;
    let Some(session_id) = positionals.first() else {
        return Err(CliError::usage("info requires a session id"));
    };
    let json = args.iter().any(|arg| arg == "--json");
    Ok(InfoArgs {
        dir,
        session_id: session_id.clone(),
        json,
    })
}

fn parse_transcript_args(args: &[String]) -> Result<TranscriptArgs, CliError> {
    let dir = eager_parse_cli_flag("--cwd", args).map(PathBuf::from);
    let positionals = collect_positionals(args)?;
    let Some(session_id) = positionals.first() else {
        return Err(CliError::usage("transcript requires a session id"));
    };
    let json = args.iter().any(|arg| arg == "--json");
    Ok(TranscriptArgs {
        dir,
        session_id: session_id.clone(),
        json,
    })
}

fn parse_exec_args(args: &[String]) -> Result<ExecArgs, CliError> {
    let cwd = eager_parse_cli_flag("--cwd", args)
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
    let shell = eager_parse_cli_flag("--shell", args).unwrap_or_else(default_shell);
    let json = args.iter().any(|arg| arg == "--json");
    let env_values = collect_env_values(args)?;
    let env_vars = parse_env_vars(&env_values).map_err(CliError::usage)?;
    let positionals = collect_exec_positionals(args)?;
    let Some(first) = positionals.first() else {
        return Err(CliError::usage("exec requires a command"));
    };
    let (command, rest) = extract_args_after_double_dash(first, &positionals[1..]);
    let mut command_parts = vec![command];
    command_parts.extend(rest);
    let command = command_parts.join(" ").trim().to_string();
    if command.is_empty() {
        return Err(CliError::usage("exec requires a command"));
    }

    Ok(ExecArgs {
        cwd,
        shell,
        command,
        env_vars,
        json,
    })
}

fn parse_edit_args(args: &[String]) -> Result<EditArgs, CliError> {
    let cwd = eager_parse_cli_flag("--cwd", args)
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
    let raw_file_path = eager_parse_cli_flag("--file", args)
        .ok_or_else(|| CliError::usage("edit requires --file"))?;
    let old_string = eager_parse_cli_flag("--old", args)
        .ok_or_else(|| CliError::usage("edit requires --old"))?;
    let new_string = eager_parse_cli_flag("--new", args)
        .ok_or_else(|| CliError::usage("edit requires --new"))?;
    let replace_all = args.iter().any(|arg| arg == "--replace-all");
    let json = args.iter().any(|arg| arg == "--json");

    Ok(EditArgs {
        file_path: resolve_against_cwd(&cwd, &raw_file_path),
        old_string,
        new_string,
        replace_all,
        json,
    })
}

fn parse_agent_spawn_args(args: &[String]) -> Result<AgentSpawnArgs, CliError> {
    let cwd = eager_parse_cli_flag("--cwd", args)
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
    let shell = eager_parse_cli_flag("--shell", args).unwrap_or_else(default_shell);
    let model = eager_parse_cli_flag("--model", args).unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let agent_type = eager_parse_cli_flag("--type", args)
        .ok_or_else(|| CliError::usage("agent spawn requires --type"))?;
    let json = args.iter().any(|arg| arg == "--json");
    let env_values = collect_env_values(args)?;
    let env_vars = parse_env_vars(&env_values).map_err(CliError::usage)?;
    let positionals = collect_agent_spawn_positionals(args)?;
    let Some(first) = positionals.first() else {
        return Err(CliError::usage("agent spawn requires a prompt"));
    };
    let (command, rest) = extract_args_after_double_dash(first, &positionals[1..]);
    let mut prompt_parts = vec![command];
    prompt_parts.extend(rest);
    let prompt = prompt_parts.join(" ").trim().to_string();
    if prompt.is_empty() {
        return Err(CliError::usage("agent spawn requires a prompt"));
    }

    Ok(AgentSpawnArgs {
        cwd,
        shell,
        model,
        agent_type,
        prompt,
        env_vars,
        json,
    })
}

fn parse_agent_wait_args(args: &[String]) -> Result<AgentWaitArgs, CliError> {
    let json = args.iter().any(|arg| arg == "--json");
    let positionals = collect_positionals(args)?;
    let Some(agent_id) = positionals.first() else {
        return Err(CliError::usage("agent wait requires an agent id"));
    };

    Ok(AgentWaitArgs {
        agent_id: agent_id.clone(),
        json,
    })
}

fn parse_agent_send_args(args: &[String]) -> Result<AgentSendArgs, CliError> {
    let json = args.iter().any(|arg| arg == "--json");
    let positionals = collect_positionals(args)?;
    let Some(agent_id) = positionals.first() else {
        return Err(CliError::usage("agent send requires an agent id"));
    };
    let Some(first) = positionals.get(1) else {
        return Err(CliError::usage("agent send requires a prompt"));
    };
    let (command, rest) = extract_args_after_double_dash(first, &positionals[2..]);
    let mut prompt_parts = vec![command];
    prompt_parts.extend(rest);
    let prompt = prompt_parts.join(" ").trim().to_string();
    if prompt.is_empty() {
        return Err(CliError::usage("agent send requires a prompt"));
    }

    Ok(AgentSendArgs {
        agent_id: agent_id.clone(),
        prompt,
        json,
    })
}

fn parse_agent_close_args(args: &[String]) -> Result<AgentCloseArgs, CliError> {
    let json = args.iter().any(|arg| arg == "--json");
    let positionals = collect_positionals(args)?;
    let Some(agent_id) = positionals.first() else {
        return Err(CliError::usage("agent close requires an agent id"));
    };

    Ok(AgentCloseArgs {
        agent_id: agent_id.clone(),
        json,
    })
}

fn parse_mcp_route_args(args: &[String]) -> Result<McpRouteArgs, CliError> {
    let json = args.iter().any(|arg| arg == "--json");
    let positionals = collect_positionals(args)?;
    let Some(tool_name) = positionals.first() else {
        return Err(CliError::usage("mcp route requires a tool name"));
    };

    Ok(McpRouteArgs {
        tool_name: tool_name.clone(),
        json,
    })
}

fn parse_mcp_call_args(args: &[String]) -> Result<McpCallArgs, CliError> {
    let cwd = eager_parse_cli_flag("--cwd", args)
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
    let shell = eager_parse_cli_flag("--shell", args).unwrap_or_else(default_shell);
    let input = parse_string_flag("--input", args)?;
    let json = args.iter().any(|arg| arg == "--json");
    let env_values = collect_env_values(args)?;
    let env_vars = parse_env_vars(&env_values).map_err(CliError::usage)?;
    let positionals = collect_mcp_call_positionals(args)?;
    let Some(tool_name) = positionals.first() else {
        return Err(CliError::usage("mcp call requires a tool name"));
    };

    Ok(McpCallArgs {
        cwd,
        shell,
        tool_name: tool_name.clone(),
        input,
        env_vars,
        json,
    })
}

fn parse_setup_args(args: &[String]) -> Result<SetupArgs, CliError> {
    let force = args.iter().any(|arg| arg == "--force");
    let json = args.iter().any(|arg| arg == "--json");
    let positionals = collect_setup_positionals(args)?;
    if !positionals.is_empty() {
        return Err(CliError::usage(
            "setup does not accept positional arguments",
        ));
    }

    Ok(SetupArgs { force, json })
}

fn parse_resume_flag(args: &[String]) -> Result<Option<String>, CliError> {
    let short = parse_string_flag("-r", args)?;
    let long = parse_string_flag("--resume", args)?;
    match (short, long) {
        (Some(_), Some(_)) => Err(CliError::usage("resume session specified more than once")),
        (Some(value), None) | (None, Some(value)) => Ok(Some(value)),
        (None, None) => Ok(None),
    }
}

fn collect_env_values(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut values = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "-e" | "--env" => {
                let Some(value) = args.get(idx + 1) else {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                };
                values.push(value.clone());
                idx += 2;
            }
            arg if arg.starts_with("--env=") => {
                values.push(arg["--env=".len()..].to_string());
                idx += 1;
            }
            _ => idx += 1,
        }
    }

    Ok(values)
}

fn collect_setup_positionals(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut positionals = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--force" | "--json" => idx += 1,
            other => {
                positionals.push(other.to_string());
                idx += 1;
            }
        }
    }

    Ok(positionals)
}

fn collect_startup_positionals(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut positionals = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--cwd" | "--shell" | "--model" | "-e" | "--env" | "-r" | "--resume" => {
                if args.get(idx + 1).is_none() {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                }
                idx += 2;
            }
            "-p" | "--print" | "-c" | "--continue" => idx += 1,
            arg if arg.starts_with("--cwd=")
                || arg.starts_with("--shell=")
                || arg.starts_with("--model=")
                || arg.starts_with("--env=")
                || arg.starts_with("--resume=") =>
            {
                idx += 1;
            }
            "--" => {
                positionals.extend(args[idx + 1..].iter().cloned());
                break;
            }
            other => {
                positionals.push(other.to_string());
                idx += 1;
            }
        }
    }

    Ok(positionals)
}

fn collect_positionals(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut positionals = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--cwd" | "--shell" | "--model" | "-e" | "--env" => {
                if args.get(idx + 1).is_none() {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                }
                idx += 2;
            }
            "--json" => idx += 1,
            arg if arg.starts_with("--cwd=")
                || arg.starts_with("--shell=")
                || arg.starts_with("--model=")
                || arg.starts_with("--env=") =>
            {
                idx += 1;
            }
            "--" => {
                positionals.push("--".to_string());
                positionals.extend(args[idx + 1..].iter().cloned());
                break;
            }
            other => {
                positionals.push(other.to_string());
                idx += 1;
            }
        }
    }

    Ok(positionals)
}

fn collect_chat_positionals(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut positionals = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--cwd" | "--shell" | "--model" | "--resume" | "-e" | "--env" => {
                if args.get(idx + 1).is_none() {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                }
                idx += 2;
            }
            "--json" => idx += 1,
            arg if arg.starts_with("--cwd=")
                || arg.starts_with("--shell=")
                || arg.starts_with("--model=")
                || arg.starts_with("--resume=")
                || arg.starts_with("--env=") =>
            {
                idx += 1;
            }
            other => {
                positionals.push(other.to_string());
                idx += 1;
            }
        }
    }

    Ok(positionals)
}

fn collect_exec_positionals(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut positionals = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--cwd" | "--shell" | "-e" | "--env" => {
                if args.get(idx + 1).is_none() {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                }
                idx += 2;
            }
            "--json" => idx += 1,
            arg if arg.starts_with("--cwd=")
                || arg.starts_with("--shell=")
                || arg.starts_with("--env=") =>
            {
                idx += 1;
            }
            "--" => {
                positionals.push("--".to_string());
                positionals.extend(args[idx + 1..].iter().cloned());
                break;
            }
            other => {
                positionals.push(other.to_string());
                idx += 1;
            }
        }
    }

    Ok(positionals)
}

fn collect_agent_spawn_positionals(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut positionals = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--cwd" | "--shell" | "--model" | "--type" | "-e" | "--env" => {
                if args.get(idx + 1).is_none() {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                }
                idx += 2;
            }
            "--json" => idx += 1,
            arg if arg.starts_with("--cwd=")
                || arg.starts_with("--shell=")
                || arg.starts_with("--model=")
                || arg.starts_with("--type=")
                || arg.starts_with("--env=") =>
            {
                idx += 1
            }
            "--" => {
                positionals.push("--".to_string());
                positionals.extend(args[idx + 1..].iter().cloned());
                break;
            }
            other => {
                positionals.push(other.to_string());
                idx += 1;
            }
        }
    }
    Ok(positionals)
}

fn collect_mcp_call_positionals(args: &[String]) -> Result<Vec<String>, CliError> {
    let mut positionals = Vec::new();
    let mut idx = 0;
    while idx < args.len() {
        match args[idx].as_str() {
            "--cwd" | "--shell" | "--input" | "-e" | "--env" => {
                if args.get(idx + 1).is_none() {
                    return Err(CliError::usage(format!("missing value for {}", args[idx])));
                }
                idx += 2;
            }
            "--json" => idx += 1,
            arg if arg.starts_with("--cwd=")
                || arg.starts_with("--shell=")
                || arg.starts_with("--input=")
                || arg.starts_with("--env=") =>
            {
                idx += 1;
            }
            "--" => {
                positionals.push("--".to_string());
                positionals.extend(args[idx + 1..].iter().cloned());
                break;
            }
            other => {
                positionals.push(other.to_string());
                idx += 1;
            }
        }
    }

    Ok(positionals)
}

fn parse_usize_flag(flag_name: &str, args: &[String]) -> Result<Option<usize>, CliError> {
    let Some(raw) = eager_parse_cli_flag(flag_name, args) else {
        return Ok(None);
    };

    raw.parse::<usize>()
        .map(Some)
        .map_err(|_| CliError::usage(format!("invalid value for {flag_name}: {raw}")))
}

fn parse_string_flag(flag_name: &str, args: &[String]) -> Result<Option<String>, CliError> {
    for (idx, arg) in args.iter().enumerate() {
        if let Some(rest) = arg.strip_prefix(&format!("{flag_name}=")) {
            return Ok(Some(rest.to_string()));
        }

        if arg == flag_name {
            let Some(value) = args.get(idx + 1) else {
                return Err(CliError::usage(format!("missing value for {flag_name}")));
            };
            return Ok(Some(value.clone()));
        }
    }

    Ok(None)
}

fn execute_setup(args: SetupArgs) -> Result<SetupReport> {
    if !args.force {
        if let Some(existing) = load_anthropic_api_key() {
            return Ok(setup_report_for_existing(existing.source));
        }
    }

    let Some(api_key) = read_setup_api_key()? else {
        return Ok(SetupReport {
            configured: false,
            saved: false,
            source: "none".to_string(),
            path: None,
        });
    };

    let path = save_anthropic_api_key(&api_key).with_context(|| {
        format!(
            "saving Anthropic API key to {}",
            anthropic_api_key_path().display()
        )
    })?;
    Ok(SetupReport {
        configured: true,
        saved: true,
        source: AnthropicApiKeySource::Config.label().to_string(),
        path: Some(path.display().to_string()),
    })
}

fn execute_run(args: RunArgs) -> Result<RunReport> {
    maybe_prompt_for_live_setup(prompt_wants_live_mode(&args.prompt))?;
    let model = effective_session_model(&args.model);
    let outcome = run_with_options(RunOptions {
        prompt: &args.prompt,
        history: &[],
        api_history: None,
        cwd: &args.cwd,
        shell: &args.shell,
        env_vars: &args.env_vars,
        model: &model,
    })?;
    let persisted = write_session(&args.cwd, &model, &args.prompt, &outcome)
        .with_context(|| format!("persisting run for {}", args.cwd.display()))?;
    Ok(persisted.report)
}

fn print_run(report: RunReport, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        println!("{}", report.assistant_text);
    }
    Ok(())
}

fn should_use_interactive_chat(args: &ChatArgs) -> bool {
    !args.json && stdin().is_terminal()
}

fn execute_chat(args: ChatArgs) -> Result<ChatReport, CliError> {
    let prompts = read_chat_prompts()?;
    if prompts.is_empty() {
        return Err(CliError::usage(
            "chat requires at least one prompt on stdin",
        ));
    }
    let mut state = start_chat_session(&args)?;
    for prompt in prompts {
        run_chat_turn(&args, &mut state, &prompt)?;
    }
    Ok(finish_chat_report(state))
}

fn execute_interactive_chat(
    args: ChatArgs,
    initial_prompt: Option<String>,
) -> Result<(), CliError> {
    let mut state = start_chat_session(&args)?;
    let mut announced_session_id = state
        .handle
        .as_ref()
        .map(|handle| handle.session_id.clone());

    print_interactive_banner(&state, &args, announced_session_id.as_deref())?;

    if let Some(prompt) = initial_prompt.as_deref() {
        print_interactive_user_prompt(prompt)?;
        if matches!(
            handle_interactive_command(prompt, &args, &mut state, &mut announced_session_id)?,
            InteractiveCommandResult::Continue
        ) {
            let assistant_text = run_chat_turn(&args, &mut state, prompt)?;
            maybe_print_session_line(&state, &mut announced_session_id)?;
            print_interactive_assistant(&state, announced_session_id.as_deref(), &assistant_text)?;
        }
    }

    let stdin = stdin();
    let mut reader = stdin.lock();
    loop {
        print_interactive_prompt()?;
        stdout()
            .flush()
            .map_err(|err| CliError::internal(err.into()))?;

        let Some(prompt) = read_interactive_prompt(&mut reader)? else {
            println!();
            break;
        };
        if prompt.trim().is_empty() {
            continue;
        }
        if is_exit_prompt(&prompt) {
            break;
        }

        if matches!(
            handle_interactive_command(&prompt, &args, &mut state, &mut announced_session_id)?,
            InteractiveCommandResult::HandledLocally
        ) {
            continue;
        }

        let assistant_text = run_chat_turn(&args, &mut state, &prompt)?;
        maybe_print_session_line(&state, &mut announced_session_id)?;
        print_interactive_assistant(&state, announced_session_id.as_deref(), &assistant_text)?;
    }

    Ok(())
}

fn print_chat(report: ChatReport, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        for assistant_text in report.assistant_texts {
            if assistant_text.ends_with('\n') {
                print!("{assistant_text}");
            } else {
                println!("{assistant_text}");
            }
        }
    }
    Ok(())
}

fn start_chat_session(args: &ChatArgs) -> Result<ChatSessionState, CliError> {
    if let Some(session_id) = args.resume_session_id.as_deref() {
        let handle = load_session_handle(session_id, args.cwd.as_deref())
            .map_err(CliError::internal)?
            .ok_or_else(|| {
                CliError::internal(anyhow::anyhow!("session not found: {session_id}"))
            })?;
        let history =
            load_conversation_history(&handle.session_path).map_err(CliError::internal)?;
        let api_history =
            load_live_request_history(&handle.session_path).map_err(CliError::internal)?;
        let cwd = PathBuf::from(&handle.cwd);
        let model = handle.model.clone();
        return Ok(ChatSessionState {
            handle: Some(handle),
            cwd,
            model,
            history,
            api_history,
            assistant_texts: Vec::new(),
            next_turn_ms: now_ms(),
        });
    }

    let cwd = args
        .cwd
        .clone()
        .or_else(|| std::env::current_dir().ok())
        .ok_or_else(|| CliError::usage("could not resolve current directory"))?;
    Ok(ChatSessionState {
        handle: None,
        cwd,
        model: effective_session_model(&args.model),
        history: Vec::new(),
        api_history: None,
        assistant_texts: Vec::new(),
        next_turn_ms: now_ms() + 1,
    })
}

fn run_chat_turn(
    args: &ChatArgs,
    state: &mut ChatSessionState,
    prompt: &str,
) -> Result<String, CliError> {
    maybe_prompt_for_live_setup(prompt_wants_live_mode(prompt)).map_err(CliError::internal)?;
    let outcome = run_with_options(RunOptions {
        prompt,
        history: &state.history,
        api_history: state.api_history.as_deref(),
        cwd: &state.cwd,
        shell: &args.shell,
        env_vars: &args.env_vars,
        model: &state.model,
    })
    .map_err(CliError::internal)?;

    if let Some(handle) = state.handle.as_ref() {
        append_turn(handle, prompt, &outcome, state.next_turn_ms).map_err(CliError::internal)?;
        state.next_turn_ms += 1;
    } else {
        let persisted = write_session(&state.cwd, &state.model, prompt, &outcome)
            .map_err(CliError::internal)?;
        state.handle = Some(SessionHandle {
            session_id: persisted.report.session_id,
            session_path: PathBuf::from(persisted.report.session_path),
            cwd: persisted.report.cwd,
            model: persisted.report.model,
        });
    }

    extend_history(&mut state.history, prompt, &outcome.assistant_text);
    state.api_history = outcome.api_messages.clone();
    state.assistant_texts.push(outcome.assistant_text.clone());
    Ok(outcome.assistant_text)
}

fn finish_chat_report(state: ChatSessionState) -> ChatReport {
    let handle = state
        .handle
        .expect("chat report requires at least one completed turn");
    build_chat_report(&handle, state.assistant_texts)
}

fn read_interactive_prompt(reader: &mut impl BufRead) -> Result<Option<String>, CliError> {
    let mut input = String::new();
    let read = reader
        .read_line(&mut input)
        .map_err(|err| CliError::internal(err.into()))?;
    if read == 0 {
        return Ok(None);
    }
    Ok(Some(
        input
            .trim_end_matches('\n')
            .trim_end_matches('\r')
            .to_string(),
    ))
}

fn is_exit_prompt(prompt: &str) -> bool {
    matches!(prompt.trim(), "/exit" | "/quit")
}

fn handle_interactive_command(
    prompt: &str,
    args: &ChatArgs,
    state: &mut ChatSessionState,
    announced_session_id: &mut Option<String>,
) -> Result<InteractiveCommandResult, CliError> {
    match prompt.trim() {
        "/help" | "/?" => {
            print_interactive_help_panel(state, args, announced_session_id.as_deref())?;
            Ok(InteractiveCommandResult::HandledLocally)
        }
        "/status" => {
            print_interactive_status_panel(state, args, announced_session_id.as_deref())?;
            Ok(InteractiveCommandResult::HandledLocally)
        }
        "/clear" => {
            clear_interactive_screen()?;
            print_interactive_banner(state, args, announced_session_id.as_deref())?;
            Ok(InteractiveCommandResult::HandledLocally)
        }
        _ => Ok(InteractiveCommandResult::Continue),
    }
}

fn print_interactive_banner(
    state: &ChatSessionState,
    args: &ChatArgs,
    session_id: Option<&str>,
) -> Result<(), CliError> {
    let mut out = stdout();
    let rule = ui_rule('=');
    let title = ui_accent("nano-claude-code");
    let version = ui_dim("v0.1.0");
    let mode = interactive_mode_label();

    writeln!(out, "{} {}", title, version).map_err(|err| CliError::internal(err.into()))?;
    writeln!(out, "{rule}").map_err(|err| CliError::internal(err.into()))?;
    for line in interactive_art_lines() {
        writeln!(out, "{}", ui_dim(line)).map_err(|err| CliError::internal(err.into()))?;
    }
    writeln!(out, "{rule}").map_err(|err| CliError::internal(err.into()))?;
    writeln!(
        out,
        "{} {}",
        ui_label("workspace"),
        truncate_middle(&display_workspace_path(&state.cwd), ui_content_width())
    )
    .map_err(|err| CliError::internal(err.into()))?;
    writeln!(out, "{} {}", ui_label("model"), state.model)
        .map_err(|err| CliError::internal(err.into()))?;
    writeln!(out, "{} {}", ui_label("shell"), args.shell)
        .map_err(|err| CliError::internal(err.into()))?;
    writeln!(out, "{} {}", ui_label("mode"), mode).map_err(|err| CliError::internal(err.into()))?;
    writeln!(
        out,
        "{} {}",
        ui_label("session"),
        session_id
            .map(str::to_string)
            .unwrap_or_else(|| "created on first message".to_string())
    )
    .map_err(|err| CliError::internal(err.into()))?;
    writeln!(out).map_err(|err| CliError::internal(err.into()))?;
    for line in interactive_quick_start_lines() {
        writeln!(out, "{line}").map_err(|err| CliError::internal(err.into()))?;
    }
    writeln!(out).map_err(|err| CliError::internal(err.into()))?;
    out.flush().map_err(|err| CliError::internal(err.into()))
}

fn print_interactive_prompt() -> Result<(), CliError> {
    let mut out = stdout();
    write!(out, "{} ", ui_prompt("you>")).map_err(|err| CliError::internal(err.into()))?;
    out.flush().map_err(|err| CliError::internal(err.into()))
}

fn print_interactive_user_prompt(prompt: &str) -> Result<(), CliError> {
    let mut out = stdout();
    writeln!(out, "{} {}", ui_prompt("you>"), prompt)
        .map_err(|err| CliError::internal(err.into()))?;
    out.flush().map_err(|err| CliError::internal(err.into()))
}

fn maybe_print_session_line(
    state: &ChatSessionState,
    announced_session_id: &mut Option<String>,
) -> Result<(), CliError> {
    if announced_session_id.is_some() {
        return Ok(());
    }

    if let Some(handle) = state.handle.as_ref() {
        let mut out = stdout();
        writeln!(out, "{} {}", ui_label("Session:"), handle.session_id)
            .map_err(|err| CliError::internal(err.into()))?;
        writeln!(out).map_err(|err| CliError::internal(err.into()))?;
        *announced_session_id = Some(handle.session_id.clone());
    }

    Ok(())
}

fn print_interactive_assistant(
    state: &ChatSessionState,
    session_id: Option<&str>,
    assistant_text: &str,
) -> Result<(), CliError> {
    let mut out = stdout();
    writeln!(out, "{} {}", ui_prompt("assistant>"), ui_section_rule())
        .map_err(|err| CliError::internal(err.into()))?;
    for line in assistant_text.lines() {
        writeln!(out, "{} {}", ui_dim("|"), line).map_err(|err| CliError::internal(err.into()))?;
    }
    if assistant_text.is_empty() || assistant_text.ends_with('\n') {
        writeln!(out, "{}", ui_dim("|")).map_err(|err| CliError::internal(err.into()))?;
    }
    writeln!(out, "{}", interactive_status_footer(state, session_id))
        .map_err(|err| CliError::internal(err.into()))?;
    writeln!(out).map_err(|err| CliError::internal(err.into()))?;
    out.flush().map_err(|err| CliError::internal(err.into()))
}

fn print_interactive_help_panel(
    state: &ChatSessionState,
    args: &ChatArgs,
    session_id: Option<&str>,
) -> Result<(), CliError> {
    let mut out = stdout();
    writeln!(out, "{} {}", ui_label("help"), ui_section_rule())
        .map_err(|err| CliError::internal(err.into()))?;
    for line in interactive_help_lines(state, args, session_id) {
        writeln!(out, "{line}").map_err(|err| CliError::internal(err.into()))?;
    }
    writeln!(out).map_err(|err| CliError::internal(err.into()))?;
    out.flush().map_err(|err| CliError::internal(err.into()))
}

fn print_interactive_status_panel(
    state: &ChatSessionState,
    args: &ChatArgs,
    session_id: Option<&str>,
) -> Result<(), CliError> {
    let mut out = stdout();
    writeln!(out, "{} {}", ui_label("status"), ui_section_rule())
        .map_err(|err| CliError::internal(err.into()))?;
    writeln!(
        out,
        "{} {}",
        ui_label("workspace"),
        truncate_middle(&display_workspace_path(&state.cwd), ui_content_width())
    )
    .map_err(|err| CliError::internal(err.into()))?;
    writeln!(out, "{} {}", ui_label("model"), state.model)
        .map_err(|err| CliError::internal(err.into()))?;
    writeln!(out, "{} {}", ui_label("shell"), args.shell)
        .map_err(|err| CliError::internal(err.into()))?;
    writeln!(out, "{} {}", ui_label("mode"), interactive_mode_label())
        .map_err(|err| CliError::internal(err.into()))?;
    writeln!(
        out,
        "{} {}",
        ui_label("session"),
        session_id
            .map(str::to_string)
            .unwrap_or_else(|| "created on first message".to_string())
    )
    .map_err(|err| CliError::internal(err.into()))?;
    writeln!(out).map_err(|err| CliError::internal(err.into()))?;
    out.flush().map_err(|err| CliError::internal(err.into()))
}

fn clear_interactive_screen() -> Result<(), CliError> {
    if stdout().is_terminal() {
        let mut out = stdout();
        write!(out, "\x1b[2J\x1b[H").map_err(|err| CliError::internal(err.into()))?;
        out.flush().map_err(|err| CliError::internal(err.into()))?;
    }
    Ok(())
}

fn interactive_art_lines() -> &'static [&'static str] {
    &[
        "         .          .",
        "   .       ____        .        _",
        "          / __ \\___  ____ _____(_)___",
        "     /\\_/\\\\ / / / / _ \\/ __ `/ __/ / __ \\",
        "    ( o.o )/ /_/ /  __/ /_/ / /_/ / /_/ /",
        "     > ^ </_____/\\___/\\__,_/\\__/_/\\____/",
    ]
}

fn interactive_quick_start_lines() -> Vec<String> {
    vec![
        ui_label("quick start").to_string(),
        "  Ask naturally about the repo, then let the app inspect, edit, test, and spawn helpers."
            .to_string(),
        format!(
            "  {}  show terminal commands and startup shortcuts",
            ui_prompt("/help")
        ),
        format!(
            "  {} show workspace, model, mode, and session",
            ui_prompt("/status")
        ),
        format!("  {} list the built-in demo tools", ui_prompt("/tools")),
        format!("  {}  quit the session", ui_prompt("/exit")),
    ]
}

fn interactive_help_lines(
    state: &ChatSessionState,
    _args: &ChatArgs,
    session_id: Option<&str>,
) -> Vec<String> {
    vec![
        format!(
            "{} {}",
            ui_label("slash"),
            "/help  /status  /clear  /tools  /exit"
        ),
        format!("{} {}", ui_label("startup"), "nano-claude-code"),
        format!(
            "{} {}",
            ui_label("prompt"),
            "nano-claude-code \"inspect this repo and summarize startup\""
        ),
        format!(
            "{} {}",
            ui_label("one-shot"),
            "nano-claude-code -p \"run one non-interactive prompt\""
        ),
        format!("{} {}", ui_label("continue"), "nano-claude-code -c"),
        format!(
            "{} {}",
            ui_label("resume"),
            session_id
                .map(|id| format!("nano-claude-code -r {id}"))
                .unwrap_or_else(|| "nano-claude-code -r <session-id>".to_string())
        ),
        format!(
            "{} {}",
            ui_label("workspace"),
            truncate_middle(&display_workspace_path(&state.cwd), ui_content_width())
        ),
    ]
}

fn interactive_mode_label() -> String {
    if load_anthropic_api_key().is_some() {
        ui_success("live")
    } else {
        ui_dim("mock")
    }
}

fn interactive_status_footer(state: &ChatSessionState, session_id: Option<&str>) -> String {
    let session = session_id
        .map(short_session_id)
        .unwrap_or_else(|| "pending".to_string());
    let model = truncate_middle(&state.model, 28);
    ui_dim(&format!(
        "[session {session}] [model {model}] [mode {}]",
        if load_anthropic_api_key().is_some() {
            "live"
        } else {
            "mock"
        }
    ))
}

fn short_session_id(session_id: &str) -> String {
    session_id.chars().take(8).collect()
}

fn ui_color_enabled() -> bool {
    stdout().is_terminal() && env::var_os("NO_COLOR").is_none()
}

fn ui_style(text: &str, code: &str) -> String {
    if ui_color_enabled() {
        format!("\x1b[{code}m{text}\x1b[0m")
    } else {
        text.to_string()
    }
}

fn ui_accent(text: &str) -> String {
    ui_style(text, "1;38;5;216")
}

fn ui_prompt(text: &str) -> String {
    ui_style(text, "1;38;5;117")
}

fn ui_label(text: &str) -> String {
    ui_style(text, "1;38;5;180")
}

fn ui_success(text: &str) -> String {
    ui_style(text, "1;38;5;114")
}

fn ui_dim(text: &str) -> String {
    ui_style(text, "2;38;5;247")
}

fn ui_width() -> usize {
    env::var("COLUMNS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value >= 40)
        .map(|value| value.min(88))
        .unwrap_or(72)
}

fn ui_content_width() -> usize {
    ui_width().saturating_sub(14)
}

fn ui_rule(ch: char) -> String {
    ch.to_string().repeat(ui_width())
}

fn ui_section_rule() -> String {
    "-".repeat(ui_width().saturating_sub(12).max(16))
}

fn truncate_middle(value: &str, max_width: usize) -> String {
    let chars: Vec<char> = value.chars().collect();
    if chars.len() <= max_width {
        return value.to_string();
    }
    if max_width <= 5 {
        return chars.iter().take(max_width).collect();
    }

    let left = (max_width - 3) / 2;
    let right = max_width - 3 - left;
    let prefix: String = chars.iter().take(left).collect();
    let suffix: String = chars[chars.len().saturating_sub(right)..].iter().collect();
    format!("{prefix}...{suffix}")
}

fn display_workspace_path(path: &Path) -> String {
    let display = path.to_string_lossy();
    let Ok(home) = env::var("HOME") else {
        return display.into_owned();
    };
    if home.is_empty() {
        return display.into_owned();
    }

    if let Ok(relative) = path.strip_prefix(&home) {
        if relative.as_os_str().is_empty() {
            "~".to_string()
        } else {
            format!("~/{}", relative.display())
        }
    } else {
        display.into_owned()
    }
}

fn print_setup(report: SetupReport, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|err| CliError::internal(err.into()))?
        );
        return Ok(());
    }

    if report.saved {
        println!(
            "Saved Anthropic API key to {}",
            report.path.as_deref().unwrap_or_default()
        );
    } else if report.configured && report.source == AnthropicApiKeySource::Env.label() {
        println!("Anthropic API key already available from ANTHROPIC_API_KEY");
    } else if report.configured {
        println!(
            "Anthropic API key already saved at {}",
            report.path.as_deref().unwrap_or_default()
        );
    } else {
        println!("No Anthropic API key saved");
    }

    Ok(())
}

fn effective_session_model(requested_model: &str) -> String {
    effective_model(requested_model)
}

fn extend_history(history: &mut Vec<ConversationMessage>, prompt: &str, assistant_text: &str) {
    history.push(ConversationMessage {
        role: MessageRole::User,
        content: prompt.to_string(),
    });
    history.push(ConversationMessage {
        role: MessageRole::Assistant,
        content: assistant_text.to_string(),
    });
}

fn execute_exec(args: ExecArgs) -> Result<ExecReport> {
    execute_shell(&args.command, &args.cwd, &args.shell, &args.env_vars)
        .with_context(|| format!("executing command in {}", args.cwd.display()))
}

fn print_exec(report: ExecReport, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        print!("{}", report.stdout);
        eprint!("{}", report.stderr);
    }
    Ok(())
}

fn execute_edit(args: EditArgs) -> Result<EditReport> {
    perform_edit(
        &args.file_path,
        &args.old_string,
        &args.new_string,
        args.replace_all,
    )
}

fn print_edit(report: EditReport, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        println!("Edited {}", report.file_path);
    }
    Ok(())
}

fn resolve_against_cwd(cwd: &Path, raw_path: &str) -> PathBuf {
    let file_path = PathBuf::from(raw_path);
    if file_path.is_absolute() {
        file_path
    } else {
        cwd.join(file_path)
    }
}

fn read_chat_prompts() -> Result<Vec<String>, CliError> {
    let mut input = String::new();
    stdin()
        .read_to_string(&mut input)
        .map_err(|err| CliError::internal(err.into()))?;
    Ok(input
        .lines()
        .map(|line| line.trim_end_matches('\r').to_string())
        .filter(|line| !line.trim().is_empty())
        .collect())
}

fn setup_report_for_existing(source: AnthropicApiKeySource) -> SetupReport {
    let path = if source == AnthropicApiKeySource::Config {
        Some(anthropic_api_key_path().display().to_string())
    } else {
        None
    };
    SetupReport {
        configured: true,
        saved: false,
        source: source.label().to_string(),
        path,
    }
}

fn prompt_wants_live_mode(prompt: &str) -> bool {
    let trimmed = prompt.trim_start();
    !trimmed.is_empty() && !trimmed.starts_with('/')
}

fn maybe_prompt_for_live_setup(should_prompt: bool) -> Result<()> {
    if !should_prompt || load_anthropic_api_key().is_some() {
        return Ok(());
    }

    let Some(api_key) = prompt_for_api_key_on_tty(
        "Anthropic API key not configured.",
        "Paste your Anthropic API key to enable live mode, or press Enter to stay on the mock path: ",
    )?
    else {
        return Ok(());
    };

    let path = save_anthropic_api_key(&api_key).with_context(|| {
        format!(
            "saving Anthropic API key to {}",
            anthropic_api_key_path().display()
        )
    })?;
    let _ = write_line_to_tty(&format!(
        "Saved Anthropic API key to {}. Live mode is now enabled.",
        path.display()
    ));
    Ok(())
}

fn read_setup_api_key() -> Result<Option<String>> {
    if stdin().is_terminal() {
        return prompt_for_api_key_on_tty(
            "Paste your Anthropic API key to enable live mode.",
            "Anthropic API key: ",
        );
    }

    let mut input = String::new();
    stdin().read_to_string(&mut input)?;
    let trimmed = input.trim();
    if trimmed.is_empty() {
        Ok(None)
    } else {
        Ok(Some(trimmed.to_string()))
    }
}

fn prompt_for_api_key_on_tty(header: &str, prompt: &str) -> Result<Option<String>> {
    let mut tty = match OpenOptions::new().read(true).write(true).open("/dev/tty") {
        Ok(file) => file,
        Err(_) => return Ok(None),
    };
    writeln!(tty, "{header}")?;
    write!(tty, "{prompt}")?;
    tty.flush()?;

    let mut reader = BufReader::new(tty.try_clone()?);
    let mut input = String::new();
    reader.read_line(&mut input)?;
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    Ok(Some(trimmed.to_string()))
}

fn write_line_to_tty(message: &str) -> Result<()> {
    let mut tty = OpenOptions::new().write(true).open("/dev/tty")?;
    writeln!(tty, "{message}")?;
    Ok(())
}

fn run_agent_command(args: &[String]) -> Result<(), CliError> {
    match args.first().map(String::as_str) {
        Some("types") => {
            let json = args[1..].iter().any(|arg| arg == "--json");
            print_agent_types(built_in_agents(), json)
        }
        Some("spawn") => {
            let cmd = parse_agent_spawn_args(&args[1..])?;
            let json = cmd.json;
            maybe_prompt_for_live_setup(prompt_wants_live_mode(&cmd.prompt))
                .map_err(CliError::internal)?;
            let record = spawn_agent_with_options(
                &cmd.agent_type,
                &cmd.cwd,
                &cmd.prompt,
                &cmd.shell,
                &cmd.model,
                &cmd.env_vars,
            )
            .map_err(CliError::internal)?;
            print_agent_record(record, json)
        }
        Some("wait") => {
            let cmd = parse_agent_wait_args(&args[1..])?;
            let record = wait_agent(&cmd.agent_id).map_err(CliError::internal)?;
            print_optional_agent_record(record, cmd.json)
        }
        Some("send") => {
            let cmd = parse_agent_send_args(&args[1..])?;
            let record = send_agent(&cmd.agent_id, &cmd.prompt).map_err(CliError::internal)?;
            print_optional_agent_record(record, cmd.json)
        }
        Some("close") => {
            let cmd = parse_agent_close_args(&args[1..])?;
            let record = close_agent(&cmd.agent_id).map_err(CliError::internal)?;
            print_optional_agent_close_record(record, cmd.json)
        }
        Some("ps") => {
            let json = args[1..].iter().any(|arg| arg == "--json");
            let records = list_agent_records().map_err(CliError::internal)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string(&records)
                        .map_err(|err| CliError::internal(err.into()))?
                );
            } else {
                for record in records {
                    println!(
                        "{}\t{}\t{}",
                        record.agent_id, record.agent_type, record.status
                    );
                }
            }
            Ok(())
        }
        Some(other) => Err(CliError::usage(format!("unknown agent command: {other}"))),
        None => Err(CliError::usage("agent requires a subcommand")),
    }
}

fn run_mcp_command(args: &[String]) -> Result<(), CliError> {
    match args.first().map(String::as_str) {
        Some("route") => {
            let cmd = parse_mcp_route_args(&args[1..])?;
            let route = route_mcp_tool(&cmd.tool_name).map_err(CliError::internal)?;
            print_mcp_route(route, cmd.json)
        }
        Some("call") => {
            let cmd = parse_mcp_call_args(&args[1..])?;
            let report = call_mcp_tool(
                &cmd.tool_name,
                cmd.input.as_deref(),
                &cmd.cwd,
                &cmd.shell,
                &cmd.env_vars,
            )
            .map_err(CliError::internal)?;
            print_mcp_call(report, cmd.json)
        }
        Some(other) => Err(CliError::usage(format!("unknown mcp command: {other}"))),
        None => Err(CliError::usage("mcp requires a subcommand")),
    }
}

fn print_agent_types(types: Vec<AgentTypeInfo>, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&types).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        for agent in types {
            println!("{}\t{}", agent.agent_type, agent.when_to_use);
        }
    }
    Ok(())
}

fn print_agent_record(record: AgentRecord, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&record).map_err(|err| CliError::internal(err.into()))?
        );
    } else {
        println!("{}", record.agent_id);
    }
    Ok(())
}

fn print_optional_agent_record(record: Option<AgentRecord>, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&record).map_err(|err| CliError::internal(err.into()))?
        );
    } else if let Some(record) = record {
        println!("{}", record.response);
    }
    Ok(())
}

fn print_optional_agent_close_record(
    record: Option<AgentRecord>,
    json: bool,
) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&record).map_err(|err| CliError::internal(err.into()))?
        );
    } else if let Some(record) = record {
        println!("{}", record.agent_id);
    }
    Ok(())
}

fn print_mcp_route(report: McpRouteReport, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|err| CliError::internal(err.into()))?
        );
    } else if let Some(tool_name) = report.tool_name {
        println!("{}\t{}\t{}", report.server_name, tool_name, report.route);
    } else {
        println!("{}\t{}", report.server_name, report.route);
    }
    Ok(())
}

fn print_mcp_call(report: McpCallReport, json: bool) -> Result<(), CliError> {
    if json {
        println!(
            "{}",
            serde_json::to_string(&report).map_err(|err| CliError::internal(err.into()))?
        );
        return Ok(());
    }

    match report.result {
        McpCallResult::BashExec { report } => print_exec(report, false),
        McpCallResult::FileEdit { report } => print_edit(report, false),
        McpCallResult::AgentTypes { types } => print_agent_types(types, false),
        McpCallResult::AgentSpawn { record } => print_agent_record(record, false),
        McpCallResult::AgentWait { record } | McpCallResult::AgentSend { record } => {
            print_optional_agent_record(record, false)
        }
        McpCallResult::AgentClose { record } => print_optional_agent_close_record(record, false),
        McpCallResult::AgentPs { records } => {
            for record in records {
                println!(
                    "{}\t{}\t{}",
                    record.agent_id, record.agent_type, record.status
                );
            }
            Ok(())
        }
        McpCallResult::SessionInfo { info } => {
            if let Some(info) = info {
                println!("{}", info.summary);
            }
            Ok(())
        }
        McpCallResult::SessionTranscript { scan } => {
            if let Some(scan) = scan {
                print!("{}", scan.transcript);
            }
            Ok(())
        }
        McpCallResult::SessionList { sessions } => {
            for session in sessions {
                println!("{}\t{}", session.session_id, session.summary);
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::display_workspace_path;
    use std::env;
    use std::path::PathBuf;

    #[test]
    fn display_workspace_path_uses_tilde_for_home_descendants() {
        let home = env::var("HOME").expect("HOME");
        let path = PathBuf::from(&home).join("demo").join("repo");
        assert_eq!(display_workspace_path(&path), "~/demo/repo");
    }

    #[test]
    fn display_workspace_path_leaves_external_paths_alone() {
        let path = PathBuf::from("/tmp/nano-claude-code-demo");
        assert_eq!(display_workspace_path(&path), "/tmp/nano-claude-code-demo");
    }
}
