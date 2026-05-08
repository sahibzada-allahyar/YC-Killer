use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::{Result, anyhow, bail};
use serde::Serialize;

use crate::file_tools::{
    apply_head_limit, expand_path, ripgrep, split_glob_patterns, to_relative_path,
};

const FILE_NOT_FOUND_CWD_NOTE: &str = "Note: your current working directory is";
const VCS_DIRECTORIES_TO_EXCLUDE: &[&str] = &[".git", ".svn", ".hg", ".bzr", ".jj", ".sl"];

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GrepSearchOutput {
    pub mode: String,
    pub num_files: usize,
    pub filenames: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_lines: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_matches: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied_limit: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied_offset: Option<usize>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct GlobSearchOutput {
    pub duration_ms: usize,
    pub num_files: usize,
    pub filenames: Vec<String>,
    pub truncated: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrepOutputMode {
    Content,
    FilesWithMatches,
    Count,
}

impl GrepOutputMode {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Content => "content",
            Self::FilesWithMatches => "files_with_matches",
            Self::Count => "count",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct GrepSearchArgs {
    pub cwd: PathBuf,
    pub pattern: String,
    pub path: Option<String>,
    pub glob: Option<String>,
    pub output_mode: Option<GrepOutputMode>,
    pub context_before: Option<usize>,
    pub context_after: Option<usize>,
    pub context: Option<usize>,
    pub show_line_numbers: Option<bool>,
    pub case_insensitive: bool,
    pub file_type: Option<String>,
    pub head_limit: Option<usize>,
    pub offset: usize,
    pub multiline: bool,
}

#[derive(Debug, Clone)]
pub struct GlobSearchArgs {
    pub cwd: PathBuf,
    pub pattern: String,
    pub path: Option<String>,
    pub limit: usize,
    pub offset: usize,
}

pub fn grep_search(args: GrepSearchArgs) -> Result<GrepSearchOutput> {
    let absolute_path = resolve_search_path(&args.cwd, args.path.as_deref())?;
    validate_grep_path(&absolute_path, args.path.as_deref(), &args.cwd)?;

    let output_mode = args.output_mode.unwrap_or(GrepOutputMode::FilesWithMatches);
    let mut rg_args = vec!["--hidden".to_string()];
    for directory in VCS_DIRECTORIES_TO_EXCLUDE {
        rg_args.push("--glob".to_string());
        rg_args.push(format!("!{directory}"));
    }
    rg_args.push("--max-columns".to_string());
    rg_args.push("500".to_string());

    if args.multiline {
        rg_args.push("-U".to_string());
        rg_args.push("--multiline-dotall".to_string());
    }
    if args.case_insensitive {
        rg_args.push("-i".to_string());
    }

    match output_mode {
        GrepOutputMode::FilesWithMatches => rg_args.push("-l".to_string()),
        GrepOutputMode::Count => rg_args.push("-c".to_string()),
        GrepOutputMode::Content => {
            if args.show_line_numbers.unwrap_or(true) {
                rg_args.push("-n".to_string());
            }
            if let Some(context) = args.context {
                rg_args.push("-C".to_string());
                rg_args.push(context.to_string());
            } else {
                if let Some(before) = args.context_before {
                    rg_args.push("-B".to_string());
                    rg_args.push(before.to_string());
                }
                if let Some(after) = args.context_after {
                    rg_args.push("-A".to_string());
                    rg_args.push(after.to_string());
                }
            }
        }
    }

    if args.pattern.starts_with('-') {
        rg_args.push("-e".to_string());
        rg_args.push(args.pattern.clone());
    } else {
        rg_args.push(args.pattern.clone());
    }

    if let Some(file_type) = &args.file_type {
        rg_args.push("--type".to_string());
        rg_args.push(file_type.clone());
    }

    if let Some(glob) = &args.glob {
        for glob_pattern in split_glob_patterns(glob) {
            rg_args.push("--glob".to_string());
            rg_args.push(glob_pattern);
        }
    }

    let mut results = ripgrep(&rg_args, &absolute_path)?;
    if env_is_test() && !matches!(output_mode, GrepOutputMode::FilesWithMatches) {
        results.sort();
    }
    match output_mode {
        GrepOutputMode::Content => {
            let (limited, applied_limit) = apply_head_limit(&results, args.head_limit, args.offset);
            let final_lines = limited
                .into_iter()
                .map(|line| relativize_grep_line(&line, &args.cwd))
                .collect::<Vec<_>>();
            Ok(GrepSearchOutput {
                mode: GrepOutputMode::Content.as_str().to_string(),
                num_files: 0,
                filenames: Vec::new(),
                content: Some(final_lines.join("\n")),
                num_lines: Some(final_lines.len()),
                num_matches: None,
                applied_limit,
                applied_offset: (args.offset > 0).then_some(args.offset),
            })
        }
        GrepOutputMode::Count => {
            let (limited, applied_limit) = apply_head_limit(&results, args.head_limit, args.offset);
            let final_lines = limited
                .into_iter()
                .map(|line| relativize_count_line(&line, &args.cwd))
                .collect::<Vec<_>>();
            let (num_files, num_matches) =
                final_lines.iter().fold((0usize, 0usize), |acc, line| {
                    let Some(index) = line.rfind(':') else {
                        return acc;
                    };
                    let count = line[index + 1..].parse::<usize>().unwrap_or(0);
                    (acc.0 + 1, acc.1 + count)
                });
            Ok(GrepSearchOutput {
                mode: GrepOutputMode::Count.as_str().to_string(),
                num_files,
                filenames: Vec::new(),
                content: Some(final_lines.join("\n")),
                num_lines: None,
                num_matches: Some(num_matches),
                applied_limit,
                applied_offset: (args.offset > 0).then_some(args.offset),
            })
        }
        GrepOutputMode::FilesWithMatches => {
            let sorted = sort_match_paths(results, env_is_test())?;
            let (limited, applied_limit) = apply_head_limit(&sorted, args.head_limit, args.offset);
            let filenames = limited
                .into_iter()
                .map(|path| to_relative_path(Path::new(&path), &args.cwd))
                .collect::<Vec<_>>();
            Ok(GrepSearchOutput {
                mode: GrepOutputMode::FilesWithMatches.as_str().to_string(),
                num_files: filenames.len(),
                filenames,
                content: None,
                num_lines: None,
                num_matches: None,
                applied_limit,
                applied_offset: (args.offset > 0).then_some(args.offset),
            })
        }
    }
}

pub fn glob_search(args: GlobSearchArgs) -> Result<GlobSearchOutput> {
    let absolute_path = resolve_search_path(&args.cwd, args.path.as_deref())?;
    validate_glob_path(&absolute_path, args.path.as_deref(), &args.cwd)?;

    let rg_args = vec![
        "--files".to_string(),
        "--glob".to_string(),
        args.pattern.clone(),
        "--sort=modified".to_string(),
        "--no-ignore".to_string(),
        "--hidden".to_string(),
    ];

    let relative_paths = ripgrep(&rg_args, &absolute_path)?;
    let absolute_paths = relative_paths
        .into_iter()
        .map(|path| {
            let candidate = PathBuf::from(&path);
            if candidate.is_absolute() {
                candidate
            } else {
                absolute_path.join(candidate)
            }
        })
        .collect::<Vec<_>>();
    let truncated = absolute_paths.len() > args.offset + args.limit;
    let filenames = absolute_paths
        .into_iter()
        .skip(args.offset)
        .take(args.limit)
        .map(|path| to_relative_path(&path, &args.cwd))
        .collect::<Vec<_>>();

    Ok(GlobSearchOutput {
        duration_ms: 0,
        num_files: filenames.len(),
        filenames,
        truncated,
    })
}

fn resolve_search_path(cwd: &Path, raw_path: Option<&str>) -> Result<PathBuf> {
    let expanded = match raw_path {
        Some(path) => expand_path(path, Some(cwd))?,
        None => cwd.to_string_lossy().to_string(),
    };
    Ok(PathBuf::from(expanded))
}

fn validate_grep_path(path: &Path, raw_path: Option<&str>, cwd: &Path) -> Result<()> {
    if raw_path.is_none() {
        return Ok(());
    }
    if path.exists() {
        return Ok(());
    }
    bail!(
        "Path does not exist: {}. {FILE_NOT_FOUND_CWD_NOTE} {}.",
        raw_path.unwrap_or_default(),
        cwd.to_string_lossy()
    )
}

fn validate_glob_path(path: &Path, raw_path: Option<&str>, cwd: &Path) -> Result<()> {
    if raw_path.is_none() {
        return Ok(());
    }
    let metadata = fs::metadata(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            anyhow!(
                "Directory does not exist: {}. {FILE_NOT_FOUND_CWD_NOTE} {}.",
                raw_path.unwrap_or_default(),
                cwd.to_string_lossy()
            )
        } else {
            error.into()
        }
    })?;
    if metadata.is_dir() {
        return Ok(());
    }
    bail!("Path is not a directory: {}", raw_path.unwrap_or_default());
}

fn relativize_grep_line(line: &str, cwd: &Path) -> String {
    let Some(index) = line.find(':') else {
        return line.to_string();
    };
    let file_path = &line[..index];
    let rest = &line[index..];
    format!("{}{}", to_relative_path(Path::new(file_path), cwd), rest)
}

fn relativize_count_line(line: &str, cwd: &Path) -> String {
    let Some(index) = line.rfind(':') else {
        return line.to_string();
    };
    let file_path = &line[..index];
    let rest = &line[index..];
    format!("{}{}", to_relative_path(Path::new(file_path), cwd), rest)
}

fn sort_match_paths(paths: Vec<String>, deterministic: bool) -> Result<Vec<String>> {
    let mut entries = paths
        .into_iter()
        .map(|path| {
            let mtime = fs::metadata(&path)
                .ok()
                .and_then(|metadata| metadata.modified().ok())
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_millis())
                .unwrap_or(0);
            (path, mtime)
        })
        .collect::<Vec<_>>();

    entries.sort_by(|left, right| {
        if deterministic {
            left.0.cmp(&right.0)
        } else {
            right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        }
    });

    Ok(entries.into_iter().map(|(path, _)| path).collect())
}

fn env_is_test() -> bool {
    std::env::var("NODE_ENV").ok().as_deref() == Some("test")
}
