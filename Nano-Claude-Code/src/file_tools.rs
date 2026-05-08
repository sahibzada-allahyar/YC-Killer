use std::env;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Result, anyhow, bail};
use serde::Serialize;
use unicode_normalization::UnicodeNormalization;

const DEFAULT_GREP_HEAD_LIMIT: usize = 250;
const DIFF_CONTEXT_LINES: usize = 3;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum FileEncoding {
    #[serde(rename = "utf8")]
    Utf8,
    #[serde(rename = "utf16le")]
    Utf16Le,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub enum LineEndingType {
    #[serde(rename = "CRLF")]
    CrLf,
    #[serde(rename = "LF")]
    Lf,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct FileWithMetadata {
    pub content: String,
    pub encoding: FileEncoding,
    pub line_endings: LineEndingType,
    pub file_exists: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct DiffHunk {
    pub old_start: usize,
    pub old_lines: usize,
    pub new_start: usize,
    pub new_lines: usize,
    pub lines: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct TextReadFile {
    pub file_path: String,
    pub content: String,
    pub num_lines: usize,
    pub start_line: usize,
    pub total_lines: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct TextReadOutput {
    #[serde(rename = "type")]
    pub type_name: String,
    pub file: TextReadFile,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum FileWriteType {
    #[serde(rename = "create")]
    Create,
    #[serde(rename = "update")]
    Update,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct FileWriteReport {
    #[serde(rename = "type")]
    pub type_name: FileWriteType,
    pub file_path: String,
    pub content: String,
    pub structured_patch: Vec<DiffHunk>,
    pub original_file: Option<String>,
}

pub fn expand_path(raw_path: &str, base_dir: Option<&Path>) -> Result<String> {
    let actual_base_dir = base_dir
        .map(PathBuf::from)
        .unwrap_or_else(|| env::current_dir().expect("current directory"));

    if raw_path.contains('\0') || actual_base_dir.to_string_lossy().contains('\0') {
        bail!("Path contains null bytes");
    }

    let trimmed = raw_path.trim();
    if trimmed.is_empty() {
        return Ok(normalize_nfc_path(&normalize_path(&actual_base_dir)));
    }

    let expanded = if trimmed == "~" {
        homedir_path()?
    } else if let Some(rest) = trimmed.strip_prefix("~/") {
        normalize_path(&homedir_path()?.join(rest))
    } else {
        let candidate = PathBuf::from(trimmed);
        if candidate.is_absolute() {
            normalize_path(&candidate)
        } else {
            normalize_path(&actual_base_dir.join(candidate))
        }
    };

    Ok(normalize_nfc_path(&expanded))
}

pub fn to_relative_path(absolute_path: &Path, cwd: &Path) -> String {
    let Ok(relative) = absolute_path.strip_prefix(cwd) else {
        return normalize_nfc_path(absolute_path);
    };

    let rendered = relative.to_string_lossy();
    if rendered.starts_with("..") {
        normalize_nfc_path(absolute_path)
    } else {
        rendered.nfc().collect()
    }
}

pub fn contains_path_traversal(path: &str) -> bool {
    path.split(['/', '\\']).any(|segment| segment == "..")
}

pub fn read_file_for_edit(file_path: &Path) -> Result<FileWithMetadata> {
    match fs::read(file_path) {
        Ok(bytes) => {
            let encoding = detect_encoding(&bytes);
            let raw = decode_bytes(&bytes, encoding);
            let line_endings = detect_line_endings(sample_head(&raw, 4096));
            Ok(FileWithMetadata {
                content: raw.replace("\r\n", "\n"),
                encoding,
                line_endings,
                file_exists: true,
            })
        }
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(FileWithMetadata {
            content: String::new(),
            encoding: FileEncoding::Utf8,
            line_endings: LineEndingType::Lf,
            file_exists: false,
        }),
        Err(err) => Err(err.into()),
    }
}

pub fn write_text_content(
    file_path: &Path,
    content: &str,
    encoding: FileEncoding,
    endings: LineEndingType,
) -> Result<()> {
    let mut to_write = content.to_string();
    if endings == LineEndingType::CrLf {
        to_write = content.replace("\r\n", "\n").replace('\n', "\r\n");
    }

    let bytes = match encoding {
        FileEncoding::Utf8 => to_write.into_bytes(),
        FileEncoding::Utf16Le => {
            let mut bytes = Vec::with_capacity(to_write.len() * 2);
            for unit in to_write.encode_utf16() {
                bytes.extend_from_slice(&unit.to_le_bytes());
            }
            bytes
        }
    };

    fs::write(file_path, bytes)?;
    Ok(())
}

pub fn read_text_file(
    file_path: &Path,
    offset: usize,
    limit: Option<usize>,
) -> Result<TextReadOutput> {
    let raw = String::from_utf8_lossy(&fs::read(file_path)?).into_owned();
    let text = raw.strip_prefix('\u{feff}').unwrap_or(&raw);
    let line_offset = if offset == 0 { 0 } else { offset - 1 };
    let end_line = limit
        .map(|value| line_offset.saturating_add(value))
        .unwrap_or(usize::MAX);

    let mut selected_lines = Vec::new();
    let mut line_index = 0usize;
    let mut start = 0usize;

    while let Some(relative_newline) = text[start..].find('\n') {
        let newline = start + relative_newline;
        if line_index >= line_offset && line_index < end_line {
            let mut line = text[start..newline].to_string();
            if line.ends_with('\r') {
                line.pop();
            }
            selected_lines.push(line);
        }
        line_index += 1;
        start = newline + 1;
    }

    if line_index >= line_offset && line_index < end_line {
        let mut line = text[start..].to_string();
        if line.ends_with('\r') {
            line.pop();
        }
        selected_lines.push(line);
    }
    line_index += 1;

    Ok(TextReadOutput {
        type_name: "text".to_string(),
        file: TextReadFile {
            file_path: normalize_nfc_path(file_path),
            content: selected_lines.join("\n"),
            num_lines: selected_lines.len(),
            start_line: offset,
            total_lines: line_index,
        },
    })
}

pub fn write_file(file_path: &Path, content: &str) -> Result<FileWriteReport> {
    let meta = read_file_for_edit(file_path)?;
    let (type_name, original_file, encoding) = if meta.file_exists {
        (
            FileWriteType::Update,
            Some(meta.content.clone()),
            meta.encoding,
        )
    } else {
        (FileWriteType::Create, None, FileEncoding::Utf8)
    };

    write_text_content(file_path, content, encoding, LineEndingType::Lf)?;

    let structured_patch = if let Some(original) = &original_file {
        build_display_patch(&normalize_nfc_path(file_path), original, content)
    } else {
        Vec::new()
    };

    Ok(FileWriteReport {
        type_name,
        file_path: normalize_nfc_path(file_path),
        content: content.to_string(),
        structured_patch,
        original_file,
    })
}

pub fn split_glob_patterns(glob: &str) -> Vec<String> {
    let mut patterns = Vec::new();

    for raw_pattern in glob.split_whitespace() {
        if raw_pattern.contains('{') && raw_pattern.contains('}') {
            patterns.push(raw_pattern.to_string());
            continue;
        }

        for part in raw_pattern.split(',') {
            if !part.is_empty() {
                patterns.push(part.to_string());
            }
        }
    }

    patterns
}

pub fn apply_head_limit<T: Clone>(
    items: &[T],
    limit: Option<usize>,
    offset: usize,
) -> (Vec<T>, Option<usize>) {
    if limit == Some(0) {
        return (items.iter().skip(offset).cloned().collect(), None);
    }

    let effective_limit = limit.unwrap_or(DEFAULT_GREP_HEAD_LIMIT);
    let sliced: Vec<T> = items
        .iter()
        .skip(offset)
        .take(effective_limit)
        .cloned()
        .collect();
    let truncated = items.len().saturating_sub(offset) > effective_limit;
    (sliced, truncated.then_some(effective_limit))
}

pub fn build_display_patch(file_path: &str, old_content: &str, new_content: &str) -> Vec<DiffHunk> {
    build_patch(
        file_path,
        &convert_leading_tabs_to_spaces(old_content),
        &convert_leading_tabs_to_spaces(new_content),
    )
}

pub fn convert_leading_tabs_to_spaces(content: &str) -> String {
    if !content.contains('\t') {
        return content.to_string();
    }

    content
        .split('\n')
        .map(|line| {
            let tabs = line.chars().take_while(|ch| *ch == '\t').count();
            format!("{}{}", "  ".repeat(tabs), &line[tabs..])
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub fn ripgrep(args: &[String], target: &Path) -> Result<Vec<String>> {
    let output = Command::new("rg")
        .args(args)
        .arg(target)
        .output()
        .map_err(|err| anyhow!("failed to run rg: {err}"))?;

    match output.status.code().unwrap_or(-1) {
        0 | 1 => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            Ok(stdout
                .lines()
                .filter(|line| !line.is_empty())
                .map(ToString::to_string)
                .collect())
        }
        _ => {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            Err(anyhow!(if stderr.is_empty() {
                "ripgrep failed".to_string()
            } else {
                stderr
            }))
        }
    }
}

fn build_patch(_file_path: &str, old_content: &str, new_content: &str) -> Vec<DiffHunk> {
    if old_content == new_content {
        return Vec::new();
    }

    let old_lines = split_lines_preserve_trailing_empty(old_content);
    let new_lines = split_lines_preserve_trailing_empty(new_content);

    let mut prefix = 0usize;
    while prefix < old_lines.len()
        && prefix < new_lines.len()
        && old_lines[prefix] == new_lines[prefix]
    {
        prefix += 1;
    }

    let mut old_suffix = old_lines.len();
    let mut new_suffix = new_lines.len();
    while old_suffix > prefix
        && new_suffix > prefix
        && old_lines[old_suffix - 1] == new_lines[new_suffix - 1]
    {
        old_suffix -= 1;
        new_suffix -= 1;
    }

    let context_start = prefix.saturating_sub(DIFF_CONTEXT_LINES);
    let old_context_end = old_lines.len().min(old_suffix + DIFF_CONTEXT_LINES);
    let new_context_end = new_lines.len().min(new_suffix + DIFF_CONTEXT_LINES);

    let mut lines = Vec::new();
    for line in &old_lines[context_start..prefix] {
        lines.push(format!(" {line}"));
    }
    for line in &old_lines[prefix..old_suffix] {
        lines.push(format!("-{line}"));
    }
    for line in &new_lines[prefix..new_suffix] {
        lines.push(format!("+{line}"));
    }

    let trailing_context = (old_context_end - old_suffix).min(new_context_end - new_suffix);
    for line in &old_lines[old_suffix..old_suffix + trailing_context] {
        lines.push(format!(" {line}"));
    }

    vec![DiffHunk {
        old_start: context_start + 1,
        old_lines: old_context_end - context_start,
        new_start: context_start + 1,
        new_lines: new_context_end - context_start,
        lines,
    }]
}

fn normalize_nfc_path(path: &Path) -> String {
    path.to_string_lossy().nfc().collect()
}

fn normalize_path(path: &Path) -> PathBuf {
    let mut normalized = if path.is_absolute() {
        PathBuf::from("/")
    } else {
        PathBuf::new()
    };
    let mut segments = Vec::<std::ffi::OsString>::new();

    for component in path.components() {
        match component {
            std::path::Component::RootDir | std::path::Component::Prefix(_) => {}
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                if let Some(previous) = segments.pop() {
                    if previous == ".." && !path.is_absolute() {
                        segments.push(previous);
                        segments.push("..".into());
                    }
                } else if !path.is_absolute() {
                    segments.push("..".into());
                }
            }
            std::path::Component::Normal(segment) => segments.push(segment.to_os_string()),
        }
    }

    for segment in segments {
        normalized.push(segment);
    }

    if normalized.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        normalized
    }
}

fn homedir_path() -> Result<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow!("HOME is not set"))
}

fn detect_encoding(bytes: &[u8]) -> FileEncoding {
    if bytes.len() >= 2 && bytes[0] == 0xff && bytes[1] == 0xfe {
        FileEncoding::Utf16Le
    } else {
        FileEncoding::Utf8
    }
}

fn decode_bytes(bytes: &[u8], encoding: FileEncoding) -> String {
    match encoding {
        FileEncoding::Utf8 => String::from_utf8_lossy(bytes).into_owned(),
        FileEncoding::Utf16Le => {
            let units: Vec<u16> = bytes
                .chunks_exact(2)
                .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
                .collect();
            String::from_utf16_lossy(&units)
        }
    }
}

fn detect_line_endings(content: &str) -> LineEndingType {
    let mut crlf_count = 0usize;
    let mut lf_count = 0usize;
    let chars: Vec<char> = content.chars().collect();

    for idx in 0..chars.len() {
        if chars[idx] == '\n' {
            if idx > 0 && chars[idx - 1] == '\r' {
                crlf_count += 1;
            } else {
                lf_count += 1;
            }
        }
    }

    if crlf_count > lf_count {
        LineEndingType::CrLf
    } else {
        LineEndingType::Lf
    }
}

fn sample_head(value: &str, max_chars: usize) -> &str {
    let end = value
        .char_indices()
        .nth(max_chars)
        .map(|(idx, _)| idx)
        .unwrap_or(value.len());
    &value[..end]
}

fn split_lines_preserve_trailing_empty(content: &str) -> Vec<String> {
    content.split('\n').map(ToString::to_string).collect()
}
