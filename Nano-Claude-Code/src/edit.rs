use std::path::Path;

use anyhow::{Result, anyhow, bail};
use serde::Serialize;

use crate::file_tools::{DiffHunk, build_display_patch, read_file_for_edit, write_text_content};

const LEFT_SINGLE_CURLY_QUOTE: char = '‘';
const RIGHT_SINGLE_CURLY_QUOTE: char = '’';
const LEFT_DOUBLE_CURLY_QUOTE: char = '“';
const RIGHT_DOUBLE_CURLY_QUOTE: char = '”';

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct EditReport {
    pub file_path: String,
    pub actual_old_string: String,
    pub actual_new_string: String,
    pub replace_all: bool,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct EditToolReport {
    pub file_path: String,
    pub old_string: String,
    pub new_string: String,
    pub original_file: String,
    pub structured_patch: Vec<DiffHunk>,
    pub user_modified: bool,
    pub replace_all: bool,
}

pub fn normalize_quotes(value: &str) -> String {
    value
        .replace([LEFT_SINGLE_CURLY_QUOTE, RIGHT_SINGLE_CURLY_QUOTE], "'")
        .replace([LEFT_DOUBLE_CURLY_QUOTE, RIGHT_DOUBLE_CURLY_QUOTE], "\"")
}

pub fn find_actual_string(file_content: &str, search_string: &str) -> Option<String> {
    if file_content.contains(search_string) {
        return Some(search_string.to_string());
    }

    let normalized_search = normalize_quotes(search_string);
    let normalized_file = normalize_quotes(file_content);
    let search_index = normalized_file.find(&normalized_search)?;
    let start_chars = normalized_file[..search_index].chars().count();
    let search_chars = search_string.chars().count();

    Some(slice_chars(file_content, start_chars, search_chars))
}

pub fn preserve_quote_style(old_string: &str, actual_old_string: &str, new_string: &str) -> String {
    if old_string == actual_old_string {
        return new_string.to_string();
    }

    let has_double_quotes = actual_old_string.contains(LEFT_DOUBLE_CURLY_QUOTE)
        || actual_old_string.contains(RIGHT_DOUBLE_CURLY_QUOTE);
    let has_single_quotes = actual_old_string.contains(LEFT_SINGLE_CURLY_QUOTE)
        || actual_old_string.contains(RIGHT_SINGLE_CURLY_QUOTE);

    if !has_double_quotes && !has_single_quotes {
        return new_string.to_string();
    }

    let mut result = new_string.to_string();
    if has_double_quotes {
        result = apply_curly_double_quotes(&result);
    }
    if has_single_quotes {
        result = apply_curly_single_quotes(&result);
    }
    result
}

pub fn apply_edit_to_file(
    original_content: &str,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> String {
    if !new_string.is_empty() {
        return replace_text(original_content, old_string, new_string, replace_all);
    }

    let strip_trailing_newline =
        !old_string.ends_with('\n') && original_content.contains(&format!("{old_string}\n"));

    if strip_trailing_newline {
        replace_text(
            original_content,
            &format!("{old_string}\n"),
            new_string,
            replace_all,
        )
    } else {
        replace_text(original_content, old_string, new_string, replace_all)
    }
}

pub fn perform_edit(
    file_path: &Path,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<EditReport> {
    let report = perform_edit_tool(file_path, old_string, new_string, replace_all)?;
    Ok(EditReport {
        file_path: report.file_path,
        actual_old_string: report.old_string,
        actual_new_string: report.new_string,
        replace_all: report.replace_all,
    })
}

pub fn perform_edit_tool(
    file_path: &Path,
    old_string: &str,
    new_string: &str,
    replace_all: bool,
) -> Result<EditToolReport> {
    if old_string == new_string {
        bail!("No changes to make: old_string and new_string are exactly the same.");
    }

    let file = read_file_for_edit(file_path)?;
    if !file.file_exists && !old_string.is_empty() {
        bail!("File does not exist.");
    }

    if file.file_exists && old_string.is_empty() && !file.content.trim().is_empty() {
        bail!("Cannot create new file - file already exists.");
    }

    let actual_old_string = if old_string.is_empty() {
        String::new()
    } else {
        find_actual_string(&file.content, old_string)
            .ok_or_else(|| anyhow!("String to replace not found in file.\nString: {old_string}"))?
    };

    if !actual_old_string.is_empty() {
        let matches = file.content.match_indices(&actual_old_string).count();
        if matches > 1 && !replace_all {
            bail!(
                "Found {matches} matches of the string to replace, but replace_all is false. To replace all occurrences, set replace_all to true. To replace only one occurrence, please provide more context to uniquely identify the instance.\nString: {old_string}"
            );
        }
    }

    let actual_new_string = preserve_quote_style(old_string, &actual_old_string, new_string);
    let updated_file = if old_string.is_empty() {
        new_string.to_string()
    } else {
        apply_edit_to_file(
            &file.content,
            &actual_old_string,
            &actual_new_string,
            replace_all,
        )
    };

    let allow_empty_create = !file.file_exists && old_string.is_empty() && new_string.is_empty();
    if updated_file == file.content && !allow_empty_create {
        bail!("String not found in file. Failed to apply edit.");
    }

    write_text_content(file_path, &updated_file, file.encoding, file.line_endings)?;

    Ok(EditToolReport {
        file_path: file_path.to_string_lossy().to_string(),
        old_string: actual_old_string,
        new_string: actual_new_string,
        original_file: file.content.clone(),
        structured_patch: build_display_patch(
            &file_path.to_string_lossy(),
            &file.content,
            &updated_file,
        ),
        user_modified: false,
        replace_all,
    })
}

fn replace_text(content: &str, search: &str, replace: &str, replace_all: bool) -> String {
    if replace_all {
        content.replace(search, replace)
    } else {
        content.replacen(search, replace, 1)
    }
}

fn apply_curly_double_quotes(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let mut result = String::new();

    for (idx, ch) in chars.iter().enumerate() {
        if *ch == '"' {
            result.push(if is_opening_context(&chars, idx) {
                LEFT_DOUBLE_CURLY_QUOTE
            } else {
                RIGHT_DOUBLE_CURLY_QUOTE
            });
        } else {
            result.push(*ch);
        }
    }

    result
}

fn apply_curly_single_quotes(value: &str) -> String {
    let chars: Vec<char> = value.chars().collect();
    let mut result = String::new();

    for (idx, ch) in chars.iter().enumerate() {
        if *ch == '\'' {
            let prev = if idx > 0 { Some(chars[idx - 1]) } else { None };
            let next = chars.get(idx + 1).copied();
            let prev_is_letter = prev.is_some_and(char::is_alphabetic);
            let next_is_letter = next.is_some_and(char::is_alphabetic);

            if prev_is_letter && next_is_letter {
                result.push(RIGHT_SINGLE_CURLY_QUOTE);
            } else {
                result.push(if is_opening_context(&chars, idx) {
                    LEFT_SINGLE_CURLY_QUOTE
                } else {
                    RIGHT_SINGLE_CURLY_QUOTE
                });
            }
        } else {
            result.push(*ch);
        }
    }

    result
}

fn is_opening_context(chars: &[char], index: usize) -> bool {
    if index == 0 {
        return true;
    }

    matches!(
        chars[index - 1],
        ' ' | '\t' | '\n' | '\r' | '(' | '[' | '{' | '\u{2014}' | '\u{2013}'
    )
}

fn slice_chars(value: &str, start_chars: usize, len_chars: usize) -> String {
    let start = byte_index_for_char(value, start_chars);
    let end = byte_index_for_char(value, start_chars + len_chars);
    value[start..end].to_string()
}

fn byte_index_for_char(value: &str, char_index: usize) -> usize {
    if char_index == 0 {
        return 0;
    }

    value
        .char_indices()
        .nth(char_index)
        .map(|(idx, _)| idx)
        .unwrap_or(value.len())
}
