use std::path::{Path, PathBuf};

use anyhow::{Result, bail};

use crate::file_tools::contains_path_traversal;

const MAX_WORKTREE_SLUG_LENGTH: usize = 64;

pub fn validate_worktree_slug(slug: &str) -> Result<()> {
    if slug.len() > MAX_WORKTREE_SLUG_LENGTH {
        bail!(
            "Invalid worktree name: must be {MAX_WORKTREE_SLUG_LENGTH} characters or fewer (got {})",
            slug.len()
        );
    }

    for segment in slug.split('/') {
        if segment == "." || segment == ".." {
            bail!(
                "Invalid worktree name \"{slug}\": must not contain \".\" or \"..\" path segments"
            );
        }

        if segment.is_empty()
            || !segment
                .chars()
                .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
        {
            bail!(
                "Invalid worktree name \"{slug}\": each \"/\"-separated segment must be non-empty and contain only letters, digits, dots, underscores, and dashes"
            );
        }
    }

    Ok(())
}

pub fn generate_tmux_session_name(repo_path: &Path, branch: &str) -> String {
    let repo_name = repo_path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or_default();
    format!("{repo_name}_{branch}").replace(['/', '.'], "_")
}

pub fn worktree_branch_name(slug: &str) -> String {
    format!("worktree-{}", flatten_slug(slug))
}

pub fn worktree_path(repo_root: &Path, slug: &str) -> PathBuf {
    repo_root
        .join(".claude")
        .join("worktrees")
        .join(flatten_slug(slug))
}

pub fn should_skip_symlink_dir(dir: &str) -> bool {
    contains_path_traversal(dir)
}

fn flatten_slug(slug: &str) -> String {
    slug.replace('/', "+")
}
