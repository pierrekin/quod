//! Persisted list of recently-opened workspace anchors.
//!
//! Stored at `$XDG_STATE_HOME/qui/recents.json` (falling back to
//! `~/.local/state/qui/recents.json`). Best-effort: any I/O error treats
//! the store as empty and continues — recents are convenience, not
//! correctness.
//!
//! Each entry is one anchor (project or standalone program). Anchors are
//! identified by `(kind, absolute path)`; bumping recency on an existing
//! entry rewrites the timestamp without producing a duplicate.
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

const CAP: usize = 20;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnchorKind {
    Project,
    Program,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentAnchor {
    pub kind: AnchorKind,
    pub path: PathBuf,
    /// Unix seconds. i64 keeps the JSON readable.
    pub last_opened: i64,
    /// For Project anchors: the program-name within the project that was
    /// last activated through this anchor. Carried so a future "reopen
    /// most recent" can land on the same program.
    pub last_active_program: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Recents {
    #[serde(default, alias = "folders")]
    pub anchors: Vec<RecentAnchor>,
}

impl Recents {
    pub fn load() -> Self {
        let Some(path) = store_path() else { return Self::default() };
        let Ok(data) = fs::read_to_string(&path) else { return Self::default() };
        // serde_json silently tolerates extra/missing fields; the
        // `alias = "folders"` lets old `Recents { folders: [...] }`
        // files load — entries from the previous schema will lack `kind`
        // and be discarded by `from_str` returning default. Best-effort.
        serde_json::from_str(&data).unwrap_or_default()
    }

    pub fn save(&self) -> std::io::Result<()> {
        let Some(path) = store_path() else { return Ok(()) };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let data = serde_json::to_string_pretty(self).unwrap_or_default();
        fs::write(&path, data)
    }

    /// Bump (or insert) a Project anchor. `active_program` is the name
    /// of the program activated through this project right now, if any.
    pub fn note_project(&mut self, path: PathBuf, active_program: Option<String>) {
        self.note(AnchorKind::Project, path, active_program);
    }

    /// Bump (or insert) a standalone Program anchor.
    pub fn note_program(&mut self, file: PathBuf) {
        self.note(AnchorKind::Program, file, None);
    }

    fn note(&mut self, kind: AnchorKind, path: PathBuf, active_program: Option<String>) {
        let now = unix_now();
        // Carry over the prior last_active_program when the new one is
        // None — opening a project then immediately switching shouldn't
        // lose what the user had picked last time.
        let prior = self
            .anchors
            .iter()
            .find(|a| a.kind == kind && a.path == path)
            .and_then(|a| a.last_active_program.clone());
        self.anchors.retain(|a| !(a.kind == kind && a.path == path));
        self.anchors.insert(
            0,
            RecentAnchor {
                kind,
                path,
                last_opened: now,
                last_active_program: active_program.or(prior),
            },
        );
        self.anchors.truncate(CAP);
    }
}

fn unix_now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

fn store_path() -> Option<PathBuf> {
    if let Ok(state) = std::env::var("XDG_STATE_HOME") {
        if !state.is_empty() {
            return Some(Path::new(&state).join("qui").join("recents.json"));
        }
    }
    let home = std::env::var("HOME").ok()?;
    Some(
        Path::new(&home)
            .join(".local")
            .join("state")
            .join("qui")
            .join("recents.json"),
    )
}

/// Human-friendly "2d ago", "in the future", etc. for display.
pub fn relative_time(then: i64) -> String {
    let now = unix_now();
    let delta = now - then;
    if delta < 0 {
        return "future".into();
    }
    if delta < 60 {
        return "just now".into();
    }
    if delta < 3600 {
        return format!("{}m ago", delta / 60);
    }
    if delta < 86400 {
        return format!("{}h ago", delta / 3600);
    }
    if delta < 86400 * 30 {
        return format!("{}d ago", delta / 86400);
    }
    if delta < 86400 * 365 {
        return format!("{}mo ago", delta / (86400 * 30));
    }
    format!("{}y ago", delta / (86400 * 365))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn note_project_dedupes_by_path() {
        let mut r = Recents::default();
        r.note_project(PathBuf::from("/a"), Some("foo".into()));
        r.note_project(PathBuf::from("/a"), Some("bar".into()));
        assert_eq!(r.anchors.len(), 1);
        assert_eq!(r.anchors[0].last_active_program.as_deref(), Some("bar"));
    }

    #[test]
    fn note_carries_prior_program_when_new_is_none() {
        let mut r = Recents::default();
        r.note_project(PathBuf::from("/a"), Some("foo".into()));
        r.note_project(PathBuf::from("/a"), None);
        assert_eq!(r.anchors[0].last_active_program.as_deref(), Some("foo"));
    }

    #[test]
    fn note_program_distinct_from_note_project_at_same_path() {
        let mut r = Recents::default();
        r.note_project(PathBuf::from("/a"), None);
        r.note_program(PathBuf::from("/a"));
        assert_eq!(r.anchors.len(), 2);
    }

    #[test]
    fn most_recent_is_first() {
        let mut r = Recents::default();
        r.note_project(PathBuf::from("/older"), None);
        r.note_project(PathBuf::from("/newer"), None);
        assert_eq!(r.anchors[0].path, PathBuf::from("/newer"));
        assert_eq!(r.anchors[1].path, PathBuf::from("/older"));
    }

    #[test]
    fn cap_bounds_the_list() {
        let mut r = Recents::default();
        for i in 0..(CAP + 5) {
            r.note_project(PathBuf::from(format!("/p{i}")), None);
        }
        assert_eq!(r.anchors.len(), CAP);
    }
}
