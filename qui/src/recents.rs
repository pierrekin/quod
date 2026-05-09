//! Persisted list of recently-opened folders and files.
//!
//! Stored at `$XDG_STATE_HOME/qui/recents.json` (falling back to
//! `~/.local/state/qui/recents.json`). Best-effort: any I/O error
//! treats the store as empty and continues — recents are convenience,
//! not correctness.
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

const CAP: usize = 20;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecentFolder {
    pub path: PathBuf,
    /// Unix seconds. Stored as i64 to keep the JSON readable.
    pub last_opened: i64,
    pub last_active_program: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Recents {
    #[serde(default)]
    pub folders: Vec<RecentFolder>,
}

impl Recents {
    pub fn load() -> Self {
        let Some(path) = store_path() else { return Self::default() };
        let Ok(data) = fs::read_to_string(&path) else { return Self::default() };
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

    pub fn note_folder(&mut self, path: PathBuf, active_program: Option<String>) {
        let now = unix_now();
        // Carry over previous last_active_program if the new one is None
        // (e.g. opened folder then immediately switched away — don't lose
        // the prior pick).
        let prior = self
            .folders
            .iter()
            .find(|f| f.path == path)
            .and_then(|f| f.last_active_program.clone());
        self.folders.retain(|f| f.path != path);
        self.folders.insert(
            0,
            RecentFolder {
                path,
                last_opened: now,
                last_active_program: active_program.or(prior),
            },
        );
        self.folders.truncate(CAP);
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
