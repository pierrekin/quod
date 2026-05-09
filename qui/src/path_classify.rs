//! Classify a user-supplied path into a workspace [`Anchor`].
//!
//! The CLI accepts N positional anchor paths; each one resolves to either:
//!
//!   - `.toml` file or directory containing `quod.toml` → `Project` anchor.
//!     The project's display name is read from the toml's required
//!     top-level `name` field (loaded by an injected reader so this module
//!     stays decoupled from the toml-loading library).
//!   - `.json` file → `Program` anchor (standalone).
//!   - anything else → hard error.

use std::fs;
use std::path::{Path, PathBuf};

use crate::workspace::Anchor;

#[derive(Debug)]
pub enum ClassifyError {
    NotFound(PathBuf),
    UnsupportedKind(PathBuf),
    BadProject { root: PathBuf, reason: String },
    Canonicalize(PathBuf, std::io::Error),
}

impl std::fmt::Display for ClassifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ClassifyError::NotFound(p) => write!(f, "path does not exist: {}", p.display()),
            ClassifyError::UnsupportedKind(p) => write!(
                f,
                "not a quod project (.toml) or program (.json): {}",
                p.display()
            ),
            ClassifyError::BadProject { root, reason } => {
                write!(f, "invalid quod project at {}: {}", root.display(), reason)
            }
            ClassifyError::Canonicalize(p, e) => {
                write!(f, "cannot canonicalize {}: {}", p.display(), e)
            }
        }
    }
}

/// Read the required top-level `name` from a `quod.toml`. Returns the
/// trimmed string on success, or a human-readable error describing what's
/// wrong (missing key, wrong type, parse failure). Does not validate the
/// rest of the toml — that's the LSP's job at workspace-add time.
pub fn read_project_name(toml_path: &Path) -> Result<String, String> {
    let text = fs::read_to_string(toml_path)
        .map_err(|e| format!("read {}: {}", toml_path.display(), e))?;
    let parsed: toml::Value = toml::from_str(&text)
        .map_err(|e| format!("parse {}: {}", toml_path.display(), e))?;
    match parsed.get("name") {
        Some(toml::Value::String(s)) if !s.is_empty() => Ok(s.clone()),
        Some(toml::Value::String(_)) => Err("`name` is empty".into()),
        Some(_) => Err("`name` must be a string".into()),
        None => Err("missing required top-level key `name`".into()),
    }
}

/// Resolve `~` and a leading `~/` against `$HOME`. Other `~user` forms
/// are not expanded — pass them through unchanged.
pub fn expand_path(text: &str) -> PathBuf {
    if let Some(rest) = text.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return Path::new(&home).join(rest);
        }
    }
    if text == "~" {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home);
        }
    }
    PathBuf::from(text)
}

/// Classify `raw` into an [`Anchor`]. `read_project_name` is called for
/// project paths to fetch the display name; pass a closure that loads the
/// toml. Decoupling lets us unit-test classification with a stub.
pub fn classify<F>(raw: &str, read_project_name: F) -> Result<Anchor, ClassifyError>
where
    F: FnOnce(&Path) -> Result<String, String>,
{
    let path = expand_path(raw);
    if !path.exists() {
        return Err(ClassifyError::NotFound(path));
    }
    let path = path
        .canonicalize()
        .map_err(|e| ClassifyError::Canonicalize(path.clone(), e))?;

    if path.is_file() {
        return classify_file(path, read_project_name);
    }
    if path.is_dir() {
        return classify_dir(path, read_project_name);
    }
    Err(ClassifyError::UnsupportedKind(path))
}

fn classify_file<F>(path: PathBuf, read_project_name: F) -> Result<Anchor, ClassifyError>
where
    F: FnOnce(&Path) -> Result<String, String>,
{
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
    let stem = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
    if ext == "toml" && stem == "quod.toml" {
        let root = path
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| ClassifyError::BadProject {
                root: path.clone(),
                reason: "quod.toml has no parent directory".into(),
            })?;
        let name = read_project_name(&path).map_err(|reason| ClassifyError::BadProject {
            root: root.clone(),
            reason,
        })?;
        return Ok(Anchor::Project { root, name });
    }
    if ext == "json" {
        return Ok(Anchor::Program { file: path });
    }
    Err(ClassifyError::UnsupportedKind(path))
}

fn classify_dir<F>(path: PathBuf, read_project_name: F) -> Result<Anchor, ClassifyError>
where
    F: FnOnce(&Path) -> Result<String, String>,
{
    let toml = path.join("quod.toml");
    if !toml.is_file() {
        return Err(ClassifyError::UnsupportedKind(path));
    }
    let name = read_project_name(&toml).map_err(|reason| ClassifyError::BadProject {
        root: path.clone(),
        reason,
    })?;
    Ok(Anchor::Project { root: path, name })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn stub_name(_: &Path) -> Result<String, String> {
        Ok("stub".into())
    }

    #[test]
    fn classify_dir_with_quod_toml_is_project() {
        let tmp = tempfile_dir();
        fs::write(tmp.join("quod.toml"), "name = \"X\"").unwrap();
        let a = classify(tmp.to_str().unwrap(), |_| Ok("X".into())).unwrap();
        match a {
            Anchor::Project { root, name } => {
                assert_eq!(name, "X");
                assert_eq!(root.canonicalize().unwrap(), tmp.canonicalize().unwrap());
            }
            _ => panic!("expected Project, got {:?}", a),
        }
    }

    #[test]
    fn classify_dir_without_quod_toml_errors() {
        let tmp = tempfile_dir();
        let err = classify(tmp.to_str().unwrap(), stub_name).unwrap_err();
        assert!(matches!(err, ClassifyError::UnsupportedKind(_)));
    }

    #[test]
    fn classify_quod_toml_file_is_project() {
        let tmp = tempfile_dir();
        let toml = tmp.join("quod.toml");
        fs::write(&toml, "name = \"X\"").unwrap();
        let a = classify(toml.to_str().unwrap(), |_| Ok("X".into())).unwrap();
        assert!(matches!(a, Anchor::Project { .. }));
    }

    #[test]
    fn classify_other_toml_file_errors() {
        let tmp = tempfile_dir();
        let other = tmp.join("Cargo.toml");
        fs::write(&other, "").unwrap();
        let err = classify(other.to_str().unwrap(), stub_name).unwrap_err();
        assert!(matches!(err, ClassifyError::UnsupportedKind(_)));
    }

    #[test]
    fn classify_json_is_program() {
        let tmp = tempfile_dir();
        let p = tmp.join("loose.json");
        fs::write(&p, "{}").unwrap();
        let a = classify(p.to_str().unwrap(), stub_name).unwrap();
        assert!(matches!(a, Anchor::Program { .. }));
    }

    #[test]
    fn classify_missing_path_errors() {
        let err = classify("/definitely/does/not/exist", stub_name).unwrap_err();
        assert!(matches!(err, ClassifyError::NotFound(_)));
    }

    #[test]
    fn classify_propagates_project_name_read_error() {
        let tmp = tempfile_dir();
        fs::write(tmp.join("quod.toml"), "").unwrap();
        let err = classify(tmp.to_str().unwrap(), |_| {
            Err("missing required top-level key `name`".into())
        })
        .unwrap_err();
        match err {
            ClassifyError::BadProject { reason, .. } => {
                assert!(reason.contains("name"));
            }
            other => panic!("expected BadProject, got {other:?}"),
        }
    }

    #[test]
    fn read_project_name_extracts_top_level_string() {
        let tmp = tempfile_dir();
        let toml = tmp.join("quod.toml");
        fs::write(&toml, "name = \"My Project\"\n[build]\nprofile = 2\n").unwrap();
        assert_eq!(read_project_name(&toml).unwrap(), "My Project");
    }

    #[test]
    fn read_project_name_errors_when_missing() {
        let tmp = tempfile_dir();
        let toml = tmp.join("quod.toml");
        fs::write(&toml, "[build]\nprofile = 2\n").unwrap();
        let err = read_project_name(&toml).unwrap_err();
        assert!(err.contains("missing required top-level key `name`"), "got: {err}");
    }

    #[test]
    fn read_project_name_errors_on_wrong_type() {
        let tmp = tempfile_dir();
        let toml = tmp.join("quod.toml");
        fs::write(&toml, "name = 42\n").unwrap();
        let err = read_project_name(&toml).unwrap_err();
        assert!(err.contains("string"), "got: {err}");
    }

    #[test]
    fn read_project_name_ignores_program_name_shadow() {
        // The historical loader had a `name` shadowing bug; this nails
        // down that the *top-level* `name` is what we read, not any
        // `[[program]] name = ...`.
        let tmp = tempfile_dir();
        let toml = tmp.join("quod.toml");
        fs::write(
            &toml,
            "name = \"Top\"\n[[program]]\nname = \"inner\"\nfile = \"p.json\"\n",
        )
        .unwrap();
        assert_eq!(read_project_name(&toml).unwrap(), "Top");
    }

    /// Make a unique temp dir under `/tmp/qui-test-<pid>-<n>/`. Avoids
    /// pulling in the `tempfile` crate just for tests.
    fn tempfile_dir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "qui-test-{}-{}-{}",
            std::process::id(),
            n,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }
}
