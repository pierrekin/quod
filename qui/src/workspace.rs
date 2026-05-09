//! Workspace data model.
//!
//! A workspace is a *set of anchors*. Each anchor is either:
//!
//!   - `Project { root, name }` — a directory containing a `quod.toml`
//!     (resolved at add time so the display name is fixed).
//!   - `Program { file }` — an absolute path to a quod program JSON.
//!
//! Anchor uniqueness is keyed on `(kind, absolute path)`. Adding the same
//! anchor twice is a no-op. Programs reached *through* anchors are NOT
//! deduped — the same `foo.json` brought in by two projects plus directly
//! shows three times in the picker.
//!
//! Active state is `(anchor_context, program)`:
//!   - For a project-routed program: `anchor_context = Some(ProjectRef {root, name})`,
//!     program is named via `ProgramSpec.name`.
//!   - For a standalone Program anchor: `anchor_context = None`, program
//!     is named by the `.json` filename stem.
//!
//! Two activations are *different* whenever their `anchor_context` or
//! `program` identity differs — the same `foo.json` reached three ways
//! is three independent active states because the project context shapes
//! how the program renders.

use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Anchor {
    Project { root: PathBuf, name: String },
    Program { file: PathBuf },
}

/// Identity slice of an Anchor — the part used for set-membership and
/// removal lookups. Display name on a Project anchor isn't part of identity.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum AnchorId {
    Project(PathBuf),
    Program(PathBuf),
}

impl Anchor {
    pub fn id(&self) -> AnchorId {
        match self {
            Anchor::Project { root, .. } => AnchorId::Project(root.clone()),
            Anchor::Program { file } => AnchorId::Program(file.clone()),
        }
    }

    /// The absolute path this anchor identifies — `quod.toml`'s parent
    /// for Project, the `.json` for Program.
    pub fn path(&self) -> &Path {
        match self {
            Anchor::Project { root, .. } => root,
            Anchor::Program { file } => file,
        }
    }
}

#[derive(Debug, Default)]
pub struct Workspace {
    anchors: Vec<Anchor>,
}

impl Workspace {
    pub fn anchors(&self) -> &[Anchor] {
        &self.anchors
    }

    /// Add an anchor. Returns `false` if the anchor (by identity) was
    /// already present — second add of the same anchor is a no-op.
    pub fn add(&mut self, anchor: Anchor) -> bool {
        let id = anchor.id();
        if self.anchors.iter().any(|a| a.id() == id) {
            return false;
        }
        self.anchors.push(anchor);
        true
    }

    /// Remove an anchor by identity. Returns `false` if not present.
    pub fn remove(&mut self, id: &AnchorId) -> bool {
        let before = self.anchors.len();
        self.anchors.retain(|a| a.id() != *id);
        self.anchors.len() != before
    }

    /// Iterate Project anchors only, in insertion order. Useful for
    /// rendering the project headers in the picker.
    pub fn projects(&self) -> impl Iterator<Item = (&Path, &str)> {
        self.anchors.iter().filter_map(|a| match a {
            Anchor::Project { root, name } => Some((root.as_path(), name.as_str())),
            Anchor::Program { .. } => None,
        })
    }

    /// Iterate standalone Program anchors only, in insertion order.
    pub fn standalone_programs(&self) -> impl Iterator<Item = &Path> {
        self.anchors.iter().filter_map(|a| match a {
            Anchor::Program { file } => Some(file.as_path()),
            Anchor::Project { .. } => None,
        })
    }
}

/// Identifies a Project anchor in the active state. Carried alongside
/// the active program so the status bar / picker know which project
/// context the program is loaded under.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ProjectRef {
    pub root: PathBuf,
    pub name: String,
}

/// `.json` filename stem (e.g. `/path/to/foo.json` → `foo`). Falls back
/// to the full filename if there's no extension to strip.
pub fn program_name_from_file(file: &Path) -> String {
    file.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| file.to_string_lossy().into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn proj(p: &str, name: &str) -> Anchor {
        Anchor::Project { root: PathBuf::from(p), name: name.into() }
    }
    fn prog(p: &str) -> Anchor {
        Anchor::Program { file: PathBuf::from(p) }
    }

    #[test]
    fn add_dedupes_by_identity_not_display_name() {
        let mut ws = Workspace::default();
        assert!(ws.add(proj("/a/b", "Foo")));
        // Same root, different name → still a duplicate.
        assert!(!ws.add(proj("/a/b", "Renamed")));
        assert_eq!(ws.anchors().len(), 1);
    }

    #[test]
    fn add_keeps_project_and_program_with_same_path_separate() {
        // Pathologically: a Program anchor at the same path as a Project's
        // root. Different kinds → different identities → both coexist.
        let mut ws = Workspace::default();
        assert!(ws.add(proj("/a/b", "P")));
        assert!(ws.add(prog("/a/b")));
        assert_eq!(ws.anchors().len(), 2);
    }

    #[test]
    fn remove_returns_false_when_absent() {
        let mut ws = Workspace::default();
        assert!(!ws.remove(&AnchorId::Project(PathBuf::from("/missing"))));
    }

    #[test]
    fn remove_distinguishes_kind() {
        let mut ws = Workspace::default();
        ws.add(proj("/a", "A"));
        ws.add(prog("/a"));
        assert!(ws.remove(&AnchorId::Program(PathBuf::from("/a"))));
        assert_eq!(ws.anchors().len(), 1);
        assert!(matches!(ws.anchors()[0], Anchor::Project { .. }));
    }

    #[test]
    fn projects_iter_yields_project_anchors_only() {
        let mut ws = Workspace::default();
        ws.add(proj("/a", "A"));
        ws.add(prog("/x.json"));
        ws.add(proj("/b", "B"));
        let names: Vec<&str> = ws.projects().map(|(_, n)| n).collect();
        assert_eq!(names, vec!["A", "B"]);
    }

    #[test]
    fn standalone_programs_iter_yields_program_anchors_only() {
        let mut ws = Workspace::default();
        ws.add(proj("/a", "A"));
        ws.add(prog("/x.json"));
        ws.add(prog("/y.json"));
        let files: Vec<&Path> = ws.standalone_programs().collect();
        assert_eq!(files, vec![Path::new("/x.json"), Path::new("/y.json")]);
    }

    #[test]
    fn program_name_from_file_strips_json_extension() {
        assert_eq!(program_name_from_file(Path::new("/a/b/foo.json")), "foo");
        assert_eq!(program_name_from_file(Path::new("foo.json")), "foo");
        assert_eq!(program_name_from_file(Path::new("noext")), "noext");
    }
}
