//! Open-anchor modal: a file/dir browser pane (top) over a recents
//! pane (bottom). Tab toggles focus between them. Returns a path on
//! commit; the App classifies it into a workspace Anchor (Project for
//! `.toml` / dir-with-quod.toml, Program for `.json`) and adds it.
//!
//! The browser filters to: dirs (for navigation) + files named exactly
//! `quod.toml` + files with the `.json` extension. No "magical"
//! is-project highlighting — the user is selecting a file, not a dir.
use std::path::{Path, PathBuf};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph};

#[derive(Debug)]
pub enum Outcome {
    /// User picked a path. App classifies it into an Anchor and adds it.
    Open(PathBuf),
    /// User asked to remove the workspace anchor at this index.
    /// Index is into the modal's `workspace` vec — the App needs to
    /// translate it back to an `AnchorId`.
    RemoveWorkspace(usize),
    Cancel,
    Continue,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Focus {
    Files,
    Workspace,
    Recents,
}

/// One row in the Workspace pane — an anchor currently in the workspace.
#[derive(Debug, Clone)]
pub struct WorkspaceEntry {
    /// `>` for project, `-` for standalone program.
    pub glyph: char,
    pub display_name: String,
    pub display_path: String,
}

#[derive(Debug, Clone)]
pub enum Entry {
    Parent,                            // "../" — step out
    Dir { name: String },              // subdir — step in on commit
    File { name: String },             // quod.toml or *.json — commits as Anchor
}

#[derive(Debug, Clone)]
pub struct RecentEntry {
    pub path: PathBuf,
    pub display: String,
    pub time_ago: String,
}

pub struct OpenModal {
    pub input: String,
    pub entries: Vec<Entry>,
    pub workspace: Vec<WorkspaceEntry>,
    pub recents: Vec<RecentEntry>,
    pub focus: Focus,
    pub fs_cursor: usize,
    pub workspace_cursor: usize,
    pub recents_cursor: usize,
    pub error: Option<String>,
}

impl OpenModal {
    pub fn new(
        initial: PathBuf,
        workspace: Vec<WorkspaceEntry>,
        recents: Vec<RecentEntry>,
    ) -> Self {
        let input = initial.to_string_lossy().into_owned();
        let entries = scan(&initial);
        Self {
            input,
            entries,
            workspace,
            recents,
            focus: Focus::Files,
            fs_cursor: 0,
            workspace_cursor: 0,
            recents_cursor: 0,
            error: None,
        }
    }

    pub fn set_error(&mut self, msg: impl Into<String>) {
        self.error = Some(msg.into());
    }

    /// Re-list the directory implied by the current input. If the input
    /// is a partial path (e.g. user is mid-typing), we list the longest
    /// existing ancestor — same as fzf-style file pickers.
    fn rescan(&mut self) {
        let path = expand(&self.input);
        let dir = if path.is_dir() {
            Some(path)
        } else {
            path.parent()
                .filter(|p| p.is_dir())
                .map(Path::to_path_buf)
        };
        match dir {
            Some(d) => {
                self.entries = scan(&d);
                if self.fs_cursor >= self.entries.len() {
                    self.fs_cursor = 0;
                }
            }
            None => {
                self.entries.clear();
                self.fs_cursor = 0;
            }
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Outcome {
        self.error = None;
        match key.code {
            KeyCode::Esc => Outcome::Cancel,
            KeyCode::Tab => {
                self.cycle_focus(true);
                Outcome::Continue
            }
            KeyCode::BackTab => {
                self.cycle_focus(false);
                Outcome::Continue
            }
            KeyCode::Up => {
                match self.focus {
                    Focus::Files => self.fs_cursor = self.fs_cursor.saturating_sub(1),
                    Focus::Workspace => {
                        self.workspace_cursor = self.workspace_cursor.saturating_sub(1)
                    }
                    Focus::Recents => {
                        self.recents_cursor = self.recents_cursor.saturating_sub(1)
                    }
                }
                Outcome::Continue
            }
            KeyCode::Down => {
                match self.focus {
                    Focus::Files => {
                        if self.fs_cursor + 1 < self.entries.len() {
                            self.fs_cursor += 1;
                        }
                    }
                    Focus::Workspace => {
                        if self.workspace_cursor + 1 < self.workspace.len() {
                            self.workspace_cursor += 1;
                        }
                    }
                    Focus::Recents => {
                        if self.recents_cursor + 1 < self.recents.len() {
                            self.recents_cursor += 1;
                        }
                    }
                }
                Outcome::Continue
            }
            KeyCode::Enter => self.commit(),
            KeyCode::Char('x') if self.focus == Focus::Workspace => {
                if self.workspace_cursor < self.workspace.len() {
                    return Outcome::RemoveWorkspace(self.workspace_cursor);
                }
                Outcome::Continue
            }
            KeyCode::Backspace if self.focus == Focus::Files => {
                self.input.pop();
                self.rescan();
                Outcome::Continue
            }
            KeyCode::Char('u')
                if key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.focus == Focus::Files =>
            {
                self.input.clear();
                self.rescan();
                Outcome::Continue
            }
            KeyCode::Char(c)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && self.focus == Focus::Files =>
            {
                self.input.push(c);
                self.rescan();
                Outcome::Continue
            }
            _ => Outcome::Continue,
        }
    }

    /// Tab through the panes that have content. Skip empty panes so
    /// `tab` always lands on something useful.
    fn cycle_focus(&mut self, forward: bool) {
        let order = [Focus::Files, Focus::Workspace, Focus::Recents];
        let cur = order.iter().position(|f| *f == self.focus).unwrap_or(0);
        let n = order.len();
        for step in 1..=n {
            let next = if forward {
                order[(cur + step) % n]
            } else {
                order[(cur + n - step) % n]
            };
            let has_content = match next {
                Focus::Files => true,
                Focus::Workspace => !self.workspace.is_empty(),
                Focus::Recents => !self.recents.is_empty(),
            };
            if has_content {
                self.focus = next;
                return;
            }
        }
    }

    fn commit(&mut self) -> Outcome {
        match self.focus {
            Focus::Files => {
                let entry = match self.entries.get(self.fs_cursor) {
                    Some(e) => e.clone(),
                    None => return Outcome::Continue,
                };
                // Use the directory the listing was actually drawn from
                // (which may be the parent of `input` if the user is
                // mid-typing a partial filename).
                let typed = expand(&self.input);
                let base = if typed.is_dir() {
                    typed
                } else {
                    typed
                        .parent()
                        .filter(|p| p.is_dir())
                        .map(Path::to_path_buf)
                        .unwrap_or(typed)
                };
                match entry {
                    Entry::Parent => match base.parent() {
                        Some(p) => {
                            self.input = p.to_string_lossy().into_owned();
                            self.rescan();
                            Outcome::Continue
                        }
                        None => Outcome::Continue,
                    },
                    Entry::Dir { name } => {
                        let target = base.join(&name);
                        self.input = target.to_string_lossy().into_owned();
                        self.rescan();
                        Outcome::Continue
                    }
                    Entry::File { name } => Outcome::Open(base.join(&name)),
                }
            }
            Focus::Recents => match self.recents.get(self.recents_cursor) {
                Some(r) => Outcome::Open(r.path.clone()),
                None => Outcome::Continue,
            },
            // Workspace pane uses `x` (not enter) to remove. Enter is a
            // no-op there — the anchor is already in the workspace.
            Focus::Workspace => Outcome::Continue,
        }
    }

    pub fn render(&self, frame: &mut Frame<'_>, full_area: Rect) {
        let need_h = 14;
        let need_w = 36;
        if full_area.width < need_w || full_area.height < need_h {
            return;
        }
        let w = full_area.width.saturating_sub(2).min(80);
        let h = full_area.height.saturating_sub(2).min(28).max(need_h);
        let x = full_area.x + (full_area.width - w) / 2;
        let y = full_area.y + (full_area.height - h) / 2;
        let area = Rect { x, y, width: w, height: h };

        frame.render_widget(Clear, area);
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" workspace anchors ")
            .title_alignment(Alignment::Center);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Vertical layout:
        //   path (1) · sep (1) · [error (1)] · files (min 4)
        //   · sep (1) · "Workspace" header (1) · workspace (≤ N)
        //   · sep (1) · "Recent" header (1) · recents (≤ N)
        //   · sep (1) · footer (1)
        let workspace_h = (self.workspace.len() as u16).clamp(1, 5);
        let recents_h = (self.recents.len() as u16).clamp(1, 5);
        let mut constraints: Vec<Constraint> = vec![
            Constraint::Length(1),
            Constraint::Length(1),
        ];
        if self.error.is_some() {
            constraints.push(Constraint::Length(1));
        }
        constraints.push(Constraint::Min(3));
        constraints.push(Constraint::Length(1));
        constraints.push(Constraint::Length(1));
        constraints.push(Constraint::Length(workspace_h));
        constraints.push(Constraint::Length(1));
        constraints.push(Constraint::Length(1));
        constraints.push(Constraint::Length(recents_h));
        constraints.push(Constraint::Length(1));
        constraints.push(Constraint::Length(1));
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(inner);

        let dim = Style::default().fg(Color::DarkGray);
        let label_style = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
        let focused_hl = Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let unfocused_hl = Style::default().fg(Color::DarkGray);

        let mut idx = 0usize;

        // Path
        let mut path_spans = vec![Span::styled("Path: ", label_style), Span::raw(self.input.clone())];
        if self.focus == Focus::Files {
            path_spans.push(Span::styled("█", dim));
        }
        frame.render_widget(Paragraph::new(Line::from(path_spans)), chunks[idx]);
        idx += 1;

        // Sep
        frame.render_widget(
            Paragraph::new(Span::styled("─".repeat(inner.width as usize), dim)),
            chunks[idx],
        );
        idx += 1;

        // Error
        if let Some(err) = &self.error {
            frame.render_widget(
                Paragraph::new(Span::styled(err.clone(), Style::default().fg(Color::Red))),
                chunks[idx],
            );
            idx += 1;
        }

        // Files
        let files_area = chunks[idx];
        idx += 1;
        let entries: Vec<ListItem> = self
            .entries
            .iter()
            .map(|e| ListItem::new(render_entry(e)))
            .collect();
        let mut state = ListState::default();
        if !entries.is_empty() {
            state.select(Some(self.fs_cursor.min(entries.len() - 1)));
        }
        let style = if self.focus == Focus::Files { focused_hl } else { unfocused_hl };
        frame.render_stateful_widget(
            List::new(entries).highlight_style(style),
            files_area,
            &mut state,
        );

        // Sep
        frame.render_widget(
            Paragraph::new(Span::styled("─".repeat(inner.width as usize), dim)),
            chunks[idx],
        );
        idx += 1;

        // Workspace header
        frame.render_widget(
            Paragraph::new(Span::styled("Workspace", label_style)),
            chunks[idx],
        );
        idx += 1;

        // Workspace anchors
        let ws_area = chunks[idx];
        idx += 1;
        let ws_items: Vec<ListItem> = if self.workspace.is_empty() {
            vec![ListItem::new(Span::styled("(empty)", dim))]
        } else {
            self.workspace
                .iter()
                .map(|w| {
                    let avail = ws_area.width as usize;
                    let lhs = format!("{} {}", w.glyph, w.display_name);
                    let lhs_w = lhs.chars().count();
                    let path_w = w.display_path.chars().count();
                    let pad = avail.saturating_sub(lhs_w + path_w + 1);
                    ListItem::new(Line::from(vec![
                        Span::raw(lhs),
                        Span::raw(" ".repeat(pad)),
                        Span::styled(w.display_path.clone(), dim),
                    ]))
                })
                .collect()
        };
        let mut ws_state = ListState::default();
        if !self.workspace.is_empty() {
            ws_state.select(Some(self.workspace_cursor.min(self.workspace.len() - 1)));
        }
        let ws_style = if self.focus == Focus::Workspace { focused_hl } else { unfocused_hl };
        frame.render_stateful_widget(
            List::new(ws_items).highlight_style(ws_style),
            ws_area,
            &mut ws_state,
        );

        // Sep
        frame.render_widget(
            Paragraph::new(Span::styled("─".repeat(inner.width as usize), dim)),
            chunks[idx],
        );
        idx += 1;

        // Recent header
        frame.render_widget(
            Paragraph::new(Span::styled("Recent", label_style)),
            chunks[idx],
        );
        idx += 1;

        // Recents
        let recents_area = chunks[idx];
        idx += 1;
        let r_items: Vec<ListItem> = self
            .recents
            .iter()
            .map(|r| {
                let avail = recents_area.width as usize;
                let label_w = r.display.chars().count();
                let time_w = r.time_ago.chars().count();
                let pad = avail.saturating_sub(label_w + time_w + 1);
                ListItem::new(Line::from(vec![
                    Span::raw(r.display.clone()),
                    Span::raw(" ".repeat(pad)),
                    Span::styled(r.time_ago.clone(), dim),
                ]))
            })
            .collect();
        let mut r_state = ListState::default();
        if !self.recents.is_empty() {
            r_state.select(Some(self.recents_cursor.min(self.recents.len() - 1)));
        }
        let r_style = if self.focus == Focus::Recents { focused_hl } else { unfocused_hl };
        frame.render_stateful_widget(
            List::new(r_items).highlight_style(r_style),
            recents_area,
            &mut r_state,
        );

        // Sep
        frame.render_widget(
            Paragraph::new(Span::styled("─".repeat(inner.width as usize), dim)),
            chunks[idx],
        );
        idx += 1;

        // Footer — `x` only meaningful when the Workspace pane is focused.
        let mut hints: Vec<(&str, &str)> = vec![
            ("↵", "add"),
            ("⇥", "switch pane"),
        ];
        if self.focus == Focus::Workspace {
            hints.push(("x", "remove"));
        }
        hints.push(("esc", "cancel"));
        render_footer(frame, chunks[idx], &hints);
    }
}

fn render_entry(e: &Entry) -> Line<'static> {
    let dim = Style::default().fg(Color::DarkGray);
    match e {
        Entry::Parent => Line::from(Span::raw("../")),
        Entry::Dir { name } => Line::from(Span::raw(format!("{name}/"))),
        Entry::File { name } => Line::from(vec![
            Span::styled("- ", dim),
            Span::raw(name.clone()),
        ]),
    }
}

fn render_footer(frame: &mut Frame<'_>, area: Rect, hints: &[(&str, &str)]) {
    let key_style = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
    let dim = Style::default().fg(Color::DarkGray);
    let mut spans: Vec<Span> = Vec::new();
    for (i, (k, label)) in hints.iter().enumerate() {
        if i > 0 {
            spans.push(Span::styled(" · ", dim));
        }
        spans.push(Span::styled(*k, key_style));
        spans.push(Span::raw(" "));
        spans.push(Span::styled(*label, dim));
    }
    frame.render_widget(
        Paragraph::new(Line::from(spans)).alignment(Alignment::Center),
        area,
    );
}

fn scan(path: &Path) -> Vec<Entry> {
    let mut entries = vec![Entry::Parent];
    if let Ok(rd) = std::fs::read_dir(path) {
        let mut dirs: Vec<String> = Vec::new();
        let mut files: Vec<String> = Vec::new();
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() {
                dirs.push(name);
            } else if ft.is_file() && is_anchor_file(&name) {
                files.push(name);
            }
        }
        dirs.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));
        files.sort_by(|a, b| a.to_lowercase().cmp(&b.to_lowercase()));
        for name in dirs {
            entries.push(Entry::Dir { name });
        }
        for name in files {
            entries.push(Entry::File { name });
        }
    }
    entries
}

/// True iff `name` is a file the picker should surface — either the
/// literal `quod.toml` (a Project anchor) or any `*.json` (a candidate
/// Program anchor; the loader will reject non-program JSON at commit).
fn is_anchor_file(name: &str) -> bool {
    if name == "quod.toml" {
        return true;
    }
    Path::new(name)
        .extension()
        .and_then(|s| s.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("json"))
        .unwrap_or(false)
}

fn expand(text: &str) -> PathBuf {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn tempdir() -> PathBuf {
        use std::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "qui-openmodal-test-{}-{}-{}",
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

    #[test]
    fn is_anchor_file_accepts_quod_toml_and_json() {
        assert!(is_anchor_file("quod.toml"));
        assert!(is_anchor_file("foo.json"));
        assert!(is_anchor_file("FOO.JSON")); // case-insensitive ext
    }

    #[test]
    fn is_anchor_file_rejects_other_tomls_and_random_files() {
        assert!(!is_anchor_file("Cargo.toml"));
        assert!(!is_anchor_file("pyproject.toml"));
        assert!(!is_anchor_file("README.md"));
        assert!(!is_anchor_file("script.sh"));
        assert!(!is_anchor_file(""));
    }

    #[test]
    fn scan_includes_only_anchor_files_plus_dirs_and_parent() {
        let dir = tempdir();
        fs::create_dir(dir.join("subdir")).unwrap();
        fs::write(dir.join("quod.toml"), "name = \"x\"").unwrap();
        fs::write(dir.join("foo.json"), "{}").unwrap();
        fs::write(dir.join("Cargo.toml"), "").unwrap();
        fs::write(dir.join("README.md"), "").unwrap();

        let entries = scan(&dir);

        // Parent first, then dirs, then files.
        assert!(matches!(entries[0], Entry::Parent));

        let names: Vec<&str> = entries
            .iter()
            .filter_map(|e| match e {
                Entry::Dir { name } => Some(name.as_str()),
                Entry::File { name } => Some(name.as_str()),
                Entry::Parent => None,
            })
            .collect();
        assert!(names.contains(&"subdir"));
        assert!(names.contains(&"quod.toml"));
        assert!(names.contains(&"foo.json"));
        assert!(!names.contains(&"Cargo.toml"));
        assert!(!names.contains(&"README.md"));
    }

    #[test]
    fn scan_hides_dot_entries() {
        let dir = tempdir();
        fs::create_dir(dir.join(".hidden")).unwrap();
        fs::write(dir.join(".secret.json"), "{}").unwrap();
        let entries = scan(&dir);
        let names: Vec<&str> = entries
            .iter()
            .filter_map(|e| match e {
                Entry::Dir { name } => Some(name.as_str()),
                Entry::File { name } => Some(name.as_str()),
                Entry::Parent => None,
            })
            .collect();
        assert!(names.is_empty(), "got {names:?}");
    }
}
