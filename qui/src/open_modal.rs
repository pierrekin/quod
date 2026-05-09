//! Open-project modal: a directory browser pane (top) over a recents
//! pane (bottom). Tab toggles focus between them. Returns a path on
//! commit; the App decides what to do with it (validate, add to
//! workspace, set up active program, etc.).
use std::path::{Path, PathBuf};

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph};

#[derive(Debug)]
pub enum Outcome {
    /// User picked a path. App should validate and add as a project.
    Open(PathBuf),
    Cancel,
    Continue,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Focus {
    Files,
    Recents,
}

#[derive(Debug, Clone)]
pub enum Entry {
    Cur,                                          // "./"
    Parent,                                       // "../"
    Dir { name: String, is_project: bool },       // subdir; ✓ if it has quod.toml
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
    pub recents: Vec<RecentEntry>,
    pub focus: Focus,
    pub fs_cursor: usize,
    pub recents_cursor: usize,
    pub error: Option<String>,
}

impl OpenModal {
    pub fn new(initial: PathBuf, recents: Vec<RecentEntry>) -> Self {
        let input = initial.to_string_lossy().into_owned();
        let entries = scan(&initial);
        Self {
            input,
            entries,
            recents,
            focus: Focus::Files,
            fs_cursor: 0,
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
                self.focus = match self.focus {
                    Focus::Files => {
                        if self.recents.is_empty() { Focus::Files } else { Focus::Recents }
                    }
                    Focus::Recents => Focus::Files,
                };
                Outcome::Continue
            }
            KeyCode::Up => {
                match self.focus {
                    Focus::Files => self.fs_cursor = self.fs_cursor.saturating_sub(1),
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
                    Focus::Recents => {
                        if self.recents_cursor + 1 < self.recents.len() {
                            self.recents_cursor += 1;
                        }
                    }
                }
                Outcome::Continue
            }
            KeyCode::Enter => self.commit(),
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
                    Entry::Cur => Outcome::Open(base),
                    Entry::Parent => match base.parent() {
                        Some(p) => {
                            self.input = p.to_string_lossy().into_owned();
                            self.rescan();
                            Outcome::Continue
                        }
                        None => Outcome::Continue,
                    },
                    Entry::Dir { name, is_project } => {
                        let target = base.join(&name);
                        if is_project {
                            Outcome::Open(target)
                        } else {
                            self.input = target.to_string_lossy().into_owned();
                            self.rescan();
                            Outcome::Continue
                        }
                    }
                }
            }
            Focus::Recents => match self.recents.get(self.recents_cursor) {
                Some(r) => Outcome::Open(r.path.clone()),
                None => Outcome::Continue,
            },
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
            .title(" open project ")
            .title_alignment(Alignment::Center);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        // Vertical layout:
        //   path (1) · sep (1) · [error (1)] · files (min 4)
        //   · sep (1) · "Recent" header (1) · recents (≤ N)
        //   · sep (1) · footer (1)
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

        // Footer
        render_footer(
            frame,
            chunks[idx],
            &[
                ("↵", "open"),
                ("⇥", "switch pane"),
                ("esc", "cancel"),
            ],
        );
    }
}

fn render_entry(e: &Entry) -> Line<'static> {
    match e {
        Entry::Cur => Line::from(Span::raw("./")),
        Entry::Parent => Line::from(Span::raw("../")),
        Entry::Dir { name, is_project } => {
            let mut spans = vec![Span::raw(format!("{}/", name))];
            if *is_project {
                spans.push(Span::styled(
                    "  ✓",
                    Style::default().fg(Color::Green),
                ));
            }
            Line::from(spans)
        }
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
    let mut entries = vec![Entry::Cur, Entry::Parent];
    if let Ok(rd) = std::fs::read_dir(path) {
        let mut dirs: Vec<(String, bool)> = Vec::new();
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue;
            }
            let Ok(ft) = e.file_type() else { continue };
            if !ft.is_dir() {
                continue;
            }
            let is_project = path.join(&name).join("quod.toml").is_file();
            dirs.push((name, is_project));
        }
        dirs.sort_by(|a, b| a.0.to_lowercase().cmp(&b.0.to_lowercase()));
        for (name, is_project) in dirs {
            entries.push(Entry::Dir { name, is_project });
        }
    }
    entries
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
