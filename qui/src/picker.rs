//! Modal picker overlay.
//!
//! Two shapes share this widget:
//!
//! - **Flat**: no headers; every item is selectable. Filter substring-
//!   matches against `search_key`. Used by the add-project prompt.
//!
//! - **Nested**: items grouped under non-selectable headers, with an
//!   optional non-selectable placeholder when a group is empty. Filter
//!   keeps a group iff at least one of its selectables matches; in that
//!   case the header is kept; placeholders are always hidden when
//!   filtering. Used by the program picker.
use std::collections::BTreeSet;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, List, ListItem, ListState, Paragraph};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ItemKind {
    Header,
    Selectable,
    Placeholder,
}

#[derive(Debug, Clone)]
pub struct PickerItem {
    pub label: String,
    pub detail: Option<String>,
    pub search_key: String,
    pub kind: ItemKind,
    /// Items sharing the same `group` are filtered together. For headers
    /// it's their own index; for selectables/placeholders it's the index
    /// of their parent header.
    pub group: usize,
    pub indent: u16,
    /// Caller-defined index. For programs, this points back into the
    /// workspace's flat program list.
    pub user_data: Option<usize>,
}

impl PickerItem {
    pub fn header(label: impl Into<String>, idx: usize) -> Self {
        let label = label.into();
        Self {
            search_key: label.clone(),
            label,
            detail: None,
            kind: ItemKind::Header,
            group: idx,
            indent: 0,
            user_data: None,
        }
    }

    pub fn selectable(label: impl Into<String>, group: usize) -> Self {
        let label = label.into();
        Self {
            search_key: label.clone(),
            label,
            detail: None,
            kind: ItemKind::Selectable,
            group,
            indent: 2,
            user_data: None,
        }
    }

    /// Selectable rendered at the top level (no parent header). Behaves
    /// like a degenerate group: its `group` index is its own item index.
    pub fn standalone(label: impl Into<String>, idx: usize) -> Self {
        let label = label.into();
        Self {
            search_key: label.clone(),
            label,
            detail: None,
            kind: ItemKind::Selectable,
            group: idx,
            indent: 0,
            user_data: None,
        }
    }

    pub fn placeholder(label: impl Into<String>, group: usize) -> Self {
        Self {
            label: label.into(),
            detail: None,
            search_key: String::new(),
            kind: ItemKind::Placeholder,
            group,
            indent: 2,
            user_data: None,
        }
    }

    pub fn with_user_data(mut self, data: usize) -> Self {
        self.user_data = Some(data);
        self
    }
}

pub struct Picker {
    pub title: String,
    pub items: Vec<PickerItem>,
    pub input: String,
    /// Cursor index into the *filtered* list. Always points to a
    /// Selectable when one exists.
    pub cursor: usize,
    /// Each entry is `(hotkey, label)`. Hotkeys are rendered in cyan,
    /// labels dim. Empty list = no footer.
    pub footer: Vec<(String, String)>,
    pub error: Option<String>,
    /// Index into `items` of the currently-active row (the program the
    /// app is rendering). Renders a `*` in the gutter and applies the
    /// active-row style. None = no active row.
    pub active_item: Option<usize>,
}

#[derive(Debug)]
pub enum Outcome {
    Item(usize), // index into self.items (always Selectable)
    Cancel,
    Continue,
}

impl Picker {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            items: Vec::new(),
            input: String::new(),
            cursor: 0,
            footer: vec![("↵".into(), "select".into()), ("esc".into(), "cancel".into())],
            error: None,
            active_item: None,
        }
    }

    pub fn with_active(mut self, idx: Option<usize>) -> Self {
        self.active_item = idx;
        self
    }

    pub fn with_items(mut self, items: Vec<PickerItem>) -> Self {
        self.items = items;
        self.snap_cursor_to_selectable(0, true);
        self
    }

    pub fn with_footer(mut self, hints: Vec<(String, String)>) -> Self {
        self.footer = hints;
        self
    }

    pub fn set_error(&mut self, msg: impl Into<String>) {
        self.error = Some(msg.into());
    }

    #[allow(dead_code)] // wired by the membership editor (next commit)
    pub fn refresh_items(&mut self, items: Vec<PickerItem>) {
        self.items = items;
        self.snap_cursor_to_selectable(0, true);
    }

    pub fn filtered(&self) -> Vec<usize> {
        let nested = self.items.iter().any(|it| it.kind == ItemKind::Header);
        if self.input.is_empty() {
            // No filter: show everything (placeholders included so the
            // empty-project case shows up).
            return (0..self.items.len()).collect();
        }
        let needle = self.input.to_lowercase();
        if !nested {
            return self
                .items
                .iter()
                .enumerate()
                .filter(|(_, it)| {
                    it.kind == ItemKind::Selectable
                        && it.search_key.to_lowercase().contains(&needle)
                })
                .map(|(i, _)| i)
                .collect();
        }
        // Nested: a group survives iff one of its selectables matches.
        let mut group_match: BTreeSet<usize> = BTreeSet::new();
        for it in &self.items {
            if it.kind == ItemKind::Selectable
                && it.search_key.to_lowercase().contains(&needle)
            {
                group_match.insert(it.group);
            }
        }
        let mut out = Vec::new();
        for (i, it) in self.items.iter().enumerate() {
            match it.kind {
                ItemKind::Header => {
                    if group_match.contains(&it.group) {
                        out.push(i);
                    }
                }
                ItemKind::Selectable => {
                    if it.search_key.to_lowercase().contains(&needle) {
                        out.push(i);
                    }
                }
                ItemKind::Placeholder => {} // hidden under any non-empty filter
            }
        }
        out
    }

    /// Move cursor to the nearest Selectable in the filtered list.
    /// `from` is an index into `filtered`. `forward` controls direction
    /// of search; we fall back to the other direction if needed.
    fn snap_cursor_to_selectable(&mut self, from: usize, forward: bool) {
        let filtered = self.filtered();
        if filtered.is_empty() {
            self.cursor = 0;
            return;
        }
        let is_sel = |i: usize| {
            self.items
                .get(filtered[i])
                .map(|it| it.kind == ItemKind::Selectable)
                .unwrap_or(false)
        };
        let try_dir = |start: usize, step: isize, len: usize| -> Option<usize> {
            let mut i = start as isize;
            while i >= 0 && (i as usize) < len {
                if is_sel(i as usize) {
                    return Some(i as usize);
                }
                i += step;
            }
            None
        };
        let len = filtered.len();
        let start = from.min(len - 1);
        let primary = if forward {
            try_dir(start, 1, len).or_else(|| try_dir(start, -1, len))
        } else {
            try_dir(start, -1, len).or_else(|| try_dir(start, 1, len))
        };
        self.cursor = primary.unwrap_or(0);
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Outcome {
        self.error = None;
        match key.code {
            KeyCode::Esc => Outcome::Cancel,
            KeyCode::Up => {
                if self.cursor > 0 {
                    self.snap_cursor_to_selectable(self.cursor - 1, false);
                }
                Outcome::Continue
            }
            KeyCode::Down => {
                let len = self.filtered().len();
                if len > 0 && self.cursor + 1 < len {
                    self.snap_cursor_to_selectable(self.cursor + 1, true);
                }
                Outcome::Continue
            }
            KeyCode::Enter => {
                let f = self.filtered();
                if let Some(&idx) = f.get(self.cursor) {
                    if matches!(
                        self.items.get(idx).map(|it| it.kind),
                        Some(ItemKind::Selectable)
                    ) {
                        return Outcome::Item(idx);
                    }
                }
                Outcome::Continue
            }
            KeyCode::Backspace => {
                self.input.pop();
                self.snap_cursor_to_selectable(0, true);
                Outcome::Continue
            }
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.clear();
                self.snap_cursor_to_selectable(0, true);
                Outcome::Continue
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.input.push(c);
                self.snap_cursor_to_selectable(0, true);
                Outcome::Continue
            }
            _ => Outcome::Continue,
        }
    }

    pub fn render(&self, frame: &mut Frame<'_>, full_area: Rect) {
        let need_h = 6 + self.error.is_some() as u16;
        let need_w = 18;
        if full_area.width < need_w || full_area.height < need_h {
            return;
        }
        let w = full_area.width.saturating_sub(2).min(70);
        let h = full_area.height.saturating_sub(2).min(20).max(need_h);
        let x = full_area.x + (full_area.width - w) / 2;
        let y = full_area.y + (full_area.height - h) / 2;
        let area = Rect { x, y, width: w, height: h };

        frame.render_widget(Clear, area);
        let block = Block::default()
            .borders(Borders::ALL)
            .title(format!(" {} ", self.title))
            .title_alignment(Alignment::Center);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let mut constraints: Vec<Constraint> = vec![
            Constraint::Length(1), // input
            Constraint::Length(1), // separator
        ];
        if self.error.is_some() {
            constraints.push(Constraint::Length(1));
        }
        constraints.push(Constraint::Min(1)); // list
        constraints.push(Constraint::Length(1)); // footer
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints(constraints)
            .split(inner);

        let prompt = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
        let dim = Style::default().fg(Color::DarkGray);

        // Input line with a block cursor.
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("› ", prompt),
                Span::raw(self.input.clone()),
                Span::styled("█", dim),
            ])),
            chunks[0],
        );
        frame.render_widget(
            Paragraph::new(Span::styled("─".repeat(inner.width as usize), dim)),
            chunks[1],
        );

        let mut idx = 2;
        if let Some(err) = &self.error {
            frame.render_widget(
                Paragraph::new(Span::styled(err.clone(), Style::default().fg(Color::Red))),
                chunks[idx],
            );
            idx += 1;
        }

        let list_area = chunks[idx];
        let filtered = self.filtered();
        let header_style = Style::default()
            .fg(Color::Yellow)
            .add_modifier(Modifier::BOLD);
        let placeholder_style = Style::default()
            .fg(Color::DarkGray)
            .add_modifier(Modifier::ITALIC);
        // Per 01-program-picker.md: the active row gets a left-edge `*`
        // in the gutter and the row spans render bold + green.
        let active_style = Style::default()
            .fg(Color::Green)
            .add_modifier(Modifier::BOLD);

        // Reserve the leftmost column as a 2-char gutter for the active
        // marker. Headers/placeholders still render but never get the
        // marker.
        const GUTTER: &str = "  ";
        const GUTTER_ACTIVE: &str = "* ";

        let items: Vec<ListItem> = filtered
            .iter()
            .map(|i| {
                let it = &self.items[*i];
                let is_active = self.active_item == Some(*i);
                let gutter = if is_active { GUTTER_ACTIVE } else { GUTTER };
                let gutter_span = if is_active {
                    Span::styled(gutter, active_style)
                } else {
                    Span::raw(gutter)
                };
                let indent = " ".repeat(it.indent as usize);
                let line = match it.kind {
                    ItemKind::Header => {
                        // `> Name        ~/abs/path`
                        let label_w = 2 /* "> " */ + it.label.chars().count();
                        let label_span = Span::styled(
                            format!("> {}", it.label),
                            header_style,
                        );
                        match &it.detail {
                            Some(d) => {
                                let avail = (list_area.width as usize)
                                    .saturating_sub(label_w + GUTTER.len() + 2);
                                let trimmed = truncate_left(d, avail);
                                let pad = (list_area.width as usize)
                                    .saturating_sub(label_w + trimmed.chars().count() + GUTTER.len() + 1);
                                Line::from(vec![
                                    gutter_span,
                                    label_span,
                                    Span::raw(" ".repeat(pad)),
                                    Span::styled(trimmed, dim),
                                ])
                            }
                            None => Line::from(vec![gutter_span, label_span]),
                        }
                    }
                    ItemKind::Placeholder => Line::from(vec![
                        gutter_span,
                        Span::raw(indent),
                        Span::styled(it.label.clone(), placeholder_style),
                    ]),
                    ItemKind::Selectable => {
                        // `  - Name           rel/path`  (under a header)
                        // `- Name             /abs/path` (top-level standalone)
                        let bullet = "- ";
                        let label_w = it.indent as usize
                            + bullet.chars().count()
                            + it.label.chars().count();
                        let label_style = if is_active { active_style } else { Style::default() };
                        let label_spans = vec![
                            Span::raw(indent),
                            Span::styled(bullet, if is_active { active_style } else { dim }),
                            Span::styled(it.label.clone(), label_style),
                        ];
                        match &it.detail {
                            Some(d) => {
                                let avail = (list_area.width as usize)
                                    .saturating_sub(label_w + GUTTER.len() + 2);
                                let trimmed = truncate_left(d, avail);
                                let pad = (list_area.width as usize)
                                    .saturating_sub(label_w + trimmed.chars().count() + GUTTER.len() + 1);
                                let mut spans = vec![gutter_span];
                                spans.extend(label_spans);
                                spans.push(Span::raw(" ".repeat(pad)));
                                spans.push(Span::styled(
                                    trimmed,
                                    if is_active { active_style } else { dim },
                                ));
                                Line::from(spans)
                            }
                            None => {
                                let mut spans = vec![gutter_span];
                                spans.extend(label_spans);
                                Line::from(spans)
                            }
                        }
                    }
                };
                ListItem::new(line)
            })
            .collect();

        let mut state = ListState::default();
        if !filtered.is_empty() {
            state.select(Some(self.cursor.min(filtered.len() - 1)));
        }
        let list = List::new(items).highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );
        frame.render_stateful_widget(list, list_area, &mut state);

        idx += 1;
        let key_style = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
        let mut footer_spans: Vec<Span> = Vec::new();
        for (i, (k, label)) in self.footer.iter().enumerate() {
            if i > 0 {
                footer_spans.push(Span::styled(" · ", dim));
            }
            footer_spans.push(Span::styled(k.clone(), key_style));
            footer_spans.push(Span::raw(" "));
            footer_spans.push(Span::styled(label.clone(), dim));
        }
        frame.render_widget(
            Paragraph::new(Line::from(footer_spans)).alignment(Alignment::Center),
            chunks[idx],
        );
    }
}

/// Trim `s` to at most `max` chars, preserving the right-hand side and
/// prefixing an ellipsis if anything was dropped. Empty `max` returns "".
fn truncate_left(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    let skip = count - (max - 1);
    let mut out = String::with_capacity(s.len());
    out.push('…');
    out.extend(s.chars().skip(skip));
    out
}
