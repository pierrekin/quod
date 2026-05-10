//! Help overlay listing the global chord set. Triggered by `?` from any
//! non-text-input context (per `03-control-architecture.md`). Closes on
//! `esc` / `ctrl-c` / `?` again / any other key.
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

use crate::footer;

#[derive(Debug)]
pub enum Outcome {
    Continue,
    Close,
}

pub struct HelpModal;

impl HelpModal {
    pub fn new() -> Self {
        Self
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Esc => Outcome::Close,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => Outcome::Close,
            KeyCode::Char('?') => Outcome::Close,
            _ => Outcome::Continue,
        }
    }

    pub fn render(&self, frame: &mut Frame<'_>, full_area: Rect) {
        let need_h = (CHORDS.len() as u16) + 7;
        let need_w = 50;
        if full_area.width < need_w || full_area.height < need_h {
            return;
        }
        let w = full_area.width.saturating_sub(2).min(64);
        let h = full_area.height.saturating_sub(2).min(need_h + 2);
        let x = full_area.x + (full_area.width - w) / 2;
        let y = full_area.y + (full_area.height - h) / 2;
        let area = Rect { x, y, width: w, height: h };

        frame.render_widget(Clear, area);
        let block = Block::default()
            .borders(Borders::ALL)
            .title(" help ")
            .title_alignment(Alignment::Center);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(1),
                Constraint::Length(2),
            ])
            .split(inner);

        let key_style = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
        let dim = Style::default().fg(Color::DarkGray);

        let mut lines: Vec<Line> = Vec::new();
        for (chord, label) in CHORDS {
            let pad = (inner.width as usize)
                .saturating_sub(chord.chars().count() + label.chars().count() + 4);
            lines.push(Line::from(vec![
                Span::raw(" "),
                Span::styled(*chord, key_style),
                Span::raw(" ".repeat(pad)),
                Span::styled(*label, dim),
                Span::raw(" "),
            ]));
        }
        frame.render_widget(Paragraph::new(lines), chunks[0]);
        footer::render(frame, chunks[1], &[("ctrl-c", "close")]);
    }
}

/// (chord, what it does). Edit when adding new chords. Order is what the
/// help modal renders top-to-bottom.
const CHORDS: &[(&str, &str)] = &[
    ("ctrl-o", "open program"),
    ("ctrl-p", "choose program"),
    ("ctrl-c or esc", "cancel"),
    ("ctrl-q", "quit"),
    ("ctrl-x", "close tab"),
    ("ctrl-n", "switch focus"),
    ("ctrl-[ and ctrl-]", "cycle tabs"),
    ("ctrl-shift-[ and ctrl-shift-]", "cycle subtabs"),
    ("?", "show help"),
];
