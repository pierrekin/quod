//! Help overlay listing the global chord set. Triggered by `?` from any
//! non-text-input context (per `03-control-architecture.md`). Closes on
//! `esc` / `ctrl-c` / `?` again / any other key.
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph};

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
        let need_h = (CHORDS.len() as u16) + 6;
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
                Constraint::Length(1),
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
        frame.render_widget(
            Paragraph::new(Span::styled("esc to close", dim)).alignment(Alignment::Center),
            chunks[1],
        );
    }
}

/// (chord, what it does). Edit when adding new chords. Order is what the
/// help modal renders top-to-bottom.
const CHORDS: &[(&str, &str)] = &[
    ("ctrl-p", "program picker"),
    ("ctrl-o", "workspace anchors (add / remove)"),
    ("ctrl-c", "cancel / back one level"),
    ("ctrl-q", "quit"),
    ("ctrl-x", "close active tab"),
    ("ctrl-n", "switch focus between nav and body"),
    ("ctrl-[ / ctrl-]", "cycle tabs"),
    ("ctrl-shift-[ / ctrl-shift-]", "cycle projections"),
    ("?", "this help"),
];
