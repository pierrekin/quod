//! Error modal per `06-loose-ends.md`. Shows a title + body, dismissed
//! with ctrl-c / esc (recoverable) or ctrl-q (fatal). Fatal modals
//! reply with `Outcome::Quit`; recoverable ones with `Outcome::Close`.
//!
//! In-modal errors (e.g. an Add-Project failure inside the membership
//! editor) don't use this — they render as a toast inside the existing
//! modal via its `set_error` hook. This module is for app-level errors.
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};

use crate::footer;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum Severity {
    Recoverable,
    Fatal,
}

#[derive(Debug)]
pub enum Outcome {
    Continue,
    Close,
    Quit,
}

pub struct ErrorModal {
    pub title: String,
    pub body: String,
    pub severity: Severity,
}

impl ErrorModal {
    pub fn recoverable(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: body.into(),
            severity: Severity::Recoverable,
        }
    }

    #[allow(dead_code)] // wired when the LSP-died path materializes
    pub fn fatal(title: impl Into<String>, body: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            body: body.into(),
            severity: Severity::Fatal,
        }
    }

    pub fn handle_key(&mut self, key: KeyEvent) -> Outcome {
        match key.code {
            KeyCode::Esc => Outcome::Close,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => Outcome::Close,
            KeyCode::Char('q') if key.modifiers.contains(KeyModifiers::CONTROL) => Outcome::Quit,
            _ => Outcome::Continue,
        }
    }

    pub fn render(&self, frame: &mut Frame<'_>, full_area: Rect) {
        let need_h: u16 = 9;
        let need_w: u16 = 40;
        if full_area.width < need_w || full_area.height < need_h {
            return;
        }
        let w = full_area.width.saturating_sub(2).min(70);
        let h = full_area.height.saturating_sub(2).min(14).max(need_h);
        let x = full_area.x + (full_area.width - w) / 2;
        let y = full_area.y + (full_area.height - h) / 2;
        let area = Rect { x, y, width: w, height: h };

        frame.render_widget(Clear, area);

        let border_color = match self.severity {
            Severity::Recoverable => Color::Yellow,
            Severity::Fatal => Color::Red,
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(border_color))
            .title(format!(" {} ", self.title))
            .title_alignment(Alignment::Center);
        let inner = block.inner(area);
        frame.render_widget(block, area);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(2)])
            .split(inner);

        let body_style = Style::default().fg(Color::White);
        frame.render_widget(
            Paragraph::new(Span::styled(self.body.clone(), body_style)).wrap(Wrap { trim: false }),
            chunks[0],
        );

        let hints: &[(&str, &str)] = match self.severity {
            Severity::Recoverable => &[("ctrl-c", "close")],
            Severity::Fatal => &[("ctrl-c", "close"), ("ctrl-q", "quit")],
        };
        footer::render(frame, chunks[1], hints);
    }
}
