//! Universal footer renderer used by every modal and the main view.
//!
//! Reserves a 2-line area: a full-width `─` separator on top, then a
//! centered hint row of `(key, label)` pairs styled identically across
//! the app. Callers reserve `Constraint::Length(2)` for the area.
//!
//! Convention: every footer must include a "way out" hint
//! (`ctrl-c close` inside overlays, `ctrl-q quit` at the main view).
use ratatui::Frame;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

pub fn render(frame: &mut Frame<'_>, area: Rect, hints: &[(&str, &str)]) {
    let dim = Style::default().fg(Color::DarkGray);
    let key_style = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Length(1)])
        .split(area);

    frame.render_widget(
        Paragraph::new(Span::styled("─".repeat(area.width as usize), dim)),
        chunks[0],
    );

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
        chunks[1],
    );
}
