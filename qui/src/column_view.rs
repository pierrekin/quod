//! Generic columnar/joined view: a fixed set of columns and a list of
//! rows, where each row's cells are multi-line text. Designed for
//! cross-layer "joined" views — lift-trace today, proof-trace next.
//!
//! Cells are rendered with a max line budget per row; overflow is shown
//! with "…". Up/down moves the row cursor; the view auto-scrolls to keep
//! the cursor on screen.
use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};

use crate::highlight::{self, Language};
use crate::scroll;

#[derive(Debug, Clone)]
pub struct Column {
    pub label: String,
    /// `None` = flex (split remaining space evenly among flex columns).
    pub fixed_width: Option<u16>,
    pub language: Language,
}

#[derive(Debug, Clone)]
pub struct Row {
    /// One per column, multi-line text.
    pub cells: Vec<String>,
}

pub struct ColumnView {
    pub columns: Vec<Column>,
    pub rows: Vec<Row>,
    pub cursor: usize,
    pub scroll: usize,
    /// Max lines per cell rendered in a row. Cells longer than this
    /// get "…" appended to the last visible line.
    pub max_cell_lines: u16,
}

impl ColumnView {
    pub fn new(columns: Vec<Column>, rows: Vec<Row>) -> Self {
        Self {
            columns,
            rows,
            cursor: 0,
            scroll: 0,
            max_cell_lines: 6,
        }
    }

    pub fn move_up(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        }
    }

    pub fn move_down(&mut self) {
        if !self.rows.is_empty() && self.cursor + 1 < self.rows.len() {
            self.cursor += 1;
        }
    }

    /// Compute per-column widths given the available area width.
    fn column_widths(&self, total: u16) -> Vec<u16> {
        let n = self.columns.len();
        if n == 0 {
            return vec![];
        }
        // 1 char of padding between columns.
        let separators = (n.saturating_sub(1)) as u16;
        let fixed_total: u16 = self
            .columns
            .iter()
            .filter_map(|c| c.fixed_width)
            .sum();
        let flex_count = self.columns.iter().filter(|c| c.fixed_width.is_none()).count();
        let remaining = total
            .saturating_sub(fixed_total)
            .saturating_sub(separators);
        let flex_each = if flex_count > 0 {
            remaining / flex_count as u16
        } else {
            0
        };
        self.columns
            .iter()
            .map(|c| c.fixed_width.unwrap_or(flex_each))
            .collect()
    }

    /// How tall each rendered row is, in terminal lines (cells + separator).
    fn row_height(&self) -> u16 {
        self.max_cell_lines + 1 // cells + 1-line separator
    }

    fn ensure_cursor_visible(&mut self, body_height: u16) {
        let row_h = self.row_height();
        if row_h == 0 || self.rows.is_empty() {
            return;
        }
        let visible_rows = (body_height / row_h).max(1) as usize;
        self.scroll = scroll::compute_offset(
            self.scroll,
            self.cursor,
            visible_rows,
            self.rows.len(),
        );
    }

    pub fn render(&mut self, frame: &mut Frame<'_>, area: Rect) {
        if area.width < 8 || area.height < 4 {
            return;
        }
        let widths = self.column_widths(area.width);

        // Header row: 1 line + 1 separator line.
        let header_h: u16 = 2;
        let body_y = area.y + header_h;
        let body_h = area.height.saturating_sub(header_h);
        self.ensure_cursor_visible(body_h);

        // Render header.
        self.render_header(frame, area, &widths);

        // Render rows.
        let row_h = self.row_height();
        if row_h == 0 {
            return;
        }
        let mut y = body_y;
        let mut row_idx = self.scroll;
        while y + row_h <= body_y + body_h && row_idx < self.rows.len() {
            self.render_row(
                frame,
                Rect { x: area.x, y, width: area.width, height: row_h },
                &widths,
                row_idx,
                row_idx == self.cursor,
            );
            y += row_h;
            row_idx += 1;
        }
    }

    fn render_header(&self, frame: &mut Frame<'_>, area: Rect, widths: &[u16]) {
        use ratatui::text::{Line, Span};
        use ratatui::widgets::Paragraph;

        let key = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
        let dim = Style::default().fg(Color::DarkGray);

        // Header text on first line.
        let header_y = area.y;
        let mut x = area.x;
        for (col, w) in self.columns.iter().zip(widths.iter()) {
            if *w == 0 {
                continue;
            }
            let cell_area = Rect { x, y: header_y, width: *w, height: 1 };
            let label = truncate(&col.label, *w as usize);
            frame.render_widget(
                Paragraph::new(Line::from(Span::styled(label, key))),
                cell_area,
            );
            x += w;
            // Separator between columns.
            if x < area.x + area.width {
                let sep_area = Rect { x, y: header_y, width: 1, height: 1 };
                frame.render_widget(Paragraph::new(Span::styled(" ", dim)), sep_area);
                x += 1;
            }
        }
        // Underline / separator row.
        let sep_y = area.y + 1;
        let sep_area = Rect { x: area.x, y: sep_y, width: area.width, height: 1 };
        frame.render_widget(
            Paragraph::new(Span::styled(
                "─".repeat(area.width as usize),
                dim,
            )),
            sep_area,
        );
    }

    fn render_row(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        widths: &[u16],
        row_idx: usize,
        is_cursor: bool,
    ) {
        use ratatui::text::Span;
        use ratatui::widgets::Paragraph;

        let row = &self.rows[row_idx];
        let dim = Style::default().fg(Color::DarkGray);
        let highlight_style = Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD);

        let cells_h = self.max_cell_lines.min(area.height.saturating_sub(1));

        // For the cursor row, paint a thin column of cyan on the left as a marker.
        if is_cursor {
            let marker = Rect {
                x: area.x.saturating_sub(0),
                y: area.y,
                width: 1.min(area.width),
                height: cells_h,
            };
            // Reuse area for paragraph that fills with bg color.
            let bar = Paragraph::new(Span::styled("│", highlight_style));
            frame.render_widget(bar, marker);
        }

        let mut x = area.x;
        for (i, w) in widths.iter().enumerate() {
            let cell_text = row.cells.get(i).cloned().unwrap_or_default();
            let lang = self
                .columns
                .get(i)
                .map(|c| c.language)
                .unwrap_or(Language::Plain);
            let cell_area = Rect { x, y: area.y, width: *w, height: cells_h };
            self.render_cell(frame, cell_area, &cell_text, lang, is_cursor && i == 0);
            x += *w;
            if x < area.x + area.width {
                // Vertical-ish separator (just a space; visual cue from headers' sep)
                let sep_area = Rect { x, y: area.y, width: 1, height: cells_h };
                frame.render_widget(Paragraph::new(Span::raw(" ")), sep_area);
                x += 1;
            }
        }

        // Row separator at the bottom of the row's allocation.
        let sep_y = area.y + cells_h;
        if sep_y < area.y + area.height {
            let sep_area = Rect { x: area.x, y: sep_y, width: area.width, height: 1 };
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "·".repeat(area.width as usize),
                    dim,
                )),
                sep_area,
            );
        }
        let _ = is_cursor; // (reserved for further visual cue work)
    }

    fn render_cell(
        &self,
        frame: &mut Frame<'_>,
        area: Rect,
        text: &str,
        lang: Language,
        _emphasize: bool,
    ) {
        use ratatui::text::{Line, Span};
        use ratatui::widgets::Paragraph;

        if area.width == 0 || area.height == 0 {
            return;
        }
        // Get fully-styled lines from the highlighter, then apply width
        // truncation while preserving spans.
        let styled = highlight::highlight(text, lang);
        let max = area.height as usize;
        let total = styled.len();
        let width = area.width as usize;
        let mut out: Vec<Line> = Vec::new();
        for (i, line) in styled.iter().take(max).enumerate() {
            let last_visible_with_overflow = i + 1 == max && total > max;
            out.push(truncate_line(line, width, last_visible_with_overflow));
        }
        while out.len() < max {
            out.push(Line::from(Span::raw("")));
        }
        frame.render_widget(Paragraph::new(out), area);
    }
}

fn truncate(s: &str, max: usize) -> String {
    if max == 0 {
        return String::new();
    }
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    let mut out: String = s.chars().take(max - 1).collect();
    out.push('…');
    out
}

/// Take an already-styled `Line`, truncate to `max` chars (preserving
/// spans), and optionally append a "…" if there's more to indicate
/// content beneath.
fn truncate_line(
    line: &ratatui::text::Line<'_>,
    max: usize,
    suffix_ellipsis: bool,
) -> ratatui::text::Line<'static> {
    use ratatui::style::Style;
    use ratatui::text::{Line, Span};
    if max == 0 {
        return Line::from(Vec::<Span<'static>>::new());
    }
    let total: usize = line.spans.iter().map(|s| s.content.chars().count()).sum();
    let cap = if suffix_ellipsis { max.saturating_sub(1) } else { max };
    let mut remaining = if total <= cap { total } else { cap };
    let mut out: Vec<Span<'static>> = Vec::new();
    for span in &line.spans {
        if remaining == 0 {
            break;
        }
        let take = span.content.chars().count().min(remaining);
        let text: String = span.content.chars().take(take).collect();
        if !text.is_empty() {
            out.push(Span::styled(text, span.style));
        }
        remaining -= take;
    }
    if total > max {
        out.push(Span::styled(
            "…",
            Style::default().fg(ratatui::style::Color::DarkGray),
        ));
    } else if suffix_ellipsis {
        // Already ran out (text fits but we want overflow indicator below).
        out.push(Span::styled(
            "…",
            Style::default().fg(ratatui::style::Color::DarkGray),
        ));
    }
    Line::from(out)
}
