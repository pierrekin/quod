//! Main body pane: sidebar (Entities + Views sections) and a tab strip
//! whose active tab fills the rest of the area.
//!
//! Per `04-tabs-and-projections.md` and `05-views.md`:
//!
//!   - The sidebar has two sections, `Entities` and `Views`. `Entities`
//!     mirrors the `quod/getProgramOutline` categories (filtered to
//!     non-empty). `Views` lists cross-cutting "apps" gated on the
//!     active program's shape (C Lift if there are source units;
//!     Binary Lift if there are binary units).
//!   - Pressing enter on a sidebar entry launches a Tab. Two kinds:
//!     `CategoryList` for entity categories and `View` for the
//!     view-set entries.
//!   - All tabs close when the active program changes.
//!   - Projections: each entity-launched tab carries a sub-strip with
//!     only `details` for now. View tabs have no sub-strip.
//!
//! This module owns the body's *state* — what tabs exist, which one's
//! active, sidebar cursors and focus — plus the rendering logic.
//! Lazy data fetches (lift traces) stay on the App and are passed in
//! via `BodyFrame`.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use crate::column_view::ColumnView;

/// One sidebar entity category. Mirrors what the LSP returns from
/// `quod/getProgramOutline`, filtered to non-empty by the App.
#[derive(Clone)]
pub struct EntityCategory {
    pub label: String,
    pub count: usize,
    pub items: Vec<String>,
}

/// One of the cross-cutting views the sidebar can launch.
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum View {
    CLift,
    BinaryLift,
}

impl View {
    pub fn label(&self) -> &'static str {
        match self {
            View::CLift => "C Lift",
            View::BinaryLift => "Binary Lift",
        }
    }
}

/// Booleans summarizing what's populated in the active program. Drives
/// which Views the sidebar exposes.
#[derive(Clone, Copy, Default)]
pub struct ProgramShape {
    pub has_source_units: bool,
    pub has_binary_units: bool,
    /// Future view (Equivalences) is gated on this; not surfaced yet.
    #[allow(dead_code)]
    pub has_equivalences: bool,
}

#[derive(Clone)]
pub struct Tab {
    pub label: String,
    pub kind: TabKind,
}

#[derive(Clone, Debug)]
pub enum TabKind {
    /// Seeded automatically when an active program is set. Auto-closes
    /// the moment the user launches a real tab from the nav.
    Welcome,
    /// `enter`-on-a-category-in-the-sidebar landed here. The body shows
    /// that category's items (the heading+summary list per 04-).
    CategoryList { category_idx: usize },
    View(View),
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum SidebarSection {
    Entities,
    Views,
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum BodyFocus {
    Sidebar,
    Tab,
}

/// Body state owned by the App. Lazy data (lift traces) lives on the
/// App and is threaded into render via `BodyFrame`.
pub struct Body {
    pub tabs: Vec<Tab>,
    pub active_tab: Option<usize>,
    pub focus: BodyFocus,
    pub sidebar_section: SidebarSection,
    pub entities_cursor: usize,
    pub views_cursor: usize,
}

impl Default for Body {
    fn default() -> Self {
        Self {
            tabs: Vec::new(),
            active_tab: None,
            focus: BodyFocus::Sidebar,
            sidebar_section: SidebarSection::Entities,
            entities_cursor: 0,
            views_cursor: 0,
        }
    }
}

impl Body {
    /// Reset to clean-slate (sidebar focus, single Welcome tab). Called
    /// whenever the active program changes — per 04-, all tabs close;
    /// then a fresh Welcome tab is seeded so the body never renders the
    /// "no tabs" empty state right after activation.
    pub fn reset_for_new_program(&mut self) {
        self.tabs.clear();
        self.tabs.push(Tab {
            label: "welcome".into(),
            kind: TabKind::Welcome,
        });
        self.active_tab = Some(0);
        self.focus = BodyFocus::Sidebar;
        self.sidebar_section = SidebarSection::Entities;
        self.entities_cursor = 0;
        self.views_cursor = 0;
    }

    /// `tab` — local-context cycle within the focused thing.
    pub fn handle_tab(&mut self) {
        match self.focus {
            BodyFocus::Sidebar => {
                self.sidebar_section = match self.sidebar_section {
                    SidebarSection::Entities => SidebarSection::Views,
                    SidebarSection::Views => SidebarSection::Entities,
                };
            }
            BodyFocus::Tab => {
                // Per 04-: tab inside the body is per-tab contextual.
                // No tabs implement context-sensitive tab yet, so this
                // is a no-op.
            }
        }
    }

    pub fn toggle_focus(&mut self) {
        self.focus = match self.focus {
            BodyFocus::Sidebar => BodyFocus::Tab,
            BodyFocus::Tab => BodyFocus::Sidebar,
        };
    }

    pub fn move_up(&mut self) {
        match self.focus {
            BodyFocus::Sidebar => match self.sidebar_section {
                SidebarSection::Entities => {
                    self.entities_cursor = self.entities_cursor.saturating_sub(1)
                }
                SidebarSection::Views => {
                    self.views_cursor = self.views_cursor.saturating_sub(1)
                }
            },
            BodyFocus::Tab => {} // per-tab nav handled by tab-specific code
        }
    }

    pub fn move_down(&mut self, entities_len: usize, views_len: usize) {
        match self.focus {
            BodyFocus::Sidebar => match self.sidebar_section {
                SidebarSection::Entities => {
                    if self.entities_cursor + 1 < entities_len {
                        self.entities_cursor += 1;
                    }
                }
                SidebarSection::Views => {
                    if self.views_cursor + 1 < views_len {
                        self.views_cursor += 1;
                    }
                }
            },
            BodyFocus::Tab => {}
        }
    }

    /// Cycle to the next/previous tab. No-op when zero or one tab.
    pub fn cycle_tab(&mut self, forward: bool) {
        let n = self.tabs.len();
        if n <= 1 {
            return;
        }
        let cur = self.active_tab.unwrap_or(0);
        self.active_tab = Some(if forward { (cur + 1) % n } else { (cur + n - 1) % n });
    }

    /// Close the active tab, leaving focus on the sidebar.
    pub fn close_active_tab(&mut self) {
        let Some(idx) = self.active_tab else { return };
        if idx >= self.tabs.len() {
            return;
        }
        self.tabs.remove(idx);
        if self.tabs.is_empty() {
            self.active_tab = None;
            self.focus = BodyFocus::Sidebar;
        } else if idx >= self.tabs.len() {
            self.active_tab = Some(self.tabs.len() - 1);
        } else {
            self.active_tab = Some(idx);
        }
    }

    /// Open a tab, focusing existing if one matches (per 04-:
    /// "focus existing if open, else open new"). Switches focus to
    /// the body so the user can interact with the freshly-launched tab.
    /// Drops any seeded Welcome tab — once the user opens a real tab,
    /// the welcome message is no longer useful.
    pub fn open_tab(&mut self, tab: Tab) {
        self.tabs.retain(|t| !matches!(t.kind, TabKind::Welcome));
        if let Some(i) = self.tabs.iter().position(|t| same_target(&t.kind, &tab.kind)) {
            self.active_tab = Some(i);
        } else {
            self.tabs.push(tab);
            self.active_tab = Some(self.tabs.len() - 1);
        }
        self.focus = BodyFocus::Tab;
    }

    /// Convenience: launch the entry under the cursor, if any.
    pub fn launch_under_cursor(
        &mut self,
        entities: &[EntityCategory],
        views: &[View],
    ) {
        match self.sidebar_section {
            SidebarSection::Entities => {
                if let Some(cat) = entities.get(self.entities_cursor) {
                    self.open_tab(Tab {
                        label: cat.label.clone(),
                        kind: TabKind::CategoryList {
                            category_idx: self.entities_cursor,
                        },
                    });
                }
            }
            SidebarSection::Views => {
                if let Some(v) = views.get(self.views_cursor) {
                    self.open_tab(Tab {
                        label: v.label().into(),
                        kind: TabKind::View(*v),
                    });
                }
            }
        }
    }
}

/// True iff two TabKinds refer to the same launchable thing — used
/// for focus-existing dedupe in `open_tab`.
fn same_target(a: &TabKind, b: &TabKind) -> bool {
    match (a, b) {
        (TabKind::CategoryList { category_idx: i }, TabKind::CategoryList { category_idx: j }) => {
            i == j
        }
        (TabKind::View(a), TabKind::View(b)) => a == b,
        _ => false,
    }
}

/// One render pass's inputs from the App. Lazy-fetched data lives on
/// the App and is borrowed in here so the body module stays
/// I/O-agnostic.
pub struct BodyFrame<'a> {
    pub entities: &'a [EntityCategory],
    pub views: &'a [View],
    pub c_lift: Option<&'a mut ColumnView>,
    pub bin_lift: Option<&'a mut ColumnView>,
}

pub fn render(body: &Body, frame: &mut Frame<'_>, area: Rect, mut data: BodyFrame<'_>) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(sidebar_width(data.entities, data.views)),
            Constraint::Min(1),
        ])
        .split(area);

    render_sidebar(body, frame, cols[0], &data);
    render_tab_area(body, frame, cols[1], &mut data);
}

fn sidebar_width(entities: &[EntityCategory], views: &[View]) -> u16 {
    let max_label = entities
        .iter()
        .map(|c| c.label.chars().count() + 1 + format!("{}", c.count).chars().count())
        .chain(views.iter().map(|v| v.label().chars().count()))
        .max()
        .unwrap_or(12);
    // `> ` glyph + label + 2-char gutter + count + padding.
    ((max_label + 8) as u16).clamp(18, 32)
}

fn render_sidebar(
    body: &Body,
    frame: &mut Frame<'_>,
    area: Rect,
    data: &BodyFrame<'_>,
) {
    let dim = Style::default().fg(Color::DarkGray);
    let header = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
    let highlight = Style::default()
        .fg(Color::Black)
        .bg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let dim_highlight = Style::default()
        .fg(Color::DarkGray)
        .add_modifier(Modifier::BOLD);

    let inner = Rect {
        x: area.x + 1,
        y: area.y + 1,
        width: area.width.saturating_sub(2),
        height: area.height.saturating_sub(2),
    };

    let mut lines: Vec<Line> = Vec::new();

    let sb_focused = body.focus == BodyFocus::Sidebar;

    // ----- Entities section -----
    lines.push(section_header(
        "Entities",
        sb_focused && body.sidebar_section == SidebarSection::Entities,
        header,
        dim,
    ));
    if data.entities.is_empty() {
        lines.push(empty_row("(none)", dim));
    } else {
        for (i, cat) in data.entities.iter().enumerate() {
            let cursor_here = sb_focused
                && body.sidebar_section == SidebarSection::Entities
                && i == body.entities_cursor;
            let row_style = if cursor_here { highlight } else if sb_focused { Style::default() } else { dim_highlight };
            lines.push(entity_row(&cat.label, cat.count, inner.width, row_style));
        }
    }

    lines.push(Line::from(""));

    // ----- Views section -----
    lines.push(section_header(
        "Views",
        sb_focused && body.sidebar_section == SidebarSection::Views,
        header,
        dim,
    ));
    if data.views.is_empty() {
        lines.push(empty_row("(none)", dim));
    } else {
        for (i, v) in data.views.iter().enumerate() {
            let cursor_here = sb_focused
                && body.sidebar_section == SidebarSection::Views
                && i == body.views_cursor;
            let row_style = if cursor_here { highlight } else if sb_focused { Style::default() } else { dim_highlight };
            lines.push(view_row(v.label(), inner.width, row_style));
        }
    }

    frame.render_widget(Paragraph::new(lines), inner);
}

fn section_header(
    name: &'static str,
    section_focused: bool,
    header: Style,
    dim: Style,
) -> Line<'static> {
    let style = if section_focused { header } else { dim };
    Line::from(Span::styled(name.to_string(), style))
}

fn empty_row(text: &'static str, dim: Style) -> Line<'static> {
    Line::from(vec![Span::raw("  "), Span::styled(text, dim)])
}

fn entity_row(label: &str, count: usize, width: u16, style: Style) -> Line<'static> {
    let lhs = format!("> {label}");
    let count_s = format!("{count}");
    let pad = (width as usize).saturating_sub(lhs.chars().count() + count_s.chars().count() + 1);
    Line::from(vec![
        Span::styled(lhs, style),
        Span::styled(" ".repeat(pad), style),
        Span::styled(count_s, style),
        Span::styled(" ", style),
    ])
}

fn view_row(label: &str, width: u16, style: Style) -> Line<'static> {
    let lhs = format!("> {label}");
    let pad = (width as usize).saturating_sub(lhs.chars().count() + 1);
    Line::from(vec![
        Span::styled(lhs, style),
        Span::styled(" ".repeat(pad), style),
    ])
}

fn render_tab_area(
    body: &Body,
    frame: &mut Frame<'_>,
    area: Rect,
    data: &mut BodyFrame<'_>,
) {
    if body.tabs.is_empty() {
        return; // body stays blank — the nav drives every other affordance
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // tab strip
            Constraint::Length(1), // projection sub-strip
            Constraint::Min(1),    // active tab body
        ])
        .split(area);

    render_tab_strip(body, frame, chunks[0]);
    render_projection_strip(body, frame, chunks[1]);
    render_active_tab_body(body, frame, chunks[2], data);
}

fn render_tab_strip(body: &Body, frame: &mut Frame<'_>, area: Rect) {
    let dim = Style::default().fg(Color::DarkGray);
    let active = Style::default()
        .fg(Color::Black)
        .bg(Color::Cyan)
        .add_modifier(Modifier::BOLD);
    let inactive = Style::default().fg(Color::Gray);

    let mut spans: Vec<Span> = Vec::new();
    for (i, t) in body.tabs.iter().enumerate() {
        let is_active = body.active_tab == Some(i);
        let style = if is_active { active } else { inactive };
        spans.push(Span::styled(format!(" {} ", t.label), style));
        spans.push(Span::styled(" ", dim));
    }
    frame.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn render_projection_strip(body: &Body, frame: &mut Frame<'_>, area: Rect) {
    let active = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);

    // Per 04-: view tabs have no projection sub-strip; entity-launched
    // tabs have a `[details]` sub-strip (only one projection for now).
    // Welcome is informational and shows no projections.
    let Some(idx) = body.active_tab else { return };
    let Some(tab) = body.tabs.get(idx) else { return };
    match &tab.kind {
        TabKind::View(_) | TabKind::Welcome => {} // strip stays empty
        TabKind::CategoryList { .. } => {
            let spans = vec![
                Span::raw(" "),
                Span::styled("[details]", active),
            ];
            frame.render_widget(Paragraph::new(Line::from(spans)), area);
        }
    }
}

fn render_active_tab_body(
    body: &Body,
    frame: &mut Frame<'_>,
    area: Rect,
    data: &mut BodyFrame<'_>,
) {
    let Some(idx) = body.active_tab else { return };
    let Some(tab) = body.tabs.get(idx) else { return };
    match &tab.kind {
        TabKind::Welcome => render_welcome(frame, area),
        TabKind::CategoryList { category_idx } => {
            render_category_list(*category_idx, frame, area, data);
        }
        TabKind::View(View::CLift) => {
            render_view_lift(frame, area, data.c_lift.as_deref_mut(), CLIFT_BANNER);
        }
        TabKind::View(View::BinaryLift) => {
            render_view_lift(frame, area, data.bin_lift.as_deref_mut(), BINLIFT_BANNER);
        }
    }
}

fn render_welcome(frame: &mut Frame<'_>, area: Rect) {
    let dim = Style::default().fg(Color::DarkGray);
    let lines = vec![
        Line::from(""),
        Line::from(vec![Span::raw(" "), Span::raw("Welcome to qui.")]),
        Line::from(""),
        Line::from(vec![
            Span::raw(" "),
            Span::styled(
                "To get started, open a new tab using the navigator.",
                dim,
            ),
        ]),
    ];
    frame.render_widget(Paragraph::new(lines), area);
}

const CLIFT_BANNER: &str =
    "  layer-A (C source) → layer-B (structured) → layer-C (core)";
const BINLIFT_BANNER: &str =
    "  Ghidra decompile → layer-A (CFn) → layer-B → layer-C";

fn render_category_list(
    category_idx: usize,
    frame: &mut Frame<'_>,
    area: Rect,
    data: &BodyFrame<'_>,
) {
    let dim = Style::default().fg(Color::DarkGray);
    let header = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
    let cat = match data.entities.get(category_idx) {
        Some(c) => c,
        None => {
            frame.render_widget(
                Paragraph::new(Span::styled("(stale tab — category gone)", dim)),
                area,
            );
            return;
        }
    };
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(vec![
        Span::raw(" "),
        Span::styled(cat.label.clone(), header),
        Span::raw(" "),
        Span::styled(format!("({})", cat.count), dim),
    ]));
    lines.push(Line::from(""));
    if cat.items.is_empty() {
        lines.push(Line::from(Span::styled("  (empty)", dim)));
    } else {
        for item in &cat.items {
            lines.push(Line::from(vec![Span::raw(" "), Span::raw(item.clone())]));
        }
    }
    frame.render_widget(Paragraph::new(lines), area);
}

fn render_view_lift(
    frame: &mut Frame<'_>,
    area: Rect,
    view_data: Option<&mut ColumnView>,
    banner: &str,
) {
    let dim = Style::default().fg(Color::DarkGray);
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(1)])
        .split(area);
    frame.render_widget(
        Paragraph::new(Span::styled(banner, dim)),
        chunks[0],
    );
    match view_data {
        Some(view) if !view.rows.is_empty() => view.render(frame, chunks[1]),
        Some(_) => {
            frame.render_widget(
                Paragraph::new(Span::styled("(no rows)", dim)),
                chunks[1],
            );
        }
        None => {
            frame.render_widget(
                Paragraph::new(Span::styled("loading…", dim)),
                chunks[1],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cat(label: &str, count: usize) -> EntityCategory {
        EntityCategory { label: label.into(), count, items: vec![] }
    }

    #[test]
    fn open_tab_focuses_existing_when_target_matches() {
        let mut body = Body::default();
        body.open_tab(Tab {
            label: "fns".into(),
            kind: TabKind::CategoryList { category_idx: 0 },
        });
        body.open_tab(Tab {
            label: "structs".into(),
            kind: TabKind::CategoryList { category_idx: 1 },
        });
        assert_eq!(body.tabs.len(), 2);
        assert_eq!(body.active_tab, Some(1));

        // Re-open the first one — should refocus, not duplicate.
        body.open_tab(Tab {
            label: "fns".into(),
            kind: TabKind::CategoryList { category_idx: 0 },
        });
        assert_eq!(body.tabs.len(), 2);
        assert_eq!(body.active_tab, Some(0));
    }

    #[test]
    fn close_active_tab_picks_a_neighbor() {
        let mut body = Body::default();
        body.open_tab(Tab { label: "a".into(), kind: TabKind::View(View::CLift) });
        body.open_tab(Tab { label: "b".into(), kind: TabKind::View(View::BinaryLift) });
        body.active_tab = Some(0);
        body.close_active_tab();
        assert_eq!(body.tabs.len(), 1);
        assert_eq!(body.active_tab, Some(0));
    }

    #[test]
    fn close_last_tab_drops_focus_to_sidebar() {
        let mut body = Body::default();
        body.open_tab(Tab { label: "x".into(), kind: TabKind::View(View::CLift) });
        body.close_active_tab();
        assert!(body.tabs.is_empty());
        assert_eq!(body.active_tab, None);
        assert_eq!(body.focus, BodyFocus::Sidebar);
    }

    #[test]
    fn cycle_tab_wraps() {
        let mut body = Body::default();
        body.open_tab(Tab { label: "a".into(), kind: TabKind::View(View::CLift) });
        body.open_tab(Tab { label: "b".into(), kind: TabKind::View(View::BinaryLift) });
        body.active_tab = Some(0);
        body.cycle_tab(true);
        assert_eq!(body.active_tab, Some(1));
        body.cycle_tab(true);
        assert_eq!(body.active_tab, Some(0));
        body.cycle_tab(false);
        assert_eq!(body.active_tab, Some(1));
    }

    #[test]
    fn tab_within_sidebar_swaps_section() {
        let mut body = Body::default();
        assert_eq!(body.sidebar_section, SidebarSection::Entities);
        body.handle_tab();
        assert_eq!(body.sidebar_section, SidebarSection::Views);
        body.handle_tab();
        assert_eq!(body.sidebar_section, SidebarSection::Entities);
    }

    #[test]
    fn launch_under_cursor_opens_a_view_tab() {
        let mut body = Body::default();
        body.sidebar_section = SidebarSection::Views;
        body.views_cursor = 0;
        body.launch_under_cursor(&[cat("fns", 0)], &[View::CLift, View::BinaryLift]);
        assert_eq!(body.tabs.len(), 1);
        match &body.tabs[0].kind {
            TabKind::View(View::CLift) => {}
            other => panic!("expected View(CLift), got {other:?}"),
        }
    }

    #[test]
    fn reset_seeds_a_welcome_tab() {
        let mut body = Body::default();
        body.open_tab(Tab { label: "x".into(), kind: TabKind::View(View::CLift) });
        body.entities_cursor = 5;
        body.reset_for_new_program();
        assert_eq!(body.tabs.len(), 1);
        assert!(matches!(body.tabs[0].kind, TabKind::Welcome));
        assert_eq!(body.active_tab, Some(0));
        assert_eq!(body.entities_cursor, 0);
        assert_eq!(body.focus, BodyFocus::Sidebar);
    }

    #[test]
    fn opening_a_real_tab_drops_welcome() {
        let mut body = Body::default();
        body.reset_for_new_program();
        assert!(matches!(body.tabs[0].kind, TabKind::Welcome));
        body.open_tab(Tab { label: "fns".into(), kind: TabKind::CategoryList { category_idx: 0 } });
        assert_eq!(body.tabs.len(), 1);
        assert!(matches!(body.tabs[0].kind, TabKind::CategoryList { category_idx: 0 }));
        assert_eq!(body.active_tab, Some(0));
    }
}
