mod column_view;
mod highlight;
mod lsp;
mod open_modal;
mod picker;
mod recents;

use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use lsp_types::{ClientCapabilities, ClientInfo, InitializeParams, InitializeResult};
use ratatui::Frame;
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use serde_json::{Value, json};

use crate::column_view::{Column, ColumnView, Row};
use crate::open_modal::{OpenModal, RecentEntry};
use crate::picker::{Picker, PickerItem};
use crate::recents::{Recents, relative_time};

fn main() -> io::Result<()> {
    let server_cmd = std::env::var("QUI_QUOD_CMD").unwrap_or_else(|_| "quod".into());
    let arg = std::env::args().nth(1);

    let mut app = App::new(server_cmd);
    if let Err(e) = app.bootstrap(arg.as_deref()) {
        eprintln!("qui: {e}");
        std::process::exit(2);
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let result = app.run(&mut terminal);

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();

    app.teardown();
    let _ = app.recents.save();
    result
}

// ---------- State ----------

struct App {
    server_cmd: String,
    client: Option<lsp::Client>,
    workspace: Workspace,
    overlay: Option<Overlay>,
    recents: Recents,
    error: Option<String>,
    /// Selected category index in the body sidebar (Outline mode).
    sidebar_cursor: usize,
    /// Which body view is showing.
    body_mode: BodyMode,
    /// Lazily-fetched lift trace for the active program. Cleared on
    /// active-program change.
    lift_trace: Option<ColumnView>,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum BodyMode {
    Outline,
    LiftTrace,
}

enum Overlay {
    ProgramPicker(Picker),
    OpenProject(OpenModal),
}

#[derive(Default)]
struct Workspace {
    projects: Vec<ProjectInfo>,
    programs: Vec<ProgramEntry>,
    active: Option<Active>,
}

#[derive(Clone)]
struct ProjectInfo {
    path: PathBuf,
    name: String,
}

#[derive(Clone)]
struct ProgramEntry {
    project: String,
    project_path: PathBuf,
    name: String,
    file: String,
}

#[derive(Clone)]
struct Active {
    label: String,
    project: Option<String>,
    summary: String,
    outline: Vec<OutlineCategory>,
}

#[derive(Clone)]
struct OutlineCategory {
    label: String,
    count: usize,
    items: Vec<String>,
}

impl App {
    fn new(server_cmd: String) -> Self {
        Self {
            server_cmd,
            client: None,
            workspace: Workspace::default(),
            overlay: None,
            recents: Recents::load(),
            error: None,
            sidebar_cursor: 0,
            body_mode: BodyMode::Outline,
            lift_trace: None,
        }
    }

    /// Translate the optional CLI arg into starting state. Per the spec:
    ///   - no arg: walk cwd ancestors for quod.toml; if found, seed
    ///   - dir w/ quod.toml or .toml file: seed that project
    ///   - dir w/o quod.toml: open the open-project modal at that dir
    ///   - anything else: hard error
    fn bootstrap(&mut self, arg: Option<&str>) -> Result<(), String> {
        self.spawn_client_with(&[])?;

        let seed = match arg {
            None => find_ancestor_toml(&std::env::current_dir().unwrap_or_default())
                .map(SeedAction::AddProject),
            Some(s) => Some(self.classify_path(s)?),
        };

        match seed {
            Some(SeedAction::AddProject(p)) => {
                self.add_project(&p)?;
                self.maybe_auto_select();
            }
            Some(SeedAction::OpenAt(initial)) => {
                self.open_open_project_modal(Some(initial));
            }
            None => {}
        }
        Ok(())
    }

    fn classify_path(&self, raw: &str) -> Result<SeedAction, String> {
        let path = expand_path(raw);
        if !path.exists() {
            return Err(format!("path does not exist: {}", path.display()));
        }
        if path.is_dir() {
            if path.join("quod.toml").is_file() {
                return Ok(SeedAction::AddProject(path));
            }
            return Ok(SeedAction::OpenAt(path));
        }
        if path.is_file() {
            if path.extension().and_then(|s| s.to_str()) == Some("toml") {
                let parent = path
                    .parent()
                    .map(Path::to_path_buf)
                    .ok_or_else(|| format!("toml has no parent dir: {}", path.display()))?;
                return Ok(SeedAction::AddProject(parent));
            }
            return Err(format!("not a project (.toml or dir): {}", path.display()));
        }
        Err(format!("unrecognized path: {}", path.display()))
    }

    fn run<B: ratatui::backend::Backend>(&mut self, terminal: &mut Terminal<B>) -> io::Result<()> {
        loop {
            // Hard invariant: never sit in the main view with no active
            // program. The program picker is the entry point that backs
            // every empty/no-active state.
            self.ensure_no_dead_end();

            terminal.draw(|f| self.draw(f))?;
            if !event::poll(Duration::from_millis(250))? {
                continue;
            }
            let Event::Key(key) = event::read()? else { continue };
            if key.kind != KeyEventKind::Press {
                continue;
            }
            if self.overlay.is_some() {
                if self.handle_overlay_key(key) {
                    return Ok(());
                }
            } else {
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        return Ok(());
                    }
                    KeyCode::Char('o') => self.open_open_project_modal(None),
                    KeyCode::Char('p') => self.open_program_picker(),
                    KeyCode::Char('v') => self.cycle_body_mode(),
                    KeyCode::Up | KeyCode::Char('k') => self.body_up(),
                    KeyCode::Down | KeyCode::Char('j') => self.body_down(),
                    _ => {}
                }
            }
        }
    }

    fn body_up(&mut self) {
        match self.body_mode {
            BodyMode::Outline => {
                if self.sidebar_cursor > 0 {
                    self.sidebar_cursor -= 1;
                }
            }
            BodyMode::LiftTrace => {
                if let Some(t) = self.lift_trace.as_mut() {
                    t.move_up();
                }
            }
        }
    }

    fn body_down(&mut self) {
        match self.body_mode {
            BodyMode::Outline => {
                if let Some(active) = &self.workspace.active {
                    let len = active.outline.len();
                    if len > 0 && self.sidebar_cursor + 1 < len {
                        self.sidebar_cursor += 1;
                    }
                }
            }
            BodyMode::LiftTrace => {
                if let Some(t) = self.lift_trace.as_mut() {
                    t.move_down();
                }
            }
        }
    }

    fn cycle_body_mode(&mut self) {
        self.body_mode = match self.body_mode {
            BodyMode::Outline => BodyMode::LiftTrace,
            BodyMode::LiftTrace => BodyMode::Outline,
        };
    }

    /// Lazy-fetch the lift trace from the server. Idempotent.
    fn ensure_lift_trace(&mut self) {
        if self.lift_trace.is_some() || self.workspace.active.is_none() {
            return;
        }
        let Some(client) = self.client.as_mut() else { return };
        let resp = match client.request("quod/getLiftTrace", json!({})) {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(format!("getLiftTrace: {e}"));
                return;
            }
        };
        self.lift_trace = Some(parse_lift_trace(&resp));
    }

    fn ensure_no_dead_end(&mut self) {
        if self.workspace.active.is_none() && self.overlay.is_none() {
            self.open_program_picker();
        }
    }

    fn teardown(&mut self) {
        if let Some(client) = self.client.take() {
            let _ = client.shutdown();
        }
    }

    // ---------- LSP plumbing ----------

    fn spawn_client_with(&mut self, projects: &[&Path]) -> Result<(), String> {
        if let Some(client) = self.client.take() {
            let _ = client.shutdown();
        }
        let mut client = lsp::Client::spawn(&self.server_cmd, &["lsp"])
            .map_err(|e| e.to_string())?;
        let folders: Vec<lsp_types::WorkspaceFolder> = projects
            .iter()
            .filter_map(|p| {
                let uri: lsp_types::Uri = format!("file://{}", p.display()).parse().ok()?;
                Some(lsp_types::WorkspaceFolder {
                    uri,
                    name: p.file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| p.to_string_lossy().into_owned()),
                })
            })
            .collect();
        let init_params = InitializeParams {
            process_id: Some(std::process::id()),
            client_info: Some(ClientInfo {
                name: "qui".into(),
                version: Some(env!("CARGO_PKG_VERSION").into()),
            }),
            capabilities: ClientCapabilities {
                experimental: Some(json!({"qui": {"version": 1}})),
                workspace: Some(lsp_types::WorkspaceClientCapabilities {
                    workspace_folders: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            },
            workspace_folders: if folders.is_empty() { None } else { Some(folders) },
            ..Default::default()
        };
        let raw = client
            .request(
                "initialize",
                serde_json::to_value(&init_params).expect("InitializeParams serializes"),
            )
            .map_err(|e| e.to_string())?;
        let _: InitializeResult = serde_json::from_value(raw.clone())
            .map_err(|e| format!("malformed InitializeResult: {e}"))?;
        client
            .notify("initialized", json!({}))
            .map_err(|e| e.to_string())?;
        self.client = Some(client);
        self.workspace.programs = parse_programs(&raw);
        Ok(())
    }

    fn add_project(&mut self, path: &Path) -> Result<(), String> {
        let path = path
            .canonicalize()
            .map_err(|e| format!("canonicalize: {e}"))?;
        if !path.join("quod.toml").is_file() {
            return Err(format!("no quod.toml in {}", path.display()));
        }
        if self.workspace.projects.iter().any(|p| p.path == path) {
            return Err(format!("already open: {}", path.display()));
        }

        let folder = lsp_types::WorkspaceFolder {
            uri: format!("file://{}", path.display())
                .parse()
                .map_err(|e: <lsp_types::Uri as std::str::FromStr>::Err| e.to_string())?,
            name: path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| path.to_string_lossy().into_owned()),
        };
        let client = self.client.as_mut().ok_or("no LSP client")?;
        client
            .notify(
                "workspace/didChangeWorkspaceFolders",
                json!({"event": {"added": [folder], "removed": []}}),
            )
            .map_err(|e| e.to_string())?;
        let resp = client
            .request("quod/listPrograms", json!({}))
            .map_err(|e| e.to_string())?;
        self.workspace.programs = parse_programs_field(&resp);

        let name = path
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.to_string_lossy().into_owned());
        self.workspace.projects.push(ProjectInfo { path: path.clone(), name });
        self.recents.note_folder(path, None);
        self.error = None;
        Ok(())
    }

    fn remove_project(&mut self, path: &Path) -> Result<(), String> {
        let folder = lsp_types::WorkspaceFolder {
            uri: format!("file://{}", path.display())
                .parse()
                .map_err(|e: <lsp_types::Uri as std::str::FromStr>::Err| e.to_string())?,
            name: path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default(),
        };
        let client = self.client.as_mut().ok_or("no LSP client")?;
        client
            .notify(
                "workspace/didChangeWorkspaceFolders",
                json!({"event": {"added": [], "removed": [folder]}}),
            )
            .map_err(|e| e.to_string())?;
        let resp = client
            .request("quod/listPrograms", json!({}))
            .map_err(|e| e.to_string())?;
        self.workspace.programs = parse_programs_field(&resp);

        self.workspace.projects.retain(|p| p.path != path);
        if let Some(a) = &self.workspace.active {
            if let Some(proj) = &a.project {
                if !self.workspace.programs.iter().any(|p| &p.project == proj) {
                    self.workspace.active = None;
                }
            }
        }
        Ok(())
    }

    fn set_active(&mut self, project: &str, name: &str) -> Result<(), String> {
        let client = self.client.as_mut().ok_or("no LSP client")?;
        let resp = client
            .request(
                "quod/setActiveProgram",
                json!({"project": project, "name": name}),
            )
            .map_err(|e| e.to_string())?;
        let outline_resp = client
            .request("quod/getProgramOutline", json!({}))
            .map_err(|e| e.to_string())?;
        let outline = parse_outline(&outline_resp);
        self.workspace.active = Some(parse_active(&resp, outline)?);
        self.sidebar_cursor = 0;
        self.lift_trace = None; // invalidate; refetched lazily on first LiftTrace draw
        if let Some(p) = self.workspace.projects.iter().find(|p| p.name == project) {
            self.recents.note_folder(p.path.clone(), Some(name.to_string()));
        }
        Ok(())
    }

    fn maybe_auto_select(&mut self) {
        if self.workspace.active.is_none() && self.workspace.programs.len() == 1 {
            let p = self.workspace.programs[0].clone();
            let _ = self.set_active(&p.project, &p.name);
        }
    }

    // ---------- Overlays ----------

    fn open_program_picker(&mut self) {
        let items = self.build_program_items();
        let footer = self.program_picker_footer();
        self.overlay = Some(Overlay::ProgramPicker(
            Picker::new("programs")
                .with_items(items)
                .with_footer(footer),
        ));
    }

    fn build_program_items(&self) -> Vec<PickerItem> {
        let mut items: Vec<PickerItem> = Vec::new();
        for project in &self.workspace.projects {
            let header_idx = items.len();
            items.push(PickerItem::header(format!("{}/", project.name), header_idx));
            let mut had_any = false;
            for (i, prog) in self.workspace.programs.iter().enumerate() {
                if prog.project_path != project.path {
                    continue;
                }
                had_any = true;
                let mut item = PickerItem::selectable(prog.name.clone(), header_idx)
                    .with_user_data(i);
                item.detail = Some(prog.file.clone());
                items.push(item);
            }
            if !had_any {
                items.push(PickerItem::placeholder("{ no programs }", header_idx));
            }
        }
        items
    }

    fn program_picker_footer(&self) -> Vec<(String, String)> {
        if self.workspace.projects.is_empty() {
            vec![
                ("o".into(), "add project".into()),
                ("q".into(), "quit".into()),
            ]
        } else {
            vec![
                ("↵".into(), "select".into()),
                ("o".into(), "add".into()),
                ("x".into(), "remove project".into()),
                ("esc".into(), "cancel".into()),
            ]
        }
    }

    fn open_open_project_modal(&mut self, initial: Option<PathBuf>) {
        let initial = initial.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        let recents: Vec<RecentEntry> = self
            .recents
            .folders
            .iter()
            .map(|f| RecentEntry {
                path: f.path.clone(),
                display: display_path(&f.path),
                time_ago: relative_time(f.last_opened),
            })
            .collect();
        self.overlay = Some(Overlay::OpenProject(OpenModal::new(initial, recents)));
    }

    fn refresh_program_picker_items(&mut self) {
        let items = self.build_program_items();
        let footer = self.program_picker_footer();
        if let Some(Overlay::ProgramPicker(picker)) = self.overlay.as_mut() {
            picker.refresh_items(items);
            picker.footer = footer;
        }
    }

    fn remove_focused_in_picker(&mut self) {
        // Only meaningful from the program picker.
        let Some(Overlay::ProgramPicker(picker)) = self.overlay.as_ref() else { return };
        let filtered = picker.filtered();
        let Some(item_idx) = filtered.get(picker.cursor).copied() else { return };
        let Some(item) = picker.items.get(item_idx).cloned() else { return };
        let project_path: Option<PathBuf> = match item.user_data {
            Some(prog_idx) => self
                .workspace
                .programs
                .get(prog_idx)
                .map(|p| p.project_path.clone()),
            None => picker.items.get(item.group).and_then(|hdr| {
                let stripped = hdr.label.strip_suffix('/').unwrap_or(&hdr.label);
                self.workspace
                    .projects
                    .iter()
                    .find(|p| p.name == stripped)
                    .map(|p| p.path.clone())
            }),
        };
        let Some(path) = project_path else { return };
        if let Err(e) = self.remove_project(&path) {
            if let Some(Overlay::ProgramPicker(p)) = self.overlay.as_mut() {
                p.set_error(format!("remove: {e}"));
            }
            return;
        }
        self.refresh_program_picker_items();
    }

    fn handle_overlay_key(&mut self, key: crossterm::event::KeyEvent) -> bool {
        // ctrl-c always quits, even from inside an overlay.
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return true;
        }
        // o/x shortcuts inside the program picker take precedence over filter typing.
        if let Some(Overlay::ProgramPicker(_)) = &self.overlay {
            if key.modifiers.is_empty() {
                match key.code {
                    KeyCode::Char('o') => {
                        self.open_open_project_modal(None);
                        return false;
                    }
                    KeyCode::Char('x') => {
                        self.remove_focused_in_picker();
                        return false;
                    }
                    _ => {}
                }
            }
        }

        match self.overlay.as_mut() {
            Some(Overlay::ProgramPicker(p)) => {
                let outcome = p.handle_key(key);
                self.process_picker_outcome(outcome);
                false
            }
            Some(Overlay::OpenProject(m)) => {
                let outcome = m.handle_key(key);
                self.process_open_modal_outcome(outcome);
                false
            }
            None => false,
        }
    }

    fn process_picker_outcome(&mut self, outcome: picker::Outcome) {
        match outcome {
            picker::Outcome::Continue => {}
            picker::Outcome::Cancel => {
                self.overlay = None; // invariant will reopen if active is None
            }
            picker::Outcome::Item(idx) => {
                let prog = match self.overlay.as_ref() {
                    Some(Overlay::ProgramPicker(p)) => p
                        .items
                        .get(idx)
                        .and_then(|it| it.user_data)
                        .and_then(|i| self.workspace.programs.get(i).cloned()),
                    _ => None,
                };
                match prog {
                    Some(p) => match self.set_active(&p.project, &p.name) {
                        Ok(()) => self.overlay = None,
                        Err(e) => {
                            if let Some(Overlay::ProgramPicker(picker)) = self.overlay.as_mut() {
                                picker.set_error(e);
                            }
                        }
                    },
                    None => {} // header/placeholder; ignore
                }
            }
        }
    }

    fn process_open_modal_outcome(&mut self, outcome: open_modal::Outcome) {
        match outcome {
            open_modal::Outcome::Continue => {}
            open_modal::Outcome::Cancel => {
                self.overlay = None; // invariant will reopen program picker if no active
            }
            open_modal::Outcome::Open(path) => {
                // Accept either a dir or a .toml file (step up to its parent).
                let dir = if path.is_file()
                    && path.extension().and_then(|s| s.to_str()) == Some("toml")
                {
                    match path.parent().map(Path::to_path_buf) {
                        Some(p) => p,
                        None => {
                            if let Some(Overlay::OpenProject(m)) = self.overlay.as_mut() {
                                m.set_error(format!("toml has no parent: {}", path.display()));
                            }
                            return;
                        }
                    }
                } else {
                    path
                };
                match self.add_project(&dir) {
                    Ok(()) => {
                        self.overlay = None;
                        self.maybe_auto_select();
                    }
                    Err(e) => {
                        if let Some(Overlay::OpenProject(m)) = self.overlay.as_mut() {
                            m.set_error(e);
                        }
                    }
                }
            }
        }
    }

    // ---------- Render ----------

    fn draw(&mut self, frame: &mut Frame<'_>) {
        let area = frame.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // status
                Constraint::Min(1),    // body
                Constraint::Length(1), // keymap
            ])
            .split(area);
        self.draw_status(frame, chunks[0]);
        self.draw_body(frame, chunks[1]);
        self.draw_keymap(frame, chunks[2]);
        if let Some(overlay) = &self.overlay {
            match overlay {
                Overlay::ProgramPicker(p) => p.render(frame, area),
                Overlay::OpenProject(m) => m.render(frame, area),
            }
        }
    }

    fn draw_status(&self, frame: &mut Frame<'_>, area: Rect) {
        let style = Style::default().fg(Color::Black).bg(Color::Cyan);
        let line = if let Some(a) = &self.workspace.active {
            let mut spans = vec![Span::styled("  ", style)];
            if let Some(proj) = &a.project {
                spans.push(Span::styled(
                    proj.clone(),
                    style.add_modifier(Modifier::BOLD),
                ));
                spans.push(Span::styled(" · ", style));
            } else {
                spans.push(Span::styled("(file mode) · ", style));
            }
            spans.push(Span::styled(
                a.label.clone(),
                style.add_modifier(Modifier::BOLD),
            ));
            spans.push(Span::styled(" · ", style));
            spans.push(Span::styled(a.summary.clone(), style));
            Line::from(spans)
        } else {
            Line::from(Span::styled(
                "  qui",
                style.add_modifier(Modifier::BOLD),
            ))
        };
        frame.render_widget(Paragraph::new(line).style(style), area);
    }

    fn draw_body(&mut self, frame: &mut Frame<'_>, area: Rect) {
        if self.workspace.active.is_none() {
            return; // picker overlay covers
        }
        match self.body_mode {
            BodyMode::Outline => self.draw_body_outline(frame, area),
            BodyMode::LiftTrace => {
                self.ensure_lift_trace();
                self.draw_body_lift_trace(frame, area);
            }
        }
    }

    fn draw_body_outline(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(active) = &self.workspace.active else { return };
        let sidebar_w = sidebar_width(&active.outline).min(area.width / 2).max(10);
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Length(sidebar_w),
                Constraint::Min(1),
            ])
            .split(area);
        self.draw_sidebar(frame, cols[0], &active.outline);
        self.draw_detail(frame, cols[1], &active.outline);
    }

    fn draw_body_lift_trace(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let dim = Style::default().fg(Color::DarkGray);
        let header = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);

        // Top: small banner so the user knows this is the lift-trace mode.
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(1),
            ])
            .split(area);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled(" lift trace ", header),
                Span::styled(
                    "  layer-A (C source) → layer-B (structured) → layer-C (core)",
                    dim,
                ),
            ])),
            chunks[0],
        );
        frame.render_widget(
            Paragraph::new(Span::styled(
                "─".repeat(area.width as usize),
                dim,
            )),
            chunks[1],
        );

        match self.lift_trace.as_mut() {
            Some(view) if !view.rows.is_empty() => {
                view.render(frame, chunks[2]);
            }
            Some(_) => {
                frame.render_widget(
                    Paragraph::new(Span::styled(
                        "(no rows — this program has no layer-A/B/C joins)",
                        dim,
                    )),
                    chunks[2],
                );
            }
            None => {
                frame.render_widget(
                    Paragraph::new(Span::styled("loading…", dim)),
                    chunks[2],
                );
            }
        }
    }

    fn draw_sidebar(&self, frame: &mut Frame<'_>, area: Rect, outline: &[OutlineCategory]) {
        let dim = Style::default().fg(Color::DarkGray);
        let header = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
        let highlight = Style::default()
            .fg(Color::Black)
            .bg(Color::Cyan)
            .add_modifier(Modifier::BOLD);

        let inner = Rect {
            x: area.x + 1,
            y: area.y + 1,
            width: area.width.saturating_sub(2),
            height: area.height.saturating_sub(2),
        };
        let mut lines: Vec<Line> = Vec::new();
        lines.push(Line::from(Span::styled("elements", header)));
        lines.push(Line::from(""));
        for (i, cat) in outline.iter().enumerate() {
            let label_w = cat.label.chars().count();
            let count_str = format!("{}", cat.count);
            let count_w = count_str.chars().count();
            let avail = (inner.width as usize).saturating_sub(2);
            let pad = avail.saturating_sub(label_w + count_w);
            let line = if i == self.sidebar_cursor {
                Line::from(vec![
                    Span::styled(format!(" {}", cat.label), highlight),
                    Span::styled(" ".repeat(pad), highlight),
                    Span::styled(format!("{} ", count_str), highlight),
                ])
            } else {
                Line::from(vec![
                    Span::raw(format!(" {}", cat.label)),
                    Span::raw(" ".repeat(pad)),
                    Span::styled(format!("{} ", count_str), dim),
                ])
            };
            lines.push(line);
        }
        frame.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_detail(&self, frame: &mut Frame<'_>, area: Rect, outline: &[OutlineCategory]) {
        let dim = Style::default().fg(Color::DarkGray);
        let header = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);

        let inner = Rect {
            x: area.x + 2,
            y: area.y + 1,
            width: area.width.saturating_sub(3),
            height: area.height.saturating_sub(2),
        };
        let cat = outline.get(self.sidebar_cursor);
        let mut lines: Vec<Line> = Vec::new();
        match cat {
            Some(c) => {
                lines.push(Line::from(vec![
                    Span::styled(c.label.clone(), header),
                    Span::raw(" "),
                    Span::styled(format!("({})", c.count), dim),
                ]));
                lines.push(Line::from(""));
                if c.items.is_empty() {
                    lines.push(Line::from(Span::styled("(empty)", dim)));
                } else {
                    for item in &c.items {
                        lines.push(Line::from(Span::raw(item.clone())));
                    }
                }
            }
            None => {
                lines.push(Line::from(Span::styled("(no category)", dim)));
            }
        }
        if let Some(err) = &self.error {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                err.clone(),
                Style::default().fg(Color::Red),
            )));
        }
        frame.render_widget(Paragraph::new(lines), inner);
    }

    /// Always reflects the *background* surface — the main view's
    /// bindings. Doesn't change when an overlay is up; the overlay has
    /// its own footer.
    fn draw_keymap(&self, frame: &mut Frame<'_>, area: Rect) {
        let dim = Style::default().fg(Color::DarkGray);
        let key = Style::default().fg(Color::Cyan);
        let mut spans: Vec<Span> = vec![Span::raw(" ")];
        let mut add = |k: &'static str, label: &'static str, last: bool| {
            spans.push(Span::styled(k, key));
            spans.push(Span::raw(" "));
            spans.push(Span::styled(label, dim));
            if !last {
                spans.push(Span::styled(" · ", dim));
            }
        };
        add("o", "add project", false);
        add("p", "pick program", false);
        add("q", "quit", true);
        let _ = add;
        frame.render_widget(
            Paragraph::new(Line::from(spans)).alignment(Alignment::Left),
            area,
        );
    }
}

// ---------- Helpers ----------

enum SeedAction {
    AddProject(PathBuf),
    OpenAt(PathBuf),
}

fn parse_programs(initialize_resp: &Value) -> Vec<ProgramEntry> {
    parse_programs_array(
        initialize_resp
            .pointer("/capabilities/experimental/quod/programs")
            .and_then(Value::as_array),
    )
}

fn parse_programs_field(resp: &Value) -> Vec<ProgramEntry> {
    parse_programs_array(resp.get("programs").and_then(Value::as_array))
}

fn parse_programs_array(arr: Option<&Vec<Value>>) -> Vec<ProgramEntry> {
    arr.map(|a| {
        a.iter()
            .filter_map(|v| {
                Some(ProgramEntry {
                    project: v.get("project")?.as_str()?.to_string(),
                    project_path: PathBuf::from(v.get("projectPath")?.as_str()?),
                    name: v.get("name")?.as_str()?.to_string(),
                    file: v.get("file")?.as_str()?.to_string(),
                })
            })
            .collect()
    })
    .unwrap_or_default()
}

fn parse_active(resp: &Value, outline: Vec<OutlineCategory>) -> Result<Active, String> {
    let label = resp
        .get("label")
        .and_then(Value::as_str)
        .ok_or("missing label")?
        .to_string();
    let project = resp
        .get("project")
        .and_then(Value::as_str)
        .map(str::to_string);
    let s = resp.get("summary").cloned().unwrap_or_default();
    let summary = format!(
        "{} fns · {} structured · {} externs · {} structs · {} claims · {} equivs",
        s.get("functions").and_then(Value::as_i64).unwrap_or(0),
        s.get("structuredFunctions").and_then(Value::as_i64).unwrap_or(0),
        s.get("externs").and_then(Value::as_i64).unwrap_or(0),
        s.get("structs").and_then(Value::as_i64).unwrap_or(0),
        s.get("claims").and_then(Value::as_i64).unwrap_or(0),
        s.get("equivalences").and_then(Value::as_i64).unwrap_or(0),
    );
    Ok(Active { label, project, summary, outline })
}

fn parse_lift_trace(resp: &Value) -> ColumnView {
    let rows: Vec<Row> = resp
        .get("rows")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let hash = v.get("hash")?.as_str().unwrap_or("").to_string();
                    let name = v.get("name")?.as_str().unwrap_or("").to_string();
                    let layer_a = v.get("layerA").and_then(Value::as_str).unwrap_or("").to_string();
                    let layer_b = v.get("layerB").and_then(Value::as_str).unwrap_or("").to_string();
                    let layer_c = v.get("layerC").and_then(Value::as_str).unwrap_or("").to_string();
                    let id = if hash.is_empty() { name.clone() } else { format!("{} · {}", hash, name) };
                    Some(Row {
                        cells: vec![id, layer_a, layer_b, layer_c],
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let columns = vec![
        Column {
            label: "hash · name".into(),
            fixed_width: Some(22),
            language: highlight::Language::Plain,
        },
        Column {
            label: "layer-A (C)".into(),
            fixed_width: None,
            language: highlight::Language::C,
        },
        Column {
            label: "layer-B".into(),
            fixed_width: None,
            language: highlight::Language::QuodScript,
        },
        Column {
            label: "layer-C".into(),
            fixed_width: None,
            language: highlight::Language::QuodScript,
        },
    ];
    ColumnView::new(columns, rows)
}

fn parse_outline(resp: &Value) -> Vec<OutlineCategory> {
    resp.get("categories")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    Some(OutlineCategory {
                        label: v.get("label")?.as_str()?.to_string(),
                        count: v.get("count")?.as_u64()? as usize,
                        items: v
                            .get("items")?
                            .as_array()?
                            .iter()
                            .filter_map(|s| s.as_str().map(str::to_string))
                            .collect(),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}

fn find_ancestor_toml(start: &Path) -> Option<PathBuf> {
    let mut cur = Some(start);
    while let Some(p) = cur {
        if p.join("quod.toml").is_file() {
            return Some(p.to_path_buf());
        }
        cur = p.parent();
    }
    None
}

fn expand_path(text: &str) -> PathBuf {
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

fn sidebar_width(outline: &[OutlineCategory]) -> u16 {
    // Longest "label  count" plus padding (1 left, 2 right) and a min.
    let needed = outline
        .iter()
        .map(|c| c.label.chars().count() + format!("{}", c.count).chars().count() + 4)
        .max()
        .unwrap_or(16);
    needed.clamp(16, 28) as u16
}

fn display_path(p: &Path) -> String {
    if let Ok(home) = std::env::var("HOME") {
        if let Ok(stripped) = p.strip_prefix(&home) {
            return format!("~/{}", stripped.display());
        }
    }
    p.display().to_string()
}
