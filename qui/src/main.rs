mod column_view;
mod highlight;
mod lsp;
mod open_modal;
mod path_classify;
mod picker;
mod recents;
mod workspace;

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
use crate::path_classify::{ClassifyError, classify, read_project_name};
use crate::picker::{Picker, PickerItem};
use crate::recents::{Recents, relative_time};
use crate::workspace::{Anchor, AnchorId, ProjectRef, Workspace, program_name_from_file};

fn main() -> io::Result<()> {
    let server_cmd = std::env::var("QUI_QUOD_CMD").unwrap_or_else(|_| "quod".into());
    let args: Vec<String> = std::env::args().skip(1).collect();

    let mut app = App::new(server_cmd);
    if let Err(e) = app.bootstrap(&args) {
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
    /// Set of workspace anchors — Project and standalone Program.
    workspace: Workspace,
    /// LSP-derived view of project-routed programs. Refreshed whenever
    /// the workspace folder set changes. Standalone programs aren't here
    /// — they're enumerated from `workspace.standalone_programs()`.
    programs: Vec<ProgramEntry>,
    /// The currently-active program, keyed by `(anchor_context, name)`.
    active: Option<Active>,
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
    /// Same idea, but for the binary→quod pipeline.
    bin_lift_trace: Option<ColumnView>,
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum BodyMode {
    Outline,
    LiftTrace,
    BinLiftTrace,
}

enum Overlay {
    ProgramPicker(Picker),
    OpenProject(OpenModal),
}

#[derive(Clone)]
struct ProgramEntry {
    /// `Config.name` of the project containing this program. Display only.
    project: String,
    /// Absolute path to the project root — identity, not display.
    project_path: PathBuf,
    /// `ProgramSpec.name`.
    name: String,
    /// File path *relative to the project root*.
    file: String,
}

#[derive(Clone)]
struct Active {
    /// Display label for the active program. Equal to `program_name`
    /// for project-routed; absolute file path for standalone.
    label: String,
    /// `Some(...)` when reached through a Project anchor; `None` for
    /// standalone Program anchors.
    anchor_context: Option<ProjectRef>,
    summary: String,
    outline: Vec<OutlineCategory>,
}

#[derive(Clone)]
struct OutlineCategory {
    label: String,
    count: usize,
    items: Vec<String>,
}

/// Tags `PickerItem.user_data` so the picker outcome handler can
/// dispatch to the right `setActiveProgram` shape (project-routed
/// vs standalone). Encoded as a `usize` index over a flat
/// `selectables` table the picker mirrors.
#[derive(Clone)]
enum SelectableProgram {
    /// Index into `App.programs`.
    Routed(usize),
    /// Absolute path to a standalone Program anchor.
    Standalone(PathBuf),
}

impl App {
    fn new(server_cmd: String) -> Self {
        Self {
            server_cmd,
            client: None,
            workspace: Workspace::default(),
            programs: Vec::new(),
            active: None,
            overlay: None,
            recents: Recents::load(),
            error: None,
            sidebar_cursor: 0,
            body_mode: BodyMode::Outline,
            lift_trace: None,
            bin_lift_trace: None,
        }
    }

    /// Translate the CLI args into starting state. Each positional arg is
    /// classified into an Anchor (Project for `quod.toml` / dir-with-
    /// quod.toml, Program for `.json`). With no args, walk the cwd's
    /// ancestors for a `quod.toml` and seed that as a single Project.
    fn bootstrap(&mut self, args: &[String]) -> Result<(), String> {
        self.spawn_client_with(&[])?;

        if args.is_empty() {
            if let Some(root) = find_ancestor_toml(&std::env::current_dir().unwrap_or_default()) {
                let name = read_project_name(&root.join("quod.toml")).map_err(|e| {
                    format!("seed project at {}: {}", root.display(), e)
                })?;
                self.add_anchor(Anchor::Project { root, name })?;
                self.maybe_auto_select();
            }
            return Ok(());
        }

        for arg in args {
            let anchor = classify(arg, |toml| read_project_name(toml))
                .map_err(|e: ClassifyError| e.to_string())?;
            self.add_anchor(anchor)?;
        }
        self.maybe_auto_select();
        Ok(())
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
            BodyMode::BinLiftTrace => {
                if let Some(t) = self.bin_lift_trace.as_mut() {
                    t.move_up();
                }
            }
        }
    }

    fn body_down(&mut self) {
        match self.body_mode {
            BodyMode::Outline => {
                if let Some(active) = &self.active {
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
            BodyMode::BinLiftTrace => {
                if let Some(t) = self.bin_lift_trace.as_mut() {
                    t.move_down();
                }
            }
        }
    }

    fn cycle_body_mode(&mut self) {
        self.body_mode = match self.body_mode {
            BodyMode::Outline => BodyMode::LiftTrace,
            BodyMode::LiftTrace => BodyMode::BinLiftTrace,
            BodyMode::BinLiftTrace => BodyMode::Outline,
        };
    }

    /// Lazy-fetch the lift trace from the server. Idempotent.
    fn ensure_lift_trace(&mut self) {
        if self.lift_trace.is_some() || self.active.is_none() {
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

    fn ensure_bin_lift_trace(&mut self) {
        if self.bin_lift_trace.is_some() || self.active.is_none() {
            return;
        }
        let Some(client) = self.client.as_mut() else { return };
        let resp = match client.request("quod/getBinLiftTrace", json!({})) {
            Ok(v) => v,
            Err(e) => {
                self.error = Some(format!("getBinLiftTrace: {e}"));
                return;
            }
        };
        self.bin_lift_trace = Some(parse_bin_lift_trace(&resp));
    }

    fn ensure_no_dead_end(&mut self) {
        if self.active.is_none() && self.overlay.is_none() {
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
        self.programs = parse_programs(&raw);
        Ok(())
    }

    /// Add an Anchor to the workspace. For Project anchors this notifies
    /// the LSP via `workspace/didChangeWorkspaceFolders` and refreshes
    /// `programs`. For Program anchors there's no LSP interaction —
    /// activation happens later via `setActiveProgram({file})`.
    fn add_anchor(&mut self, anchor: Anchor) -> Result<(), String> {
        match &anchor {
            Anchor::Project { root, .. } => {
                let folder = lsp_types::WorkspaceFolder {
                    uri: format!("file://{}", root.display())
                        .parse()
                        .map_err(|e: <lsp_types::Uri as std::str::FromStr>::Err| e.to_string())?,
                    name: root
                        .file_name()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_else(|| root.to_string_lossy().into_owned()),
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
                self.programs = parse_programs_field(&resp);
                self.recents.note_project(root.clone(), None);
            }
            Anchor::Program { file } => {
                self.recents.note_program(file.clone());
            }
        }

        if !self.workspace.add(anchor.clone()) {
            return Err(format!("already open: {}", anchor.path().display()));
        }
        self.error = None;
        Ok(())
    }

    /// Remove an anchor by identity. For Project anchors this notifies
    /// the LSP and refreshes `programs`. Drops `active` if it was rooted
    /// in (or was) the removed anchor.
    fn remove_anchor(&mut self, id: &AnchorId) -> Result<(), String> {
        match id {
            AnchorId::Project(root) => {
                let folder = lsp_types::WorkspaceFolder {
                    uri: format!("file://{}", root.display())
                        .parse()
                        .map_err(|e: <lsp_types::Uri as std::str::FromStr>::Err| e.to_string())?,
                    name: root
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
                self.programs = parse_programs_field(&resp);

                // Drop active if its anchor_context just went away.
                if let Some(a) = &self.active {
                    if let Some(ctx) = &a.anchor_context {
                        if &ctx.root == root {
                            self.active = None;
                        }
                    }
                }
            }
            AnchorId::Program(file) => {
                if let Some(a) = &self.active {
                    // Standalone activation: anchor_context is None and
                    // label is the file path. Drop if the removed program
                    // matches the active label.
                    if a.anchor_context.is_none()
                        && Path::new(&a.label) == file.as_path()
                    {
                        self.active = None;
                    }
                }
            }
        }
        self.workspace.remove(id);
        Ok(())
    }

    fn set_active_routed(
        &mut self,
        project: &ProjectRef,
        name: &str,
    ) -> Result<(), String> {
        let client = self.client.as_mut().ok_or("no LSP client")?;
        let resp = client
            .request(
                "quod/setActiveProgram",
                json!({
                    "projectPath": project.root.display().to_string(),
                    "name": name,
                }),
            )
            .map_err(|e| e.to_string())?;
        self.finalize_set_active(resp)?;
        self.recents.note_project(project.root.clone(), Some(name.to_string()));
        Ok(())
    }

    fn set_active_standalone(&mut self, file: &Path) -> Result<(), String> {
        let client = self.client.as_mut().ok_or("no LSP client")?;
        let resp = client
            .request(
                "quod/setActiveProgram",
                json!({"file": file.display().to_string()}),
            )
            .map_err(|e| e.to_string())?;
        self.finalize_set_active(resp)?;
        self.recents.note_program(file.to_path_buf());
        Ok(())
    }

    fn finalize_set_active(&mut self, resp: Value) -> Result<(), String> {
        let client = self.client.as_mut().ok_or("no LSP client")?;
        let outline_resp = client
            .request("quod/getProgramOutline", json!({}))
            .map_err(|e| e.to_string())?;
        let outline = parse_outline(&outline_resp);
        self.active = Some(parse_active(&resp, outline)?);
        self.sidebar_cursor = 0;
        // Both columnar caches invalidate when the active program changes.
        self.lift_trace = None;
        self.bin_lift_trace = None;
        Ok(())
    }

    fn maybe_auto_select(&mut self) {
        if self.active.is_some() {
            return;
        }
        let routed: Vec<_> = self.programs.iter().cloned().collect();
        let standalone: Vec<_> = self.workspace.standalone_programs().map(Path::to_path_buf).collect();
        if routed.len() + standalone.len() == 1 {
            if let Some(p) = routed.into_iter().next() {
                let _ = self.set_active_routed(
                    &ProjectRef { root: p.project_path, name: p.project },
                    &p.name,
                );
            } else if let Some(file) = standalone.into_iter().next() {
                let _ = self.set_active_standalone(&file);
            }
        }
    }

    // ---------- Overlays ----------

    fn open_program_picker(&mut self) {
        let (items, _) = self.build_program_items();
        let footer = self.program_picker_footer();
        self.overlay = Some(Overlay::ProgramPicker(
            Picker::new("programs")
                .with_items(items)
                .with_footer(footer),
        ));
    }

    /// Build the picker rows. Returns the items plus a parallel
    /// `selectables` table that maps `user_data` indices back to either
    /// a project-routed `ProgramEntry` or a standalone Program anchor.
    fn build_program_items(&self) -> (Vec<PickerItem>, Vec<SelectableProgram>) {
        let mut items: Vec<PickerItem> = Vec::new();
        let mut selectables: Vec<SelectableProgram> = Vec::new();

        // One header per Project anchor, with its programs nested.
        for (root, name) in self.workspace.projects() {
            let header_idx = items.len();
            items.push(PickerItem::header(format!("{name}/"), header_idx));
            let mut had_any = false;
            for (i, prog) in self.programs.iter().enumerate() {
                if prog.project_path != root {
                    continue;
                }
                had_any = true;
                let sel_idx = selectables.len();
                selectables.push(SelectableProgram::Routed(i));
                let mut item = PickerItem::selectable(prog.name.clone(), header_idx)
                    .with_user_data(sel_idx);
                item.detail = Some(prog.file.clone());
                items.push(item);
            }
            if !had_any {
                items.push(PickerItem::placeholder("{ no programs }", header_idx));
            }
        }

        // Standalone Program anchors: one selectable each, under a single
        // synthetic "standalone" header so the existing picker filter
        // behaves predictably.
        let standalone: Vec<&Path> = self.workspace.standalone_programs().collect();
        if !standalone.is_empty() {
            let header_idx = items.len();
            items.push(PickerItem::header("standalone/", header_idx));
            for file in standalone {
                let display = program_name_from_file(file);
                let sel_idx = selectables.len();
                selectables.push(SelectableProgram::Standalone(file.to_path_buf()));
                let mut item = PickerItem::selectable(display, header_idx)
                    .with_user_data(sel_idx);
                item.detail = Some(display_path(file));
                items.push(item);
            }
        }

        (items, selectables)
    }

    fn program_picker_footer(&self) -> Vec<(String, String)> {
        if self.workspace.anchors().is_empty() {
            vec![
                ("o".into(), "add anchor".into()),
                ("q".into(), "quit".into()),
            ]
        } else {
            vec![
                ("↵".into(), "select".into()),
                ("o".into(), "add".into()),
                ("x".into(), "remove anchor".into()),
                ("esc".into(), "cancel".into()),
            ]
        }
    }

    fn open_open_project_modal(&mut self, initial: Option<PathBuf>) {
        let initial = initial.unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
        let recents: Vec<RecentEntry> = self
            .recents
            .anchors
            .iter()
            .map(|a| RecentEntry {
                path: a.path.clone(),
                display: display_path(&a.path),
                time_ago: relative_time(a.last_opened),
            })
            .collect();
        self.overlay = Some(Overlay::OpenProject(OpenModal::new(initial, recents)));
    }

    fn refresh_program_picker_items(&mut self) {
        let (items, _) = self.build_program_items();
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
        let (_items, selectables) = self.build_program_items();
        let id: Option<AnchorId> = match item.user_data {
            Some(sel_idx) => match selectables.get(sel_idx) {
                Some(SelectableProgram::Routed(i)) => self
                    .programs
                    .get(*i)
                    .map(|p| AnchorId::Project(p.project_path.clone())),
                Some(SelectableProgram::Standalone(file)) => {
                    Some(AnchorId::Program(file.clone()))
                }
                None => None,
            },
            None => {
                // Header row: removing the anchor identified by the header.
                let label = item.label.strip_suffix('/').unwrap_or(&item.label);
                if label == "standalone" {
                    None // header has no single anchor identity
                } else {
                    self.workspace
                        .projects()
                        .find(|(_, n)| *n == label)
                        .map(|(root, _)| AnchorId::Project(root.to_path_buf()))
                }
            }
        };
        let Some(id) = id else { return };
        if let Err(e) = self.remove_anchor(&id) {
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
                let (_items, selectables) = self.build_program_items();
                let sel = match self.overlay.as_ref() {
                    Some(Overlay::ProgramPicker(p)) => p
                        .items
                        .get(idx)
                        .and_then(|it| it.user_data)
                        .and_then(|i| selectables.get(i).cloned()),
                    _ => None,
                };
                let result: Option<Result<(), String>> = match sel {
                    Some(SelectableProgram::Routed(i)) => {
                        self.programs.get(i).cloned().map(|p| {
                            self.set_active_routed(
                                &ProjectRef { root: p.project_path, name: p.project },
                                &p.name,
                            )
                        })
                    }
                    Some(SelectableProgram::Standalone(file)) => {
                        Some(self.set_active_standalone(&file))
                    }
                    None => None,
                };
                match result {
                    Some(Ok(())) => self.overlay = None,
                    Some(Err(e)) => {
                        if let Some(Overlay::ProgramPicker(picker)) = self.overlay.as_mut() {
                            picker.set_error(e);
                        }
                    }
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
                let raw = path.to_string_lossy();
                let anchor_result = classify(&raw, |toml| read_project_name(toml))
                    .map_err(|e| e.to_string());
                let result = anchor_result.and_then(|a| self.add_anchor(a));
                match result {
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
        let line = if let Some(a) = &self.active {
            let mut spans = vec![Span::styled("  ", style)];
            if let Some(ctx) = &a.anchor_context {
                spans.push(Span::styled(
                    ctx.name.clone(),
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
        if self.active.is_none() {
            return; // picker overlay covers
        }
        match self.body_mode {
            BodyMode::Outline => self.draw_body_outline(frame, area),
            BodyMode::LiftTrace => {
                self.ensure_lift_trace();
                self.draw_body_lift_trace(frame, area);
            }
            BodyMode::BinLiftTrace => {
                self.ensure_bin_lift_trace();
                self.draw_body_bin_lift_trace(frame, area);
            }
        }
    }

    fn draw_body_outline(&self, frame: &mut Frame<'_>, area: Rect) {
        let Some(active) = &self.active else { return };
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
            Paragraph::new(Span::styled("─".repeat(area.width as usize), dim)),
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

    fn draw_body_bin_lift_trace(&mut self, frame: &mut Frame<'_>, area: Rect) {
        let dim = Style::default().fg(Color::DarkGray);
        let header = Style::default()
            .fg(Color::Cyan)
            .add_modifier(Modifier::BOLD);
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
                Span::styled(" bin lift trace ", header),
                Span::styled(
                    "  Ghidra decompile → layer-A (CFn) → layer-B → layer-C",
                    dim,
                ),
            ])),
            chunks[0],
        );
        frame.render_widget(
            Paragraph::new(Span::styled("─".repeat(area.width as usize), dim)),
            chunks[1],
        );

        match self.bin_lift_trace.as_mut() {
            Some(view) if !view.rows.is_empty() => {
                view.render(frame, chunks[2]);
            }
            Some(_) => {
                frame.render_widget(
                    Paragraph::new(Span::styled(
                        "(no rows — this program has no binary_units)",
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
        add("o", "add anchor", false);
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
                    project: v.get("projectName")?.as_str()?.to_string(),
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
    let project_name = resp.get("projectName").and_then(Value::as_str).map(str::to_string);
    let project_path = resp.get("projectPath").and_then(Value::as_str).map(PathBuf::from);
    let anchor_context = match (project_name, project_path) {
        (Some(name), Some(root)) => Some(ProjectRef { root, name }),
        _ => None,
    };
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
    Ok(Active { label, anchor_context, summary, outline })
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

fn parse_bin_lift_trace(resp: &Value) -> ColumnView {
    let rows: Vec<Row> = resp
        .get("rows")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let hash = v.get("hash")?.as_str().unwrap_or("").to_string();
                    let name = v.get("name")?.as_str().unwrap_or("").to_string();
                    let bin = v.get("bin").and_then(Value::as_str).unwrap_or("").to_string();
                    let layer_a = v.get("layerA").and_then(Value::as_str).unwrap_or("").to_string();
                    let layer_b = v.get("layerB").and_then(Value::as_str).unwrap_or("").to_string();
                    let layer_c = v.get("layerC").and_then(Value::as_str).unwrap_or("").to_string();
                    let id = if hash.is_empty() {
                        name.clone()
                    } else {
                        format!("{} · {}", hash, name)
                    };
                    Some(Row {
                        cells: vec![id, bin, layer_a, layer_b, layer_c],
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
            // Ghidra decompile is C-shaped; tree-sitter-c gets us mostly there.
            label: "bin (decompile)".into(),
            fixed_width: None,
            language: highlight::Language::C,
        },
        Column {
            label: "layer-A (CFn)".into(),
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
