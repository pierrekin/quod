mod body;
mod column_view;
mod error_modal;
mod footer;
mod help_modal;
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
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use serde_json::{Value, json};

use crate::body::{Body, BodyFocus, BodyFrame, EntityCategory, ProgramShape, View};
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
    /// the workspace folder set changes.
    programs: Vec<ProgramEntry>,
    /// The currently-active program, keyed by `(anchor_context, name)`.
    active: Option<Active>,
    /// Body pane state: sidebar (Entities + Views) plus tab strip.
    body: Body,
    /// Categories for the active program's Entities sidebar section.
    /// Filtered to non-empty by the App; empty when no active program.
    entities: Vec<EntityCategory>,
    /// Views available for the active program — gated by ProgramShape.
    views: Vec<View>,
    /// What's populated in the active program. Drives `views`.
    program_shape: ProgramShape,
    overlay: Option<Overlay>,
    recents: Recents,
    /// Lazily-fetched C-Lift table for the active program. Cleared on
    /// active-program change.
    c_lift: Option<ColumnView>,
    /// Same idea, for the binary→quod pipeline.
    bin_lift: Option<ColumnView>,
}

enum Overlay {
    ProgramPicker(Picker),
    OpenProject(OpenModal),
    Help(help_modal::HelpModal),
    Error(error_modal::ErrorModal),
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
            body: Body::default(),
            entities: Vec::new(),
            views: Vec::new(),
            program_shape: ProgramShape::default(),
            overlay: None,
            recents: Recents::load(),
            c_lift: None,
            bin_lift: None,
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
            } else if let Some(action) = classify_key(key) {
                match action {
                    Action::Quit => return Ok(()),
                    Action::OpenMembership => self.open_open_project_modal(None),
                    Action::OpenPicker => self.open_program_picker(),
                    Action::OpenHelp => {
                        self.overlay = Some(Overlay::Help(help_modal::HelpModal::new()));
                    }
                    Action::ToggleNavBody => self.body.toggle_focus(),
                    Action::CycleTabPrev => self.body.cycle_tab(false),
                    Action::CycleTabNext => self.body.cycle_tab(true),
                    Action::CloseTab => self.body.close_active_tab(),
                    Action::TabKey { forward } => self.body.handle_tab(forward),
                    Action::Launch => {
                        self.body.launch_under_cursor(&self.entities, &self.views);
                    }
                    Action::BodyUp => self.body_up(),
                    Action::BodyDown => self.body_down(),
                    // Per-tab projection cycling: only `details` exists
                    // today, so these are no-ops until per-category
                    // projection lists land.
                    Action::CycleProjPrev | Action::CycleProjNext => {}
                }
            }
        }
    }

    fn body_up(&mut self) {
        if let Some(view) = self.body_active_view_mut() {
            view.move_up();
        } else {
            self.body.move_up();
        }
    }

    fn body_down(&mut self) {
        if let Some(view) = self.body_active_view_mut() {
            view.move_down();
        } else {
            self.body.move_down(self.entities.len(), self.views.len());
        }
    }

    /// When the body is focused on a view tab, returns the view's
    /// ColumnView so arrow keys scroll it. Otherwise None.
    fn body_active_view_mut(&mut self) -> Option<&mut ColumnView> {
        if self.body.focus != BodyFocus::Tab {
            return None;
        }
        let idx = self.body.active_tab?;
        let kind = self.body.tabs.get(idx)?.kind.clone();
        match kind {
            body::TabKind::View(View::CLift) => self.c_lift.as_mut(),
            body::TabKind::View(View::BinaryLift) => self.bin_lift.as_mut(),
            body::TabKind::CategoryList { .. } | body::TabKind::Welcome => None,
        }
    }

    /// Lazy-fetch the C-Lift table for the active program. Idempotent.
    fn ensure_c_lift(&mut self) {
        if self.c_lift.is_some() || self.active.is_none() {
            return;
        }
        let Some(client) = self.client.as_mut() else { return };
        let resp = match client.request("quod/getLiftTrace", json!({})) {
            Ok(v) => v,
            Err(e) => {
                self.show_recoverable_error("getLiftTrace failed", e.to_string());
                return;
            }
        };
        self.c_lift = Some(parse_lift_trace(&resp));
    }

    fn ensure_bin_lift(&mut self) {
        if self.bin_lift.is_some() || self.active.is_none() {
            return;
        }
        let Some(client) = self.client.as_mut() else { return };
        let resp = match client.request("quod/getBinLiftTrace", json!({})) {
            Ok(v) => v,
            Err(e) => {
                self.show_recoverable_error("getBinLiftTrace failed", e.to_string());
                return;
            }
        };
        self.bin_lift = Some(parse_bin_lift_trace(&resp));
    }

    fn show_recoverable_error(&mut self, title: impl Into<String>, body: impl Into<String>) {
        self.overlay = Some(Overlay::Error(error_modal::ErrorModal::recoverable(
            title, body,
        )));
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
        Ok(())
    }

    /// Remove an anchor by identity. For Project anchors this notifies
    /// the LSP and refreshes `programs`. Drops `active` if it was rooted
    /// in (or was) the removed anchor.
    #[allow(dead_code)] // wired up when the picker grows `x` removal
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
        let entities = parse_outline_to_entities(&outline_resp);

        let shape_resp = client
            .request("quod/getActiveProgramShape", json!({}))
            .map_err(|e| e.to_string())?;
        let shape = parse_program_shape(&shape_resp);

        self.active = Some(parse_active(&resp)?);
        self.entities = entities;
        self.program_shape = shape;
        self.views = available_views(shape);
        // Per 04-: all tabs close when the active program changes.
        self.body.reset_for_new_program();
        // Per-view caches invalidate.
        self.c_lift = None;
        self.bin_lift = None;
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
        let (items, selectables) = self.build_program_items();
        let footer = self.program_picker_footer();
        let active_item = self.compute_active_item_index(&items, &selectables);
        self.overlay = Some(Overlay::ProgramPicker(
            Picker::new("choose program")
                .with_items(items)
                .with_footer(footer)
                .with_active(active_item),
        ));
    }

    /// Build the picker rows. Returns the items plus a parallel
    /// `selectables` table that maps `user_data` indices back to either
    /// a project-routed `ProgramEntry` or a standalone Program anchor.
    fn build_program_items(&self) -> (Vec<PickerItem>, Vec<SelectableProgram>) {
        let mut items: Vec<PickerItem> = Vec::new();
        let mut selectables: Vec<SelectableProgram> = Vec::new();

        // One header per Project anchor, with its programs nested. The
        // header carries the abs path to `quod.toml` as detail, rendered
        // right-aligned with `~`-substitution by the picker.
        for (root, name) in self.workspace.projects() {
            let header_idx = items.len();
            let mut header = PickerItem::header(name.to_string(), header_idx);
            header.detail = Some(display_path(&root.join("quod.toml")));
            items.push(header);
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

        // Standalone Program anchors render at the top level — no header,
        // just `- name    /abs/path` rows. Each is its own filter group.
        for file in self.workspace.standalone_programs() {
            let display = program_name_from_file(file);
            let sel_idx = selectables.len();
            selectables.push(SelectableProgram::Standalone(file.to_path_buf()));
            let item_idx = items.len();
            let mut item = PickerItem::standalone(display, item_idx).with_user_data(sel_idx);
            item.detail = Some(display_path(file));
            items.push(item);
        }

        (items, selectables)
    }

    /// Find the picker `items` index that corresponds to the currently
    /// active program, if any. Looks up the active state's
    /// `(anchor_context, label)` against the `selectables` table.
    fn compute_active_item_index(
        &self,
        items: &[PickerItem],
        selectables: &[SelectableProgram],
    ) -> Option<usize> {
        let active = self.active.as_ref()?;
        for (item_idx, item) in items.iter().enumerate() {
            let Some(sel_idx) = item.user_data else { continue };
            let Some(sel) = selectables.get(sel_idx) else { continue };
            let matches = match sel {
                SelectableProgram::Routed(i) => {
                    let Some(prog) = self.programs.get(*i) else { continue };
                    matches!(
                        &active.anchor_context,
                        Some(ctx)
                            if ctx.root == prog.project_path && active.label == prog.name
                    )
                }
                SelectableProgram::Standalone(file) => {
                    active.anchor_context.is_none()
                        && Path::new(&active.label) == file.as_path()
                }
            };
            if matches {
                return Some(item_idx);
            }
        }
        None
    }

    fn program_picker_footer(&self) -> Vec<(&'static str, &'static str)> {
        let mut hints = vec![("ctrl-o", "open")];
        // When the workspace is empty, ctrl-c / esc would dead-end (the
        // picker reopens immediately) — surface ctrl-q as the way out.
        // Otherwise ctrl-c closes back to the active program.
        if self.workspace.anchors().is_empty() {
            hints.push(("ctrl-q", "quit"));
        } else {
            hints.push(("ctrl-c", "close"));
        }
        hints
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

    #[allow(dead_code)] // wired up by the membership editor (next commit)
    fn refresh_program_picker_items(&mut self) {
        let (items, _) = self.build_program_items();
        let footer = self.program_picker_footer();
        if let Some(Overlay::ProgramPicker(picker)) = self.overlay.as_mut() {
            picker.refresh_items(items);
            picker.footer = footer;
        }
    }

    fn handle_overlay_key(&mut self, key: crossterm::event::KeyEvent) -> bool {
        // ctrl-c is universal cancel/back: closes the current overlay.
        // ctrl-q always quits, even from inside an overlay.
        if key.code == KeyCode::Char('q') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return true;
        }
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            self.overlay = None;
            return false;
        }
        // ctrl-p / ctrl-o reach the picker / membership editor from anywhere,
        // including from another overlay (replaces the current one).
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            match key.code {
                KeyCode::Char('p') => {
                    self.open_program_picker();
                    return false;
                }
                KeyCode::Char('o') => {
                    self.open_open_project_modal(None);
                    return false;
                }
                _ => {}
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
            Some(Overlay::Help(h)) => {
                if let help_modal::Outcome::Close = h.handle_key(key) {
                    self.overlay = None;
                }
                false
            }
            Some(Overlay::Error(e)) => {
                match e.handle_key(key) {
                    error_modal::Outcome::Continue => false,
                    error_modal::Outcome::Close => {
                        self.overlay = None;
                        false
                    }
                    error_modal::Outcome::Quit => true,
                }
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
                Constraint::Length(2), // footer block
            ])
            .split(area);
        self.draw_status(frame, chunks[0]);
        self.draw_body(frame, chunks[1]);
        self.draw_keymap(frame, chunks[2]);
        if let Some(overlay) = &self.overlay {
            match overlay {
                Overlay::ProgramPicker(p) => p.render(frame, area),
                Overlay::OpenProject(m) => m.render(frame, area),
                Overlay::Help(h) => h.render(frame, area),
                Overlay::Error(e) => e.render(frame, area),
            }
        }
    }

    fn draw_status(&self, frame: &mut Frame<'_>, area: Rect) {
        // Cyan bg is the bar's base; per-Span fg/modifiers compose on top.
        let bg = Style::default().bg(Color::Cyan).fg(Color::Black);
        let bold = bg.add_modifier(Modifier::BOLD);
        let dim = bg.add_modifier(Modifier::DIM);
        let line = if let Some(a) = &self.active {
            let mut spans = vec![Span::styled("  ", bg)];
            if let Some(ctx) = &a.anchor_context {
                // `> Project / Program`
                spans.push(Span::styled("> ", dim));
                spans.push(Span::styled(ctx.name.clone(), bold));
                spans.push(Span::styled(" / ", dim));
                spans.push(Span::styled(a.label.clone(), bold));
            } else {
                // Standalone — bare program name, no glyph (per 02-).
                spans.push(Span::styled(a.label.clone(), bold));
            }
            // Nav state slot is empty until views/tabs land. When empty,
            // omit the trailing `|` per 02-.
            Line::from(spans)
        } else {
            Line::from(Span::styled("  qui", bold))
        };
        frame.render_widget(Paragraph::new(line).style(bg), area);
    }

    fn draw_body(&mut self, frame: &mut Frame<'_>, area: Rect) {
        if self.active.is_none() {
            return; // picker overlay covers
        }
        // Lazy-fetch any view tab that's about to render.
        if let Some(idx) = self.body.active_tab {
            if let Some(tab) = self.body.tabs.get(idx) {
                match &tab.kind {
                    body::TabKind::View(View::CLift) => self.ensure_c_lift(),
                    body::TabKind::View(View::BinaryLift) => self.ensure_bin_lift(),
                    body::TabKind::CategoryList { .. } | body::TabKind::Welcome => {}
                }
            }
        }
        let frame_data = BodyFrame {
            entities: &self.entities,
            views: &self.views,
            c_lift: self.c_lift.as_mut(),
            bin_lift: self.bin_lift.as_mut(),
        };
        body::render(&self.body, frame, area, frame_data);
    }

    /// Footer block at the bottom of the main view — same renderer as
    /// every modal. Always shows `?` for the full chord list and `ctrl-q`
    /// as the way out.
    fn draw_keymap(&self, frame: &mut Frame<'_>, area: Rect) {
        footer::render(
            frame,
            area,
            &[("?", "show help"), ("ctrl-q", "quit")],
        );
    }
}

// ---------- Key dispatch ----------

/// Top-level actions the main view's key handler can dispatch. New
/// chords become new variants here; the actual binding lives in
/// `classify_key` so it's easy to keep the chord table coherent.
enum Action {
    Quit,
    OpenMembership,
    OpenPicker,
    OpenHelp,
    CloseTab,
    ToggleNavBody,
    CycleTabPrev,
    CycleTabNext,
    CycleProjPrev,
    CycleProjNext,
    TabKey { forward: bool },
    /// `enter` — launch the highlighted sidebar entry as a tab.
    Launch,
    BodyUp,
    BodyDown,
}

/// Map a raw key event from the *main view* (no overlay focused) into
/// a logical action. Returns `None` for keys the main view doesn't bind.
/// The control architecture (`03-`) makes unqualified keys local to the
/// focused thing; the main view today owns the body, so unqualified
/// arrow / hjkl forward to body nav.
fn classify_key(key: crossterm::event::KeyEvent) -> Option<Action> {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);
    match key.code {
        KeyCode::Char('q') if ctrl => Some(Action::Quit),
        KeyCode::Char('o') if ctrl => Some(Action::OpenMembership),
        KeyCode::Char('p') if ctrl => Some(Action::OpenPicker),
        KeyCode::Char('x') if ctrl => Some(Action::CloseTab),
        KeyCode::Char('n') if ctrl => Some(Action::ToggleNavBody),
        // ctrl-shift-[ / ] cycles projections; ctrl-[ / ] cycles tabs.
        // Most terminals deliver the bracket char with the SHIFT modifier
        // for `{` / `}`, so accept both encodings.
        KeyCode::Char('[') if ctrl && shift => Some(Action::CycleProjPrev),
        KeyCode::Char(']') if ctrl && shift => Some(Action::CycleProjNext),
        KeyCode::Char('{') if ctrl => Some(Action::CycleProjPrev),
        KeyCode::Char('}') if ctrl => Some(Action::CycleProjNext),
        KeyCode::Char('[') if ctrl => Some(Action::CycleTabPrev),
        KeyCode::Char(']') if ctrl => Some(Action::CycleTabNext),
        KeyCode::Char('?') => Some(Action::OpenHelp),
        KeyCode::Tab => Some(Action::TabKey { forward: true }),
        KeyCode::BackTab => Some(Action::TabKey { forward: false }),
        KeyCode::Enter => Some(Action::Launch),
        KeyCode::Up | KeyCode::Char('k') if !ctrl => Some(Action::BodyUp),
        KeyCode::Down | KeyCode::Char('j') if !ctrl => Some(Action::BodyDown),
        _ => None,
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

fn parse_active(resp: &Value) -> Result<Active, String> {
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
    Ok(Active { label, anchor_context })
}

fn parse_program_shape(resp: &Value) -> ProgramShape {
    ProgramShape {
        has_source_units: resp.get("hasSourceUnits").and_then(Value::as_bool).unwrap_or(false),
        has_binary_units: resp.get("hasBinaryUnits").and_then(Value::as_bool).unwrap_or(false),
        has_equivalences: resp.get("hasEquivalences").and_then(Value::as_bool).unwrap_or(false),
    }
}

/// Filter the LSP outline to non-empty categories (per 05-: empty
/// categories don't appear in the sidebar).
fn parse_outline_to_entities(resp: &Value) -> Vec<EntityCategory> {
    resp.get("categories")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|v| {
                    let label = v.get("label")?.as_str()?.to_string();
                    let count = v.get("count")?.as_u64()? as usize;
                    if count == 0 {
                        return None;
                    }
                    let items = v
                        .get("items")?
                        .as_array()?
                        .iter()
                        .filter_map(|s| s.as_str().map(str::to_string))
                        .collect();
                    Some(EntityCategory { label, count, items })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Which Views are available for the active program's shape.
fn available_views(shape: ProgramShape) -> Vec<View> {
    let mut out = Vec::new();
    if shape.has_source_units {
        out.push(View::CLift);
    }
    if shape.has_binary_units {
        out.push(View::BinaryLift);
    }
    out
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

fn display_path(p: &Path) -> String {
    if let Ok(home) = std::env::var("HOME") {
        if let Ok(stripped) = p.strip_prefix(&home) {
            return format!("~/{}", stripped.display());
        }
    }
    p.display().to_string()
}
