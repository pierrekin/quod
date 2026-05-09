//! Tree-sitter-driven syntax highlighting.
//!
//! Today: C only (off-the-shelf grammar via `tree-sitter-c`). The quod
//! IR layers (B, C) fall back to plain text — a real tree-sitter
//! grammar for the IR is a separate piece of work and is the natural
//! follow-on to this spike.
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use tree_sitter_highlight::{HighlightConfiguration, HighlightEvent, Highlighter};

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Language {
    Plain,
    C,
    /// quodscript — the textual surface for quod programs.
    QuodScript,
}

/// The set of capture names the highlighter is configured to recognise.
/// Index into this list ↔ index used by `Highlight(usize)`.
const HIGHLIGHT_NAMES: &[&str] = &[
    "comment",
    "constant",
    "constant.builtin",
    "function",
    "function.builtin",
    "keyword",
    "number",
    "operator",
    "punctuation",
    "punctuation.bracket",
    "punctuation.delimiter",
    "string",
    "string.escape",
    "type",
    "type.builtin",
    "variable",
    "variable.parameter",
];

fn style_for(name: &str) -> Style {
    let base = Style::default();
    match name {
        "comment" => base.fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
        "constant" | "constant.builtin" => {
            base.fg(Color::Blue).add_modifier(Modifier::BOLD)
        }
        "function" | "function.builtin" => {
            base.fg(Color::Cyan).add_modifier(Modifier::BOLD)
        }
        "keyword" => base.fg(Color::Magenta).add_modifier(Modifier::BOLD),
        "number" => base.fg(Color::Blue),
        "operator" => base.fg(Color::White),
        "punctuation" | "punctuation.bracket" | "punctuation.delimiter" => {
            base.fg(Color::DarkGray)
        }
        "string" => base.fg(Color::Green),
        "string.escape" => base.fg(Color::Green).add_modifier(Modifier::BOLD),
        "type" | "type.builtin" => base.fg(Color::Yellow),
        "variable" => base,
        "variable.parameter" => base.fg(Color::Cyan),
        _ => base,
    }
}

fn build_c_config() -> Option<HighlightConfiguration> {
    let lang: tree_sitter::Language = tree_sitter_c::LANGUAGE.into();
    let mut cfg = HighlightConfiguration::new(
        lang,
        "c",
        tree_sitter_c::HIGHLIGHT_QUERY,
        "",
        "",
    )
    .ok()?;
    cfg.configure(HIGHLIGHT_NAMES);
    Some(cfg)
}

fn build_quodscript_config() -> Option<HighlightConfiguration> {
    let lang: tree_sitter::Language = tree_sitter_quodscript::LANGUAGE.into();
    let mut cfg = HighlightConfiguration::new(
        lang,
        "quodscript",
        tree_sitter_quodscript::HIGHLIGHTS_QUERY,
        "",
        "",
    )
    .ok()?;
    cfg.configure(HIGHLIGHT_NAMES);
    Some(cfg)
}

pub fn highlight(text: &str, lang: Language) -> Vec<Line<'static>> {
    let cfg = match lang {
        Language::Plain => return plain_lines(text),
        Language::C => build_c_config(),
        Language::QuodScript => build_quodscript_config(),
    };
    match cfg {
        Some(cfg) => highlight_with(text, &cfg).unwrap_or_else(|| plain_lines(text)),
        None => plain_lines(text),
    }
}

fn highlight_with(text: &str, cfg: &HighlightConfiguration) -> Option<Vec<Line<'static>>> {
    let mut highlighter = Highlighter::new();
    let events = highlighter
        .highlight(cfg, text.as_bytes(), None, |_| None)
        .ok()?;
    let bytes = text.as_bytes();
    let mut stack: Vec<usize> = Vec::new();
    let mut current: Vec<Span<'static>> = Vec::new();
    let mut lines: Vec<Line<'static>> = Vec::new();

    for ev in events {
        let Ok(ev) = ev else { continue };
        match ev {
            HighlightEvent::HighlightStart(h) => stack.push(h.0),
            HighlightEvent::HighlightEnd => {
                stack.pop();
            }
            HighlightEvent::Source { start, end } => {
                if start >= end || end > bytes.len() {
                    continue;
                }
                let chunk = std::str::from_utf8(&bytes[start..end]).unwrap_or("");
                let style = stack
                    .last()
                    .and_then(|i| HIGHLIGHT_NAMES.get(*i).copied())
                    .map(style_for)
                    .unwrap_or_default();
                let mut parts = chunk.split('\n').peekable();
                while let Some(part) = parts.next() {
                    if !part.is_empty() {
                        current.push(Span::styled(part.to_string(), style));
                    }
                    if parts.peek().is_some() {
                        lines.push(Line::from(std::mem::take(&mut current)));
                    }
                }
            }
        }
    }
    if !current.is_empty() || lines.is_empty() {
        lines.push(Line::from(current));
    }
    Some(lines)
}

fn plain_lines(text: &str) -> Vec<Line<'static>> {
    text.split('\n')
        .map(|l| Line::from(Span::raw(l.to_string())))
        .collect()
}
