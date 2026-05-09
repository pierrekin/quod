//! Rust bindings for tree-sitter-quodscript.
use tree_sitter_language::LanguageFn;

extern "C" {
    fn tree_sitter_quodscript() -> *const ();
}

/// Tree-sitter language function pointer.
pub const LANGUAGE: LanguageFn = unsafe { LanguageFn::from_raw(tree_sitter_quodscript) };

/// Highlights query (read at compile time from queries/highlights.scm).
pub const HIGHLIGHTS_QUERY: &str = include_str!("../../queries/highlights.scm");
