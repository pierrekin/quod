fn main() {
    let src_dir = std::path::Path::new("src");
    cc::Build::new()
        .std("c11")
        .include(src_dir)
        .file(src_dir.join("parser.c"))
        .compile("tree-sitter-quodscript");
    println!("cargo:rerun-if-changed=src/parser.c");
    println!("cargo:rerun-if-changed=grammar.js");
}
