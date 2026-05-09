/**
 * @file Quodscript grammar (textual surface for quod programs).
 *
 * Spec lives in /root/quod/LANGUAGE.md. This grammar accepts both
 * authoring qs (`fn name(...) { ... }`) and the pretty-printer output
 * used by `quod show` (no `fn` keyword, no statement terminators) —
 * the same surface, different ergonomics.
 *
 * Spike scope: enough structure for a tree-sitter highlighter to
 * classify tokens (keywords, types, identifiers, literals, comments,
 * operators). Fine-grained AST work can come later.
 */

module.exports = grammar({
  name: 'quodscript',

  word: $ => $.identifier,

  extras: $ => [
    /[ \t\r\n]+/,
    $.line_comment,
  ],

  conflicts: $ => [
    // `name { ... }` is both a struct-init and a call followed by a
    // block; no good way to disambiguate locally.
    [$.call, $.struct_init],
    // `_path` and `_primary` both reach `identifier`; declare to make
    // GLR explore both.
    [$._path, $._primary],
  ],

  rules: {
    // Top-level allows only functions (and freestanding comments via
    // `extras`). Statements/expressions at the top would fight
    // function-headers for the same prefix `name(...)`; restricting to
    // functions sidesteps the ambiguity. This matches the pretty-
    // printer's output shape — every code cell is a function body.
    source_file: $ => repeat($.function),

    line_comment: _ => token(seq('//', /[^\n]*/)),

    // ---------- Functions ----------

    function: $ => seq(
      optional('fn'),
      field('name', $._path),
      field('parameters', $.parameter_list),
      '->',
      field('return_type', $._type),
      field('body', $.block),
    ),

    // Either a plain ident (`square`) or a dotted path (`core.str.eq`).
    _path: $ => choice($.identifier, $.dotted_path),

    parameter_list: $ => seq('(', optional(commaSep1($.parameter)), ')'),
    parameter: $ => seq(
      field('name', $.identifier),
      ':',
      field('type', $._type),
    ),

    // ---------- Types ----------

    _type: $ => choice(
      $.pointer_type,
      $.primitive_type,
      $.named_type,
    ),
    primitive_type: _ => choice(
      'i1', 'i8', 'i16', 'i32', 'i64', 'void', 'bool',
    ),
    pointer_type: $ => seq($.primitive_type, '*'),
    named_type: $ => $._path,

    // ---------- Statements ----------

    block: $ => seq('{', repeat($._statement), '}'),

    _statement: $ => choice(
      $.let_statement,
      $.if_statement,
      $.while_statement,
      $.for_statement,
      $.return_statement,
      $.match_statement,
      $.with_arena_statement,
      $.store_statement,
      $.assign_statement,
      $.field_set_statement,
      $.expression_statement,
    ),

    let_statement: $ => seq(
      'let',
      field('name', $.identifier),
      ':',
      field('type', $._type),
      '=',
      field('value', $._expression),
      optional(';'),
    ),

    if_statement: $ => prec.right(seq(
      'if', '(', $._expression, ')', $.block,
      optional(seq('else', $.block)),
    )),

    while_statement: $ => seq(
      'while', '(', $._expression, ')', $.block,
    ),

    for_statement: $ => seq(
      'for', $.identifier, ':', $._type,
      'in', $._expression, '..', $._expression,
      $.block,
    ),

    return_statement: $ => prec.right(seq(
      'return',
      optional($._expression),
      optional(';'),
    )),

    match_statement: $ => seq(
      'match', $._expression, '{', repeat($.match_arm), '}',
    ),
    match_arm: $ => seq(
      choice($.enum_pattern, '_'),
      optional(seq($.identifier, repeat(seq(',', $.identifier)))),
      $.block,
    ),
    enum_pattern: $ => seq(
      $.identifier, '::', $.identifier,
    ),

    with_arena_statement: $ => seq(
      'with_arena', $.identifier,
      '(', 'capacity', '=', $._expression, ')',
      $.block,
    ),

    store_statement: $ => seq(
      'store', '(', $._expression, ',', $._expression, ')',
      optional(';'),
    ),

    assign_statement: $ => prec(1, seq(
      $.identifier, '=', $._expression, optional(';'),
    )),

    field_set_statement: $ => prec(2, seq(
      $.identifier, '.', $.identifier, '=', $._expression, optional(';'),
    )),

    expression_statement: $ => seq($._expression, optional(';')),

    // ---------- Expressions ----------

    _expression: $ => choice(
      $.binary_expression,
      $.try_expression,
      $._primary,
    ),

    binary_expression: $ => choice(
      prec.left(1, seq($._expression, '||', $._expression)),
      prec.left(2, seq($._expression, '&&', $._expression)),
      prec.left(3, seq(
        $._expression,
        choice('==', '!=', '<', '<=', '>', '>=', '<u', '<=u', '>u', '>=u'),
        $._expression,
      )),
      prec.left(4, seq($._expression, choice('+', '-'), $._expression)),
      prec.left(5, seq(
        $._expression,
        choice('*', '/', '%', '/u'),
        $._expression,
      )),
    ),

    try_expression: $ => prec(7, seq($._expression, '?')),

    _primary: $ => choice(
      $.number_literal,
      $.char_literal,
      $.bool_literal,
      $.null_literal,
      $.string_ref,
      $.builtin_call,
      $.call,
      $.struct_init,
      $.enum_init,
      $.field_access,
      $.parenthesized,
      $.dotted_path,
      $.identifier,
    ),

    number_literal: _ => token(seq(
      optional('-'),
      /\d+/,
      optional(/[iu](8|16|32|64)/),
    )),
    char_literal: _ => token(seq(
      "'",
      choice(/[^'\\]/, /\\./),
      "'",
    )),
    bool_literal: _ => choice('true', 'false'),
    null_literal: _ => 'null',

    // `&.name.path` — single token so it doesn't fight `&` (operator)
    // or the leading dot of a path.
    string_ref: _ => token(seq('&', /\.[A-Za-z_0-9][A-Za-z0-9_.]*/)),

    builtin_call: $ => choice(
      seq('load', '[', $._type, ']', '(', $._expression, ')'),
      seq('widen', '(', $._expression, 'to', $._type, ')'),
      seq('uwiden', '(', $._expression, 'to', $._type, ')'),
      seq('ptr_offset', '(', $._expression, ',', $._expression, ')'),
      seq('sizeof', '[', $._type, ']'),
    ),

    call: $ => prec(8, seq(
      field('callee', $._path),
      '(', optional(commaSep1($._expression)), ')',
    )),

    struct_init: $ => prec(8, seq(
      field('type', $._path),
      '{', optional(commaSep1($.field_init)), '}',
    )),
    field_init: $ => seq(
      field('name', $.identifier),
      ':',
      field('value', $._expression),
    ),

    enum_init: $ => seq(
      $.identifier, '::', $.identifier,
      '{', optional(commaSep1($.field_init)), '}',
    ),

    field_access: $ => prec.left(9, seq(
      $._primary, '.', $.identifier,
    )),

    parenthesized: $ => seq('(', $._expression, ')'),

    // ---------- Lexical primitives ----------

    // Dotted name like `core.str.eq` — at least one dot, so single
    // idents stay as `identifier` and don't shadow keywords. Trailing
    // segments may start with a digit (e.g. constant names like
    // `.str.arithmetic.0`).
    dotted_path: _ => token(prec(1,
      /[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_0-9][A-Za-z0-9_]*)+/,
    )),

    identifier: _ => /[A-Za-z_][A-Za-z0-9_]*/,
  },
});

function commaSep1(rule) {
  return seq(rule, repeat(seq(',', rule)));
}
