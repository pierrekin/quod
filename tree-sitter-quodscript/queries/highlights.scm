; Comments
(line_comment) @comment

; Keywords (statement openers + reserved words)
[
  "fn"
  "let"
  "if"
  "else"
  "while"
  "for"
  "in"
  "to"
  "return"
  "match"
  "with_arena"
  "capacity"
  "store"
  "load"
  "widen"
  "uwiden"
  "ptr_offset"
  "sizeof"
] @keyword

; Types
(primitive_type) @type.builtin
(named_type) @type
(pointer_type) @type

; Literals
(number_literal) @number
(char_literal) @string
(bool_literal) @constant.builtin
(null_literal) @constant.builtin
(string_ref) @string

; Operators / punctuation
[
  "+" "-" "*" "/" "%" "/u"
  "==" "!=" "<" "<=" ">" ">=" "<u" "<=u" ">u" ">=u"
  "&&" "||"
  "->" ".." "::"
  "=" "?"
] @operator

[ "{" "}" "(" ")" "[" "]" ] @punctuation.bracket
[ "," ";" ":" ] @punctuation.delimiter

; Function declarations and calls
(function name: (identifier) @function)
(function name: (dotted_path) @function)
(call callee: (identifier) @function)
(call callee: (dotted_path) @function)

; Builtin call names ("load", "widen", etc) styled as keywords above

; Parameters
(parameter name: (identifier) @variable.parameter)

; Catch-all for plain identifiers in expressions
(identifier) @variable
