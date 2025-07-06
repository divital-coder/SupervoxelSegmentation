GEOMETRIC_ALGORITHM_GRAMMAR="""
?start: statement_list
?statement_list: (statement)*
?statement: assign_var ";"
          | triangle_def ";"
          | expr ";" -> expr_statement
?assign_var: NAME "=" expr
?triangle_def: "defineTriangle" "(" arguments ")" -> triangle_definition
?expr: sum
?sum: product
    | sum "+" product -> add
    | sum "-" product -> sub
?product: power
        | product "*" power -> mul
        | product "/" power -> div
?power: atom_expr "^" power -> pow
      | atom_expr
?atom_expr: func_call
          | atom
          | "-" atom_expr -> neg
          | "(" expr ")"
?atom: NUMBER -> number
     | indexed_var
     | NAME -> var
     | point
point: "(" expr "," expr "," expr ")"
?indexed_var: NAME "_" "(" arguments ")"
func_call: FUNC_NAME "(" [arguments] ")"
FUNC_NAME.2: "sin" | "cos" | "sqrt" | "dot" | "div" | "vector" | "lin_p" | "ortho_proj" | "line_plane_intersect" | "range"
arguments: expr ("," expr)*
NAME: /(?!(sin|cos|sqrt|dot|div|vector|lin_p|ortho_proj|line_plane_intersect|range)$)[a-zA-Z][a-zA-Z0-9]*/
%import common.NUMBER
%import common.WS
%ignore WS
%ignore /#.*/
"""
