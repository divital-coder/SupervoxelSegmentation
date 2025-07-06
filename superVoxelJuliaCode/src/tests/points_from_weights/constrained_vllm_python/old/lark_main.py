from lark import Lark, Transformer, v_args
import math
import numpy as np

# Assuming geometric_utils exists and has these functions
# from geometric_utils import lin_p, orthogonal_projection_onto_plane, line_plane_intersection


# --- Mock geometric_utils for standalone testing ---
class DummyGeometricUtils:
    def lin_p(p_a, p_b, weight):
        """Linear interpolation between two points p_a and p_b using a weight."""
        p_a, p_b = np.asarray(p_a), np.asarray(p_b)
        return p_a + (p_b - p_a) * float(weight)

    def orthogonal_projection_onto_plane(point, plane_p1, plane_p2, plane_p3):
        """Projects a point onto the plane defined by three other points."""
        point = np.asarray(point)
        plane_p1, plane_p2, plane_p3 = np.asarray(plane_p1), np.asarray(plane_p2), np.asarray(plane_p3)
        
        v1 = plane_p2 - plane_p1
        v2 = plane_p3 - plane_p1
        normal = np.cross(v1, v2)
        norm_mag = np.linalg.norm(normal)
        if norm_mag == 0: # Points are collinear, plane is undefined
            # print(f"Warning: Plane undefined for projection (collinear points: {plane_p1}, {plane_p2}, {plane_p3}). Returning original point.")
            return point 
        normal = normal / norm_mag
        
        return point - np.dot(point - plane_p1, normal) * normal

    def line_plane_intersection(line_p1, line_p2, plane_p1, plane_p2, plane_p3):
        """Finds the intersection point of a line (defined by two points) and a plane (defined by three points)."""
        line_p1, line_p2 = np.asarray(line_p1), np.asarray(line_p2)
        plane_p1, plane_p2, plane_p3 = np.asarray(plane_p1), np.asarray(plane_p2), np.asarray(plane_p3)

        line_dir = line_p2 - line_p1
        
        v1 = plane_p2 - plane_p1
        v2 = plane_p3 - plane_p1
        plane_normal = np.cross(v1, v2)
        norm_mag = np.linalg.norm(plane_normal)

        if norm_mag == 0: # Plane undefined
            # print(f"Warning: Plane undefined for intersection. Returning None.")
            return None

        plane_normal = plane_normal / norm_mag
        
        dot_product = np.dot(line_dir, plane_normal)
        
        if abs(dot_product) < 1e-9: # Line is parallel to the plane (or in the plane)
            # Check if line_p1 is on the plane
            if abs(np.dot(plane_normal, line_p1 - plane_p1)) < 1e-9:
                # print("Warning: Line is in the plane. Returning line_p1 as an intersection point.")
                return line_p1 # Line is in the plane
            # print("Warning: Line is parallel to plane and not intersecting. Returning None.")
            return None # Line is parallel and not in the plane

        t = np.dot(plane_normal, plane_p1 - line_p1) / dot_product
        intersection_point = line_p1 + t * line_dir
        return intersection_point


# --- FIX: Access functions directly from the class ---
# By referencing the functions from the class `DummyGeometricUtils` instead of an
# instance, we get unbound functions. This prevents Python from implicitly
# passing `self` as the first argument, resolving the TypeError.
lin_p = DummyGeometricUtils.lin_p
orthogonal_projection_onto_plane = DummyGeometricUtils.orthogonal_projection_onto_plane
line_plane_intersection = DummyGeometricUtils.line_plane_intersection


# Apply the v_args decorator at the class level for cleaner transformer methods
@v_args(inline=True)
class CalculateTree(Transformer):
    # Import basic math operators
    from operator import add, sub, mul, truediv as div, neg
    # Define the type for numbers
    number = float

    def __init__(self):
        # Dictionary to store variables
        self.vars = {}

    # --- Transformer Methods ---
    def assign_var(self, name_token, value):
        """Assigns a value to a variable."""
        self.vars[name_token.value] = value
        return value

    def var(self, name_token):
        """Retrieves a variable's value."""
        var_name_str = name_token.value
        try:
            return self.vars[var_name_str]
        except KeyError:
            raise NameError(f"Variable not found: {var_name_str}")

    def pow(self, left, right):
        """Handles the power operator."""
        return left ** right

    # --- Terminal Handling for Function Names ---
    # These methods simply return the string value of the matched token.
    def SIN_T(self, token): return token.value
    def COS_T(self, token): return token.value
    def SQRT_T(self, token): return token.value
    def LIN_P_T(self, token): return token.value
    def ORTHO_PROJ_T(self, token): return token.value
    def LINE_PLANE_INTERSECT_T(self, token): return token.value

    # --- NEW: Method to handle parenthesized expressions ---
    def paren_group(self, *args):
        """
        Handles expressions inside parentheses.
        Differentiates between a grouped expression like (5+3)
        and a point literal like (1, 2, 3).
        'args' is a list of evaluated expressions from the 'arguments' rule.
        """
        # Note: Because of how Lark handles single vs multiple children,
        # if there's one argument, 'args' will contain that argument directly.
        # If there are multiple arguments, 'args' will be a tuple of those arguments.
        # We normalize this by checking the type.
        items = args[0] if isinstance(args[0], list) else list(args)

        if len(items) == 1:
            return items[0]  # Standard grouped expression
        elif len(items) == 3:
            return self.point_literal(*items)  # Point literal
        else:
            raise SyntaxError(
                f"Parenthesized expression must contain 1 (for grouping) or 3 (for a point) elements, but found {len(items)}."
            )

    # --- Data Structure Transformers ---
    def point_literal(self, x, y, z):
        """Creates a point tuple from three expressions."""
        # This method is no longer called directly by the parser, but by paren_group
        if not all(isinstance(coord, (int, float)) for coord in [x, y, z]):
            raise TypeError(f"Point coordinates must be numbers. Got: ({type(x)}, {type(y)}, {type(z)})")
        return (float(x), float(y), float(z))

    def arguments(self, *items):
        """
        Collects all comma-separated expressions into a list.
        This is used for function calls and now for parenthesized expressions.
        """
        return list(items)

    # --- Function Call Dispatcher ---
    def func_call(self, func_name_str, args):
        """
        Calls the appropriate Python function based on the parsed function name.
        """
        # If there are no arguments (e.g., my_func()), 'args' will be None.
        actual_args = args if args is not None else []

        # A dispatch table mapping function names to their handlers and expected arg counts
        func_dispatch = {
            "sin": (math.sin, 1),
            "cos": (math.cos, 1),
            "sqrt": (math.sqrt, 1),
            "lin_p": (lin_p, 3),
            "ortho_proj": (orthogonal_projection_onto_plane, 4),
            "line_plane_intersect": (line_plane_intersection, 5),
        }

        if func_name_str in func_dispatch:
            func, expected_arg_count = func_dispatch[func_name_str]
            
            if len(actual_args) != expected_arg_count:
                raise TypeError(f"{func_name_str}() takes exactly {expected_arg_count} arguments, but got {len(actual_args)}")

            if func_name_str in ["sin", "cos", "sqrt"]:
                if not isinstance(actual_args[0], (int, float)):
                    raise TypeError(f"{func_name_str}() requires a numeric argument.")
                if func_name_str == "sqrt" and actual_args[0] < 0:
                    raise ValueError("math domain error: sqrt of a negative number")

            if func_name_str in ["lin_p", "ortho_proj", "line_plane_intersect"]:
                for i, arg in enumerate(actual_args):
                    is_last_arg_of_linp = (func_name_str == "lin_p" and i == 2)
                    if is_last_arg_of_linp:
                        if not isinstance(arg, (int, float)):
                             raise TypeError(f"{func_name_str}() argument {i+1} (weight) must be a number.")
                    elif not (isinstance(arg, tuple) and len(arg) == 3):
                        raise TypeError(f"{func_name_str}() argument {i+1} must be a point (tuple of 3 numbers).")
            
            return func(*actual_args)
        else:
            raise NameError(f"Unknown function: {func_name_str}")

# --- Updated Lark Grammar ---
# The ambiguity between `(expr)` and `(expr, expr, expr)` has been resolved
# by using a single rule for parentheses and handling the logic in the transformer.
calc_grammar = r"""
    ?start: expr
          | NAME "=" expr    -> assign_var

    ?expr: sum

    ?sum: product
        | sum "+" product   -> add
        | sum "-" product   -> sub

    ?product: power
            | product "*" power  -> mul
            | product "/" power  -> div

    ?power: atom_expr "^" power -> pow
          | atom_expr

    ?atom_expr: func_call
              | NUMBER           -> number
              | "-" atom_expr    -> neg
              | NAME             -> var
              | "(" arguments ")"  -> paren_group // RESOLVED AMBIGUITY

    // REMOVED: The old point_literal rule is gone. Its logic is now in the transformer.

    // Function calls with optional arguments
    func_call: fun_name_rule "(" [arguments] ")"

    // Argument list: one or more comma-separated expressions.
    // This now serves both function calls and parenthesized groups.
    arguments: expr ("," expr)*

    // Rule to group all function name terminals
    ?fun_name_rule: SIN_T | COS_T | SQRT_T | LIN_P_T | ORTHO_PROJ_T | LINE_PLANE_INTERSECT_T

    // Terminals for built-in function names ('.1' gives them higher priority than NAME)
    SIN_T.1: "sin"
    COS_T.1: "cos"
    SQRT_T.1: "sqrt"
    LIN_P_T.1: "lin_p"
    ORTHO_PROJ_T.1: "ortho_proj"
    LINE_PLANE_INTERSECT_T.1: "line_plane_intersect"

    // Standard imports from the common library
    %import common.CNAME -> NAME
    %import common.NUMBER
    %import common.WS_INLINE
    %ignore WS_INLINE
"""

# Instantiate the transformer
transformer = CalculateTree()

# Initialize parser with the grammar and the transformer instance
calc_parser = Lark(calc_grammar, parser='lalr', transformer=transformer, debug=False)
calc = calc_parser.parse

def main():
    """A REPL (Read-Eval-Print Loop) for the calculator."""
    print("Calculator REPL with custom functions. Type 'exit' or press Ctrl+D to quit.")
    # The 'calc' function uses a single, persistent transformer instance, so state is maintained.
    while True:
        try:
            s = input('> ')
            if s.lower() == 'exit':
                break
        except (EOFError, KeyboardInterrupt):
            break
        if not s.strip():
            continue
        try:
            result = calc(s)
            if result is not None:
                if isinstance(result, tuple):
                    formatted_result = tuple(round(n, 4) for n in result)
                    print(formatted_result)
                else:
                    print(round(result, 4))
        except Exception as e:
            print(f"Error: {e}")

def test():
    """Runs a suite of tests to validate the parser and calculator."""
    print("--- Running tests ---")
    
    # Reset variables for each test run to ensure isolation
    global transformer
    transformer = CalculateTree()
    global calc
    calc = Lark(calc_grammar, parser='lalr', transformer=transformer, debug=False).parse

    test_cases = [
        # (input, expected_output)
        ("a = 5 + 3", 8.0),
        ("b = a * -3 + 1", -23.0),
        ("sin(b) + cos(a)", round(math.sin(-23) + math.cos(8), 4)),
        ("p1 = (1, 2, 3)", (1.0, 2.0, 3.0)),
        ("p2 = (a, b, 0)", (8.0, -23.0, 0.0)),
        ("lin_p((1,1,1), (3,3,3), 0.5)", (2.0, 2.0, 2.0)),
        ("pt_a = (0,0,0)", (0.0, 0.0, 0.0)),
        ("pt_b = (10,20,30)", (10.0, 20.0, 30.0)),
        ("lin_p(pt_a, pt_b, 0.25)", (2.5, 5.0, 7.5)),
        ("ortho_proj((1,1,10), (0,0,0), (1,0,0), (0,1,0))", (1.0, 1.0, 0.0)),
        ("line_plane_intersect((0,0,-1), (0,0,1), (-1,-1,0), (1,-1,0), (0,1,0))", (0.0, 0.0, 0.0)),
        ("2^3", 8.0),
        ("a^2", 64.0),
        ("sin()", "ERROR"),
        ("lin_p(1, 2, 3)", "ERROR"),
        ("c = d + 1", "ERROR"),
        ("sqrt(-4)", "ERROR"),
    ]
    
    all_passed = True
    for i, (t, expected) in enumerate(test_cases):
        # Reset variables before each individual test case
        transformer.vars = {}
        # Pre-load variables if needed by the test case
        if "a =" not in t: transformer.vars['a'] = 8.0
        if "b =" not in t: transformer.vars['b'] = -23.0
        if "pt_a =" not in t: transformer.vars['pt_a'] = (0.0, 0.0, 0.0)
        if "pt_b =" not in t: transformer.vars['pt_b'] = (10.0, 20.0, 30.0)

        print(f"\n[Test {i+1}] Input: '{t}'")
        try:
            result = calc(t)
            
            from lark import Tree
            if isinstance(result, Tree):
                 raise TypeError(f"Unexpected parse tree returned (missing transformer rule?): {result}")

            # --- FIX: Convert numpy arrays to tuples before rounding ---
            def to_tuple(val):
                if hasattr(val, 'tolist'):
                    return tuple(val.tolist())
                return val

            formatted_result = result
            if isinstance(result, tuple) or (hasattr(result, 'tolist') and isinstance(result.tolist(), list)):
                # Convert numpy arrays or tuples of numpy floats to tuples of floats
                tuple_result = to_tuple(result)
                formatted_result = tuple(round(float(n), 4) for n in tuple_result)
            elif result is not None:
                if hasattr(result, 'tolist'):
                    # Single numpy scalar
                    formatted_result = round(float(result.tolist()), 4)
                else:
                    formatted_result = round(result, 4)

            if expected == "ERROR":
                print(f"FAILED: Expected error, but got result: {formatted_result}")
                all_passed = False
            else:
                # Use np.allclose for float comparisons to handle precision issues
                is_close = False
                if isinstance(expected, tuple) and isinstance(formatted_result, tuple):
                    import numpy as np
                    is_close = np.allclose(np.array(formatted_result), np.array(expected), atol=1e-4)
                elif isinstance(expected, (float, int)) and isinstance(formatted_result, (float, int)):
                    import numpy as np
                    is_close = np.isclose(formatted_result, expected, atol=1e-4)

                assert is_close, f"Result {formatted_result} != Expected {expected}"
                print(f"Result: {formatted_result} -> PASSED")

        except Exception as e:
            if expected == "ERROR":
                print(f"Caught Expected Error: {type(e).__name__}: {e} -> PASSED")
            else:
                print(f"FAILED: Unexpected Error: {type(e).__name__}: {e}")
                all_passed = False
            
    print("\n--- Tests finished ---")
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")


if __name__ == '__main__':
    test()
    # main()