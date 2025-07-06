import math
import numpy as np
from lark import Lark, Transformer, v_args, LarkError
import copy
import random
from multiprocessing import Pool, cpu_count
from functools import partial

# A sentinel value to indicate that a control point has not been calculated yet.
UNINITIALIZED_POINT = np.array([-100000.0, -100000.0, -100000.0])

# --- Mock Geometric Utility Functions ---
def lin_p(p1, p2, w):
    """Linear interpolation between two points."""
    if p1 is None or p2 is None:
        return None
    p1, p2, w = np.asarray(p1), np.asarray(p2), float(w)
    return p1 * (1 - w) + p2 * w

def orthogonal_projection_onto_plane(point_to_project, p1, p2, p3):
    """Projects a point onto a plane defined by three other points."""
    if point_to_project is None or p1 is None or p2 is None or p3 is None:
        return None
    point_to_project = np.asarray(point_to_project)
    p1, p2, p3 = np.asarray(p1), np.asarray(p2), np.asarray(p3)
    
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm_mag = np.linalg.norm(normal)
    if norm_mag < 1e-9:
        return point_to_project 
    normal = normal / norm_mag
    
    dist = np.dot(point_to_project - p1, normal)
    return point_to_project - dist * normal

def line_plane_intersection(line_p1, line_p2, plane_p1, plane_p2, plane_p3):
    """Calculates the intersection of a line and a plane."""
    if line_p1 is None or line_p2 is None or plane_p1 is None or plane_p2 is None or plane_p3 is None:
        return None
    line_p1, line_p2 = np.asarray(line_p1), np.asarray(line_p2)
    plane_p1, plane_p2, plane_p3 = np.asarray(plane_p1), np.asarray(plane_p2), np.asarray(plane_p3)

    v1 = plane_p2 - plane_p1
    v2 = plane_p3 - plane_p1
    plane_normal = np.cross(v1, v2)
    
    line_dir = line_p2 - line_p1
    
    dot_product = np.dot(line_dir, plane_normal)
    if abs(dot_product) < 1e-6:
        return (line_p1 + line_p2) / 2

    w = plane_p1 - line_p1
    t = np.dot(w, plane_normal) / dot_product
    intersection_point = line_p1 + t * line_dir
    return intersection_point

# --- LARK GRAMMAR (CORRECTED) ---
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
FUNC_NAME.2: "sin" | "cos" | "sqrt" | "dot" | "div" | "vector" | "lin_p" | "ortho_proj" | "line_plane_intersect"
arguments: expr ("," expr)*
NAME: /[a-zA-Z][a-zA-Z0-9]*/
%import common.NUMBER
%import common.WS
%ignore WS
%ignore /#.*/
"""

@v_args(inline=True)
class GeometricAlgorithmTransformer(Transformer):
    number = float

    def __init__(self, context, weights):
        self.context = context
        self.new_control_points = {}
        self.vars = copy.deepcopy(self.context['read_grid_data'][self.context['current_indices']]['control_points'])
        self.vars.update(weights)
        self.vars['rx'], self.vars['ry'], self.vars['rz'] = self.context['grid_spacing']
        self.triangles = []

    # --- Robust Arithmetic Operations ---
    def add(self, a, b):
        if a is None or b is None: return None
        return a + b

    def sub(self, a, b):
        if a is None or b is None: return None
        return a - b

    def mul(self, a, b):
        if a is None or b is None: return None
        return a * b

    def pow(self, a, b):
        if a is None or b is None: return None
        return a ** b
        
    def neg(self, a):
        if a is None: return None
        return -a

    def div(self, a, b): return self._safe_divide(a, b)

    def _safe_divide(self, a, b):
        if a is None or b is None: return None
        a_arr, b_arr = np.asarray(a), np.asarray(b)
        if hasattr(b_arr, 'ndim') and b_arr.ndim > 0:
            safe_b = np.where(b_arr == 0, 1e-9, b_arr)
        else:
            safe_b = b_arr if b_arr != 0 else 1e-9
        return a_arr / safe_b
        
    def statement_list(self, *statements):
        if self.context.get('step_index') == 4:
            return {"triangles": self.triangles}
        return None

    def expr_statement(self, expr): return None

    def assign_var(self, name_token, value):
        var_name = name_token.value
        self.vars[var_name] = value
        if 'live_grid_data' in self.context and value is not None:
             self.context['live_grid_data'][self.context['current_indices']]['control_points'][var_name] = value
        return None

    def var(self, name_token):
        var_name = name_token.value
        if var_name == "originalSvCenter":
            return self._get_sv_center_at(self.context['current_indices'])
        try:
            return self.vars[var_name]
        except KeyError:
            raise NameError(f"Variable '{var_name}' not found.")

    def _get_sv_center_at(self, indices):
        spacing = self.context['grid_spacing']
        return np.array([indices[0] * spacing[0], indices[1] * spacing[1], indices[2] * spacing[2]])

    def indexed_var(self, name_token, args):
        var_name = name_token.value
        args = [int(arg) for arg in args]
        if len(args) != 3:
            raise ValueError(f"Indexed access requires 3 indices, but got {len(args)}")
        
        current_idx = np.array(self.context['current_indices'])
        target_indices = tuple(current_idx + np.array(args))

        if var_name == "originalSvCenter":
            return self._get_sv_center_at(target_indices)

        is_local_access = all(offset == 0 for offset in args)
        if is_local_access:
            if var_name in self.vars:
                return self.vars[var_name]
            else:
                raise NameError(f"Variable '{var_name}' not found in the current supervoxel context.")

        if target_indices not in self.context['read_grid_data']:
            return None 

        try:
            value = self.context['read_grid_data'][target_indices]['control_points'][var_name]
            # FIX: Check for None *before* np.allclose to avoid the TypeError.
            if value is None:
                return None
            if np.allclose(value, UNINITIALIZED_POINT): 
                return None
            return value
        except KeyError:
            return None

    def point(self, x, y, z): return np.array([float(x), float(y), float(z)])
    def arguments(self, *items): return list(items)

    def triangle_definition(self, args):
        if any(p is None for p in args): return None 

        if len(args) != 3:
            raise ValueError(f"defineTriangle requires 3 points, but got {len(args)}")
        for i, p in enumerate(args):
            if not (isinstance(p, np.ndarray) and p.shape == (3,)):
                raise TypeError(f"Argument {i+1} for defineTriangle must be a 3D point. Got: {type(p)}")
        self.triangles.append(args)
        return None

    def func_call(self, func_name_token, args):
        func_name = func_name_token.value
        actual_args = args if args is not None else []
        
        func_dispatch = {
            "sin": (math.sin, 1), "cos": (math.cos, 1), "sqrt": (math.sqrt, 1),
            "dot": (lambda a, b: np.dot(a, b), 2),
            "div": (self._safe_divide, 2),
            "vector": (lambda x, y, z: np.array([x, y, z]), 3),
            "lin_p": (lin_p, 3), 
            "ortho_proj": (orthogonal_projection_onto_plane, 4),
            "line_plane_intersect": (line_plane_intersection, 5),
        }

        if func_name in func_dispatch:
            func, expected_arg_count = func_dispatch[func_name]
            if any(arg is None for arg in actual_args): return None
            if len(actual_args) != expected_arg_count:
                raise TypeError(f"{func_name}() expects {expected_arg_count} arguments, but got {len(actual_args)}")
            return func(*actual_args)
        else:
            raise NameError(f"Unknown function: '{func_name}'")
        
def process_algorithm_script(script, context, weights):
    """Processes a single step of the algorithm script for a given supervoxel."""
    transformer = GeometricAlgorithmTransformer(context, weights)
    parser = Lark(GEOMETRIC_ALGORITHM_GRAMMAR, parser='lalr', transformer=transformer)
    try:
        return parser.parse(script)
    except LarkError as e:
        print(f"Error parsing algorithm script at indices {context['current_indices']}: {e}")
        raise

# --- Worker Functions for Parallel Processing ---

def initialize_cell(indices, persistent_point_names):
    """Initializes a single cell's data structure. Designed to be called by a multiprocessing Pool."""
    x, y, z = indices
    cp = {}
    spacing = (10.0, 10.0, 10.0)
    cp['originalSvCenter'] = np.array([x * spacing[0], y * spacing[1], z * spacing[2]])
    for name in persistent_point_names:
        cp[name] = UNINITIALIZED_POINT
    return (indices, {'control_points': cp, 'triangles': []})

def process_step_for_cell(indices, step_script, weights, read_grid_data, step_index):
    """
    Processes a single algorithm step for a single cell. This function is designed to be pure
    and safe for parallel execution. It does not modify any shared state.
    """
    context = {
        'current_indices': indices,
        'grid_spacing': (10.0, 10.0, 10.0),
        'step_index': step_index,
        'read_grid_data': read_grid_data,
    }
    
    try:
        transformer = GeometricAlgorithmTransformer(context, weights)
        parser = Lark(GEOMETRIC_ALGORITHM_GRAMMAR, parser='lalr', transformer=transformer)
        parser.parse(step_script)
        
        original_cp_keys = read_grid_data[indices]['control_points'].keys()
        final_control_points = {key: transformer.vars.get(key) for key in original_cp_keys}
        
        final_triangles = transformer.triangles
        
        return (indices, final_control_points, final_triangles)
        
    except Exception as e:
        print(f"Error processing cell {indices}: {e}")
        return (indices, None, None)

def execute_grid_algorithm(grid_dimensions, algorithm_steps, weights):
    """Initializes a grid and executes a multi-step algorithm over it using parallel processing."""
    
    persistent_point_names = [
        "svCenter", "linX", "linY", "linZ", "edgeXY", "edgeYZ", "edgeXZ", 
        "mob", "int1", "int2", "int3", "int4", "int5", "int6"
    ]

    print("--- Initializing Grid (Parallel) ---")
    all_indices = [(x, y, z) for x in range(grid_dimensions[0]) for y in range(grid_dimensions[1]) for z in range(grid_dimensions[2])]
    
    init_partial = partial(initialize_cell, persistent_point_names=persistent_point_names)

    with Pool(processes=cpu_count()) as pool:
        initialized_cells = pool.map(init_partial, all_indices)
    
    grid_data = dict(initialized_cells)
    
    for i, step_script in enumerate(algorithm_steps):
        print(f"\n--- Executing Step {i+1} (Parallel) ---")
        read_grid_data = copy.deepcopy(grid_data)

        process_partial = partial(process_step_for_cell, 
                                  step_script=step_script, 
                                  weights=weights, 
                                  read_grid_data=read_grid_data, 
                                  step_index=i)

        with Pool(processes=cpu_count()) as pool:
            step_results = pool.map(process_partial, all_indices)

        print(f"--- Synchronizing Results for Step {i+1} ---")
        for indices, final_control_points, final_triangles in step_results:
            if final_control_points is not None:
                grid_data[indices]['control_points'].update(final_control_points)
            
            if final_triangles:
                grid_data[indices]['triangles'].extend(final_triangles)
    
    return grid_data



def test_full_algorithm_a():
    """A comprehensive test case using the full algorithm specification."""
    print("--- Running Full Algorithm Test on a Grid ---")
    
    # <S1> Supervoxel Center Modification
    step1 = """
        svCenter = originalSvCenter_(0,0,0) + vector(w1 * rx * 0.5 - rx * 0.25, w2 * ry * 0.5 - ry * 0.25, w3 * rz * 0.5 - rz * 0.25);
    """
    
    # <S2> Linear and Edge Control Points for the "Negative Corner"
    step2 = """
        linX = lin_p(svCenter_(0,0,0), svCenter_(-1,0,0), w4);
        linY = lin_p(svCenter_(0,0,0), svCenter_(0,-1,0), w5);
        linZ = lin_p(svCenter_(0,0,0), svCenter_(0,0,-1), w6);

        # edgeXY is a blend of 4 supervoxel centers in the XY plane
        temp1 = lin_p(svCenter_(0,0,0), svCenter_(-1,0,0), w7);
        temp2 = lin_p(svCenter_(0,-1,0), svCenter_(-1,-1,0), w7);
        edgeXY = lin_p(temp1, temp2, w8);

        # edgeYZ is a blend of 4 supervoxel centers in the YZ plane
        temp3 = lin_p(svCenter_(0,0,0), svCenter_(0,0,-1), w9);
        temp4 = lin_p(svCenter_(0,-1,0), svCenter_(0,-1,-1), w9);
        edgeYZ = lin_p(temp3, temp4, w10);

        # edgeXZ is a blend of 4 supervoxel centers in the XZ plane
        temp5 = lin_p(svCenter_(0,0,0), svCenter_(-1,0,0), w11);
        temp6 = lin_p(svCenter_(0,0,-1), svCenter_(-1,0,-1), w11);
        edgeXZ = lin_p(temp5, temp6, w12);
    """

    # <S3> Main Oblique Point (mob) Definition
    step3 = """
        # mob is a complex blend of 8 supervoxel centers, forming the deep corner point.
        tempX00 = lin_p(svCenter_(0,0,0), svCenter_(-1,0,0), w13);
        tempX10 = lin_p(svCenter_(0,-1,0), svCenter_(-1,-1,0), w13);
        tempX01 = lin_p(svCenter_(0,0,-1), svCenter_(-1,0,-1), w13);
        tempX11 = lin_p(svCenter_(0,-1,-1), svCenter_(-1,-1,-1), w13);
        tempY0 = lin_p(tempX00, tempX10, w14);
        tempY1 = lin_p(tempX01, tempX11, w14);
        mob = lin_p(tempY0, tempY1, w15);
    """

    # <S4> Intermediate Points Definition
    step4 = """
        # Create 6 intermediate points using a variety of geometric operations for complexity.
        # This tests the corrected grammar's ability to handle complex expressions.
        int1 = ortho_proj(mob_(0,0,0), linX_(0,0,0), linY_(0,0,0), linZ_(0,0,0));
        
        # FIX: Renamed 'scalar_dot' to 'scalarDot' to conform to camelCase naming rule.
        scalarDot = dot(svCenter_(0,0,0) - mob_(0,0,0), linX_(0,0,0) - mob_(0,0,0));
        offsetX = cos(scalarDot) * w16;
        offsetY = sin(scalarDot) * w17;
        offsetZ = (sqrt(w18) ^ 2.0) * rz * 0.1;
        int2 = mob_(0,0,0) + vector(offsetX, offsetY, offsetZ);

        int3 = line_plane_intersect(svCenter_(0,0,0), mob_(0,0,0), linX_(0,0,0), edgeXY_(0,0,0), linY_(0,0,0));

        tempVec = (svCenter_(0,0,0) - mob_(0,0,0)) * w19;
        int4 = mob_(0,0,0) + div(tempVec, 2.0);

        int5 = lin_p(mob_(0,0,0), linY_(0,0,0), w20);
        int6 = lin_p(mob_(0,0,0), linZ_(0,0,0), w21);
    """

    # <S5> Triangle Definitions for a Watertight Polyhedron
    step5 = """
        # Assemble the 6 faces of the supervoxel using the "Negative Corner" pattern.
        # Each face is composed of 4 triangles, creating a sealed surface.
        
        # 1. Negative-X Face (using points from the current SV's negative corner)
        defineTriangle(svCenter_(0,0,0), mob_(0,0,0), linY_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linY_(0,0,0), edgeYZ_(0,-1,0));
        defineTriangle(svCenter_(0,0,0), edgeYZ_(0,-1,0), linZ_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linZ_(0,0,0), mob_(0,0,0));

        # 2. Negative-Y Face
        defineTriangle(svCenter_(0,0,0), mob_(0,0,0), linZ_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linZ_(0,0,0), edgeXZ_(0,0,-1));
        defineTriangle(svCenter_(0,0,0), edgeXZ_(0,0,-1), linX_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linX_(0,0,0), mob_(0,0,0));

        # 3. Negative-Z Face
        defineTriangle(svCenter_(0,0,0), mob_(0,0,0), linX_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linX_(0,0,0), edgeXY_(-1,0,0));
        defineTriangle(svCenter_(0,0,0), edgeXY_(-1,0,0), linY_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linY_(0,0,0), mob_(0,0,0));

        # 4. Positive-X Face (using points from neighbor at (1,0,0))
        defineTriangle(svCenter_(0,0,0), mob_(1,0,0), linY_(1,0,0));
        defineTriangle(svCenter_(0,0,0), linY_(1,0,0), edgeYZ_(1,-1,0));
        defineTriangle(svCenter_(0,0,0), edgeYZ_(1,-1,0), linZ_(1,0,0));
        defineTriangle(svCenter_(0,0,0), linZ_(1,0,0), mob_(1,0,0));

        # 5. Positive-Y Face (using points from neighbor at (0,1,0))
        defineTriangle(svCenter_(0,0,0), mob_(0,1,0), linZ_(0,1,0));
        defineTriangle(svCenter_(0,0,0), linZ_(0,1,0), edgeXZ_(0,1,-1));
        defineTriangle(svCenter_(0,0,0), edgeXZ_(0,1,-1), linX_(0,1,0));
        defineTriangle(svCenter_(0,0,0), linX_(0,1,0), mob_(0,1,0));

        # 6. Positive-Z Face (using points from neighbor at (0,0,1))
        defineTriangle(svCenter_(0,0,0), mob_(0,0,1), linX_(0,0,1));
        defineTriangle(svCenter_(0,0,0), linX_(0,0,1), edgeXY_(-1,0,1));
        defineTriangle(svCenter_(0,0,0), edgeXY_(-1,0,1), linY_(0,0,1));
        defineTriangle(svCenter_(0,0,0), linY_(0,0,1), mob_(0,0,1));
    """

    algorithm_steps = [step1, step2, step3, step4, step5]
    weights = {f'w{i}': random.random() for i in range(1, 22)} 
    grid_dimensions = (7, 7, 7)

    final_grid_data = execute_grid_algorithm(grid_dimensions, algorithm_steps, weights)
    
    # --- Validation ---
    print("\n--- Validation for Supervoxel (1,1,1) ---")
    # (1,1,1) is an interior cell, so it should be fully defined and have all its triangles.
    sv_data = final_grid_data[(1,1,1)]
    cp = sv_data['control_points']

    try:
        assert not np.allclose(cp['svCenter'], UNINITIALIZED_POINT), "svCenter is uninitialized"
        assert not np.allclose(cp['mob'], UNINITIALIZED_POINT), "mob is uninitialized"
        assert 'int6' in cp, "int6 was not created"
        
        # An interior cell should have 6 faces * 4 triangles/face = 24 triangles
        expected_triangles = 24
        num_triangles = len(sv_data['triangles'])

    except (AssertionError, KeyError, NameError) as e:
        print(f"\n❌ Test Failed: {e}")

def test_full_algorithm():
    """
    A simplified test case to verify the parser and execution flow with predictable values.
    All weights are set to 0.5, and algorithm steps use simple linear interpolations.
    The test validates the computed control points for an interior supervoxel (2,2,2)
    against their manually calculated expected values.
    """
    print("--- Running Simplified Parser Verification Test ---")
    
    # <S1> Simplified: svCenter is just the original center.
    step1 = "svCenter = originalSvCenter_(0,0,0);"
    
    # <S2> Corrected Step 2: Edge points are now correctly derived from svCenter (from Step 1)
    # to avoid same-step dependency violations.
    step2 = """
        linX = lin_p(svCenter_(0,0,0), svCenter_(-1,0,0), w1);
        linY = lin_p(svCenter_(0,0,0), svCenter_(0,-1,0), w2);
        linZ = lin_p(svCenter_(0,0,0), svCenter_(0,0,-1), w3);
        
        # Correctly define edge points using temporary variables based on svCenter (from Step 1)
        tempA = lin_p(svCenter_(0,0,0), svCenter_(-1,0,0), w4);
        tempB = lin_p(svCenter_(0,-1,0), svCenter_(-1,-1,0), w4);
        edgeXY = lin_p(tempA, tempB, w5);

        tempC = lin_p(svCenter_(0,0,0), svCenter_(0,0,-1), w6);
        tempD = lin_p(svCenter_(0,-1,0), svCenter_(0,-1,-1), w6);
        edgeYZ = lin_p(tempC, tempD, w7);

        tempE = lin_p(svCenter_(0,0,0), svCenter_(-1,0,0), w8);
        tempF = lin_p(svCenter_(0,0,-1), svCenter_(-1,0,-1), w8);
        edgeXZ = lin_p(tempE, tempF, w9);
    """

    # <S3> Simplified: mob is halfway between svCenter and its corner neighbor.
    # Weight index is updated to follow the new weights used in Step 2.
    step3 = "mob = lin_p(svCenter_(0,0,0), svCenter_(-1,-1,-1), w10);"

    # <S4> Simplified: Intermediate points with updated weight indices.
    step4 = """
        int1 = lin_p(mob_(0,0,0), linX_(0,0,0), w11);
        int2 = lin_p(mob_(0,0,0), linY_(0,0,0), w12);
        int3 = lin_p(mob_(0,0,0), linZ_(0,0,0), w13);
        int4 = lin_p(mob_(0,0,0), edgeXY_(0,0,0), w14);
        int5 = lin_p(mob_(0,0,0), edgeYZ_(0,0,0), w15);
        int6 = lin_p(mob_(0,0,0), edgeXZ_(0,0,0), w16);
    """

    # <S5> Triangle definitions remain structurally the same to test the "Negative Corner" assembly.
    step5 = """
        # Assemble the 6 faces of the supervoxel.
        defineTriangle(svCenter_(0,0,0), mob_(0,0,0), linY_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linY_(0,0,0), edgeYZ_(0,-1,0));
        defineTriangle(svCenter_(0,0,0), edgeYZ_(0,-1,0), linZ_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linZ_(0,0,0), mob_(0,0,0));

        defineTriangle(svCenter_(0,0,0), mob_(0,0,0), linZ_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linZ_(0,0,0), edgeXZ_(0,0,-1));
        defineTriangle(svCenter_(0,0,0), edgeXZ_(0,0,-1), linX_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linX_(0,0,0), mob_(0,0,0));

        defineTriangle(svCenter_(0,0,0), mob_(0,0,0), linX_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linX_(0,0,0), edgeXY_(-1,0,0));
        defineTriangle(svCenter_(0,0,0), edgeXY_(-1,0,0), linY_(0,0,0));
        defineTriangle(svCenter_(0,0,0), linY_(0,0,0), mob_(0,0,0));

        defineTriangle(svCenter_(0,0,0), mob_(1,0,0), linY_(1,0,0));
        defineTriangle(svCenter_(0,0,0), linY_(1,0,0), edgeYZ_(1,-1,0));
        defineTriangle(svCenter_(0,0,0), edgeYZ_(1,-1,0), linZ_(1,0,0));
        defineTriangle(svCenter_(0,0,0), linZ_(1,0,0), mob_(1,0,0));

        defineTriangle(svCenter_(0,0,0), mob_(0,1,0), linZ_(0,1,0));
        defineTriangle(svCenter_(0,0,0), linZ_(0,1,0), edgeXZ_(0,1,-1));
        defineTriangle(svCenter_(0,0,0), edgeXZ_(0,1,-1), linX_(0,1,0));
        defineTriangle(svCenter_(0,0,0), linX_(0,1,0), mob_(0,1,0));

        defineTriangle(svCenter_(0,0,0), mob_(0,0,1), linX_(0,0,1));
        defineTriangle(svCenter_(0,0,0), linX_(0,0,1), edgeXY_(-1,0,1));
        defineTriangle(svCenter_(0,0,0), edgeXY_(-1,0,1), linY_(0,0,1));
        defineTriangle(svCenter_(0,0,0), linY_(0,0,1), mob_(0,0,1));
    """

    algorithm_steps = [step1, step2, step3, step4, step5]
    # Set all weights to 0.5 for predictable calculations.
    weights = {f'w{i}': 0.5 for i in range(1, 22)} 
    grid_dimensions = (7, 7, 7) 

    final_grid_data = execute_grid_algorithm(grid_dimensions, algorithm_steps, weights)
    
    # --- Validation for the central supervoxel (2,2,2) ---
    print("\n--- Validation for Supervoxel (2,2,2) ---")
    sv_data = final_grid_data[(2,2,2)]
    cp = sv_data['control_points']

    # --- Manually calculated expected values for (2,2,2) with weights=0.5 and corrected algorithm ---
    expected_values = {
        'svCenter': np.array([20.0, 20.0, 20.0]),
        'linX':     np.array([15.0, 20.0, 20.0]),
        'linY':     np.array([20.0, 15.0, 20.0]),
        'linZ':     np.array([20.0, 20.0, 15.0]),
        'edgeXY':   np.array([15.0, 15.0, 20.0]),
        'edgeYZ':   np.array([20.0, 15.0, 15.0]),
        'edgeXZ':   np.array([15.0, 20.0, 15.0]),
        'mob':      np.array([15.0, 15.0, 15.0]),
        'int1':     np.array([15.0, 17.5, 17.5]),
        'int2':     np.array([17.5, 15.0, 17.5]),
        'int3':     np.array([17.5, 17.5, 15.0]),
        'int4':     np.array([15.0, 15.0, 17.5]),
        'int5':     np.array([17.5, 15.0, 15.0]),
        'int6':     np.array([15.0, 17.5, 15.0])
    }

    try:
        print("Checking control point values...")
        for name, expected_val in expected_values.items():
            actual_val = cp[name]
            assert np.allclose(actual_val, expected_val), f"Mismatch for '{name}'. Expected {expected_val}, got {np.round(actual_val,2)}"
            print(f"  ✅ {name}: {np.round(actual_val, 2)}")
        
        print("\nChecking triangle count...")
        expected_triangles = 24
        num_triangles = len(sv_data['triangles'])
        assert num_triangles == expected_triangles, f"Expected {expected_triangles} triangles, but found {num_triangles}"
        print(f"  ✅ Found {num_triangles} triangles (Expected: {expected_triangles})")

        print("\n✅ Simplified parser verification test passed!")

    except (AssertionError, KeyError) as e:
        print(f"\n❌ Test Failed: {e}")

if __name__ == '__main__':
    test_full_algorithm_a()
    test_full_algorithm()
