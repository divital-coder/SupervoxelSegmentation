import numpy as np
import math
import re
from concurrent.futures import ProcessPoolExecutor
import ast # For potentially parsing simple list structures if GBNF output is Python-like
import traceback # Import traceback module

# --- Configuration ---
# Default grid size, can be overridden in main call
DEFAULT_GRID_SIZE_X, DEFAULT_GRID_SIZE_Y, DEFAULT_GRID_SIZE_Z = 7, 7, 7
# Default spacing, can be overridden
DEFAULT_RX, DEFAULT_RY, DEFAULT_RZ = 1.0, 1.0, 1.0

# --- Point/Vector Operations & Math Functions ---
def vector(x, y, z):
    """Creates a numpy array representing a 3D vector."""
    return np.array([float(x), float(y), float(z)])

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

# --- Grid Creation ---
def create_grid(size_x, size_y, size_z, rx, ry, rz):
    """Creates a 3D grid storing the original_sv_center for each supervoxel."""
    grid = np.zeros((size_x, size_y, size_z), dtype=object)
    for i in range(size_x):
        for j in range(size_y):
            for k in range(size_z):
                grid[i, j, k] = vector(i * rx, j * ry, k * rz)
    return grid

# --- Expression Evaluation Context & Helpers ---
class EvaluationContext:
    """Holds the necessary context for evaluating expressions within a supervoxel's algorithm."""
    def __init__(self, current_coords, local_scope, all_control_points_map, grid_originals_map, weights_map, config_params_map):
        self.current_coords = current_coords  # (ix, iy, iz) of the supervoxel being processed
        self.local_scope = local_scope  # Dict for current supervoxel's definitions (sv_center, lin_x, temp1, etc.)
        self.all_control_points = all_control_points_map # Global map from Stage 1: {(ix,iy,iz): {cp_name: val, ...}}
        self.grid_originals = grid_originals_map    # Full grid of original_sv_center points
        self.weights = weights_map                  # Dict of weights {w1: val, ...}
        self.config_params = config_params_map      # Dict of config params {rx: val, ...}
        
        self.math_funcs = {"sin": math.sin, "cos": math.cos, "sqrt": math.sqrt}
        self.geom_funcs = {
            "vector": vector,
            "lin_p": lin_p,
            "orthogonal_projection_onto_plane": orthogonal_projection_onto_plane,
            "line_plane_intersection": line_plane_intersection,
        }

    def _parse_gbnf_neighbor_suffix(self, suffix_str_in_parens):
        """Parses GBNF neighbor suffix like '(x+1, y, z-1)' into relative offsets (dx, dy, dz)."""
        offsets = [0, 0, 0] # dx, dy, dz
        # Remove parentheses and split by comma
        parts = suffix_str_in_parens.strip("()").split(',')
        if len(parts) != 3:
            raise ValueError(f"Invalid neighbor suffix format: {suffix_str_in_parens}")
        
        for i, part_val in enumerate(parts): 
            part_val = part_val.strip()
            if   part_val.endswith("+1"): offsets[i] = 1
            elif part_val.endswith("-1"): offsets[i] = -1
            elif part_val == "x" or part_val == "y" or part_val == "z": offsets[i] = 0 
            else: raise ValueError(f"Invalid component in neighbor suffix: {part_val}")
        return tuple(offsets)

    def _parse_gbnf_triangle_point_suffix(self, suffix_str_no_underscore):
        """Parses GBNF triangle point suffix like 'xyz' or 'x+1yz-1' into relative offsets."""
        s = suffix_str_no_underscore
        offsets = [0,0,0] # dx, dy, dz

        # X component
        x_match = re.match(r"(x(\+1|-1)?)", s)
        if x_match:
            x_part = x_match.group(1)
            if x_part == "x+1": offsets[0] = 1
            elif x_part == "x-1": offsets[0] = -1
            else: offsets[0] = 0
            s = s[len(x_part):]
        else: raise ValueError(f"Missing or invalid x component in suffix '{suffix_str_no_underscore}'")
        
        # Y component
        y_match = re.match(r"(y(\+1|-1)?)", s)
        if y_match:
            y_part = y_match.group(1)
            if y_part == "y+1": offsets[1] = 1
            elif y_part == "y-1": offsets[1] = -1
            else: offsets[1] = 0
            s = s[len(y_part):]
        else: raise ValueError(f"Missing or invalid y component in suffix '{suffix_str_no_underscore}'")

        # Z component
        z_match = re.match(r"(z(\+1|-1)?)", s)
        if z_match:
            z_part = z_match.group(1)
            if z_part == "z+1": offsets[2] = 1
            elif z_part == "z-1": offsets[2] = -1
            else: offsets[2] = 0
            s = s[len(z_part):]
        else: raise ValueError(f"Missing or invalid z component in suffix '{suffix_str_no_underscore}'")
        
        if s: 
            raise ValueError(f"Invalid characters remaining in suffix '{suffix_str_no_underscore}': '{s}'")
        return tuple(offsets)

    def get_value(self, name_str):
        """Resolves a name (variable, weight, function, point component, neighbor ref) to its value."""
        if '.' in name_str:
            base_name, component = name_str.rsplit('.', 1)
            if component not in ['x', 'y', 'z']:
                raise ValueError(f"Unknown component '{component}' in '{name_str}'")
            point_val = self.get_value(base_name) 
            idx = ['x', 'y', 'z'].index(component)
            return np.asarray(point_val)[idx]

        neighbor_match = re.match(r"(\w+)_(\([\w\s\+\-,\s]+\))", name_str)
        if neighbor_match:
            base_point_name = neighbor_match.group(1)
            suffix_in_parens = neighbor_match.group(2)
            dx, dy, dz = self._parse_gbnf_neighbor_suffix(suffix_in_parens)
            
            neighbor_ix = self.current_coords[0] + dx
            neighbor_iy = self.current_coords[1] + dy
            neighbor_iz = self.current_coords[2] + dz
            
            check_x_cond = (0 <= neighbor_ix < self.grid_originals.shape[0])
            check_y_cond = (0 <= neighbor_iy < self.grid_originals.shape[1])
            check_z_cond = (0 <= neighbor_iz < self.grid_originals.shape[2])
            
            if not (check_x_cond and check_y_cond and check_z_cond):
                # print(f"--- DEBUG EvaluationContext.get_value: Out-of-bounds ---")
                # print(f"  Current Voxel Coords: {self.current_coords}")
                # print(f"  Requested GBNF Name: '{name_str}'")
                # print(f"  Calculated Neighbor Coords: ({neighbor_ix},{neighbor_iy},{neighbor_iz})")
                # print(f"  Grid Dimensions: {self.grid_originals.shape}")
                # print(f"  Check X (0 <= {neighbor_ix} < {self.grid_originals.shape[0]}): {check_x_cond}")
                # print(f"  Check Y (0 <= {neighbor_iy} < {self.grid_originals.shape[1]}): {check_y_cond}")
                # print(f"  Check Z (0 <= {neighbor_iz} < {self.grid_originals.shape[2]}): {check_z_cond}")
                # print(f"--- END DEBUG Out-of-bounds ---")
                raise ValueError(f"Neighbor coords ({neighbor_ix},{neighbor_iy},{neighbor_iz}) for '{name_str}' are out of grid bounds {self.grid_originals.shape}.")

            if base_point_name == "original_sv_center":
                return self.grid_originals[neighbor_ix, neighbor_iy, neighbor_iz]
            
            if self.all_control_points and (neighbor_ix, neighbor_iy, neighbor_iz) in self.all_control_points:
                neighbor_cps = self.all_control_points[(neighbor_ix, neighbor_iy, neighbor_iz)]
                if base_point_name in neighbor_cps:
                    return neighbor_cps[base_point_name]
            raise ValueError(f"Cannot resolve neighbor control point '{name_str}'. Neighbor CPs not found or point '{base_point_name}' undefined for neighbor ({neighbor_ix},{neighbor_iy},{neighbor_iz}).")

        triangle_point_match = re.match(r"(\b(?:lin_x|lin_y|lin_z|mob|sv_center|original_sv_center|int[1-6])\b)_((?:x|x\+1|x-1)(?:y|y\+1|y-1)(?:z|z\+1|z-1))$", name_str)
        if triangle_point_match:
            base_point_name = triangle_point_match.group(1)
            suffix_no_underscore = triangle_point_match.group(2) 
            dx, dy, dz = self._parse_gbnf_triangle_point_suffix(suffix_no_underscore)

            target_ix, target_iy, target_iz = self.current_coords[0] + dx, self.current_coords[1] + dy, self.current_coords[2] + dz
            
            if not (0 <= target_ix < self.grid_originals.shape[0] and \
                    0 <= target_iy < self.grid_originals.shape[1] and \
                    0 <= target_iz < self.grid_originals.shape[2]):
                raise ValueError(f"Target coords ({target_ix},{target_iy},{target_iz}) for triangle point spec '{name_str}' are out of grid bounds {self.grid_originals.shape}.")

            if base_point_name == "original_sv_center": 
                 return self.grid_originals[target_ix, target_iy, target_iz]

            if self.all_control_points and (target_ix, target_iy, target_iz) in self.all_control_points:
                target_cps = self.all_control_points[(target_ix, target_iy, target_iz)]
                if base_point_name in target_cps:
                    return target_cps[base_point_name]
            raise ValueError(f"Cannot resolve triangle point spec '{name_str}'. Target CP dict for ({target_ix},{target_iy},{target_iz}) not found or point '{base_point_name}' undefined in it.")

        if name_str in self.local_scope: return self.local_scope[name_str]
        if name_str in self.weights: return float(self.weights[name_str])
        if name_str in self.config_params: return float(self.config_params[name_str])
        if name_str in self.math_funcs: return self.math_funcs[name_str]
        if name_str in self.geom_funcs: return self.geom_funcs[name_str]
        if name_str == "original_sv_center": return self.grid_originals[self.current_coords]
        
        if self.all_control_points and self.current_coords in self.all_control_points:
            current_sv_cps = self.all_control_points[self.current_coords]
            if name_str in current_sv_cps:
                return current_sv_cps[name_str]

        raise ValueError(f"Unknown variable, function, or reference: '{name_str}' in current context {self.current_coords}.")


class ExpressionParser:
    """Parses and evaluates expressions based on GBNF structure using Python's eval.
    WARNING: This uses eval() and is a simplification. A full GBNF parser is more robust."""
    def __init__(self, context: EvaluationContext):
        self.context = context

    def _prepare_eval_scope(self):
        """Prepares a scope dictionary for Python's eval(), mapping all known names."""
        eval_scope = {}
        eval_scope.update(self.context.math_funcs)
        eval_scope.update(self.context.geom_funcs)
        eval_scope.update(self.context.weights)
        eval_scope.update(self.context.config_params)
        eval_scope.update(self.context.local_scope) 
        
        eval_scope['original_sv_center'] = self.context.grid_originals[self.context.current_coords]
        return eval_scope

    def _preprocess_expression_string(self, expr_str, eval_scope_keys):
        """Transforms GBNF-specific syntax in the expression string to be Python-eval compatible."""
        processed_expr = expr_str
        temp_resolution_scope = {}

        # Pattern for GBNF neighbor references: baseName_(coord_spec,coord_spec,coord_spec)
        gbnf_neighbor_pattern = re.compile(r"\b([a-zA-Z_]\w*)_(\([\w\s\+\-,\s]+\))\b")

        def gbnf_replacer_func(matchobj):
            full_match_str = matchobj.group(0) # The entire GBNF string, e.g., "original_sv_center_(x-1,y,z)"
            # base_name = matchobj.group(1) # e.g., "original_sv_center"
            # suffix = matchobj.group(2) # e.g., "(x-1,y,z)"
            
            try:
                value = self.context.get_value(full_match_str)
                py_safe_var_name = "resolved_" + re.sub(r'[^\w]', '_', full_match_str)
                
                if py_safe_var_name not in temp_resolution_scope:
                    temp_resolution_scope[py_safe_var_name] = value
                return py_safe_var_name
            except ValueError as e:
                # This error means the GBNF identifier (full_match_str) could not be resolved by get_value.
                # This is the point where out-of-bounds or other GBNF resolution issues should be caught.
                raise RuntimeError(f"GBNF Preprocessing Error: Identifier '{full_match_str}' in expression '{expr_str}' could not be resolved. Original error from get_value: ({type(e).__name__}) {e}") from e

        # Substitute all GBNF neighbor references using the replacer function
        try:
            processed_expr = gbnf_neighbor_pattern.sub(gbnf_replacer_func, processed_expr)
        except RuntimeError: # Catch errors raised by gbnf_replacer_func
            raise # Re-raise to be caught by the caller (ExpressionParser.parse)

        # Now handle point component access (e.g., point.x -> point[0])
        # This should apply to original variable names AND the newly created 'resolved_...' names
        def replace_point_component_access(match):
            base_var, component_char = match.groups()
            return f"{base_var}[{'xyz'.index(component_char)}]"
        
        # Pattern to match 'variable.component' or 'resolved_variable.component'
        # Ensure it correctly captures 'resolved_...' names which include underscores.
        point_component_pattern = re.compile(r"(\b(?:resolved_[a-zA-Z_]\w*|[a-zA-Z_]\w*))\s*\.\s*([xyz])\b")
        processed_expr = point_component_pattern.sub(replace_point_component_access, processed_expr)
        
        return processed_expr, temp_resolution_scope


    def parse(self, expr_str):
        """Evaluates the expression string."""
        current_eval_scope = self._prepare_eval_scope()
        
        try:
            # Preprocessing now directly raises RuntimeError if a GBNF identifier can't be resolved
            processed_expr_str, temp_resolved_vars = self._preprocess_expression_string(expr_str, current_eval_scope.keys())
            current_eval_scope.update(temp_resolved_vars)
        except RuntimeError as e: 
            # This RuntimeError comes from _preprocess_expression_string if a GBNF ref resolution failed.
            # print(f"ExpressionParser.parse: Preprocessing failed for '{expr_str}'. Error: {e}")
            raise # Re-raise the RuntimeError to be caught by stage1_calculate_control_points

        try:
            # print(f"DEBUG ExpressionParser.parse: About to eval. Processed string: '{processed_expr_str}'")
            # print(f"DEBUG ExpressionParser.parse: Eval scope keys: {list(current_eval_scope.keys())}")
            restricted_globals = {"__builtins__": {
                "abs": abs, "min": min, "max": max, "round": round, "len": len, 
                "float": float, "int": int, "str": str, "list": list, "dict": dict, "tuple": tuple,
                "True": True, "False": False, "None": None, "np": np 
            }}
            current_eval_scope['np'] = np
            return eval(processed_expr_str, restricted_globals, current_eval_scope)
        except Exception as e: # Catch errors from eval() itself (e.g., NameError, TypeError)
            # print(f"ExpressionParser.parse: Eval failed for processed expression '{processed_expr_str}' (original: '{expr_str}'). Error: {type(e).__name__}: {e}")
            raise # Re-raise the eval error


# --- Stage 1: Calculate Control Points for one Supervoxel ---
def stage1_calculate_control_points_for_supervoxel(args_tuple):
    """Processes point definitions for a single supervoxel."""
    (ix, iy, iz), original_sv_center_val, point_definitions_script_str, \
    global_weights, global_grid_originals, global_config_params = args_tuple

    current_sv_coords = (ix, iy, iz)
    current_sv_local_scope = {} 
    
    eval_ctx = EvaluationContext(current_sv_coords, current_sv_local_scope, 
                                 all_control_points_map=None, 
                                 grid_originals_map=global_grid_originals,
                                 weights_map=global_weights,
                                 config_params_map=global_config_params)
    
    expr_parser = ExpressionParser(eval_ctx)
    assignment_pattern = re.compile(r"^\s*(\w+)\s*=\s*(.+);")
    var_name_for_error = "" 
    expression_str_for_error = ""

    for line in point_definitions_script_str.splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue

        match = assignment_pattern.match(line)
        if match:
            var_name_for_error = match.group(1)
            expression_str_for_error = match.group(2)
            
            try:
                value = expr_parser.parse(expression_str_for_error)
                current_sv_local_scope[var_name_for_error] = value
            except Exception as e: 
                print(f"--- STAGE 1 EXCEPTION CAUGHT ---")
                print(f"Voxel: {current_sv_coords}")
                print(f"Defining Variable: '{var_name_for_error}'")
                print(f"Expression: '{expression_str_for_error}'")
                print(f"Exception Type: {type(e).__name__}")
                print(f"Exception Message: {e}")
                print(f"Full Traceback:\n{traceback.format_exc()}")
                print(f"--- END STAGE 1 EXCEPTION ---")

    control_point_names_to_extract = ["sv_center", "lin_x", "lin_y", "lin_z", "mob"] + [f"int{i}" for i in range(1, 7)]
    calculated_cps_for_this_voxel = {}
    for cp_name in control_point_names_to_extract:
        if cp_name in current_sv_local_scope:
            calculated_cps_for_this_voxel[cp_name] = current_sv_local_scope[cp_name]

    return (current_sv_coords, calculated_cps_for_this_voxel)


# --- Stage 2: Extract Triangles for one Supervoxel ---
def stage2_extract_triangles_for_supervoxel(args_tuple):
    """Extracts the 'triangs' array for a single supervoxel using pre-computed control points."""
    (ix, iy, iz), triangle_script_part_str, \
    all_cps_map_global, global_grid_originals, \
    global_weights, global_config_params = args_tuple

    current_sv_coords = (ix, iy, iz)
    current_sv_cps_from_stage1 = all_cps_map_global.get(current_sv_coords, {})

    eval_ctx = EvaluationContext(current_sv_coords, local_scope=current_sv_cps_from_stage1, 
                                 all_control_points_map=all_cps_map_global, 
                                 grid_originals_map=global_grid_originals,
                                 weights_map=global_weights,
                                 config_params_map=global_config_params)
    
    triangs_array_content_str = None
    for line in triangle_script_part_str.splitlines():
        line = line.strip()
        if line.startswith("triangs ="):
            match_array = re.match(r"^\s*triangs\s*=\s*(\[.*\])\s*;", line)
            if match_array:
                triangs_array_content_str = match_array.group(1)
            break 
    
    if not triangs_array_content_str:
        return (current_sv_coords, [])

    gbnf_point_spec_pattern = re.compile(r"\b(?:lin_x|lin_y|lin_z|mob|sv_center|original_sv_center|int[1-6])(?:_(?:x|x\+1|x-1)(?:y|y\+1|y-1)(?:z|z\+1|z-1))\b")
    resolved_triangles_list = []
    
    try:
        placeholder_map = {}
        placeholder_idx = 0

        def replacer_for_eval(match_obj):
            nonlocal placeholder_idx
            spec_str = match_obj.group(0)
            placeholder = f"__SPEC_PLACEHOLDER_{placeholder_idx}__"
            placeholder_map[placeholder] = spec_str
            placeholder_idx += 1
            return f"'{placeholder}'" 

        string_for_ast_eval = gbnf_point_spec_pattern.sub(replacer_for_eval, triangs_array_content_str)
        list_of_placeholder_triangles = ast.literal_eval(string_for_ast_eval)

        for placeholder_triangle_idx, placeholder_triangle in enumerate(list_of_placeholder_triangles):
            if not isinstance(placeholder_triangle, list) or len(placeholder_triangle) != 3:
                print(f"Stage 2 Warning: Voxel {current_sv_coords}, invalid triangle structure in parsed list (triangle index {placeholder_triangle_idx}): {placeholder_triangle}")
                continue
            
            current_resolved_triangle = []
            valid_tri = True
            original_point_spec_str_for_error = ""
            try:
                for point_in_triangle_idx, placeholder_spec in enumerate(placeholder_triangle):
                    original_point_spec_str = placeholder_map.get(placeholder_spec)
                    original_point_spec_str_for_error = original_point_spec_str 
                    if not original_point_spec_str:
                        print(f"Stage 2 Resolution Error: Voxel {current_sv_coords}, triangle index {placeholder_triangle_idx}, point index {point_in_triangle_idx}, unknown placeholder '{placeholder_spec}'.")
                        valid_tri = False; break
                    point_value = eval_ctx.get_value(original_point_spec_str)
                    current_resolved_triangle.append(point_value)
            except ValueError as e: 
                print(f"Stage 2 Resolution Error: Voxel {current_sv_coords}, triangle index {placeholder_triangle_idx}, point index {point_in_triangle_idx}, failed to resolve point spec '{original_point_spec_str_for_error}'. Error: {type(e).__name__}: {e}")
                valid_tri = False
            
            if valid_tri:
                resolved_triangles_list.append(current_resolved_triangle)

    except Exception as e:
        print(f"Stage 2 Parsing Error: Voxel {current_sv_coords}, failed to parse 'triangs' array string: '{triangs_array_content_str}'. Error: {type(e).__name__}: {e}")
        return (current_sv_coords, []) 
            
    return (current_sv_coords, resolved_triangles_list)


# --- Main Orchestration Function ---
def process_supervoxel_grid(
    grid_dimensions=(DEFAULT_GRID_SIZE_X, DEFAULT_GRID_SIZE_Y, DEFAULT_GRID_SIZE_Z),
    grid_spacing=(DEFAULT_RX, DEFAULT_RY, DEFAULT_RZ),
    full_algorithm_script="sv_center = original_sv_center; triangs = [];", 
    model_weights=None,
    parallel_execution=True,
    process_range_start=(2, 2, 2), 
    process_range_end=(6, 6, 6)    
    ):

    if model_weights is None: model_weights = {} 

    gsx, gsy, gsz = grid_dimensions
    rx_cfg, ry_cfg, rz_cfg = grid_spacing
    config_parameters = {"rx": rx_cfg, "ry": ry_cfg, "rz": rz_cfg}

    main_grid_originals = create_grid(gsx, gsy, gsz, rx_cfg, ry_cfg, rz_cfg)
    script_lines = full_algorithm_script.splitlines()
    
    point_def_script_lines = []
    triangle_def_script_lines = [] 
    
    currently_parsing_points = True
    for s_line in script_lines:
        stripped_s_line = s_line.strip()
        if not stripped_s_line or stripped_s_line.startswith("#"): continue

        if stripped_s_line.startswith("DEFINE_TRIANGLE") or stripped_s_line.startswith("triangs ="):
            currently_parsing_points = False
        
        if currently_parsing_points:
            point_def_script_lines.append(stripped_s_line)
        else:
            if stripped_s_line.startswith("triangs ="):
                 triangle_def_script_lines.append(stripped_s_line)

    point_definitions_for_stage1 = "\n".join(point_def_script_lines)
    triangle_script_for_stage2 = "\n".join(triangle_def_script_lines) 

    map_of_all_supervoxel_cps_for_debug = {}
    stage1_task_args_list = []

    start_ix, start_iy, start_iz = process_range_start
    end_ix = min(process_range_end[0] + 1, gsx)
    end_iy = min(process_range_end[1] + 1, gsy)
    end_iz = min(process_range_end[2] + 1, gsz)

    actual_start_ix = min(start_ix, gsx)
    actual_start_iy = min(start_iy, gsy)
    actual_start_iz = min(start_iz, gsz)

    print(f"Processing sub-volume: X from {actual_start_ix} to {end_ix-1}, Y from {actual_start_iy} to {end_iy-1}, Z from {actual_start_iz} to {end_iz-1}")

    for i in range(actual_start_ix, end_ix):
        for j in range(actual_start_iy, end_iy):
            for k in range(actual_start_iz, end_iz):
                original_center = main_grid_originals[i, j, k]
                args = ((i, j, k), original_center, point_definitions_for_stage1,
                        model_weights, main_grid_originals, config_parameters)
                stage1_task_args_list.append(args)
    
    print(f"Starting Stage 1: Calculating control points for {len(stage1_task_args_list)} supervoxels in sub-volume...")
    if parallel_execution and len(stage1_task_args_list) > 0 :
        with ProcessPoolExecutor() as executor:
            stage1_results_list = list(executor.map(stage1_calculate_control_points_for_supervoxel, stage1_task_args_list))
    elif len(stage1_task_args_list) > 0: 
        stage1_results_list = [stage1_calculate_control_points_for_supervoxel(task_args) for task_args in stage1_task_args_list]
    else:
        stage1_results_list = []
        print("Stage 1: No supervoxels to process in the specified sub-volume.")

    for coords, cp_dict in stage1_results_list:
        map_of_all_supervoxel_cps_for_debug[coords] = cp_dict 
    print(f"Stage 1 finished. Processed control points for {len(map_of_all_supervoxel_cps_for_debug)} supervoxels.")

    final_map_of_triangs_arrays = {}
    stage2_task_args_list = []
    if not triangle_script_for_stage2:
        print("Stage 2 Info: No 'triangs = ...' definition found in the algorithm script. Skipping triangle extraction.")
    else:
        for i in range(actual_start_ix, end_ix):
            for j in range(actual_start_iy, end_iy):
                for k in range(actual_start_iz, end_iz):
                    args = ((i, j, k), triangle_script_for_stage2,
                            map_of_all_supervoxel_cps_for_debug, 
                            main_grid_originals,
                            model_weights, config_parameters)
                    stage2_task_args_list.append(args)

        print(f"Starting Stage 2: Extracting 'triangs' arrays for {len(stage2_task_args_list)} supervoxels in sub-volume...")
        if parallel_execution and len(stage2_task_args_list) > 0:
            with ProcessPoolExecutor() as executor:
                stage2_results_list = list(executor.map(stage2_extract_triangles_for_supervoxel, stage2_task_args_list))
        elif len(stage2_task_args_list) > 0: 
            stage2_results_list = [stage2_extract_triangles_for_supervoxel(task_args) for task_args in stage2_task_args_list]
        else:
            stage2_results_list = []
            print("Stage 2: No supervoxels to process for triangle extraction in the specified sub-volume.")

        for coords, triangs_arr in stage2_results_list:
            final_map_of_triangs_arrays[coords] = triangs_arr
        print(f"Stage 2 finished. Processed 'triangs' arrays for {len(final_map_of_triangs_arrays)} supervoxels.")
    
    return final_map_of_triangs_arrays, map_of_all_supervoxel_cps_for_debug


# --- Example Usage ---
if __name__ == '__main__':
    example_llm_algorithm_output = """
    # This is an example algorithm string an LLM might generate based on GBNF.
    # It must follow the GBNF structure and conventions.

    # Stage 1: Point Definitions for ONE supervoxel
    sv_center = original_sv_center + vector(w1*rx*0.25, w2*ry*0.25, w3*rz*0.25);

    lin_x = lin_p(original_sv_center_(x-1,y,z), original_sv_center, w4);
    lin_y = lin_p(original_sv_center_(x,y-1,z), original_sv_center, w5);
    lin_z = lin_p(original_sv_center_(x,y,z-1), original_sv_center, w6);

    temp1 = w7 + w8; # Example temporary scalar variable
    mob = lin_p(lin_p(original_sv_center_(x-1,y-1,z-1), original_sv_center, temp1*0.5), 
                lin_p(original_sv_center_(x-1,y,z), original_sv_center_(x,y-1,z), w8), w9);

    int1 = orthogonal_projection_onto_plane(sv_center, lin_x, lin_y, lin_z);
    int2 = lin_p(mob, sv_center, w10);
    int3 = lin_p(lin_x, lin_y, w11); 
    int4 = lin_p(lin_y, lin_z, w12); 
    int5 = lin_p(lin_z, lin_x, w13); 
    int6 = sv_center + vector(w14*rx*0.1, w15*ry*0.1, w16*rz*0.1); 

    DEFINE_TRIANGLE(lin_x_xyz, lin_y_xyz, int1_xyz); 
    DEFINE_TRIANGLE(lin_y_xyz, lin_z_xyz, int1_xyz); 

    triangs = [
        [lin_x_xyz, lin_y_xyz, int1_xyz],
        [lin_y_xyz, lin_z_xyz, int1_xyz],
        [lin_z_xyz, lin_x_xyz, int1_xyz],
        [lin_x_xyz, mob_xyz, int2_xyz],
        [mob_xyz, lin_y_xyz, int2_xyz],
        [lin_y_xyz, mob_xyz, mob_x+1yz],  
        [int1_xyz, int2_xyz, sv_center_x-1y-1z-1] 
    ];
    """

    example_model_weights = {f"w{i}": np.random.rand() for i in range(1, 151)}
    
    print("Running example supervoxel grid processing...")
    test_grid_dimensions = (DEFAULT_GRID_SIZE_X, DEFAULT_GRID_SIZE_Y, DEFAULT_GRID_SIZE_Z) 
    test_grid_spacing = (1.0, 1.0, 1.0) 

    sub_volume_start_indices = (2, 2, 2)
    sub_volume_end_indices = (6, 6, 6) 

    all_supervoxels_triangs_data, all_supervoxels_cps_for_debug = process_supervoxel_grid(
        grid_dimensions=test_grid_dimensions,
        grid_spacing=test_grid_spacing,
        full_algorithm_script=example_llm_algorithm_output,
        model_weights=example_model_weights,
        parallel_execution=False,
        process_range_start=sub_volume_start_indices,
        process_range_end=sub_volume_end_indices
    )

    print(f"\nProcessing complete. Extracted 'triangs' arrays for {len(all_supervoxels_triangs_data)} supervoxels.")
    
    processed_voxel_count_display = 0
    # Ensure all_supervoxels_triangs_data is not empty before iterating
    if all_supervoxels_triangs_data:
        for i, (voxel_coords, triangles_list) in enumerate(all_supervoxels_triangs_data.items()):
            if processed_voxel_count_display >= 2 and len(all_supervoxels_triangs_data) > 2 : 
                print(f"\n... and results for {len(all_supervoxels_triangs_data) - processed_voxel_count_display} more supervoxels ...")
                break
            print(f"\nTriangles for supervoxel at coordinates {voxel_coords}:")
            if not triangles_list:
                print("  No triangles defined or extracted for this supervoxel.")
                cps_for_voxel = all_supervoxels_cps_for_debug.get(voxel_coords, {}) 
                if not cps_for_voxel:
                     print(f"  No control points were successfully defined for voxel {voxel_coords} in Stage 1.")
                else:
                     print(f"  Control points defined for {voxel_coords} in Stage 1: {list(cps_for_voxel.keys())}")
                
            else: 
                for tri_idx, triangle_points in enumerate(triangles_list):
                    if tri_idx >=3 and len(triangles_list) > 3: 
                        print(f"    ... and {len(triangles_list) - tri_idx} more triangles ...")
                        break
                    print(f"  Triangle {tri_idx + 1}:")
                    for pt_idx, point_coords in enumerate(triangle_points):
                        point_str = np.array2string(np.asarray(point_coords), formatter={'float_kind': lambda x: "%.3f" % x})
                        print(f"    Point {pt_idx + 1}: {point_str}")
            processed_voxel_count_display +=1
    else:
        print("No triangle data was generated for any supervoxel in the processed sub-volume.")

