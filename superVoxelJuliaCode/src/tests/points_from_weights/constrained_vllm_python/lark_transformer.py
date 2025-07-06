import math
import numpy as np
from lark import Lark, Transformer, v_args, LarkError
import copy
import random
from multiprocessing import Pool, cpu_count
from functools import partial
from grammar import GEOMETRIC_ALGORITHM_GRAMMAR
from geometric_utils import lin_p, orthogonal_projection_onto_plane, line_plane_intersection

# A sentinel value to indicate that a control point has not been calculated yet.
UNINITIALIZED_POINT = np.array([-100000.0, -100000.0, -100000.0])


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
            "range": (lambda lower, upper, weight: lower + (upper - lower) * weight, 3),
        }

        if func_name in func_dispatch:
            func, expected_arg_count = func_dispatch[func_name]
            if any(arg is None for arg in actual_args): return None
            if len(actual_args) != expected_arg_count:
                raise TypeError(f"{func_name}() expects {expected_arg_count} arguments, but got {len(actual_args)}")
            return func(*actual_args)
        else:
            raise NameError(f"Unknown function: '{func_name}'")