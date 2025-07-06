import math
import numpy as np
from lark import Lark, Transformer, v_args, LarkError
import copy
import random
from multiprocessing import Pool, cpu_count
from functools import partial
from grammar import GEOMETRIC_ALGORITHM_GRAMMAR
from geometric_utils import lin_p, orthogonal_projection_onto_plane, line_plane_intersection
from lark_transformer import GeometricAlgorithmTransformer
import re
from scipy.stats import qmc

UNINITIALIZED_POINT = np.array([-100000.0, -100000.0, -100000.0])

def get_highest_w_number(text_string):
    """
    Extracts the highest number from variables starting with 'w' in a given string.

    Args:
        text_string: The input string containing variables like w1, w2, w27, etc.

    Returns:
        The highest integer found after a 'w'.
        Returns None if no variables starting with 'w' are found.
    """
    # Find all sequences of digits that are preceded by the letter 'w'
    w_numbers = re.findall(r'w(\d+)', text_string)

    # If no such patterns are found, return None
    if not w_numbers:
        return None

    # Convert the list of found strings to a list of integers
    w_integers = [int(num) for num in w_numbers]

    # Return the maximum value from the list
    return max(w_integers)
        
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

def initialize_cell(indices, persistent_point_names, n_weights, weights):
    x, y, z = indices
    cp = {}
    spacing = (10.0, 10.0, 10.0)
    cp['originalSvCenter'] = np.array([x * spacing[0], y * spacing[1], z * spacing[2]])
    for name in persistent_point_names:
        cp[name] = UNINITIALIZED_POINT
    # Always use precomputed weights for this cell
    # Fix: If weights is a single dict (global), use it for all cells; if it's a dict of dicts, use per-cell
    if isinstance(weights, dict) and all(isinstance(v, dict) for v in weights.values()):
        cell_weights = weights[indices]
    else:
        cell_weights = weights
    return (indices, {'control_points': cp, 'triangles': [], 'weights': cell_weights})

def process_step_for_cell(indices, step_script, read_grid_data, step_index):
    context = {
        'current_indices': indices,
        'grid_spacing': (10.0, 10.0, 10.0),
        'step_index': step_index,
        'read_grid_data': read_grid_data,
    }
    weights = read_grid_data[indices]['weights']
    transformer = GeometricAlgorithmTransformer(context, weights)
    parser = Lark(GEOMETRIC_ALGORITHM_GRAMMAR, parser='lalr', transformer=transformer)
    parser.parse(step_script)
    original_cp_keys = read_grid_data[indices]['control_points'].keys()
    final_control_points = {key: transformer.vars.get(key) for key in original_cp_keys}
    final_triangles = transformer.triangles
    return (indices, final_control_points, final_triangles)

def execute_grid_algorithm(grid_dimensions, algorithm_steps, n_weights, weights=None):
    persistent_point_names = [
        "svCenter", "linX", "linY", "linZ", "edgeXY", "edgeYZ", "edgeXZ", 
        "mob", "int1", "int2", "int3", "int4", "int5", "int6",
        "int7", "int8", "int9", "int10", "int11", "int12"
    ]
    print("--- Initializing Grid (Parallel) ---")
    all_indices = [(x, y, z) for x in range(grid_dimensions[0]) for y in range(grid_dimensions[1]) for z in range(grid_dimensions[2])]
    if weights is None:
        # Precompute Sobol weights for all grid points (no multiprocessing)
        sobol = qmc.Sobol(d=n_weights, scramble=True)
        sobol_weights = sobol.random(len(all_indices))
        weights = {}
        for idx, indices in enumerate(all_indices):
            weights[indices] = {f'w{i+1}': sobol_weights[idx, i] for i in range(n_weights)}
    # If user provided a single dict of weights, use it for all cells
    init_partial = partial(initialize_cell, persistent_point_names=persistent_point_names, n_weights=n_weights, weights=weights)
    with Pool(processes=cpu_count()) as pool:
        initialized_cells = pool.map(init_partial, all_indices)
    grid_data = dict(initialized_cells)
    for i, step_script in enumerate(algorithm_steps):
        print(f"\n--- Executing Step {i+1} (Parallel) ---")
        read_grid_data = copy.deepcopy(grid_data)
        process_partial = partial(process_step_for_cell, step_script=step_script, read_grid_data=read_grid_data, step_index=i)
        with Pool(processes=cpu_count()) as pool:
            step_results = pool.map(process_partial, all_indices)
        print(f"--- Synchronizing Results for Step {i+1} ---")
        for indices, final_control_points, final_triangles in step_results:
            if final_control_points is not None:
                grid_data[indices]['control_points'].update(final_control_points)
            if final_triangles:
                grid_data[indices]['triangles'].extend(final_triangles)
    return grid_data