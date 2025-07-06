import math
import numpy as np
from lark import Lark, Transformer, v_args, LarkError
import copy
import random
from collections import Counter
import os
import json
from multiprocessing import Pool, cpu_count
from functools import partial
from grammar import GEOMETRIC_ALGORITHM_GRAMMAR
from geometric_utils import lin_p, orthogonal_projection_onto_plane, line_plane_intersection
from lark_transformer import GeometricAlgorithmTransformer
from execute_algorithm_on_data import get_highest_w_number,execute_grid_algorithm, initialize_cell, process_step_for_cell,UNINITIALIZED_POINT
from is_line_intersect import check_star_convexity,find_star_convexity_intersections
from is_water_tight import is_mesh_watertight,check_watertight
from visualize_mesh import visualize_geometry
"""
DO NOT reimplemnt is_mesh_watertight,find_star_convexity_intersections,lin_p, orthogonal_projection_onto_plane, line_plane_intersection GeometricAlgorithmTransformer execute_grid_algorithm, initialize_cell, process_step_for_cell,UNINITIALIZED_POINT
### 1. `GEOMETRIC_ALGORITHM_GRAMMAR`
- **Defined in:** constrained_vllm_python/grammar.py
- **What it is:**  
  A string variable containing the Lark grammar definition for the geometric algorithm scripting language.  
- **Usage:**  
  Used to initialize the Lark parser for parsing algorithm scripts.

---

### 2. `lin_p`
- **Defined in:** constrained_vllm_python/geometric_utils.py
- **What it does:**  
  Computes a linear interpolation between two points with a given weight.
- **Arguments:**  
  - `p_a`: First point (array-like)
  - `p_b`: Second point (array-like)
  - `weight`: Interpolation weight (float)
- **Returns:**  
  - A NumPy array representing the interpolated point.

---

### 3. `orthogonal_projection_onto_plane`
- **Defined in:** constrained_vllm_python/geometric_utils.py
- **What it does:**  
  Projects a point orthogonally onto a plane defined by three points.
- **Arguments:**  
  - `point`: The point to project (array-like)
  - `plane_p1`, `plane_p2`, `plane_p3`: Three points defining the plane (array-like)
- **Returns:**  
  - A NumPy array representing the projected point.

---

### 4. `line_plane_intersection`
- **Defined in:** constrained_vllm_python/geometric_utils.py
- **What it does:**  
  Finds the intersection point of a line (defined by two points) and a plane (defined by three points).
- **Arguments:**  
  - `line_p1`, `line_p2`: Two points defining the line (array-like)
  - `plane_p1`, `plane_p2`, `plane_p3`: Three points defining the plane (array-like)
- **Returns:**  
  - A NumPy array representing the intersection point, or `None` if no intersection.

---

### 5. `GeometricAlgorithmTransformer`
- **Defined in:** constrained_vllm_python/lark_transformer.py
- **What it does:**  
  A Lark `Transformer` subclass that interprets the parsed geometric algorithm script and executes assignments, function calls, and triangle definitions.
- **Arguments:**  
  - Initialized with a context dictionary and a weights dictionary.
- **Returns:**  
  - Used as a transformer in the Lark parser; accumulates results in its `vars` (variables) and `triangles` attributes.

---

### 6. `execute_grid_algorithm`
- **Defined in:** constrained_vllm_python/execute_algorithm_on_data.py
- **What it does:**  
  Orchestrates the execution of a multi-step geometric algorithm over a 3D grid, using multiprocessing for parallelism.
- **Arguments:**  
  - `grid_dimensions`: Tuple (x, y, z) grid size
  - `algorithm_steps`: List of script strings (one per step)
  - `weights`: Dictionary of weights
- **Returns:**  
  - A dictionary mapping grid indices to their computed control points and triangles.

---

### 7. `initialize_cell`
- **Defined in:** constrained_vllm_python/execute_algorithm_on_data.py
- **What it does:**  
  Initializes the data structure for a single grid cell, setting up control points (with uninitialized values) and triangles.
- **Arguments:**  
  - `indices`: Tuple (x, y, z) grid coordinates
  - `persistent_point_names`: List of control point names to initialize
- **Returns:**  
  - Tuple: (indices, {'control_points': ..., 'triangles': ...})

---

### 8. `process_step_for_cell`
- **Defined in:** constrained_vllm_python/execute_algorithm_on_data.py
- **What it does:**  
  Processes a single algorithm step for a single cell, updating its control points and triangles.
- **Arguments:**  
  - `indices`: Tuple (x, y, z) grid coordinates
  - `step_script`: Script string for this step
  - `weights`: Dictionary of weights
  - `read_grid_data`: The current state of the grid (read-only)
  - `step_index`: Index of the current step
- **Returns:**  
  - Tuple: (indices, final_control_points, final_triangles)

---

### 9. `UNINITIALIZED_POINT`
- **Defined in:** constrained_vllm_python/execute_algorithm_on_data.py
- **What it is:**  
  A NumPy array `np.array([-100000.0, -100000.0, -100000.0])` used as a sentinel value for uninitialized control points.

----
### 10. `find_star_convexity_intersections`

1. is_mesh_watertight

- What it does:
    Checks if a mesh (given as a list of triangles) is watertight (closed manifold, no holes).

- How to invoke:
    from is_water_tight import is_mesh_watertight

    # Each triangle is a tuple of 3 points, each point is a tuple of 3 floats (x, y, z)
    triangles = [
        ((0,0,0), (1,0,0), (0,1,0)),
        ((1,0,0), (1,1,0), (0,1,0)),
        # ... more triangles ...
    ]
    result = is_mesh_watertight(triangles)

- What to put in:
    triangles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]

- What it returns:
    True if the mesh is watertight, False otherwise.

- Example:
    triangles = [
        ((0,0,0), (1,0,0), (0,1,0)),
        ((1,0,0), (1,1,0), (0,1,0)),
        ((0,0,0), (0,1,0), (0,0,1)),
        ((0,1,0), (0,1,1), (0,0,1)),
        ((0,0,0), (0,0,1), (1,0,0)),
        ((1,0,0), (0,0,1), (1,0,1)),
        ((1,0,0), (1,0,1), (1,1,0)),
        ((1,1,0), (1,0,1), (1,1,1)),
        ((0,1,0), (1,1,0), (0,1,1)),
        ((0,1,1), (1,1,0), (1,1,1)),
        ((0,0,1), (0,1,1), (1,0,1)),
        ((1,0,1), (0,1,1), (1,1,1)),
    ]
    print(is_mesh_watertight(triangles))  # Output: True

---

2. find_star_convexity_intersections

- What it does:
    Checks if a polyhedron (given as a list of triangles) is star-convex with respect to a center point.
    For each triangle, draws a segment from the center to the triangle's barycenter and checks if it intersects any other triangle.

- How to invoke:
    from is_line_intersect import find_star_convexity_intersections


    triangles = [
        ((0,0,0), (1,0,0), (0,1,0)),
        ((1,0,0), (1,1,0), (0,1,0)),
        # ... more triangles ...
    ]
    center = (0.5, 0.5, 0.5)
    result = find_star_convexity_intersections(triangles, center)

- What to put in:
    triangles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]
    sv_center: Tuple[float, float, float]

- What it returns:
    A dictionary mapping triangle indices to lists of indices of triangles that the center-to-barycenter segment intersects.
    If the dictionary is empty, the shape is star-convex with respect to the center.

- Example:
    triangles = [
        ((1,1,1), (-1,-1,1), (-1,1,-1)),
        ((1,1,1), (1,-1,-1), (-1,1,-1)),
        ((1,1,1), (-1,-1,1), (1,-1,-1)),
        ((-1,-1,1), (-1,1,-1), (1,-1,-1)),
    ]
    center = (0,0,0)
    result = find_star_convexity_intersections(triangles, center)
    print(result)  # Output: {} (empty dict means star-convex)

    # For a non-star-convex shape, result will be non-empty, e.g. {1: [2], ...}

"""


def test_full_algorithm_a():
    """
    A comprehensive test case using a simplified, watertight algorithm,
    including validation for mesh integrity and star-convexity.
    """
    print("--- Running Simplified Watertight Algorithm Test ---")

    # Step 1: Supervoxel Center Modification
    step1 = """
        svCenter = originalSvCenter_(0,0,0) + vector(range(0.001, 0.002, w1) * rx - rx * 0.0005, range(0.001, 0.002, w2) * ry - ry * 0.0005, range(0.001, 0.002, w3) * rz - rz * 0.0005);
    """
    # Step 2: Linear and Edge Control Points (unchanged)
    step2 = """
        linX = lin_p(svCenter_(0,0,0), svCenter_(-1,0,0), range(0.49, 0.51, w4));
        linY = lin_p(svCenter_(0,0,0), svCenter_(0,-1,0), range(0.49, 0.51, w5));
        linZ = lin_p(svCenter_(0,0,0), svCenter_(0,0,-1), range(0.49, 0.51, w6));
        temp1 = lin_p(svCenter_(-1,-1,0), svCenter_(0,-1,0), range(0.49, 0.51, w7));
        temp2 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range(0.49, 0.51, w7));
        edgeXY = lin_p(temp1, temp2, range(0.49, 0.51, w8));
        temp3 = lin_p(svCenter_(0,-1,-1), svCenter_(0,0,-1), range(0.49, 0.51, w9));
        temp4 = lin_p(svCenter_(0,-1,0), svCenter_(0,0,0), range(0.49, 0.51, w9));
        edgeYZ = lin_p(temp3, temp4, range(0.49, 0.51, w10));
        temp5 = lin_p(svCenter_(-1,0,-1), svCenter_(0,0,-1), range(0.49, 0.51, w11));
        temp6 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range(0.49, 0.51, w11));
        edgeXZ = lin_p(temp5, temp6, range(0.49, 0.51, w12));
    """

    # Step 3: Main Oblique Point (mob) Definition (unchanged)
    step3 = """
        tempX00 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range(0.49, 0.51, w13));
        tempX10 = lin_p(svCenter_(-1,-1,0), svCenter_(0,-1,0), range(0.49, 0.51, w13));
        tempX01 = lin_p(svCenter_(-1,0,-1), svCenter_(0,0,-1), range(0.49, 0.51, w13));
        tempX11 = lin_p(svCenter_(-1,-1,-1), svCenter_(0,-1,-1), range(0.49, 0.51, w13));
        tempY0 = lin_p(tempX10, tempX00, range(0.49, 0.51, w14));
        tempY1 = lin_p(tempX11, tempX01, range(0.49, 0.51, w14));
        mob = lin_p(tempY1, tempY0, range(0.49, 0.51, w15));
    """

    # Step 4: NEW - Define intermediate points at the center of each face of the hexahedron.
    # These points are defined symmetrically to ensure adjacent cells compute the exact same point for their shared face.
    step4 = """
        int1 = lin_p(mob_(0,0,0), mob_(1,0,0), range(0.49, 0.51, w16));
        int2 = lin_p(int1, mob_(1,0,0), range(0.49, 0.51, w17));
        int3 = lin_p(int2, mob_(1,0,0), range(0.49, 0.51, w18));
        int4 = lin_p(int3, mob_(1,0,0), range(0.49, 0.51, w19));

        int5 = lin_p(mob_(0,0,0), mob_(0,1,0), range(0.49, 0.51, w20));
        int6 = lin_p(int5, mob_(0,1,0), range(0.49, 0.51, w21));
        int7 = lin_p(int6, mob_(0,1,0), range(0.49, 0.51, w22));
        int8 = lin_p(int7, mob_(0,1,0), range(0.49, 0.51, w23));
        
        int9 = lin_p(mob_(0,0,0), mob_(0,0,1), range(0.49, 0.51, w24));
        int10 = lin_p(int9, mob_(0,0,1), range(0.49, 0.51, w25));
        int11 = lin_p(int10, mob_(0,0,1), range(0.49, 0.51, w26));
        int12 = lin_p(int11, mob_(0,0,1), range(0.49, 0.51, w27));
        
    """

    step5 = """
# Triangles on the -Z face
# Original: defineTriangle(mob_(0,0,0), mob_(1,0,0), linZ_(0,0,0));
defineTriangle(mob_(0,0,0), int1_(0,0,0), linZ_(0,0,0));
defineTriangle(int1_(0,0,0), int2_(0,0,0), linZ_(0,0,0));
defineTriangle(int2_(0,0,0), int3_(0,0,0), linZ_(0,0,0));
defineTriangle(int3_(0,0,0), int4_(0,0,0), linZ_(0,0,0));
defineTriangle(int4_(0,0,0), mob_(1,0,0), linZ_(0,0,0));

# Original: defineTriangle(mob_(0,0,0), mob_(0,1,0), linZ_(0,0,0));
defineTriangle(mob_(0,0,0), int5_(0,0,0), linZ_(0,0,0));
defineTriangle(int5_(0,0,0), int6_(0,0,0), linZ_(0,0,0));
defineTriangle(int6_(0,0,0), int7_(0,0,0), linZ_(0,0,0));
defineTriangle(int7_(0,0,0), int8_(0,0,0), linZ_(0,0,0));
defineTriangle(int8_(0,0,0), mob_(0,1,0), linZ_(0,0,0));

# Triangles on the -Y face
# Original: defineTriangle(mob_(0,0,0), mob_(0,0,1), linY_(0,0,0));
defineTriangle(mob_(0,0,0), int9_(0,0,0), linY_(0,0,0));
defineTriangle(int9_(0,0,0), int10_(0,0,0), linY_(0,0,0));
defineTriangle(int10_(0,0,0), int11_(0,0,0), linY_(0,0,0));
defineTriangle(int11_(0,0,0), int12_(0,0,0), linY_(0,0,0));
defineTriangle(int12_(0,0,0), mob_(0,0,1), linY_(0,0,0));

# Original: defineTriangle(mob_(0,0,0), mob_(1,0,0), linY_(0,0,0));
defineTriangle(mob_(0,0,0), int1_(0,0,0), linY_(0,0,0));
defineTriangle(int1_(0,0,0), int2_(0,0,0), linY_(0,0,0));
defineTriangle(int2_(0,0,0), int3_(0,0,0), linY_(0,0,0));
defineTriangle(int3_(0,0,0), int4_(0,0,0), linY_(0,0,0));
defineTriangle(int4_(0,0,0), mob_(1,0,0), linY_(0,0,0));

# Triangles on the -X face
# Original: defineTriangle(mob_(0,0,0), mob_(0,1,0), linX_(0,0,0));
defineTriangle(mob_(0,0,0), int5_(0,0,0), linX_(0,0,0));
defineTriangle(int5_(0,0,0), int6_(0,0,0), linX_(0,0,0));
defineTriangle(int6_(0,0,0), int7_(0,0,0), linX_(0,0,0));
defineTriangle(int7_(0,0,0), int8_(0,0,0), linX_(0,0,0));
defineTriangle(int8_(0,0,0), mob_(0,1,0), linX_(0,0,0));

# Original: defineTriangle(mob_(0,0,0), mob_(0,0,1), linX_(0,0,0));
defineTriangle(mob_(0,0,0), int9_(0,0,0), linX_(0,0,0));
defineTriangle(int9_(0,0,0), int10_(0,0,0), linX_(0,0,0));
defineTriangle(int10_(0,0,0), int11_(0,0,0), linX_(0,0,0));
defineTriangle(int11_(0,0,0), int12_(0,0,0), linX_(0,0,0));
defineTriangle(int12_(0,0,0), mob_(0,0,1), linX_(0,0,0));


# --- Triangles on the shared face between (0,0,0) and (0,1,0) ---
# Edge mob_(1,1,0) -> mob_(1,0,0) is a -Y edge, negative corner is (1,0,0). Use int5-8_(1,0,0) in reverse.
# Original: defineTriangle(mob_(1,1,0), mob_(1,0,0), linZ_(0,0,0));
defineTriangle(mob_(1,1,0), int8_(1,0,0), linZ_(0,0,0));
defineTriangle(int8_(1,0,0), int7_(1,0,0), linZ_(0,0,0));
defineTriangle(int7_(1,0,0), int6_(1,0,0), linZ_(0,0,0));
defineTriangle(int6_(1,0,0), int5_(1,0,0), linZ_(0,0,0));
defineTriangle(int5_(1,0,0), mob_(1,0,0), linZ_(0,0,0));

# Edge mob_(1,1,0) -> mob_(0,1,0) is a -X edge, negative corner is (0,1,0). Use int1-4_(0,1,0) in reverse.
# Original: defineTriangle(mob_(1,1,0), mob_(0,1,0), linZ_(0,0,0));
defineTriangle(mob_(1,1,0), int4_(0,1,0), linZ_(0,0,0));
defineTriangle(int4_(0,1,0), int3_(0,1,0), linZ_(0,0,0));
defineTriangle(int3_(0,1,0), int2_(0,1,0), linZ_(0,0,0));
defineTriangle(int2_(0,1,0), int1_(0,1,0), linZ_(0,0,0));
defineTriangle(int1_(0,1,0), mob_(0,1,0), linZ_(0,0,0));

# Edge mob_(1,1,0) -> mob_(0,1,0) is a -X edge, negative corner is (0,1,0). Use int1-4_(0,1,0) in reverse.
# Original: defineTriangle(mob_(1,1,0), mob_(0,1,0), linY_(0,1,0));
defineTriangle(mob_(1,1,0), int4_(0,1,0), linY_(0,1,0));
defineTriangle(int4_(0,1,0), int3_(0,1,0), linY_(0,1,0));
defineTriangle(int3_(0,1,0), int2_(0,1,0), linY_(0,1,0));
defineTriangle(int2_(0,1,0), int1_(0,1,0), linY_(0,1,0));
defineTriangle(int1_(0,1,0), mob_(0,1,0), linY_(0,1,0));

# Edge mob_(1,1,0) -> mob_(1,1,1) is a +Z edge, negative corner is (1,1,0). Use int9-12_(1,1,0).
# Original: defineTriangle(mob_(1,1,0), mob_(1,1,1), linY_(0,1,0));
defineTriangle(mob_(1,1,0), int9_(1,1,0), linY_(0,1,0));
defineTriangle(int9_(1,1,0), int10_(1,1,0), linY_(0,1,0));
defineTriangle(int10_(1,1,0), int11_(1,1,0), linY_(0,1,0));
defineTriangle(int11_(1,1,0), int12_(1,1,0), linY_(0,1,0));
defineTriangle(int12_(1,1,0), mob_(1,1,1), linY_(0,1,0));

# Edge mob_(1,1,0) -> mob_(1,1,1) is a +Z edge, negative corner is (1,1,0). Use int9-12_(1,1,0).
# Original: defineTriangle(mob_(1,1,0), mob_(1,1,1), linX_(1,0,0));
defineTriangle(mob_(1,1,0), int9_(1,1,0), linX_(1,0,0));
defineTriangle(int9_(1,1,0), int10_(1,1,0), linX_(1,0,0));
defineTriangle(int10_(1,1,0), int11_(1,1,0), linX_(1,0,0));
defineTriangle(int11_(1,1,0), int12_(1,1,0), linX_(1,0,0));
defineTriangle(int12_(1,1,0), mob_(1,1,1), linX_(1,0,0));

# Edge mob_(1,1,0) -> mob_(1,0,0) is a -Y edge, negative corner is (1,0,0). Use int5-8_(1,0,0) in reverse.
# Original: defineTriangle(mob_(1,1,0), mob_(1,0,0), linX_(1,0,0));
defineTriangle(mob_(1,1,0), int8_(1,0,0), linX_(1,0,0));
defineTriangle(int8_(1,0,0), int7_(1,0,0), linX_(1,0,0));
defineTriangle(int7_(1,0,0), int6_(1,0,0), linX_(1,0,0));
defineTriangle(int6_(1,0,0), int5_(1,0,0), linX_(1,0,0));
defineTriangle(int5_(1,0,0), mob_(1,0,0), linX_(1,0,0));


# --- Triangles on the shared face between (0,0,0) and (0,0,1) ---
# Edge mob_(1,0,1) -> mob_(1,1,1) is a +Y edge, negative corner is (1,0,1). Use int5-8_(1,0,1).
# Original: defineTriangle(mob_(1,0,1), mob_(1,1,1), linZ_(0,0,1));
defineTriangle(mob_(1,0,1), int5_(1,0,1), linZ_(0,0,1));
defineTriangle(int5_(1,0,1), int6_(1,0,1), linZ_(0,0,1));
defineTriangle(int6_(1,0,1), int7_(1,0,1), linZ_(0,0,1));
defineTriangle(int7_(1,0,1), int8_(1,0,1), linZ_(0,0,1));
defineTriangle(int8_(1,0,1), mob_(1,1,1), linZ_(0,0,1));

# Edge mob_(1,0,1) -> mob_(0,0,1) is a -X edge, negative corner is (0,0,1). Use int1-4_(0,0,1) in reverse.
# Original: defineTriangle(mob_(1,0,1), mob_(0,0,1), linZ_(0,0,1));
defineTriangle(mob_(1,0,1), int4_(0,0,1), linZ_(0,0,1));
defineTriangle(int4_(0,0,1), int3_(0,0,1), linZ_(0,0,1));
defineTriangle(int3_(0,0,1), int2_(0,0,1), linZ_(0,0,1));
defineTriangle(int2_(0,0,1), int1_(0,0,1), linZ_(0,0,1));
defineTriangle(int1_(0,0,1), mob_(0,0,1), linZ_(0,0,1));

# Edge mob_(1,0,1) -> mob_(0,0,1) is a -X edge, negative corner is (0,0,1). Use int1-4_(0,0,1) in reverse.
# Original: defineTriangle(mob_(1,0,1), mob_(0,0,1), linY_(0,0,0));
defineTriangle(mob_(1,0,1), int4_(0,0,1), linY_(0,0,0));
defineTriangle(int4_(0,0,1), int3_(0,0,1), linY_(0,0,0));
defineTriangle(int3_(0,0,1), int2_(0,0,1), linY_(0,0,0));
defineTriangle(int2_(0,0,1), int1_(0,0,1), linY_(0,0,0));
defineTriangle(int1_(0,0,1), mob_(0,0,1), linY_(0,0,0));

# Edge mob_(1,0,1) -> mob_(1,0,0) is a -Z edge, negative corner is (1,0,0). Use int9-12_(1,0,0) in reverse.
# Original: defineTriangle(mob_(1,0,1), mob_(1,0,0), linY_(0,0,0));
defineTriangle(mob_(1,0,1), int12_(1,0,0), linY_(0,0,0));
defineTriangle(int12_(1,0,0), int11_(1,0,0), linY_(0,0,0));
defineTriangle(int11_(1,0,0), int10_(1,0,0), linY_(0,0,0));
defineTriangle(int10_(1,0,0), int9_(1,0,0), linY_(0,0,0));
defineTriangle(int9_(1,0,0), mob_(1,0,0), linY_(0,0,0));

# Edge mob_(1,0,1) -> mob_(1,1,1) is a +Y edge, negative corner is (1,0,1). Use int5-8_(1,0,1).
# Original: defineTriangle(mob_(1,0,1), mob_(1,1,1), linX_(1,0,0));
defineTriangle(mob_(1,0,1), int5_(1,0,1), linX_(1,0,0));
defineTriangle(int5_(1,0,1), int6_(1,0,1), linX_(1,0,0));
defineTriangle(int6_(1,0,1), int7_(1,0,1), linX_(1,0,0));
defineTriangle(int7_(1,0,1), int8_(1,0,1), linX_(1,0,0));
defineTriangle(int8_(1,0,1), mob_(1,1,1), linX_(1,0,0));

# Edge mob_(1,0,1) -> mob_(1,0,0) is a -Z edge, negative corner is (1,0,0). Use int9-12_(1,0,0) in reverse.
# Original: defineTriangle(mob_(1,0,1), mob_(1,0,0), linX_(1,0,0));
defineTriangle(mob_(1,0,1), int12_(1,0,0), linX_(1,0,0));
defineTriangle(int12_(1,0,0), int11_(1,0,0), linX_(1,0,0));
defineTriangle(int11_(1,0,0), int10_(1,0,0), linX_(1,0,0));
defineTriangle(int10_(1,0,0), int9_(1,0,0), linX_(1,0,0));
defineTriangle(int9_(1,0,0), mob_(1,0,0), linX_(1,0,0));


# --- Triangles on the shared face between (0,0,0) and (1,0,0) ---
# Edge mob_(0,1,1) -> mob_(1,1,1) is a +X edge, negative corner is (0,1,1). Use int1-4_(0,1,1).
# Original: defineTriangle(mob_(0,1,1), mob_(1,1,1), linZ_(0,0,1));
defineTriangle(mob_(0,1,1), int1_(0,1,1), linZ_(0,0,1));
defineTriangle(int1_(0,1,1), int2_(0,1,1), linZ_(0,0,1));
defineTriangle(int2_(0,1,1), int3_(0,1,1), linZ_(0,0,1));
defineTriangle(int3_(0,1,1), int4_(0,1,1), linZ_(0,0,1));
defineTriangle(int4_(0,1,1), mob_(1,1,1), linZ_(0,0,1));

# Edge mob_(0,1,1) -> mob_(0,0,1) is a -Y edge, negative corner is (0,0,1). Use int5-8_(0,0,1) in reverse.
# Original: defineTriangle(mob_(0,1,1), mob_(0,0,1), linZ_(0,0,1));
defineTriangle(mob_(0,1,1), int8_(0,0,1), linZ_(0,0,1));
defineTriangle(int8_(0,0,1), int7_(0,0,1), linZ_(0,0,1));
defineTriangle(int7_(0,0,1), int6_(0,0,1), linZ_(0,0,1));
defineTriangle(int6_(0,0,1), int5_(0,0,1), linZ_(0,0,1));
defineTriangle(int5_(0,0,1), mob_(0,0,1), linZ_(0,0,1));

# Edge mob_(0,1,1) -> mob_(1,1,1) is a +X edge, negative corner is (0,1,1). Use int1-4_(0,1,1).
# Original: defineTriangle(mob_(0,1,1), mob_(1,1,1), linY_(0,1,0));
defineTriangle(mob_(0,1,1), int1_(0,1,1), linY_(0,1,0));
defineTriangle(int1_(0,1,1), int2_(0,1,1), linY_(0,1,0));
defineTriangle(int2_(0,1,1), int3_(0,1,1), linY_(0,1,0));
defineTriangle(int3_(0,1,1), int4_(0,1,1), linY_(0,1,0));
defineTriangle(int4_(0,1,1), mob_(1,1,1), linY_(0,1,0));

# Edge mob_(0,1,1) -> mob_(0,1,0) is a -Z edge, negative corner is (0,1,0). Use int9-12_(0,1,0) in reverse.
# Original: defineTriangle(mob_(0,1,1), mob_(0,1,0), linY_(0,1,0));
defineTriangle(mob_(0,1,1), int12_(0,1,0), linY_(0,1,0));
defineTriangle(int12_(0,1,0), int11_(0,1,0), linY_(0,1,0));
defineTriangle(int11_(0,1,0), int10_(0,1,0), linY_(0,1,0));
defineTriangle(int10_(0,1,0), int9_(0,1,0), linY_(0,1,0));
defineTriangle(int9_(0,1,0), mob_(0,1,0), linY_(0,1,0));

# Edge mob_(0,1,1) -> mob_(0,0,1) is a -Y edge, negative corner is (0,0,1). Use int5-8_(0,0,1) in reverse.
# Original: defineTriangle(mob_(0,1,1), mob_(0,0,1), linX_(0,0,0));
defineTriangle(mob_(0,1,1), int8_(0,0,1), linX_(0,0,0));
defineTriangle(int8_(0,0,1), int7_(0,0,1), linX_(0,0,0));
defineTriangle(int7_(0,0,1), int6_(0,0,1), linX_(0,0,0));
defineTriangle(int6_(0,0,1), int5_(0,0,1), linX_(0,0,0));
defineTriangle(int5_(0,0,1), mob_(0,0,1), linX_(0,0,0));

# Edge mob_(0,1,1) -> mob_(0,1,0) is a -Z edge, negative corner is (0,1,0). Use int9-12_(0,1,0) in reverse.
# Original: defineTriangle(mob_(0,1,1), mob_(0,1,0), linX_(0,0,0));
defineTriangle(mob_(0,1,1), int12_(0,1,0), linX_(0,0,0));
defineTriangle(int12_(0,1,0), int11_(0,1,0), linX_(0,0,0));
defineTriangle(int11_(0,1,0), int10_(0,1,0), linX_(0,0,0));
defineTriangle(int10_(0,1,0), int9_(0,1,0), linX_(0,0,0));
defineTriangle(int9_(0,1,0), mob_(0,1,0), linX_(0,0,0));
    """

    algorithm_steps = [step1, step2, step3, step4,step5]
    # We only need 15 weights for this corrected algorithm.
    n_weights = get_highest_w_number(step4)
    grid_dimensions = (7, 7, 7)

    final_grid_data = execute_grid_algorithm(grid_dimensions, algorithm_steps, n_weights)

    # --- Visualization for supervoxel at (3,3,3) ---
    sv333 = final_grid_data[(3, 3, 3)]
    triangles_list = [tuple(map(tuple, tri)) for tri in sv333['triangles']]
    
    print(f"*****\n {sv333}  \n****")  
    # Prepare the output directory
    output_dir = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_viz"
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy arrays to lists for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        return obj

    sv333_serializable = {
        'control_points': to_serializable(sv333['control_points']),
        'triangles': to_serializable(sv333['triangles'])
    }

    output_path = os.path.join(output_dir, "sv333_debug.json")
    with open(output_path, "w") as f:
        json.dump(sv333_serializable, f, indent=2)

    # print(f"sv333 data saved to {output_path}")
    # control_points_dict = {
    #     'linX': tuple(sv333['control_points']['linX']),
    #     'linY': tuple(sv333['control_points']['linY']),
    #     'linZ': tuple(sv333['control_points']['linZ']),
    # }
    # # print("Control Points for sv333:")


    # visualize_geometry(triangles_list,control_points_dict)

    # --- Validation ---
    print("\n--- Running Validation on Interior Supervoxels ---")
    
    # Filter for interior cells that should be complete
    interior_cells = []
    gx, gy, gz = grid_dimensions
    for i in range(1, gx - 1):
        for j in range(1, gy - 1):
            for k in range(1, gz - 1):
                interior_cells.append(final_grid_data[(i, j, k)])

    num_interior_cells = len(interior_cells)
    print(f"Found {num_interior_cells} interior cells to validate.")

    with Pool(cpu_count()) as pool:
        # 1. Watertightness Check
        print("Checking for watertight meshes...")
        watertight_results = pool.map(check_watertight, interior_cells)
        watertight_count = sum(watertight_results)
        
        # 2. Star-Convexity Check
        print("Checking for star-convexity violations...")
        intersection_results = pool.map(check_star_convexity, interior_cells)

    # --- Reporting ---
    print("\n--- Validation Report ---")
    
    # Watertightness Report
    print(f"Watertight Meshes: {watertight_count} / {num_interior_cells}")
    if watertight_count < num_interior_cells:
        print(f"❌ Found {num_interior_cells - watertight_count} non-watertight meshes.")
    else:
        print("✅ All interior meshes are watertight.")

    # Star-Convexity Report
    intersections_found = [res for res in intersection_results if res]
    supervoxels_with_intersections = len(intersections_found)
    
    print(f"\nSupervoxels with Star-Convexity Violations: {supervoxels_with_intersections} / {num_interior_cells}")

    if supervoxels_with_intersections > 0:
        print(f"❌ Found {supervoxels_with_intersections} supervoxels with self-intersections.")
        
        # Aggregate all intersection errors to find the most common culprits
        all_intersections = Counter()
        for error_dict in intersections_found:
            for triangle_index, intersecting_triangles in error_dict.items():
                all_intersections[triangle_index] += 1
                for intersected_index in intersecting_triangles:
                    all_intersections[intersected_index] += 1
        
        print("\nMost Frequent Triangles Involved in Intersections:")
        for triangle_index, count in all_intersections.most_common(5):
            print(f"  - Triangle Index {triangle_index}: involved in {count} intersections.")
    else:
        print("✅ All interior supervoxels are star-convex.")

    print("\n--- Validation Complete ---")





























def test_full_algorithm_b():
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
 defineTriangle(mob, lin_p(mob,linX,0.5), int1);
defineTriangle(lin_p(mob,linX,0.5), linX, int1);
defineTriangle(linX, lin_p(linX,edgeXY,0.5), int1);
defineTriangle(lin_p(linX,edgeXY,0.5), edgeXY, int1);
defineTriangle(edgeXY, lin_p(edgeXY,mob,0.5), int1);
defineTriangle(lin_p(edgeXY,mob,0.5), mob, int1);
# Sub-face 2 (mob, edgeXY, linY) with apex int4
defineTriangle(mob, lin_p(mob,edgeXY,0.5), int4);
defineTriangle(lin_p(mob,edgeXY,0.5), edgeXY, int4);
defineTriangle(edgeXY, lin_p(edgeXY,linY,0.5), int4);
defineTriangle(lin_p(edgeXY,linY,0.5), linY, int4);
defineTriangle(linY, lin_p(linY,mob,0.5), int4);
defineTriangle(lin_p(linY,mob,0.5), mob, int4);

# -X Face (12 triangles using int2 and int5)
# Sub-face 1 (mob, linY, edgeYZ) with apex int2
defineTriangle(mob, lin_p(mob,linY,0.5), int2);
defineTriangle(lin_p(mob,linY,0.5), linY, int2);
defineTriangle(linY, lin_p(linY,edgeYZ,0.5), int2);
defineTriangle(lin_p(linY,edgeYZ,0.5), edgeYZ, int2);
defineTriangle(edgeYZ, lin_p(edgeYZ,mob,0.5), int2);
defineTriangle(lin_p(edgeYZ,mob,0.5), mob, int2);
# Sub-face 2 (mob, edgeYZ, linZ) with apex int5
defineTriangle(mob, lin_p(mob,edgeYZ,0.5), int5);
defineTriangle(lin_p(mob,edgeYZ,0.5), edgeYZ, int5);
defineTriangle(edgeYZ, lin_p(edgeYZ,linZ,0.5), int5);
defineTriangle(lin_p(edgeYZ,linZ,0.5), linZ, int5);
defineTriangle(linZ, lin_p(linZ,mob,0.5), int5);
defineTriangle(lin_p(linZ,mob,0.5), mob, int5);

# -Y Face (12 triangles using int3 and int6)
# Sub-face 1 (mob, linZ, edgeXZ) with apex int3
defineTriangle(mob, lin_p(mob,linZ,0.5), int3);
defineTriangle(lin_p(mob,linZ,0.5), linZ, int3);
defineTriangle(linZ, lin_p(linZ,edgeXZ,0.5), int3);
defineTriangle(lin_p(linZ,edgeXZ,0.5), edgeXZ, int3);
defineTriangle(edgeXZ, lin_p(edgeXZ,mob,0.5), int3);
defineTriangle(lin_p(edgeXZ,mob,0.5), mob, int3);
# Sub-face 2 (mob, edgeXZ, linX) with apex int6
defineTriangle(mob, lin_p(mob,edgeXZ,0.5), int6);
defineTriangle(lin_p(mob,edgeXZ,0.5), edgeXZ, int6);
defineTriangle(edgeXZ, lin_p(edgeXZ,linX,0.5), int6);
defineTriangle(lin_p(edgeXZ,linX,0.5), linX, int6);
defineTriangle(linX, lin_p(linX,mob,0.5), int6);
defineTriangle(lin_p(linX,mob,0.5), mob, int6);

# --- Positive Face Triangulations (Accessing Neighbors) ---

# +Z Face (from neighbor _(0,0,1))
defineTriangle(mob_(0,0,1), lin_p(mob_(0,0,1),linX_(0,0,1),0.5), int1_(0,0,1));
defineTriangle(lin_p(mob_(0,0,1),linX_(0,0,1),0.5), linX_(0,0,1), int1_(0,0,1));
defineTriangle(linX_(0,0,1), lin_p(linX_(0,0,1),edgeXY_(0,0,1),0.5), int1_(0,0,1));
defineTriangle(lin_p(linX_(0,0,1),edgeXY_(0,0,1),0.5), edgeXY_(0,0,1), int1_(0,0,1));
defineTriangle(edgeXY_(0,0,1), lin_p(edgeXY_(0,0,1),mob_(0,0,1),0.5), int1_(0,0,1));
defineTriangle(lin_p(edgeXY_(0,0,1),mob_(0,0,1),0.5), mob_(0,0,1), int1_(0,0,1));
defineTriangle(mob_(0,0,1), lin_p(mob_(0,0,1),edgeXY_(0,0,1),0.5), int4_(0,0,1));
defineTriangle(lin_p(mob_(0,0,1),edgeXY_(0,0,1),0.5), edgeXY_(0,0,1), int4_(0,0,1));
defineTriangle(edgeXY_(0,0,1), lin_p(edgeXY_(0,0,1),linY_(0,0,1),0.5), int4_(0,0,1));
defineTriangle(lin_p(edgeXY_(0,0,1),linY_(0,0,1),0.5), linY_(0,0,1), int4_(0,0,1));
defineTriangle(linY_(0,0,1), lin_p(linY_(0,0,1),mob_(0,0,1),0.5), int4_(0,0,1));
defineTriangle(lin_p(linY_(0,0,1),mob_(0,0,1),0.5), mob_(0,0,1), int4_(0,0,1));

# +X Face (from neighbor _(1,0,0))
defineTriangle(mob_(1,0,0), lin_p(mob_(1,0,0),linY_(1,0,0),0.5), int2_(1,0,0));
defineTriangle(lin_p(mob_(1,0,0),linY_(1,0,0),0.5), linY_(1,0,0), int2_(1,0,0));
defineTriangle(linY_(1,0,0), lin_p(linY_(1,0,0),edgeYZ_(1,0,0),0.5), int2_(1,0,0));
defineTriangle(lin_p(linY_(1,0,0),edgeYZ_(1,0,0),0.5), edgeYZ_(1,0,0), int2_(1,0,0));
defineTriangle(edgeYZ_(1,0,0), lin_p(edgeYZ_(1,0,0),mob_(1,0,0),0.5), int2_(1,0,0));
defineTriangle(lin_p(edgeYZ_(1,0,0),mob_(1,0,0),0.5), mob_(1,0,0), int2_(1,0,0));
defineTriangle(mob_(1,0,0), lin_p(mob_(1,0,0),edgeYZ_(1,0,0),0.5), int5_(1,0,0));
defineTriangle(lin_p(mob_(1,0,0),edgeYZ_(1,0,0),0.5), edgeYZ_(1,0,0), int5_(1,0,0));
defineTriangle(edgeYZ_(1,0,0), lin_p(edgeYZ_(1,0,0),linZ_(1,0,0),0.5), int5_(1,0,0));
defineTriangle(lin_p(edgeYZ_(1,0,0),linZ_(1,0,0),0.5), linZ_(1,0,0), int5_(1,0,0));
defineTriangle(linZ_(1,0,0), lin_p(linZ_(1,0,0),mob_(1,0,0),0.5), int5_(1,0,0));
defineTriangle(lin_p(linZ_(1,0,0),mob_(1,0,0),0.5), mob_(1,0,0), int5_(1,0,0));

# +Y Face (from neighbor _(0,1,0))
defineTriangle(mob_(0,1,0), lin_p(mob_(0,1,0),linZ_(0,1,0),0.5), int3_(0,1,0));
defineTriangle(lin_p(mob_(0,1,0),linZ_(0,1,0),0.5), linZ_(0,1,0), int3_(0,1,0));
defineTriangle(linZ_(0,1,0), lin_p(linZ_(0,1,0),edgeXZ_(0,1,0),0.5), int3_(0,1,0));
defineTriangle(lin_p(linZ_(0,1,0),edgeXZ_(0,1,0),0.5), edgeXZ_(0,1,0), int3_(0,1,0));
defineTriangle(edgeXZ_(0,1,0), lin_p(edgeXZ_(0,1,0),mob_(0,1,0),0.5), int3_(0,1,0));
defineTriangle(lin_p(edgeXZ_(0,1,0),mob_(0,1,0),0.5), mob_(0,1,0), int3_(0,1,0));
defineTriangle(mob_(0,1,0), lin_p(mob_(0,1,0),edgeXZ_(0,1,0),0.5), int6_(0,1,0));
defineTriangle(lin_p(mob_(0,1,0),edgeXZ_(0,1,0),0.5), edgeXZ_(0,1,0), int6_(0,1,0));
defineTriangle(edgeXZ_(0,1,0), lin_p(edgeXZ_(0,1,0),linX_(0,1,0),0.5), int6_(0,1,0));
defineTriangle(lin_p(edgeXZ_(0,1,0),linX_(0,1,0),0.5), linX_(0,1,0), int6_(0,1,0));
defineTriangle(linX_(0,1,0), lin_p(linX_(0,1,0),mob_(0,1,0),0.5), int6_(0,1,0));
defineTriangle(lin_p(linX_(0,1,0),mob_(0,1,0),0.5), mob_(0,1,0), int6_(0,1,0));

    """

    algorithm_steps = [step1, step2, step3, step4, step5]
    grid_dimensions = (7, 7, 7)

    final_grid_data = execute_grid_algorithm(grid_dimensions, algorithm_steps, 22)
    
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

    final_grid_data = execute_grid_algorithm(grid_dimensions, algorithm_steps, 22,weights=weights)
    
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
        num_triangles = len(sv_data['triangles']);
        assert num_triangles == expected_triangles, f"Expected {expected_triangles} triangles, but found {num_triangles}"
        print(f"  ✅ Found {num_triangles} triangles (Expected: {expected_triangles})")

        print("\n✅ Simplified parser verification test passed!")

    except (AssertionError, KeyError) as e:
        print(f"\n❌ Test Failed: {e}")

if __name__ == '__main__':
    test_full_algorithm_a()
    test_full_algorithm_b()
    test_full_algorithm()
