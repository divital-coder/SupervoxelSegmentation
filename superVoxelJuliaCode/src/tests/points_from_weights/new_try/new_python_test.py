import numpy as np
import trimesh # type: ignore
import random
import time # Added for pausing

# Additional trimesh imports for visualization
import trimesh.creation
# from trimesh.path import entities as path_entities # Not explicitly used

# --- Constants ---
# Small epsilon for checking degenerate segments or other floating point comparisons
PYRAYMESH_ADAPT_EPSILON = 1e-9 # Used in visualization for segment length
EPSILON_GEOMETRY = 1e-9      # Used for geometric comparisons (e.g., Möller-Trumbore)

# For reproducibility (optional, uncomment to use)
# random.seed(42)
# np.random.seed(42)

# --- Helper Functions ---
def lin_ip(p1, p2, w):
    """
    Performs linear interpolation between two 3D points p1 and p2.
    Result is p1*(1-w) + p2*w.

    Args:
        p1 (array-like): First 3D point.
        p2 (array-like): Second 3D point.
        w (float): Interpolation weight, typically between 0 and 1.

    Returns:
        np.ndarray: The 3D interpolated point.
    """
    return (1 - w) * np.array(p1) + w * np.array(p2)




def orthogonal_projection_onto_plane(
    point_to_project: tuple[float, float, float],
    plane_point1: tuple[float, float, float],
    plane_point2: tuple[float, float, float],
    plane_point3: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    """
    Calculates the orthogonal projection of a point onto a plane defined by three points.

    Args:
        point_to_project: A tuple (x, y, z) representing the point to project.
        plane_point1: A tuple (x, y, z) representing the first point defining the plane.
        plane_point2: A tuple (x, y, z) representing the second point defining the plane.
        plane_point3: A tuple (x, y, z) representing the third point defining the plane.

    Returns:
        A tuple (x, y, z) representing the orthogonally projected point,
        or None if the three plane points are collinear (don't define a plane).
    """
    p = np.array(point_to_project)
    p1 = np.array(plane_point1)
    p2 = np.array(plane_point2)
    p3 = np.array(plane_point3)

    # Vectors in the plane
    v1 = p2 - p1
    v2 = p3 - p1

    # Normal vector to the plane
    normal_vector = np.cross(v1, v2)
    normal_vector_norm_squared = np.dot(normal_vector, normal_vector)

    # Check if the plane points are collinear (normal vector is zero)
    if np.isclose(normal_vector_norm_squared, 0):
        print("Warning: The three points defining the plane are collinear.")
        # Check if the point_to_project is already on the line defined by p1, p2, p3
        # This is a simplified check; a more robust collinearity check for point_to_project
        # with the line p1-p2 would be more complex.
        # For now, if plane is not defined, we can't project.
        return None

    # Vector from a point on the plane to the point to project
    vector_to_point = p - p1

    # Distance from the point to the plane along the normal
    # t = dot(vector_to_point, normal_vector) / dot(normal_vector, normal_vector)
    # Using the formula: Proj_n(v) = (v . n / ||n||^2) * n
    # The projected point P_proj = P - Proj_n(P - P_on_plane)
    # P_proj = P - ( ((P - P_on_plane) . n) / ||n||^2 ) * n

    t = np.dot(vector_to_point, normal_vector) / normal_vector_norm_squared
    projected_point = p - t * normal_vector

    return tuple(projected_point)

def line_plane_intersection(
    line_point1: tuple[float, float, float],
    line_point2: tuple[float, float, float],
    plane_point1: tuple[float, float, float],
    plane_point2: tuple[float, float, float],
    plane_point3: tuple[float, float, float]
) -> tuple[float, float, float] | None:
    """
    Finds the intersection point of a line and a plane.
    The line is defined by two points.
    The plane is defined by three points.

    Args:
        line_point1: A tuple (x, y, z) representing the first point on the line.
        line_point2: A tuple (x, y, z) representing the second point on the line.
        plane_point1: A tuple (x, y, z) representing the first point defining the plane.
        plane_point2: A tuple (x, y, z) representing the second point defining the plane.
        plane_point3: A tuple (x, y, z) representing the third point defining the plane.

    Returns:
        A tuple (x, y, z) representing the intersection point.
        Returns None if the line is parallel to the plane and does not intersect,
        or if the line lies within the plane (infinite intersections),
        or if the three plane points are collinear.
    """
    l1 = np.array(line_point1)
    l2 = np.array(line_point2)
    p1 = np.array(plane_point1)
    p2 = np.array(plane_point2)
    p3 = np.array(plane_point3)

    # Line vector
    line_vec = l2 - l1

    # Plane vectors and normal
    plane_vec1 = p2 - p1
    plane_vec2 = p3 - p1
    plane_normal = np.cross(plane_vec1, plane_vec2)

    # Check if the plane points are collinear
    if np.all(np.isclose(plane_normal, 0)):
        print("Warning: The three points defining the plane are collinear.")
        return None

    # Check if line is parallel to the plane
    dot_product_line_normal = np.dot(line_vec, plane_normal)

    if np.isclose(dot_product_line_normal, 0):
        # Line is parallel to the plane.
        # Check if a point on the line is on the plane.
        if np.isclose(np.dot(p1 - l1, plane_normal), 0):
            print("Info: Line lies within the plane (infinite intersections).")
            return None # Or handle as infinite intersections
        else:
            print("Info: Line is parallel to the plane and does not intersect.")
            return None
    else:
        # Calculate intersection parameter t
        # t = dot(p1 - l1, plane_normal) / dot(line_vec, plane_normal)
        # We use a more stable formulation derived from:
        # (l1 + t * line_vec - p1) . plane_normal = 0
        # (l1 - p1) . plane_normal + t * (line_vec . plane_normal) = 0
        # t * (line_vec . plane_normal) = (p1 - l1) . plane_normal
        # t = ((p1 - l1) . plane_normal) / (line_vec . plane_normal)
        t = np.dot(p1 - l1, plane_normal) / dot_product_line_normal
        intersection_point = l1 + t * line_vec
        return tuple(intersection_point)

def signed_tetra_volume(v0, v1, v2, v3):
    """
    Calculates the signed volume of a tetrahedron defined by 4 vertices.
    The sign depends on the ordering of vertices (e.g., v0->v1, v0->v2, v0->v3).
    Volume = 1/6 * dot((v1-v0), cross((v2-v0), (v3-v0))).

    Args:
        v0, v1, v2, v3 (array-like): The four 3D vertices of the tetrahedron.

    Returns:
        float: The signed volume of the tetrahedron.
    """
    v0, v1, v2, v3 = np.array(v0), np.array(v1), np.array(v2), np.array(v3)
    return np.dot(v1 - v0, np.cross(v2 - v0, v3 - v0)) / 6.0

def calculate_barycenter(v0, v1, v2):
    """
    Calculates the barycenter (centroid) of a triangle defined by 3 vertices.

    Args:
        v0, v1, v2 (array-like): The three 3D vertices of the triangle.

    Returns:
        np.ndarray: The 3D barycenter of the triangle.
    """
    return (np.array(v0) + np.array(v1) + np.array(v2)) / 3.0

def segment_triangle_intersect(P0, P1, T0, T1, T2):
    """
    Checks if a line segment (P0-P1) intersects a triangle (T0-T1-T2).
    Uses a modified Möller-Trumbore algorithm for segment-triangle intersection.

    Args:
        P0 (np.ndarray): Start point of the segment.
        P1 (np.ndarray): End point of the segment.
        T0 (np.ndarray): First vertex of the triangle.
        T1 (np.ndarray): Second vertex of the triangle.
        T2 (np.ndarray): Third vertex of the triangle.

    Returns:
        bool: True if the segment intersects the triangle, False otherwise.
    """
    P0, P1 = np.array(P0), np.array(P1)
    T0, T1, T2 = np.array(T0), np.array(T1), np.array(T2)
    D = P1 - P0
    E1 = T1 - T0
    E2 = T2 - T0
    Pvec = np.cross(D, E2)
    det = np.dot(E1, Pvec)

    if abs(det) < EPSILON_GEOMETRY:
        return False
    invDet = 1.0 / det
    Tvec = P0 - T0
    u = np.dot(Tvec, Pvec) * invDet
    if u < -EPSILON_GEOMETRY or u > 1.0 + EPSILON_GEOMETRY:
        return False
    Qvec = np.cross(Tvec, E1)
    v = np.dot(D, Qvec) * invDet
    if v < -EPSILON_GEOMETRY or (u + v) > 1.0 + EPSILON_GEOMETRY:
        return False
    t_ray = np.dot(E2, Qvec) * invDet
    if -EPSILON_GEOMETRY <= t_ray <= 1.0 + EPSILON_GEOMETRY:
        if np.dot(D, D) < EPSILON_GEOMETRY**2: 
            return u >= -EPSILON_GEOMETRY and v >= -EPSILON_GEOMETRY and (u + v) <= 1.0 + EPSILON_GEOMETRY
        return True
    return False

def robust_triangle_hash(tri_coords_tuple, precision):
    """
    Creates a hashable representation of a triangle based on its sorted, rounded vertex coordinates.

    Args:
        tri_coords_tuple (tuple): A tuple of 3 vertex coordinates.
        precision (int): Number of decimal places to round coordinates to.

    Returns:
        tuple: A sorted tuple of rounded vertex coordinate tuples.
    """
    rounded_sorted_vertices = sorted([
        tuple(np.round(np.array(v_coord), precision)) for v_coord in tri_coords_tuple
    ])
    return tuple(rounded_sorted_vertices)

# --- Step 1: Define Initial Grid (sv) and Base Points (dv, now same as sv) ---
def define_sv_points(grid_dims, grid_spacings):
    """
    Defines a regular spatial grid sv (structural vertices).
    sv[i,j,k] = (i*rx, j*ry, k*rz).
    """
    print("Generating sv (structural vertex) points...")
    sv_points = np.zeros(grid_dims + (3,))
    rx, ry, rz = grid_spacings
    for i in range(grid_dims[0]):
        for j in range(grid_dims[1]):
            for k in range(grid_dims[2]):
                sv_points[i,j,k] = np.array([i * rx, j * ry, k * rz])
    return sv_points

def define_dv_points(sv_points_grid):
    """
    Defines dv points. In this simplified version, dv points are the same as sv points
    (no dislocation).
    Args:
        sv_points_grid (np.ndarray): The (Nx, Ny, Nz, 3) array of sv points.
    Returns:
        np.ndarray: A copy of sv_points_grid.
    """
    print("Defining dv points (equivalent to sv points in this version)...")
    return np.copy(sv_points_grid) # Return a copy to avoid modifying sv if dv is changed later (though not in this flow)

# --- Step 2 (was Step 3): Define Main Oblique Points (mob) ---
# These points are calculated based on dv_points (which are sv_points).
def define_mob_point_from_corners(dv_points_grid, i, j, k):
    """
    Defines a single mob point mob[i,j,k] by trilinear-like interpolation
    within the cell whose maximum corner is dv_points_grid[i,j,k].
    """
    c = {
        '000': dv_points_grid[i-1,j-1,k-1], '100': dv_points_grid[i  ,j-1,k-1],
        '010': dv_points_grid[i-1,j  ,k-1], '001': dv_points_grid[i-1,j-1,k  ],
        '110': dv_points_grid[i  ,j  ,k-1], '101': dv_points_grid[i  ,j-1,k  ],
        '011': dv_points_grid[i-1,j  ,k  ], '111': dv_points_grid[i  ,j  ,k  ]
    }
    min_w, max_w = EPSILON_GEOMETRY, 1.0 - EPSILON_GEOMETRY 
    w_m = [random.uniform(min_w, max_w) for _ in range(7)]
    p_x1 = lin_ip(c['011'], c['111'], w_m[0])
    p_x2 = lin_ip(c['001'], c['101'], w_m[1])
    p_x3 = lin_ip(c['010'], c['110'], w_m[2])
    p_x4 = lin_ip(c['000'], c['100'], w_m[3])
    p_y1 = lin_ip(p_x1, p_x2, w_m[4])
    p_y2 = lin_ip(p_x3, p_x4, w_m[5])
    mob_pt = lin_ip(p_y1, p_y2, w_m[6])
    return mob_pt

def generate_all_mob_points(dv_points_grid, grid_dims):
    """
    Generates all mob points from dv_points_grid.
    """
    print("Generating mob (main oblique) points...")
    all_mob_pts = np.zeros(grid_dims + (3,))
    for i in range(1, grid_dims[0]):
        for j in range(1, grid_dims[1]):
            for k in range(1, grid_dims[2]):
                all_mob_pts[i,j,k] = define_mob_point_from_corners(dv_points_grid, i, j, k)
    return all_mob_pts

# --- Step 3 (was Step 2 revised): Define Linear Control Points (lin) ---
def _calculate_lin_neg_point_for_axis(mob_face_coords, offset_basis_vector):
    """
    Helper function to calculate a single lin_neg point for a given axis.
    lin_neg = C_face - delta_scalar * offset_basis_vector
    delta_scalar = 1.0 + random_weight * 1.0 (since l_factor is removed)
    """
    m0, m1, m2, m3 = [np.array(p) for p in mob_face_coords]
    c_face = (m0 + m1 + m2 + m3) / 4.0
    w_depth = random.random()
    delta_scalar = 1.0 + w_depth * 1.0 
    lin_neg_pt = c_face - delta_scalar * offset_basis_vector
    return lin_neg_pt

def generate_all_revised_lin_neg_points(all_mob_pts, grid_dims, grid_spacings):
    """
    Generates 'negative' linear control points (lin_X_neg, lin_Y_neg, lin_Z_neg)
    based on mob-face centroids.
    """
    print("Generating revised lin (linear control) 'negative' points...")
    rx, ry, rz = grid_spacings
    all_lin_X_neg = np.full(grid_dims + (3,), np.nan)
    all_lin_Y_neg = np.full(grid_dims + (3,), np.nan)
    all_lin_Z_neg = np.full(grid_dims + (3,), np.nan)

    for sx_own in range(grid_dims[0]):
        for sy_own in range(grid_dims[1]):
            for sz_own in range(grid_dims[2]):
                if sx_own >= 1 and \
                   (1 <= sx_own < grid_dims[0]) and \
                   (1 <= sy_own < grid_dims[1] - 1) and \
                   (1 <= sz_own < grid_dims[2] - 1):
                    mob_face_X_coords = [
                        all_mob_pts[sx_own, sy_own,     sz_own],
                        all_mob_pts[sx_own, sy_own + 1, sz_own],
                        all_mob_pts[sx_own, sy_own + 1, sz_own + 1],
                        all_mob_pts[sx_own, sy_own,     sz_own + 1]
                    ]
                    offset_vec_X = np.array([rx, 0.0, 0.0])
                    all_lin_X_neg[sx_own, sy_own, sz_own] = _calculate_lin_neg_point_for_axis(
                        mob_face_X_coords, offset_vec_X
                    )
                if sy_own >= 1 and \
                   (1 <= sx_own < grid_dims[0] - 1) and \
                   (1 <= sy_own < grid_dims[1]) and \
                   (1 <= sz_own < grid_dims[2] - 1):
                    mob_face_Y_coords = [
                        all_mob_pts[sx_own,     sy_own, sz_own],
                        all_mob_pts[sx_own + 1, sy_own, sz_own],
                        all_mob_pts[sx_own + 1, sy_own, sz_own + 1],
                        all_mob_pts[sx_own,     sy_own, sz_own + 1]
                    ]
                    offset_vec_Y = np.array([0.0, ry, 0.0])
                    all_lin_Y_neg[sx_own, sy_own, sz_own] = _calculate_lin_neg_point_for_axis(
                        mob_face_Y_coords, offset_vec_Y
                    )
                if sz_own >= 1 and \
                   (1 <= sx_own < grid_dims[0] - 1) and \
                   (1 <= sy_own < grid_dims[1] - 1) and \
                   (1 <= sz_own < grid_dims[2]):
                    mob_face_Z_coords = [
                        all_mob_pts[sx_own,     sy_own,     sz_own],
                        all_mob_pts[sx_own + 1, sy_own,     sz_own],
                        all_mob_pts[sx_own + 1, sy_own + 1, sz_own],
                        all_mob_pts[sx_own,     sy_own + 1, sz_own]
                    ]
                    offset_vec_Z = np.array([0.0, 0.0, rz])
                    all_lin_Z_neg[sx_own, sy_own, sz_own] = _calculate_lin_neg_point_for_axis(
                        mob_face_Z_coords, offset_vec_Z
                    )
    return all_lin_X_neg, all_lin_Y_neg, all_lin_Z_neg

def get_revised_lin_points_for_poly_c(all_lin_X_neg, all_lin_Y_neg, all_lin_Z_neg, grid_dims_tuple, cx, cy, cz):
    """
    Retrieves the 6 lin points required for Poly_C construction around point (cx,cy,cz).
    """
    lin_map = {}
    def get_lin_point(lin_array, i, j, k, name_for_error):
        if not (0 <= i < grid_dims_tuple[0] and \
                0 <= j < grid_dims_tuple[1] and \
                0 <= k < grid_dims_tuple[2]):
            raise ValueError(f"Index ({i},{j},{k}) for {name_for_error} out of bounds for grid {grid_dims_tuple}.")
        point = lin_array[i,j,k]
        if np.isnan(point).any():
            raise ValueError(f"{name_for_error} at ({i},{j},{k}) is NaN.")
        return point
    try:
        lin_map['L_Xn'] = get_lin_point(all_lin_X_neg, cx,   cy,   cz,   f'L_Xn')
        lin_map['L_Xp'] = get_lin_point(all_lin_X_neg, cx+1, cy,   cz,   f'L_Xp')
        lin_map['L_Yn'] = get_lin_point(all_lin_Y_neg, cx,   cy,   cz,   f'L_Yn')
        lin_map['L_Yp'] = get_lin_point(all_lin_Y_neg, cx,   cy+1, cz,   f'L_Yp')
        lin_map['L_Zn'] = get_lin_point(all_lin_Z_neg, cx,   cy,   cz,   f'L_Zn')
        lin_map['L_Zp'] = get_lin_point(all_lin_Z_neg, cx,   cy,   cz+1, f'L_Zp')
    except IndexError as e:
        raise ValueError(f"IndexError for Poly_C at ({cx},{cy},{cz}): {e}.")
    return lin_map

def get_relevant_mob_points_for_poly_c(all_mob_pts, grid_dims_tuple, cx, cy, cz):
    """
    Retrieves the 8 mob points forming the mob-hexahedron around point (cx,cy,cz).
    """
    if not (1 <= cx < grid_dims_tuple[0] - 1 and \
            1 <= cy < grid_dims_tuple[1] - 1 and \
            1 <= cz < grid_dims_tuple[2] - 1):
        raise ValueError(
            f"Point index ({cx},{cy},{cz}) invalid for 8 surrounding mob points. "
            f"Indices must be in [1, grid_dim_axis-2]."
        )
    mob_data_map = {}
    mob_indices_map = {
        'm000': (cx, cy, cz),     'm100': (cx + 1, cy, cz),
        'm010': (cx, cy + 1, cz), 'm001': (cx, cy, cz + 1),
        'm110': (cx + 1, cy + 1, cz), 'm101': (cx + 1, cy, cz + 1),
        'm011': (cx, cy + 1, cz + 1), 'm111': (cx + 1, cy + 1, cz + 1)
    }
    for key, (mi, mj, mk) in mob_indices_map.items():
        point = all_mob_pts[mi, mj, mk]
        is_genuinely_computed_mob_location = (1 <= mi < grid_dims_tuple[0] and \
                                              1 <= mj < grid_dims_tuple[1] and \
                                              1 <= mk < grid_dims_tuple[2])
        if np.allclose(point, [0,0,0]) and is_genuinely_computed_mob_location :
             print(f"Warning: mob point {key} at ({mi},{mj},{mk}) is [0,0,0] within expected valid range.")
        mob_data_map[key] = {'coord': point, 'ijk_mob_grid': (mi, mj, mk)}
    return mob_data_map

# --- Step 4: Define the Polyhedron Poly_C ---
def define_poly_c_base_triangles(dv_center_coord, lin_map, mob_data_map):
    """
    Defines the 24 base triangles on the surface of Poly_C.
    """
    surface_base_triangles_coords = []
    face_definitions = [
        {'lin_key': 'L_Xn', 'mob_keys': ['m000', 'm001', 'm011', 'm010']},
        {'lin_key': 'L_Xp', 'mob_keys': ['m100', 'm110', 'm111', 'm101']},
        {'lin_key': 'L_Yn', 'mob_keys': ['m000', 'm100', 'm101', 'm001']},
        {'lin_key': 'L_Yp', 'mob_keys': ['m010', 'm011', 'm111', 'm110']},
        {'lin_key': 'L_Zn', 'mob_keys': ['m000', 'm010', 'm110', 'm100']},
        {'lin_key': 'L_Zp', 'mob_keys': ['m001', 'm101', 'm111', 'm011']},
    ]
    for face_def in face_definitions:
        lin_point_coord = lin_map[face_def['lin_key']]
        mob_key_cycle = face_def['mob_keys']
        for i in range(4):
            mob_p_key = mob_key_cycle[i]
            mob_q_key = mob_key_cycle[(i + 1) % 4]
            mob_p_coord = mob_data_map[mob_p_key]['coord']
            mob_q_coord = mob_data_map[mob_q_key]['coord']
            if signed_tetra_volume(dv_center_coord, lin_point_coord, mob_p_coord, mob_q_coord) < 0:
                current_base_triangle = (lin_point_coord, mob_q_coord, mob_p_coord)
            else:
                current_base_triangle = (lin_point_coord, mob_p_coord, mob_q_coord)
            surface_base_triangles_coords.append(current_base_triangle)
    return surface_base_triangles_coords

# --- Main Algorithm Execution and Verification for a Single Poly_C ---
def run_algorithm_and_verify_poly_c(dv_c_indices, grid_dims_config, grid_spacings_config):
    """
    Runs the full algorithm to generate points and then constructs and verifies
    a single Poly_C centered at dv_c_indices.
    """
    cx, cy, cz = dv_c_indices
    poly_c_id_str = f"Poly_C_sv[{cx},{cy},{cz}]" # Changed dv to sv in ID
    print(f"\n--- Processing {poly_c_id_str} ---")

    # Step 1: sv and dv points (dv is now same as sv)
    sv_points_grid = define_sv_points(grid_dims_config, grid_spacings_config)
    dv_points_grid = define_dv_points(sv_points_grid) # No l_factor
    
    # Step 2 (was Step 3): mob points
    all_mob_pts = generate_all_mob_points(dv_points_grid, grid_dims_config)

    # Step 3 (was Step 2 revised): lin points
    all_lin_X_neg, all_lin_Y_neg, all_lin_Z_neg = generate_all_revised_lin_neg_points(
        all_mob_pts, grid_dims_config, grid_spacings_config # No l_factor
    )

    center_coord = None 
    lin_map = {} # Initialize lin_map
    mob_data_map = {} # Initialize mob_data_map
    try:
        if not (0 <= cx < dv_points_grid.shape[0] and \
                0 <= cy < dv_points_grid.shape[1] and \
                0 <= cz < dv_points_grid.shape[2]):
            raise IndexError(f"Center indices ({cx},{cy},{cz}) out of bounds for grid shape {dv_points_grid.shape}.")
        center_coord = dv_points_grid[cx,cy,cz] # This is sv_points_grid[cx,cy,cz]

        lin_map = get_revised_lin_points_for_poly_c(
            all_lin_X_neg, all_lin_Y_neg, all_lin_Z_neg, grid_dims_config, cx, cy, cz)
        mob_data_map = get_relevant_mob_points_for_poly_c(all_mob_pts, grid_dims_config, cx, cy, cz)
    except (ValueError, IndexError) as e:
        print(f"ERROR ({poly_c_id_str}): Failed to get points: {e}")
        center_val_for_error = center_coord.tolist() if center_coord is not None else [0.,0.,0.]
        # Ensure lin_map and mob_data_map are populated for the return, even if empty
        return {
            "is_valid_poly_c": False, "poly_c_id": poly_c_id_str, "error_message": str(e),
            "visualization_data": {
                "dv_center": center_val_for_error,
                "base_triangles_coords_raw": [],
                "error_message": str(e),
                "lin_points": {k: v.tolist() for k,v in lin_map.items()} if lin_map else {},
                "mob_points": {k: v['coord'].tolist() for k, v in mob_data_map.items()} if mob_data_map else {}
            }
        }

    # Step 4: Define Poly_C surface triangles
    surface_base_triangles_coords = define_poly_c_base_triangles(center_coord, lin_map, mob_data_map)

    if len(surface_base_triangles_coords) != 24:
        error_msg = f"Expected 24 triangles, got {len(surface_base_triangles_coords)}."
        print(f"ERROR ({poly_c_id_str}): {error_msg}")
        return {
            "is_valid_poly_c": False,
            "poly_c_id": poly_c_id_str,
            "error_message": error_msg,
            "visualization_data": {
                "dv_center": center_coord.tolist() if center_coord is not None else [0.,0.,0.],
                "base_triangles_coords_raw": [[c.tolist() for c in tri] for tri in surface_base_triangles_coords],
                "error_message": error_msg,
                "lin_points": {k: v.tolist() for k,v in lin_map.items()} if lin_map else {},
                "mob_points": {k: v['coord'].tolist() for k, v in mob_data_map.items()} if mob_data_map else {}
            }
        }

    # --- Verification using Trimesh ---
    unique_surface_verts_map = {}
    unique_surface_verts_list = []
    rounding_precision_verts = int(-np.log10(EPSILON_GEOMETRY)) + 2 
    for tri_coords in surface_base_triangles_coords:
        for v_coord in tri_coords:
            v_tuple = tuple(np.round(np.array(v_coord), rounding_precision_verts))
            if v_tuple not in unique_surface_verts_map:
                unique_surface_verts_map[v_tuple] = len(unique_surface_verts_list)
                unique_surface_verts_list.append(np.array(v_coord))
    surface_trimesh_vertices = np.array(unique_surface_verts_list)
    surface_trimesh_faces = []
    for tri_coords in surface_base_triangles_coords:
        try:
            face_indices = [unique_surface_verts_map[tuple(np.round(np.array(v_coord), rounding_precision_verts))] for v_coord in tri_coords]
            surface_trimesh_faces.append(face_indices)
        except KeyError as e:
            error_msg_key_error = f"KeyError creating Poly_C surface faces. Vertex {e} (rounded) not found."
            print(f"ERROR ({poly_c_id_str}): {error_msg_key_error}")
            return {
                "is_valid_poly_c": False, "poly_c_id": poly_c_id_str, "error_message": error_msg_key_error,
                "visualization_data": {
                    "dv_center": center_coord.tolist() if center_coord is not None else [0.,0.,0.],
                    "base_triangles_coords_raw": [[c.tolist() for c in tri] for tri in surface_base_triangles_coords],
                    "error_message": error_msg_key_error,
                    "lin_points": {k: v.tolist() for k,v in lin_map.items()} if lin_map else {},
                    "mob_points": {k: v['coord'].tolist() for k, v in mob_data_map.items()} if mob_data_map else {}
                }
            }


    is_poly_c_valid = False 
    first_intersecting_segment_details = None 
    poly_c_surface_mesh = None
    error_msg_verification = ""

    if not surface_trimesh_faces: 
        error_msg_verification = "No faces for trimesh."
    else:
        try:
            poly_c_surface_mesh = trimesh.Trimesh(vertices=surface_trimesh_vertices,
                                                  faces=np.array(surface_trimesh_faces),
                                                  process=True)
        except Exception as e:
            error_msg_verification = f"Trimesh creation failed: {e}"
            print(f"ERROR ({poly_c_id_str}): {error_msg_verification}")
            return {
                "is_valid_poly_c": False, "poly_c_id": poly_c_id_str, "error_message": error_msg_verification,
                "visualization_data": {
                    "dv_center": center_coord.tolist() if center_coord is not None else [0.,0.,0.],
                    "base_triangles_coords_raw": [[c.tolist() for c in tri] for tri in surface_base_triangles_coords],
                    "trimesh_vertices": surface_trimesh_vertices.tolist(), 
                    "trimesh_faces": np.array(surface_trimesh_faces).tolist() if surface_trimesh_faces else [], # Ensure faces are list if empty
                    "error_message": error_msg_verification,
                    "lin_points": {k: v.tolist() for k,v in lin_map.items()} if lin_map else {},
                    "mob_points": {k: v['coord'].tolist() for k, v in mob_data_map.items()} if mob_data_map else {}
                }
            }

        is_surface_watertight = poly_c_surface_mesh.is_watertight
        V_surf, F_surf = len(poly_c_surface_mesh.vertices), len(poly_c_surface_mesh.faces)
        E_surf = len(poly_c_surface_mesh.edges_unique)
        euler_char = V_surf - E_surf + F_surf
        expected_V_surf, expected_F_surf, expected_E_surf, expected_euler_surf = 14, 24, 36, 2 
        surface_topology_correct = (np.isclose(V_surf, expected_V_surf) and 
                                    np.isclose(F_surf, expected_F_surf) and
                                    np.isclose(E_surf, expected_E_surf) and 
                                    np.isclose(euler_char, expected_euler_surf))
        surface_valid = is_surface_watertight 
        if not is_surface_watertight: error_msg_verification += "Not watertight. "
        if not surface_topology_correct: error_msg_verification += f"Topology incorrect (V={V_surf},E={E_surf},F={F_surf},Euler={euler_char:.0f}). "
        
        polyhedron_geometry_correct = True 
        if F_surf > 0 :
            for i in range(len(surface_base_triangles_coords)):
                triangle_i_coords = surface_base_triangles_coords[i] 
                barycenter_i = calculate_barycenter(*triangle_i_coords)
                segment_start = center_coord 
                segment_end = barycenter_i
                for j in range(len(surface_base_triangles_coords)):
                    if i == j: continue 
                    triangle_j_coords = surface_base_triangles_coords[j]
                    if segment_triangle_intersect(segment_start, segment_end,
                                                  triangle_j_coords[0], triangle_j_coords[1], triangle_j_coords[2]):
                        polyhedron_geometry_correct = False
                        error_msg_verification += f"Self-intersection: center-bary(tri_{i}) intersects tri_{j}. "
                        if first_intersecting_segment_details is None: 
                            first_intersecting_segment_details = {
                                "segment": (segment_start.tolist(), segment_end.tolist()),
                                "triangle_i_coords": [c.tolist() for c in triangle_i_coords],
                                "triangle_j_coords": [c.tolist() for c in triangle_j_coords]
                            }
                        break 
                if not polyhedron_geometry_correct: break 
        elif F_surf == 0: 
            polyhedron_geometry_correct = False 
            error_msg_verification += "No faces for geometry check. "
        is_poly_c_valid = surface_valid and polyhedron_geometry_correct and surface_topology_correct
    
    if is_poly_c_valid: print(f"✅ {poly_c_id_str} is valid.")
    else: print(f"⚠️ {poly_c_id_str} has issues: {error_msg_verification}")

    vis_data_dict = {
        "dv_center": center_coord.tolist() if center_coord is not None else [0.,0.,0.],
        "base_triangles_coords_raw": [[c.tolist() for c in tri] for tri in surface_base_triangles_coords],
        "trimesh_vertices": poly_c_surface_mesh.vertices.tolist() if poly_c_surface_mesh and hasattr(poly_c_surface_mesh, 'vertices') and poly_c_surface_mesh.vertices is not None else [],
        "trimesh_faces": poly_c_surface_mesh.faces.tolist() if poly_c_surface_mesh and hasattr(poly_c_surface_mesh, 'faces') and poly_c_surface_mesh.faces is not None else [],
        "intersecting_segment_info": first_intersecting_segment_details,
        "lin_points": {k: v.tolist() for k,v in lin_map.items()} if lin_map else {},
        "mob_points": {k: v['coord'].tolist() for k, v in mob_data_map.items()} if mob_data_map else {},
        "error_message": error_msg_verification if not is_poly_c_valid else ""
    }
    
    debug_data_dict = {}
    slice_possible = True
    if not ('all_lin_X_neg' in locals() and all_lin_X_neg is not None and \
            'all_mob_pts' in locals() and all_mob_pts is not None):
        slice_possible = False
    
    if slice_possible:
        # Ensure indices for slicing are within bounds
        min_idx_slice = [idx - 1 for idx in [cx, cy, cz]]
        max_idx_slice_exclusive = [idx + 2 for idx in [cx, cy, cz]] # Slice is exclusive at end

        if not (all(m >= 0 for m in min_idx_slice) and \
                max_idx_slice_exclusive[0] <= grid_dims_config[0] and \
                max_idx_slice_exclusive[1] <= grid_dims_config[1] and \
                max_idx_slice_exclusive[2] <= grid_dims_config[2]):
            slice_possible = False

    if slice_possible:
        debug_data_dict["all_lin_X_neg_sample"] = all_lin_X_neg[cx-1:cx+2, cy-1:cy+2, cz-1:cz+2].tolist()
        debug_data_dict["all_mob_pts_sample"] = all_mob_pts[cx-1:cx+2, cy-1:cy+2, cz-1:cz+2].tolist()
    else:
        debug_data_dict["all_lin_X_neg_sample"] = "N/A (Indices out of bounds or data missing for sample)"
        debug_data_dict["all_mob_pts_sample"] = "N/A (Indices out of bounds or data missing for sample)"


    return {
        "is_valid_poly_c": is_poly_c_valid,
        "poly_c_id": poly_c_id_str,
        "error_message": error_msg_verification if not is_poly_c_valid else "",
        "visualization_data": vis_data_dict,
        "debug_data": debug_data_dict
    }

# --- Visualization Functions (largely unchanged, ensure 'dv_center' key is handled) ---
def visualize_single_intersection_and_pause(intersection_info, center_coord_list, 
                                            min_dimension_for_radius,
                                            cylinder_radius_factor=0.01):
    center_coord = np.array(center_coord_list)
    poly_c_center_str = f"sv_C ({center_coord[0]:.2f}, {center_coord[1]:.2f}, {center_coord[2]:.2f})" 
    print(f"  Displaying specific intersecting triangles and segment for Poly_C at {poly_c_center_str}. Close window to continue.")
    scene = trimesh.Scene()

    def add_triangle_to_scene(tri_coords_list, color, scene_obj, name="triangle"):
        try:
            vertices = np.array(tri_coords_list)
            if vertices.shape != (3,3): return
            faces = np.array([[0, 1, 2]])
            mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            mesh.visual.face_colors = color
            scene_obj.add_geometry(mesh, geom_name=name)
        except Exception as e: print(f"    Warning: Could not create mesh for {name}: {e}")

    tri_i_coords = intersection_info.get("triangle_i_coords")
    if tri_i_coords: add_triangle_to_scene(tri_i_coords, [255, 255, 0, 200], scene, name="Triangle_i") 
    tri_j_coords = intersection_info.get("triangle_j_coords")
    if tri_j_coords: add_triangle_to_scene(tri_j_coords, [255, 165, 0, 200], scene, name="Triangle_j") 
    seg_start_list, seg_end_list = intersection_info["segment"]
    seg_start, seg_end = np.array(seg_start_list), np.array(seg_end_list)
    if np.linalg.norm(seg_start - seg_end) > PYRAYMESH_ADAPT_EPSILON:
        radius = min_dimension_for_radius * cylinder_radius_factor
        try:
            segment_cyl = trimesh.creation.cylinder(radius=max(radius, PYRAYMESH_ADAPT_EPSILON*10), segment=[seg_start, seg_end], sections=8)
            segment_cyl.visual.face_colors = [255, 0, 0, 200]
            scene.add_geometry(segment_cyl, geom_name="Intersecting_Segment")
        except Exception as e: print(f"    Warning: Could not create cylinder for segment: {e}")
    if scene.is_empty: return
    scene.show()
    time.sleep(0.1)

def visualize_poly_c_with_highlight_and_pause(vis_data_single_poly_c,
                                              min_dimension_for_radius,
                                              cylinder_radius_factor=0.01):
    center_coord_list = vis_data_single_poly_c.get('dv_center') 
    if center_coord_list is None or len(center_coord_list) != 3:
        center_coord_for_name = np.array([0.0, 0.0, 0.0])
    else:
        center_coord_for_name = np.array(center_coord_list)
    poly_c_center_str = f"sv_C ({center_coord_for_name[0]:.2f}, {center_coord_for_name[1]:.2f}, {center_coord_for_name[2]:.2f})" 
    print(f"  Displaying full Poly_C at {poly_c_center_str}. Close window to continue.")
    scene = trimesh.Scene()
    base_triangles_coords_list_of_lists = vis_data_single_poly_c.get("base_triangles_coords_raw", []) 
    intersection_info = vis_data_single_poly_c.get("intersecting_segment_info") 
    trimesh_verts_list = vis_data_single_poly_c.get("trimesh_vertices", [])
    trimesh_faces_list = vis_data_single_poly_c.get("trimesh_faces", [])
    poly_c_mesh_created = False
    if trimesh_verts_list and trimesh_faces_list:
        try:
            mesh_poly_c = trimesh.Trimesh(vertices=np.array(trimesh_verts_list), faces=np.array(trimesh_faces_list), process=False)
            mesh_poly_c.visual.face_colors = [170, 170, 200, 220] 
            scene.add_geometry(mesh_poly_c, geom_name=f"Poly_C_Surface_{poly_c_center_str}")
            poly_c_mesh_created = True
        except Exception as e: poly_c_mesh_created = False 
    if not poly_c_mesh_created and base_triangles_coords_list_of_lists:
        local_vertices, local_faces, local_face_colors = [], [], []
        vertex_to_index_map_local, next_vertex_index_local = {}, 0
        vis_rounding_precision = int(-np.log10(EPSILON_GEOMETRY)) + 3 
        def get_local_vertex_index(coord_list_fn): 
            nonlocal next_vertex_index_local, local_vertices, vertex_to_index_map_local 
            coord_tuple = tuple(np.round(np.array(coord_list_fn), vis_rounding_precision))
            if coord_tuple in vertex_to_index_map_local: return vertex_to_index_map_local[coord_tuple]
            else:
                local_vertices.append(list(coord_list_fn)); vertex_to_index_map_local[coord_tuple] = next_vertex_index_local 
                idx_to_return = next_vertex_index_local; next_vertex_index_local += 1; return idx_to_return
        highlight_tri_i_hash, highlight_tri_j_hash = None, None
        if intersection_info:
            if intersection_info.get("triangle_i_coords"): highlight_tri_i_hash = robust_triangle_hash(tuple(map(tuple,intersection_info["triangle_i_coords"])), vis_rounding_precision)
            if intersection_info.get("triangle_j_coords"): highlight_tri_j_hash = robust_triangle_hash(tuple(map(tuple,intersection_info["triangle_j_coords"])), vis_rounding_precision)
        for tri_coords_list_single in base_triangles_coords_list_of_lists: 
            if not tri_coords_list_single or len(tri_coords_list_single) != 3 or not all(v and len(v)==3 for v in tri_coords_list_single): continue
            try:
                tri_coords_for_hash = tuple(map(tuple, tri_coords_list_single))
                face_indices = [get_local_vertex_index(v) for v in tri_coords_list_single]
                local_faces.append(face_indices)
            except Exception: continue
            color_default, color_tri_i, color_tri_j = [170,170,200,220], [255,255,0,255], [255,165,0,255]
            current_tri_hash = robust_triangle_hash(tri_coords_for_hash, vis_rounding_precision)
            if highlight_tri_i_hash and current_tri_hash == highlight_tri_i_hash: local_face_colors.append(color_tri_i)
            elif highlight_tri_j_hash and current_tri_hash == highlight_tri_j_hash: local_face_colors.append(color_tri_j)
            else: local_face_colors.append(color_default)
        if local_vertices and local_faces:
            try:
                mesh_poly_c_raw = trimesh.Trimesh(vertices=np.array(local_vertices), faces=np.array(local_faces), process=False)
                if len(local_face_colors) == len(mesh_poly_c_raw.faces): mesh_poly_c_raw.visual.face_colors = np.array(local_face_colors)
                scene.add_geometry(mesh_poly_c_raw, geom_name=f"Poly_C_Surface_Raw_{poly_c_center_str}")
                poly_c_mesh_created = True
            except Exception: pass
    if intersection_info:
        seg_start_list, seg_end_list = intersection_info["segment"]
        seg_start, seg_end = np.array(seg_start_list), np.array(seg_end_list)
        if np.linalg.norm(seg_start - seg_end) > PYRAYMESH_ADAPT_EPSILON:
            radius = min_dimension_for_radius * cylinder_radius_factor
            try:
                segment_cyl = trimesh.creation.cylinder(radius=max(radius, PYRAYMESH_ADAPT_EPSILON*10), segment=[seg_start, seg_end], sections=8)
                segment_cyl.visual.face_colors = [255,0,0,200]
                scene.add_geometry(segment_cyl, geom_name=f"Intersecting_Segment_{poly_c_center_str}")
            except Exception: pass
    if not poly_c_mesh_created and not (intersection_info and scene.geometry): return 
    scene.show() 
    time.sleep(0.1)

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Running Main Block (Simplified Algorithm) ---")
    grid_dims_main = (5, 5, 5) 
    grid_spacings_main = (1.0, 1.0, 1.0)

    if any(d < 4 for d in grid_dims_main):
        print("ERROR: grid_dims must be at least (4,4,4) for chosen center index logic.")
        exit()
        
    center_indices_main = tuple(d // 2 for d in grid_dims_main)
    print(f"Selected center indices: {center_indices_main} for grid {grid_dims_main}")

    result = run_algorithm_and_verify_poly_c(
        dv_c_indices=center_indices_main, 
        grid_dims_config=grid_dims_main,
        grid_spacings_config=grid_spacings_main
    )
        
    print(f"\n--- Summary ---")
    print(f"Poly_C ID: {result.get('poly_c_id', 'N/A')}")
    print(f"Is Valid Poly_C: {result.get('is_valid_poly_c', False)}")
    if not result.get('is_valid_poly_c', False):
        print(f"Error Message: {result.get('error_message', 'No error message provided.')}")
    
    vis_data_res = result.get('visualization_data', {})
    if vis_data_res.get('intersecting_segment_info'):
        print("Found intersecting segment (center to barycenter heuristic).")
    
    if result.get('is_valid_poly_c', False):
        print(f"Poly_C successfully generated and verified.")
    elif vis_data_res.get('base_triangles_coords_raw'):
         print(f"Poly_C generated but has issues. Mesh data potentially available.")
    else:
        print(f"Poly_C could not be fully generated or has critical errors early on.")
    print("---------------------------------------\n")

    if result and result.get("visualization_data"):
        print(f"\n--- Starting Trimesh Visualization for the generated Poly_C ---")
        min_spacing_main = min(grid_spacings_main) if grid_spacings_main else 1.0
        
        vis_data_main = result["visualization_data"]
        intersection_details_main = vis_data_main.get("intersecting_segment_info") 
        center_for_current_poly_c_list_main = vis_data_main.get("dv_center")

        if center_for_current_poly_c_list_main is None or not center_for_current_poly_c_list_main :
            print(f"Skipping visualization for Poly_C {result.get('poly_c_id', 'Unknown ID')} due to missing 'dv_center'.")
        else:
            print(f"\nTrimesh Visualizing Poly_C ({result.get('poly_c_id', 'Unknown ID')})")
            if intersection_details_main:
                print(f"  Intersection detected. Visualizing problematic elements...")
                visualize_single_intersection_and_pause(
                    intersection_info=intersection_details_main,
                    center_coord_list=center_for_current_poly_c_list_main, 
                    min_dimension_for_radius=min_spacing_main
                )
            print(f"  Now preparing visualization for the full Poly_C...")
            visualize_poly_c_with_highlight_and_pause(
                vis_data_single_poly_c=vis_data_main, 
                min_dimension_for_radius=min_spacing_main
            )
        print("\n--- Poly_C Trimesh visualization prepared/attempted. ---")
    else:
        print("No Poly_C results were generated to visualize.")

