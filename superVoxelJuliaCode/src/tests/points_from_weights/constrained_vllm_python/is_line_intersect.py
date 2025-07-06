import numpy as np
from typing import List, Tuple, Dict

# --- Constants ---
# Epsilon for geometric floating-point comparisons
EPSILON_GEOMETRY = 1e-9

# --- Helper Functions (from your provided code) ---

def calculate_barycenter(v0: Tuple[float, ...], v1: Tuple[float, ...], v2: Tuple[float, ...]) -> np.ndarray:
    """
    Calculates the barycenter (centroid) of a triangle defined by 3 vertices.

    Args:
        v0, v1, v2: The three 3D vertices of the triangle.

    Returns:
        np.ndarray: The 3D barycenter of the triangle.
    """
    return (np.array(v0) + np.array(v1) + np.array(v2)) / 3.0

def segment_triangle_intersect(P0: Tuple[float, ...], P1: Tuple[float, ...], T0: Tuple[float, ...], T1: Tuple[float, ...], T2: Tuple[float, ...]) -> bool:
    """
    Checks if a line segment (P0-P1) intersects a triangle (T0-T1-T2).
    Uses a modified Möller-Trumbore algorithm for segment-triangle intersection.

    Args:
        P0: Start point of the segment.
        P1: End point of the segment.
        T0: First vertex of the triangle.
        T1: Second vertex of the triangle.
        T2: Third vertex of the triangle.

    Returns:
        bool: True if the segment intersects the triangle, False otherwise.
    """
    # Convert tuples to numpy arrays for vector operations
    P0, P1 = np.array(P0), np.array(P1)
    T0, T1, T2 = np.array(T0), np.array(T1), np.array(T2)
    
    # Segment's direction vector
    D = P1 - P0
    # Triangle edge vectors
    E1 = T1 - T0
    E2 = T2 - T0
    
    # Calculate determinant
    Pvec = np.cross(D, E2)
    det = np.dot(E1, Pvec)

    # If determinant is near zero, ray is parallel to triangle plane
    if abs(det) < EPSILON_GEOMETRY:
        return False
        
    invDet = 1.0 / det
    
    # Calculate u parameter
    Tvec = P0 - T0
    u = np.dot(Tvec, Pvec) * invDet
    # Check if u is outside the triangle bounds
    if u < -EPSILON_GEOMETRY or u > 1.0 + EPSILON_GEOMETRY:
        return False
        
    # Calculate v parameter
    Qvec = np.cross(Tvec, E1)
    v = np.dot(D, Qvec) * invDet
    # Check if v is outside the triangle bounds
    if v < -EPSILON_GEOMETRY or (u + v) > 1.0 + EPSILON_GEOMETRY:
        return False
        
    # Calculate t parameter (distance along the segment)
    t_ray = np.dot(E2, Qvec) * invDet
    
    # An intersection occurs if t is between 0 and 1 (inclusive).
    # This means the intersection point lies on the segment P0-P1.
    if -EPSILON_GEOMETRY <= t_ray <= 1.0 + EPSILON_GEOMETRY:
        # Additional check for degenerate segment (P0 == P1)
        if np.dot(D, D) < EPSILON_GEOMETRY**2:
            # If segment is a point, check if it's inside the triangle
            return u >= -EPSILON_GEOMETRY and v >= -EPSILON_GEOMETRY and (u + v) <= 1.0 + EPSILON_GEOMETRY
        return True
        
    return False

# --- Main Function ---

def find_star_convexity_intersections(
    triangles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]],
    sv_center: Tuple[float, float, float]
) -> Dict[int, List[int]]:
    """
    Checks if a polyhedron, defined by a list of triangles, is star-convex with respect
    to a central point (sv_center).

    It does this by creating a line segment from the sv_center to the barycenter of each
    triangle and verifying that this segment does not intersect any *other* triangle in the list.

    Args:
        triangles: A list of triangles. Each triangle is a tuple of 3 vertices,
                   and each vertex is a tuple of 3 float coordinates (x, y, z).
                   The user prompt mentioned a tuple of length 4, but 3D coordinates
                   are assumed here.
        sv_center: The coordinates of the central point (x, y, z) to check
                   star-convexity from.

    Returns:
        A dictionary where each key is the index of a "source" triangle whose
        line-to-barycenter causes an intersection. The corresponding value is a list
        of indices of the triangles that this line intersects.
        Returns an empty dictionary if no intersections are found (i.e., the shape
        is star-convex with respect to the center point).
    """
    intersections = {}
    num_triangles = len(triangles)

    for i in range(num_triangles):
        # Get the vertices of the source triangle
        tri_i_v0, tri_i_v1, tri_i_v2 = triangles[i]

        # Calculate its barycenter
        barycenter_i = calculate_barycenter(tri_i_v0, tri_i_v1, tri_i_v2)

        # Define the line segment from the center to the barycenter
        segment_start = sv_center
        segment_end = barycenter_i

        # Check this segment against all other triangles
        for j in range(num_triangles):
            # A segment cannot intersect its own triangle
            if i == j:
                continue

            # Get the vertices of the target triangle to check against
            tri_j_v0, tri_j_v1, tri_j_v2 = triangles[j]

            # If the segment intersects another triangle, log the error
            if segment_triangle_intersect(segment_start, segment_end, tri_j_v0, tri_j_v1, tri_j_v2):
                if i not in intersections:
                    intersections[i] = []
                intersections[i].append(j)

    return intersections


def check_star_convexity(cell_data):
    """Helper function to check a cell for star-convexity violations."""
    center = tuple(cell_data['control_points']['svCenter'])
    # A cell with no triangles cannot have intersections.
    if not cell_data['triangles']:
        return {}
    # Convert numpy arrays to tuples for the check
    triangles = [tuple(map(tuple, tri)) for tri in cell_data['triangles']]
    return find_star_convexity_intersections(triangles, center)


# --- Example Usage ---
if __name__ == '__main__':
    # --- Test Case 1: A valid, simple tetrahedron (should have no intersections) ---
    print("--- Testing a valid tetrahedron (star-convex shape) ---")
    center_point_valid = (0.0, 0.0, 0.0)
    tetra_vertices = [
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0)
    ]
    # Faces of the tetrahedron
    valid_triangles = [
        (tetra_vertices[0], tetra_vertices[2], tetra_vertices[1]),
        (tetra_vertices[0], tetra_vertices[3], tetra_vertices[2]),
        (tetra_vertices[0], tetra_vertices[1], tetra_vertices[3]),
        (tetra_vertices[1], tetra_vertices[2], tetra_vertices[3])
    ]

    intersection_results_valid = find_star_convexity_intersections(valid_triangles, center_point_valid)
    print(f"Intersection dictionary: {intersection_results_valid}")
    if not intersection_results_valid:
        print("Result: OK. No intersections found as expected.\n")
    else:
        print("Result: FAIL. Unexpected intersections found.\n")


    # --- Test Case 2: An invalid, non-convex shape (should have intersections) ---
    print("--- Testing an invalid shape (non-star-convex) ---")
    center_point_invalid = (0.0, 0.5, 0.0)
    # A base, a "spike" and a "folding back" triangle that will cause intersection
    invalid_triangles = [
        # Base triangle
        ((-2.0, 0.0, -2.0), (2.0, 0.0, -2.0), (0.0, 0.0, 2.0)), # Index 0
        # Left side
        ((-2.0, 0.0, -2.0), (0.0, 0.0, 2.0), (0.0, 2.0, 0.0)), # Index 1
        # Right side that folds back over the base
        ((2.0, 0.0, -2.0), (0.0, 0.0, 2.0), (0.0, -1.0, 0.0))  # Index 2
    ]
    
    # The line from the center to the barycenter of triangle 1 should intersect triangle 2.
    intersection_results_invalid = find_star_convexity_intersections(invalid_triangles, center_point_invalid)
    print(f"Intersection dictionary: {intersection_results_invalid}")
    if intersection_results_invalid:
        print("Result: OK. Intersections found as expected.")
    else:
        print("Result: FAIL. No intersections were found.")

