import numpy as np
import trimesh
from typing import List, Tuple

# Define a small epsilon for floating-point comparisons to handle precision issues.
GEOMETRY_EPSILON = 1e-9

def is_mesh_watertight(
    triangles: List[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]]
) -> bool:
    """
    Checks if a mesh defined by a list of triangles is watertight.

    A watertight mesh is a closed manifold mesh, meaning it has no holes,
    each edge is shared by exactly two faces, and it encloses a volume.

    Args:
        triangles: A list of triangles. Each triangle is a tuple of three points,
                   and each point is a tuple of three floats (x, y, z).
                   Note: The user prompt mentioned 4-element tuples for points,
                   but 3D geometry requires 3-element tuples (x, y, z). This
                   function assumes 3D points.

    Returns:
        bool: True if the mesh is watertight, False otherwise. Returns False
              if the input is empty or if a valid mesh cannot be constructed.
    """
    if not triangles:
        print("Warning: The list of triangles is empty.")
        return False

    # --- Vertex and Face Creation ---
    # We convert the list of triangle coordinates into a list of unique vertices
    # and a list of faces that reference those vertices by index. This is the
    # standard format required by libraries like trimesh.

    unique_vertices_map = {}
    vertex_list = []
    face_list = []
    
    # Precision for rounding vertices to handle floating-point inaccuracies
    rounding_precision = int(-np.log10(GEOMETRY_EPSILON)) + 2

    for tri_coords in triangles:
        face_indices = []
        for vertex_coord in tri_coords:
            # Round the vertex coordinates to a consistent precision to ensure
            # that vertices that are extremely close are treated as identical.
            # This is crucial for creating a valid manifold mesh.
            v_tuple = tuple(np.round(np.array(vertex_coord), rounding_precision))

            if v_tuple not in unique_vertices_map:
                # If we haven't seen this vertex before, add it to our list
                # and store its index in our map.
                unique_vertices_map[v_tuple] = len(vertex_list)
                vertex_list.append(list(vertex_coord))
            
            # Add the index of the vertex to the current face.
            face_indices.append(unique_vertices_map[v_tuple])
        
        face_list.append(face_indices)

    # --- Trimesh Validation ---
    if not vertex_list or not face_list:
        print("Warning: Could not construct any vertices or faces from the input.")
        return False

    try:
        # Create a Trimesh object from the vertices and faces.
        # The 'process=True' argument tells trimesh to perform basic processing,
        # which includes building the mesh topology (edges, etc.).
        mesh = trimesh.Trimesh(vertices=np.array(vertex_list),
                               faces=np.array(face_list),
                               process=True)
        
        # The 'is_watertight' property checks if the mesh is a closed manifold.
        return mesh.is_watertight

    except Exception as e:
        # If trimesh fails to create the mesh (e.g., due to degenerate faces
        # or other invalid geometry), it's certainly not watertight.
        print(f"An error occurred during mesh creation: {e}")
        return False

def check_watertight(cell_data):
    """Helper function to check if a cell's mesh is watertight."""
    # A valid interior cell must have triangles to be watertight.
    if not cell_data['triangles']:
        return False
    # Convert numpy arrays to tuples for the check
    triangles = [tuple(map(tuple, tri)) for tri in cell_data['triangles']]
    return is_mesh_watertight(triangles)

# --- Example Usage ---
if __name__ == '__main__':
    # Example 1: A simple, watertight cube (defined by 12 triangles)
    watertight_cube_triangles = [
        # Front face
        ((0, 0, 0), (1, 1, 0), (1, 0, 0)),
        ((0, 0, 0), (0, 1, 0), (1, 1, 0)),
        # Back face
        ((0, 0, 1), (1, 0, 1), (1, 1, 1)),
        ((0, 0, 1), (1, 1, 1), (0, 1, 1)),
        # Left face
        ((0, 0, 0), (0, 1, 1), (0, 1, 0)),
        ((0, 0, 0), (0, 0, 1), (0, 1, 1)),
        # Right face
        ((1, 0, 0), (1, 1, 0), (1, 1, 1)),
        ((1, 0, 0), (1, 1, 1), (1, 0, 1)),
        # Bottom face
        ((0, 0, 0), (1, 0, 0), (1, 0, 1)),
        ((0, 0, 0), (1, 0, 1), (0, 0, 1)),
        # Top face
        ((0, 1, 0), (0, 1, 1), (1, 1, 1)),
        ((0, 1, 0), (1, 1, 1), (1, 1, 0))
    ]

    is_it_watertight = is_mesh_watertight(watertight_cube_triangles)
    print(f"Is the cube mesh watertight? -> {is_it_watertight}") # Expected: True
    
    print("-" * 20)

    # Example 2: A mesh with a hole (missing one triangle from the cube)
    non_watertight_triangles = watertight_cube_triangles[:-1]

    is_it_watertight = is_mesh_watertight(non_watertight_triangles)
    print(f"Is the incomplete mesh watertight? -> {is_it_watertight}") # Expected: False

    print("-" * 20)
    
    # Example 3: Empty list
    is_it_watertight = is_mesh_watertight([])
    print(f"Is an empty list watertight? -> {is_it_watertight}") # Expected: False

