import numpy as np
import trimesh
import colorsys

# This more specific import is needed for the robust text creation method
try:
    from trimesh.path.entities import Text
    from trimesh.path.path import Path2D
    TEXT_SUPPORT = True
except ImportError:
    TEXT_SUPPORT = False
    print("Warning: Could not import text creation modules from trimesh.path.")
    print("Text labels for points will not be displayed.")
    print("Consider `pip install trimesh[easy]` for full functionality.")


def visualize_geometry(triangles_list, points_dict=None):
    """
    Visualizes a mesh from triangles and a set of named points.

    Each triangle is given a unique, contrasting color. Each point is
    represented by a yellow sphere with a text label for its name.

    Args:
        triangles_list (list): A list of triangles, where each triangle is a
                               tuple of 3 points (each point is a tuple of
                               3 floats).
        points_dict (dict, optional): A dictionary where keys are point names
                                      (str) and values are their 3D coordinates
                                      (tuple of 3 floats). Defaults to None.
    """
    # Initialize the scene
    scene = trimesh.Scene()

    # --- Process and Add Triangles ---
    if triangles_list:
        num_triangles = len(triangles_list)
        
        # Generate a set of visually distinct colors for the triangles
        colors = []
        for i in range(num_triangles):
            hue = i / num_triangles
            rgb_float = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            colors.append([int(c * 255) for c in rgb_float] + [255])

        # Add each triangle to the scene with its unique color
        for i, tri_coords in enumerate(triangles_list):
            try:
                vertices = np.array(tri_coords)
                if vertices.shape != (3, 3):
                    print(f"Skipping triangle {i} due to incorrect shape: {vertices.shape}")
                    continue

                faces = np.array([[0, 1, 2]])
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
                mesh.visual.face_colors = colors[i]
                scene.add_geometry(mesh, geom_name=f"triangle_{i}")

            except Exception as e:
                print(f"Could not create mesh for triangle {i}: {e}")
    else:
        print("The list of triangles is empty.")

    # --- Process and Add Points with Labels (Legend) ---
    if points_dict:
        point_color = [255, 255, 0, 255]  # Bright yellow for visibility
        
        print("\nVisualizing Points (Legend):")
        for name, coords in points_dict.items():
            try:
                # Ensure coords is a numpy array of float
                coords_np = np.array(coords, dtype=float)
                print(f"- {name}: {tuple(coords_np)}")
                # Create a small sphere to represent the point
                point_sphere = trimesh.creation.icosphere(subdivisions=2, radius=0.1)
                point_sphere.visual.face_colors = point_color
                
                transform = trimesh.transformations.translation_matrix(coords_np)
                point_sphere.apply_transform(transform)
                scene.add_geometry(point_sphere, geom_name=name)

                # Create 3D text only if the required modules were imported
                if TEXT_SUPPORT:
                    text_color = [255, 255, 255, 255] # White for text

                    # Create a 2D path object containing the text.
                    text_path = Path2D(
                        entities=[Text(text=name, height=0.2, origin=0)],
                        vertices=[[0, 0]]
                    )
                    
                    # Extrude the 2D text path into a 3D mesh
                    text_geom = text_path.extrude(height=0.02) # Give text some thickness

                    # The extruded mesh is created at the origin. We need to move it.
                    text_bounds = text_geom.bounds
                    text_center = text_bounds.mean(axis=0)
                    
                    # Position text slightly above and to the side of the point sphere
                    text_transform = trimesh.transformations.translation_matrix(
                        coords_np + np.array([0.15, 0, 0.15]) - text_center
                    )
                    text_geom.apply_transform(text_transform)
                    text_geom.visual.face_colors = text_color
                    scene.add_geometry(text_geom, geom_name=f"{name}_label")

            except Exception as e:
                print(f"Could not create visualization for point '{name}': {e}")


    # --- Show the Scene ---
    if scene.is_empty:
        print("\nScene is empty. No valid geometry was added.")
        return

    print(f"\nDisplaying {len(scene.geometry)} geometries. Close the window to exit.")
    scene.show()

# # --- Example Usage ---
# if __name__ == '__main__':
#     # 1. Create a list of 72 random triangles for demonstration
#     example_triangles = []
#     for _ in range(72):
#         triangle = tuple(tuple(np.random.rand(3) * 10) for _ in range(3))
#         example_triangles.append(triangle)

#     # 2. Create a dictionary of named points
#     example_points = {
#         'Origin': (0, 0, 0),
#         'Center': (5, 5, 5),
#         'Peak': (5, 10, 5),
#         'Corner': (10, 0, 10)
#     }

#     # 3. Visualize both the triangles and the points
#     visualize_geometry(example_triangles, example_points)
