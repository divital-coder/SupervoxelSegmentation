#
# A full Python implementation for Statistical Shape Variability Analysis
# based on the morphomatics library.
#
# This script consolidates the expert-level workflow described in the provided
# technical guides. It demonstrates a complete pipeline from raw triangle data
# to the construction and interpretation of a Statistical Shape Model (SSM).
#
# Required Dependencies:
# pip install numpy
# pip install trimesh
# pip install pyvista
# pip install morphomatics
#
# Note: morphomatics relies on JAX. If you have a compatible GPU, you can
# install JAX with GPU support for significant performance improvements:
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#

import os
import shutil
import glob
import numpy as np
import trimesh
import pyvista as pv
from morphomatics.geom import Surface
from morphomatics.manifold import DifferentialCoords
from morphomatics.stats import StatisticalShapeModel
from typing import List, Tuple

# ---
# Stage 1: From Raw Data to a Structured Mesh Object
# ---

def create_trimesh_from_raw_triangles(raw_triangles: List[Tuple[Tuple]]) -> trimesh.Trimesh:
    """
    Converts a list of triangles, where each triangle is a set of three 3D points,
    into a trimesh.Trimesh object.

    This function is critical for handling the user's input format. It performs
    the essential steps of vertex deduplication and face re-indexing to create
    a standardized mesh structure (vertices and faces arrays).

    Args:
        raw_triangles (list of tuples):
            A list of triangles, e.g., [[(x1,y1,z1), (x2,y2,z2), (x3,y3,z3)],...].

    Returns:
        trimesh.Trimesh: A mesh object representing the input geometry.
    """
    vertices = []
    faces = []
    vertex_map = {} # Tracks unique vertices and their new indices

    # Iterate through each triangle in the raw data
    for triangle in raw_triangles:
        face_indices = []
        # Iterate through each vertex of the triangle
        for vertex_coords in triangle:
            vertex_tuple = tuple(vertex_coords) # Make hashable for dictionary keys

            # If we haven't seen this vertex, add it to our list and record its index.
            if vertex_tuple not in vertex_map:
                vertex_map[vertex_tuple] = len(vertices)
                vertices.append(list(vertex_tuple))

            # Append the index of the current vertex to this face's list
            face_indices.append(vertex_map[vertex_tuple])

        faces.append(face_indices)

    # Convert lists to NumPy arrays
    vertices_np = np.array(vertices, dtype=np.float64)
    faces_np = np.array(faces, dtype=np.int64)

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=faces_np, process=True)
    return mesh

# ---
# Stage 2: Persisting and Staging Mesh Data
# ---

def save_mesh_cohort(list_of_meshes: List[trimesh.Trimesh], output_dir: str = "data/meshes"):
    """
    Saves a list of trimesh.Trimesh objects as .ply files in a specified directory.
    This intermediate file-based step is recommended for modularity, validation,
    and alignment with the official morphomatics workflow.

    Args:
        list_of_meshes (list): A list of trimesh.Trimesh objects.
        output_dir (str): The directory where .ply files will be saved.
    """
    # Clean up previous runs
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i, mesh_obj in enumerate(list_of_meshes):
        # Using zfill for consistent sorting, e.g., shape_01.ply, shape_02.ply
        filename = f"shape_{str(i+1).zfill(2)}.ply"
        filepath = os.path.join(output_dir, filename)
        # The export method handles the file format based on the extension.
        mesh_obj.export(filepath)
    print(f"Successfully saved {len(list_of_meshes)} meshes to '{output_dir}'.")


def load_staged_meshes_as_surfaces(input_dir: str = "data/meshes") -> List[Surface]:
    """
    Loads all .ply files from a directory, converts them into morphomatics.geom.Surface
    objects, and validates their correspondence. This function combines loading
    and the final conversion step.

    Args:
        input_dir (str): The directory containing the .ply mesh files.

    Returns:
        list: A list of morphomatics.geom.Surface objects ready for analysis.
    """
    # Find all files ending with .ply in the specified directory
    mesh_files = sorted(glob.glob(os.path.join(input_dir, "*.ply")))
    if not mesh_files:
        raise FileNotFoundError(f"No .ply files found in '{input_dir}'.")

    surface_list = []
    reference_faces = None

    for f_path in mesh_files:
        # Load the mesh using pyvista, the recommended bridge to morphomatics
        pv_mesh = pv.read(f_path)
        
        # Extract vertices directly.
        vertices = pv_mesh.points.astype(np.float64)
        
        # Extract and reformat faces. PyVista's faces array is 1D and padded.
        try:
            faces = pv_mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int64)
        except ValueError:
            raise ValueError(f"Mesh in {f_path} does not appear to be triangular.")

        # --- CRITICAL VALIDATION: Ensure Correspondence ---
        # All meshes must have the same connectivity (face matrix).
        if reference_faces is None:
            reference_faces = faces
        elif not np.array_equal(reference_faces, faces):
            raise ValueError(
                f"Mesh correspondence error: {f_path} has different connectivity "
                "from the first mesh. All meshes must have the same topology."
            )

        # Instantiate the morphomatics Surface object and add to list
        surface_list.append(Surface(v=vertices, f=faces))

    print(f"Successfully loaded {len(surface_list)} meshes and converted to morphomatics Surfaces.")
    return surface_list


# ---
# Stage 3 & 4: Analysis and Interpretation
# ---

def perform_shape_variability_analysis(
    raw_mesh_cohort: List[List[Tuple[Tuple]]], temp_directory: str
) -> Tuple[StatisticalShapeModel, List[Surface], float]:
    """
    Executes the full statistical shape variability analysis pipeline.

    CRITICAL ASSUMPTION: This function assumes the input meshes are in
    point-to-point correspondence. This means each mesh must be a deformation
    of a common template, having the same number of vertices and the same
    face connectivity. The data generation process must ensure this.

    Args:
        raw_mesh_cohort (list): A list where each element is a raw triangle
                                list for one mesh.

    Returns:
        tuple: A tuple containing:
            - ssm (StatisticalShapeModel): The constructed model object.
            - surfaces (list): The list of input surfaces.
            - total_variability (float): The sum of all mode variances, a single
                                         number summarizing the total shape variability.
    """
    # Stage 1: Convert all raw mesh data to Trimesh objects
    print("--- STAGE 1: Converting raw data to Trimesh objects ---")
    trimesh_cohort = [create_trimesh_from_raw_triangles(raw_mesh) for raw_mesh in raw_mesh_cohort]

    # Stage 2: Persist to disk and reload as morphomatics Surfaces
    print("\n--- STAGE 2: Saving and reloading meshes ---")
    save_mesh_cohort(trimesh_cohort, output_dir=temp_directory)
    surfaces = load_staged_meshes_as_surfaces(input_dir=temp_directory)

    if len(surfaces) < 2:
        print("Error: At least two surfaces are required for statistical analysis.")
        return None, None, -1.0

    # Stage 3: Define the Shape Space for Analysis
    print("\n--- STAGE 3: Defining the Shape Space ---")
    # We choose `DifferentialCoords`. As per the technical guide, this is the
    # superior choice for a scale-sensitive analysis of "form" (size-and-shape)
    # as it preserves scale information within a non-linear Riemannian framework.
    space_def = lambda ref: DifferentialCoords(ref)
    print("Using 'DifferentialCoords' for a scale-sensitive, non-linear analysis.")
    
    # Stage 4: Construct the Statistical Shape Model (SSM)
    print("\n--- STAGE 4: Constructing the Statistical Shape Model ---")
    ssm = StatisticalShapeModel(space_def)
    # The .construct() method performs the complex optimization to find the
    # intrinsic mean and principal modes of variation (Principal Geodesic Analysis).
    print("Constructing the model... This may take some time.")
    ssm.construct(surfaces)
    print("Model construction complete.")

    # Calculate the summary number for total variability
    total_variability = np.sum(ssm.variances)

    # Clean up intermediate files
    shutil.rmtree(temp_directory)
    
    return ssm, surfaces, total_variability


def visualize_ssm_mode(ssm: StatisticalShapeModel, surfaces: List[Surface], mode_index: int = 0):
    """
    Generates and displays meshes representing the variation along a specific mode.
    This function visualizes the mean shape and shapes at +/- 2 standard deviations.
    """
    if not ssm:
        print("SSM model is not available for visualization.")
        return

    print(f"\nVisualizing variation along mode {mode_index}...")
    
    # Calculate the standard deviation for the chosen mode
    std_dev = np.sqrt(ssm.variances[mode_index])
    
    # Get the mean coordinates in the shape space and the mode vector
    mean_coords = ssm.mean_coords
    mode_vector = ssm.modes[mode_index]
    
    # Set up the plotter for 3 side-by-side plots
    plotter = pv.Plotter(shape=(1, 3), window_size=[1800, 600])
    
    # --- FIX: Manually create the PyVista-compatible face array ---
    # The ssm.mean.f attribute is an (M, 3) array of faces. PyVista requires
    # a 1D array where each face is padded with the number of vertices (3).
    # We create this once since all meshes share the same connectivity.
    faces = ssm.mean.f
    pyvista_faces = np.hstack((np.full((faces.shape[0], 1), 3), faces)).ravel()

    # --- Generate and plot shape at -2 standard deviations ---
    plotter.subplot(0, 0)
    plotter.add_text(f"Mean - 2*SD (Mode {mode_index})", font_size=20)
    # Use the exponential map to traverse the manifold from the mean
    coords_neg = ssm.space.connec.exp(mean_coords, -2 * std_dev * mode_vector)
    # Convert shape space coordinates back to vertex coordinates
    v_neg = ssm.space.from_coords(coords_neg)
    mesh_neg = pv.PolyData(v_neg, pyvista_faces)
    plotter.add_mesh(mesh_neg, color='cornflowerblue', show_edges=True)
    plotter.view_isometric()

    # --- Plot the mean shape ---
    plotter.subplot(0, 1)
    plotter.add_text("Mean Shape", font_size=20)
    mesh_mean = pv.PolyData(ssm.mean.v, pyvista_faces)
    plotter.add_mesh(mesh_mean, color='lightgrey', show_edges=True)
    plotter.view_isometric()

    # --- Generate and plot shape at +2 standard deviations ---
    plotter.subplot(0, 2)
    plotter.add_text(f"Mean + 2*SD (Mode {mode_index})", font_size=20)
    coords_pos = ssm.space.connec.exp(mean_coords, 2 * std_dev * mode_vector)
    v_pos = ssm.space.from_coords(coords_pos)
    mesh_pos = pv.PolyData(v_pos, pyvista_faces)
    plotter.add_mesh(mesh_pos, color='salmon', show_edges=True)
    plotter.view_isometric()

    plotter.link_views() # Link cameras for easy comparison
    print(f"Displaying variation. Close the PyVista window to continue.")
    plotter.show()


# ---
# Test Cases and Main Execution
# ---

def generate_test_data() -> List[List[Tuple[Tuple]]]:
    """
    Generates a synthetic cohort of corresponding meshes with known variability.
    The variation is a simple stretching along the X-axis. This provides a
    perfect test case where the first mode of variation should capture this stretch.
    """
    print("Generating synthetic test data...")
    # A simple 2x2 square on the XY plane, made of two triangles
    p00 = (0., 0., 0.)
    p10 = (1., 0., 0.)
    p01 = (0., 1., 0.)
    p11 = (1., 1., 0.)
    
    base_raw_mesh = [
      (p00, p10, p01),  # Triangle 1
      (p10, p11, p01)   # Triangle 2
    ]
    base_vertices = np.array([p00, p10, p01, p11])

    cohort = []
    # Generate 10 meshes, stretching them progressively along the X-axis
    for i in range(10):
        scale_factor = 1.0 + (i * 0.1) # from 1.0 to 1.9
        
        # Create a transformation matrix to scale only the X coordinate
        transform = np.eye(4)
        transform[0, 0] = scale_factor
        
        # Apply transformation
        new_vertices = trimesh.transformations.transform_points(base_vertices, transform)
        
        # Reconstruct the raw triangle list from the new vertices.
        # This ensures all generated meshes have IDENTICAL connectivity,
        # satisfying the correspondence prerequisite.
        new_raw_mesh = [
            (tuple(new_vertices[0]), tuple(new_vertices[1]), tuple(new_vertices[2])),
            (tuple(new_vertices[1]), tuple(new_vertices[3]), tuple(new_vertices[2])),
        ]
        cohort.append(new_raw_mesh)
    
    print(f"Generated a cohort of {len(cohort)} meshes with controlled X-axis stretching.")
    return cohort


if __name__ == '__main__':
    # 1. Generate synthetic data that meets the correspondence prerequisite
    test_cohort_data = generate_test_data()
    
    # 2. Run the complete analysis pipeline
    ssm_model, surfaces, total_variability = perform_shape_variability_analysis(test_cohort_data,temp_directory)
    
    # 3. Print the results
    if ssm_model:
        print("\n\n--- SSM Analysis Results ---")
        print(f"Mean shape has {ssm_model.mean.v.shape[0]} vertices and {ssm_model.mean.f.shape[0]} faces.")
        print(f"Number of variation modes found: {len(ssm_model.modes)}")
        
        print("\nVariances (Eigenvalues) for each mode:")
        for i, var in enumerate(ssm_model.variances):
            print(f"  Mode {i}: {var:.6f}")

        print(f"\nTotal Shape Variability (Sum of Variances): {total_variability:.6f}")

        print("\nCoefficients for each input shape (projection onto modes):")
        # print(ssm_model.coeffs) # This can be very large, so we print the shape
        print(f"  Shape of coefficients matrix: {ssm_model.coeffs.shape} (num_shapes x num_modes)")

        # 4. Visualize the most significant modes of variation
        # Note: This will open an interactive PyVista window.
        # You can comment this out if you are running in a non-GUI environment.
        try:
            # Visualize the first (most significant) mode
            visualize_ssm_mode(ssm_model, surfaces, mode_index=0)
            
            # Visualize the second mode (which should be minor for this test case)
            if len(ssm_model.modes) > 1:
                visualize_ssm_mode(ssm_model, surfaces, mode_index=1)
        except Exception as e:
            print(f"\nCould not run visualization. This may be because you are in a non-GUI environment.")
            print(f"Error: {e}")
