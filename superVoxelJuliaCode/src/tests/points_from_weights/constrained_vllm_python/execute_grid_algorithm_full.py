
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
from algo_validation_utils import extract_step_blocks
from supervoxel_shape_variability_metric import prepare_gpa_input_array, calculate_procrustes_variability
import multiprocessing
# Import the new shape variability analysis function
from shape_variability_analisis_b import perform_shape_variability_analysis

def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(x) for x in obj]
    return obj


def process_intersections(error_dict):
    """
    Processes a single dictionary of intersections and returns a Counter.
    This function is designed to be used as a worker in a multiprocessing pool.
    """
    local_counter = Counter()
    for triangle_index, intersecting_triangles in error_dict.items():
        local_counter[triangle_index] += 1
        for intersected_index in intersecting_triangles:
            local_counter[intersected_index] += 1
    return local_counter


def save_sv_json(final_grid_data, idx_tuple, folder_path):
    sv = final_grid_data[idx_tuple]
    sv_serializable = {
        'control_points': to_serializable(sv['control_points']),
        'triangles': to_serializable(sv['triangles'])
    }
    fname = f"sv{idx_tuple[0]}{idx_tuple[1]}{idx_tuple[2]}.json"
    with open(os.path.join(folder_path, fname), 'w') as f:
        json.dump(sv_serializable, f, indent=2)



def execute_grid_algorithm_full(algorithm_text,temp_directory,grid_dimensions = (11, 11, 11),output_example_json_folder_path=None):
    """
    Executes a geometric algorithm on a grid of specified dimensions and returns the final grid data.
    output_example_json_path if provided saves single example of the output grid data to a JSON file. for visualization purposes.
    """

    steps= extract_step_blocks(algorithm_text)
    algorithm_steps = [steps["S1"], steps["S2"], steps["S3"], steps["S4"], steps["S5"]]
    n_weights = get_highest_w_number(steps["S4"])
    final_grid_data = execute_grid_algorithm(grid_dimensions, algorithm_steps, n_weights)



    if output_example_json_folder_path is not None:
        save_sv_json(final_grid_data, (3, 3, 3), output_example_json_folder_path)
        save_sv_json(final_grid_data, (4, 4, 4), output_example_json_folder_path)
        save_sv_json(final_grid_data, (5, 5, 5), output_example_json_folder_path)


    # Filter in for interior cells that should be complete
    interior_cells = []
    gx, gy, gz = grid_dimensions
    for i in range(1, gx - 1):
        for j in range(1, gy - 1):
            for k in range(1, gz - 1):
                interior_cells.append(final_grid_data[(i, j, k)])

    num_interior_cells = len(interior_cells)
    report = []
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
    report.append("\n--- Validation Report ---")
    
    # Watertightness Report
    watertight_msg = f"Watertight Meshes: {watertight_count} / {num_interior_cells}"
    print(watertight_msg)
    report.append(watertight_msg)
    if watertight_count < num_interior_cells:
        watertight_fail_msg = f"❌ Found {num_interior_cells - watertight_count} non-watertight meshes."
        print(watertight_fail_msg)
        report.append(watertight_fail_msg)
    else:
        watertight_ok_msg = "✅ All interior meshes are watertight."
        print(watertight_ok_msg)
        report.append(watertight_ok_msg)

    # Star-Convexity Report
    intersections_found = [res for res in intersection_results if res]
    supervoxels_with_intersections = len(intersections_found)
    
    star_convex_msg = f"\nSupervoxels with Star-Convexity Violations: {supervoxels_with_intersections} / {num_interior_cells}"
    print(star_convex_msg)
    report.append(star_convex_msg)

    if supervoxels_with_intersections > 0:
        star_convex_fail_msg = f"❌ Found {supervoxels_with_intersections} supervoxels with self-intersections."
        print(star_convex_fail_msg)
        report.append(star_convex_fail_msg)
        
        # Aggregate all intersection errors to find the most common culprits using multiprocessing
        with Pool(cpu_count()) as pool:
            counters = pool.map(process_intersections, intersections_found)
        all_intersections = Counter()
        for c in counters:
            all_intersections.update(c)
        
        freq_triangles_msg = "\nMost Frequent Triangles Involved in Intersections:"
        print(freq_triangles_msg)
        report.append(freq_triangles_msg)

        # Prepare the most common triangles list
        most_common = list(all_intersections.most_common(5))
        # Multiprocessing cannot pickle local functions, so do this sequentially
        triangle_msgs = []
        for triangle_index, count in most_common:
            msg = f"  - Triangle Index {triangle_index}: involved in {count} intersections."
            print(msg)
            triangle_msgs.append(msg)
        report.extend(triangle_msgs)



    else:
        star_convex_ok_msg = "✅ All interior supervoxels are star-convex."
        print(star_convex_ok_msg)
        report.append(star_convex_ok_msg)

    print("\n--- Validation Complete ---")
    report.append("\n--- Validation Complete ---")



    is_correct = watertight_count == num_interior_cells and supervoxels_with_intersections == 0
    variability_score = -1.0
    # If all meshes are watertight and star-convex, calculate shape variability metric using mesh-based method
    if is_correct:
        # Prepare the raw mesh cohort: each mesh is a list of triangles (each triangle is a list of 3 points)
        raw_mesh_cohort = []
        for cell in interior_cells:
            triangles = cell['triangles']
            # Assume the original sv center is the first control point
            sv_center = np.array(cell['control_points']["originalSvCenter"])
            # Subtract sv_center from each point in each triangle
            triangles_centered = [
                [np.array(pt) - sv_center for pt in tri]
                for tri in triangles
            ]
            # Convert back to lists for serialization/compatibility
            triangles_centered = [ [pt.tolist() for pt in tri] for tri in triangles_centered ]
            raw_mesh_cohort.append(triangles_centered)

        # Call the new shape variability analysis function
        ssm_model, surfaces, total_variability = perform_shape_variability_analysis(raw_mesh_cohort,temp_directory)


        if ssm_model is not None:
            variability_score = total_variability
            print(f"Shape Variability Metric (SSM Total Variability): {variability_score}")
            report.append(f"Shape Variability Metric (SSM Total Variability): {variability_score}")

    return is_correct, variability_score, " ".join(report)

if __name__ == "__main__":
    # Example usage

#3.569912776472611e-16



    def get_algorithm_text(ws1d,ws1u,ws2d,ws2u,ws3d,ws3u,ws4d,ws4u):

        return f"""
        <S1>
            svCenter = originalSvCenter_(0,0,0) + vector(range({ws1d}, {ws1u}, w1) * rx - rx, range({ws1d}, {ws1u}, w2) * ry - ry , range({ws1d}, {ws1u}, w3) * rz - rz);

        </S1>
        
        <S2>
            linX = lin_p(svCenter_(0,0,0), svCenter_(-1,0,0), range({ws2d}, {ws2u}, w4));
            linY = lin_p(svCenter_(0,0,0), svCenter_(0,-1,0), range({ws2d}, {ws2u}, w5));
            linZ = lin_p(svCenter_(0,0,0), svCenter_(0,0,-1), range({ws2d}, {ws2u}, w6));
            temp1 = lin_p(svCenter_(-1,-1,0), svCenter_(0,-1,0), range({ws2d}, {ws2u}, w7));
            temp2 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range({ws2d}, {ws2u}, w7));
            temp3 = lin_p(svCenter_(0,-1,-1), svCenter_(0,0,-1), range({ws2d}, {ws2u}, w9));
            temp4 = lin_p(svCenter_(0,-1,0), svCenter_(0,0,0), range({ws2d}, {ws2u}, w9));
            temp5 = lin_p(svCenter_(-1,0,-1), svCenter_(0,0,-1), range({ws2d}, {ws2u}, w11));
            temp6 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range({ws2d}, {ws2u}, w11));
        </S2>
        
        <S3>
            tempX00 = lin_p(svCenter_(-1,0,0), svCenter_(0,0,0), range({ws3d}, {ws3u}, w13));
            tempX10 = lin_p(svCenter_(-1,-1,0), svCenter_(0,-1,0), range({ws3d}, {ws3u}, w13));
            tempX01 = lin_p(svCenter_(-1,0,-1), svCenter_(0,0,-1), range({ws3d}, {ws3u}, w13));
            tempX11 = lin_p(svCenter_(-1,-1,-1), svCenter_(0,-1,-1), range({ws3d}, {ws3u}, w13));
            tempY0 = lin_p(tempX10, tempX00, range({ws3d}, {ws3u}, w14));
            tempY1 = lin_p(tempX11, tempX01, range({ws3d}, {ws3u}, w14));
            mob = lin_p(tempY1, tempY0, range({ws3d}, {ws3u}, w15));
        </S3>
        
        <S4>
            int1 = lin_p(mob_(0,0,0), mob_(1,0,0), range({ws4d}, {ws4u}, w16));
            int2 = lin_p(int1, mob_(1,0,0), range({ws4d}, {ws4u}, w17));
            int3 = lin_p(int2, mob_(1,0,0), range({ws4d}, {ws4u}, w18));
            int4 = lin_p(int3, mob_(1,0,0), range({ws4d}, {ws4u}, w19));

            int5 = lin_p(mob_(0,0,0), mob_(0,1,0), range({ws4d}, {ws4u}, w20));
            int6 = lin_p(int5, mob_(0,1,0), range({ws4d}, {ws4u}, w21));
            int7 = lin_p(int6, mob_(0,1,0), range({ws4d}, {ws4u}, w22));
            int8 = lin_p(int7, mob_(0,1,0), range({ws4d}, {ws4u}, w23));

            int9 = lin_p(mob_(0,0,0), mob_(0,0,1), range({ws4d}, {ws4u}, w24));
            int10 = lin_p(int9, mob_(0,0,1), range({ws4d}, {ws4u}, w25));
            int11 = lin_p(int10, mob_(0,0,1), range({ws4d}, {ws4u}, w26));
            int12 = lin_p(int11, mob_(0,0,1), range({ws4d}, {ws4u}, w27));
        </S4>
        
        <S5>
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
        </S5>
        """
    

    ws1d=0.01
    ws1u=0.02
    ws2d=0.49
    ws2u=0.51
    ws3d=0.49
    ws3u=0.51

    ws4d=0.49
    ws4u=0.51
    algorithm_text_a=get_algorithm_text(ws1d,ws1u,ws2d,ws2u,ws3d,ws3u,ws4d,ws4u)

    ws1d=0.01
    ws1u=0.05
    ws2d=0.47
    ws2u=0.53
    ws3d=0.47
    ws3u=0.55

    ws4d=0.49
    ws4u=0.51
    algorithm_text_b=get_algorithm_text(ws1d,ws1u,ws2d,ws2u,ws3d,ws3u,ws4d,ws4u)


    ws1d=0.01
    ws1u=0.05
    ws2d=0.47
    ws2u=0.53
    ws3d=0.47
    ws3u=0.55

    ws4d=0.1
    ws4u=0.8
    algorithm_text_c=get_algorithm_text(ws1d,ws1u,ws2d,ws2u,ws3d,ws3u,ws4d,ws4u)


    ws1d=0.01
    ws1u=0.99
    ws2d=0.1
    ws2u=0.7
    ws3d=0.1
    ws3u=0.9

    ws4d=0.1
    ws4u=0.8
    algorithm_text_d=get_algorithm_text(ws1d,ws1u,ws2d,ws2u,ws3d,ws3u,ws4d,ws4u)

    temp="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_viz/temp"
    is_correct, variability_score, report = execute_grid_algorithm_full(algorithm_text_a,temp,(8, 8, 8))

    is_correct_b, variability_score_b, report_b = execute_grid_algorithm_full(algorithm_text_b,temp,(8, 8, 8))
    is_correct_c, variability_score_c, report_c = execute_grid_algorithm_full(algorithm_text_c,temp,(8, 8, 8),output_example_json_folder_path="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_viz/algo_c")

    is_correct_d, variability_score_d, report_d = execute_grid_algorithm_full(algorithm_text_d,temp,(8, 8, 8),output_example_json_folder_path="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_viz/algo_d")

    print(f"Algorithm A: Correct={is_correct}, Variability Score={variability_score}")
    print(f"Algorithm B: Correct={is_correct_b}, Variability Score={variability_score_b}")
    print(f"Algorithm C: Correct={is_correct_c}, Variability Score={variability_score_c}")
    print(f"Algorithm D: Correct={is_correct_d}, Variability Score={variability_score_d} report={report_d}")

    assert variability_score < variability_score_b, (
        f"Expected algorithm_text_a to have smaller variability score than b, "
        f"but got {variability_score} >= {variability_score_b}"
    )
    # assert math.isclose(variability_score_b, variability_score_c, rel_tol=0.1), (
    #     f"Expected algorithm_text_c to have similar variability score to b, "
    #     f"but got {variability_score_b} vs {variability_score_c}"
    # )
    print(f"Algorithm Correctness: {is_correct}, Variability Score: {variability_score}")
    print("Report:")
    print(report)
