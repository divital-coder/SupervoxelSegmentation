# -*- coding: utf-8 -*-
"""
code is adapted from https://github.com/theochem/procrustes/blob/main/procrustes/generalized.py

Shape Variability Analysis using Generalized Procrustes Analysis (GPA)

This script quantifies the variability within a collection of 3D shapes. Each 
shape is defined by a dictionary of named 3D coordinates (landmarks). The method
is to represent each shape as a k x 3 matrix (k landmarks, 3 dimensions), 
and then use a constrained Generalized Procrustes Analysis to compute the 
Procrustes Sum of Squares (PSS) without alignment. This PSS serves as a direct, 
geometrically-grounded metric of overall shape flexibility.

Methodology:
1.  **Shape Matrix Preparation**: Each shape's landmark coordinates are transformed 
    into a k x 3 matrix. This is done by first centering the shape by 
    subtracting its 'originalSvCenter' from all other landmarks. This process 
    preserves the user-defined translational invariance. The collection of shapes
    is represented as a 3D NumPy array of shape (n, k, 3), where n is the 
    number of shapes.

2.  **Constrained Generalized Procrustes Analysis (GPA)**: The `qc-procrustes` 
    invariance to scale and translation is disabled as library is not doing it automatically

3.  **Variability Metric**: The primary output of the constrained GPA is the 
    Procrustes Sum of Squares (PSS). This value is the sum of squared 
    Euclidean distances between the landmarks of each shape and the 
    corresponding landmarks of the computed consensus (mean) shape. It is a 
    single, scalar value that quantifies the overall "flexibility" or 
    dissimilarity of the shapes, sensitive to deformation, orientation, and 
    scale.

Mathematical Properties:
-   **Translation Invariance (Custom)**: The analysis is invariant to the 
    absolute translation of each shape because every shape is centered 
    relative to its own 'originalSvCenter' during pre-processing.
-   **Scale Sensitivity**: The analysis is sensitive to the scale of the input 
    coordinates. If all shapes are scaled up, the resulting PSS will increase.
-   **Rotation Sensitivity**: The analysis is sensitive to the relative 
    orientation of the shapes. This is by design as we put only 1 iteration for algorithm
Dependencies:
-   numpy
-   qc-procrustes
"""

import json
import numpy as np
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from procrustes import orthogonal
from procrustes.utils import _check_arraytypes


def generalized(
    array_list: List[np.ndarray],
    ref: Optional[np.ndarray] = None,
    tol: float = 1.0e-7,
    n_iter: int = 200,
    check_finite: bool = True,
) -> Tuple[List[np.ndarray], float]:
    r"""Generalized Procrustes Analysis.

    Parameters
    ----------
    array_list : List
        The list of 2D-array which is going to be transformed.
    ref : ndarray, optional
        The reference array to initialize the first iteration. If None, the first array in
        `array_list` will be used.
    tol: float, optional
        Tolerance value to stop the iterations.
    n_iter: int, optional
        Number of total iterations.
    check_finite : bool, optional
        If true, convert the input to an array, checking for NaNs or Infs.

    Returns
    -------
    array_aligned : List
        A list of transformed arrays with generalized Procrustes analysis.
    new_distance_gpa: float
        The distance for matching all the transformed arrays with generalized Procrustes analysis.

    Notes
    -----
    Given a set of matrices, :math:`\mathbf{A}_1, \mathbf{A}_2, \cdots, \mathbf{A}_k` with
    :math:`k > 2`,  the objective is to minimize in order to superimpose pairs of matrices.

    .. math::
        \min \quad = \sum_{i<j}^{j} {\left\| \mathbf{A}_i \mathbf{T}_i  -
         \mathbf{A}_j \mathbf{T}_j \right\| }^2

    This function implements the Equation (20) and the corresponding algorithm in  Gower's paper.

    """
    # check input arrays
    _check_arraytypes(*array_list)
    # check finite
    if check_finite:
        array_list = [np.asarray_chkfinite(arr) for arr in array_list]

    if n_iter <= 0:
        raise ValueError("Number of iterations should be a positive number.")
    if ref is None:
        # the first array will be used to build the initial ref
        array_aligned = [array_list[0]] + [
            _orthogonal(arr, array_list[0]) for arr in array_list[1:]
        ]
        ref = np.mean(array_aligned, axis=0)
    else:
        array_aligned = [None] * len(array_list)
        ref = ref.copy()

    distance_gpa = np.inf
    for _ in np.arange(n_iter):
        # align to ref
        array_aligned = [_orthogonal(arr, ref) for arr in array_list]
        # the mean
        new_ref = np.mean(array_aligned, axis=0)
        # todo: double check if the error is defined in the right way
        # the error
        new_distance_gpa = np.square(ref - new_ref).sum()
        if distance_gpa != np.inf and np.abs(new_distance_gpa - distance_gpa) < tol:
            break
        ref = new_ref
        distance_gpa = new_distance_gpa
    return array_aligned, new_distance_gpa


def _orthogonal(arr_a: np.ndarray, arr_b: np.ndarray) -> np.ndarray:
    """Orthogonal Procrustes transformation and returns the transformed array."""
    res = orthogonal(arr_a, arr_b, translate=False, scale=False, unpad_col=False, unpad_row=False)
    return np.dot(res["new_a"], res["t"])


def prepare_gpa_input_array(dicts: list, verbose: bool = False) -> np.ndarray:
    """
    Preprocesses a list of landmark dictionaries and constructs a 3D data array.

    For each dictionary, it computes coordinate vectors relative to a defined 
    origin ('originalSvCenter') and arranges them into a k x 3 matrix. These
    matrices are then stacked into a single (n, k, 3) NumPy array suitable for
    Generalized Procrustes Analysis.

    Args:
        dicts: A list of dictionaries. Each inner dictionary should contain 3D
               coordinate vectors for landmarks, including 'originalSvCenter'.
        verbose: If True, prints detailed progress messages.

    Returns:
        A 3D NumPy array of shape (n, k, 3), where n is the number of valid
        shapes, k is the number of common landmarks, and 3 corresponds to the
        x, y, z dimensions. Returns None if the input is insufficient or invalid.
    """
    # --- 1. Input Validation ---

    if not dicts or len(dicts) < 2:
        return None

    valid_dicts = [d for d in dicts if d is not None and 'originalSvCenter' in d]
    if len(valid_dicts) < 2:
        return None

    # --- 2. Robust Landmark Key Identification ---
    # Create a canonical key order from the superset of all keys across all valid dicts.
    all_keys = set()
    for d in valid_dicts:
        all_keys.update(d.keys())
    if 'originalSvCenter' not in all_keys:
        return None
    all_keys.remove('originalSvCenter')
    sorted_keys = sorted(list(all_keys))
    if not sorted_keys:
        return None

    # --- 3. Data Processing into (n, k, 3) Array ---
    processed_matrices = []
    for i, d in enumerate(valid_dicts):
        origin = np.array(d['originalSvCenter'], dtype=float)
        shape_matrix = []
        for key in sorted_keys:
            point = np.array(d.get(key, origin), dtype=float)
            shape_matrix.append(point - origin)
        processed_matrices.append(np.array(shape_matrix))
    if len(processed_matrices) < 2:
        return None
    return np.stack(processed_matrices, axis=0)


def calculate_procrustes_variability(
    data_array: np.ndarray,
    verbose: bool = False) -> float:
    """
    Calculates shape variability using a constrained Generalized Procrustes Analysis.

    This function computes the Procrustes Sum of Squares (PSS) without performing
    rotational or scaling alignment. The resulting PSS is a measure of the total
    geometric dissimilarity in the dataset.

    Args:
        data_array: A 3D NumPy array of shape (n, k, 3) representing the
                    collection of n shapes.
        verbose: If True, prints detailed progress messages.

    Returns:
        A float representing the Procrustes Sum of Squares, which serves as the
        variability metric. Returns 0.0 on error.
    """
    if data_array is None or not isinstance(data_array, np.ndarray) or data_array.ndim != 3:
        return 0.0
    n_shapes = data_array.shape[0]
    if n_shapes < 2:
        return 0.0
    list_of_matrices = [data_array[i, :, :] for i in range(n_shapes)]
    # Call GPA with n_iter as a positional argument (per qc-procrustes API)
    # Use only the required positional argument, and set n_iter as a keyword argument
    # The function returns (aligned_arrays, sum_of_squares)
    _, variability_metric = generalized(
        list_of_matrices,
        n_iter=1
    )
    return variability_metric



# # --- Script runs as a direct test case, no argument parsing, no try/except, prints result ---
# json_path = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_viz/sv333_debug_synth.json"
# with open(json_path, 'r') as f:
#     data = json.load(f)
# dicts = [d.get("control_points") for d in data]
# data_array = prepare_gpa_input_array(dicts, verbose=False)
# if data_array is not None:
#     variability_score = calculate_procrustes_variability(data_array, verbose=False)
#     print(variability_score)
