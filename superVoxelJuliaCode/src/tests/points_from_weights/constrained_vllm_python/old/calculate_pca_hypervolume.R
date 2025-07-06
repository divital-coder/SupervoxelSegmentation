# Load required libraries
# Ensure you have installed these packages:
# install.packages("volesti")
# install.packages("jsonlite")
# install.packages("stats")
library(volesti)
library(jsonlite)
library(stats)


#' Preprocess supervoxel lists and construct data matrix
#'
#' This function takes a list of named lists (each containing 3D coordinate vectors and 'originalSvCenter'),
#' subtracts the origin from all other coordinates, and returns a matrix suitable for PCA.
#'
#' @param dicts A list of named lists. Each inner list should contain 3D coordinate vectors, including 'originalSvCenter'.
#' @return A matrix where each row is a flattened, origin-centered vector from a supervoxel.
get_data_matrix_from_supervoxel_lists <- function(dicts) {
  # --- 1. Input Validation ---
  if (is.null(dicts) || length(dicts) < 2) {
    message("Error: Not enough valid supervoxel lists to construct data matrix. Need at least 2.")
    return(NULL)
  }

  # --- 2. Data Preprocessing ---
  processed <- lapply(dicts, function(d) {
    origin <- as.numeric(d[['originalSvCenter']])
    if (is.null(origin)) {
      message("Warning: 'originalSvCenter' not found in a list element. Skipping.")
      return(NULL)
    }
    new_d <- list()
    for (key in names(d)) {
      if (key != 'originalSvCenter') {
        new_d[[key]] <- as.numeric(d[[key]]) - origin
      }
    }
    return(new_d)
  })
  processed <- processed[!sapply(processed, is.null)]
  if (length(processed) < 2) {
    message("Error: Not enough valid supervoxel lists after preprocessing. Need at least 2.")
    return(NULL)
  }

  keys <- sort(names(processed[[1]]))
  vector_list <- lapply(processed, function(d) {
    unlist(d[keys])
  })
  data_matrix <- do.call(rbind, vector_list)
  if (nrow(data_matrix) < 2) {
    message("Error: Not enough valid supervoxel lists to construct data matrix. Need at least 2.")
    return(NULL)
  }
  return(data_matrix)
}


#' Calculate the hypervolume of PCA EIGENVECTORS.
#'
#' This function implements the method from the supporting document.
#' It calculates the convex hull hypervolume of the EIGENVECTORS themselves, which define
#' the principal axes of variance. This is a robust method that will not return 0 unless k=0.
#'
#' @param data_matrix A numeric matrix where each row is a sample and each column is a feature.
#' @param variance_threshold The cumulative variance the principal components should explain.
#' @param error_tolerance The desired relative error for the volume estimation.
#' @param seed An optional integer for reproducibility.
#' @return A numeric value representing the hypervolume.
calculate_eigenvector_hypervolume <- function(data_matrix, variance_threshold = 0.95, error_tolerance = 0.08, seed = NULL) {
  if (is.null(data_matrix) || !is.matrix(data_matrix) || nrow(data_matrix) < 2) {
    message("Error: Input must be a matrix with at least 2 rows.")
    return(0.0)
  }
  
  # --- 1. PCA and Eigenvector Selection ---
  message("Performing PCA...")
  pca_result <- stats::prcomp(data_matrix, center = TRUE, scale. = TRUE)
  
  eigenvalues <- pca_result$sdev^2
  cumulative_variance <- cumsum(eigenvalues / sum(eigenvalues))
  k <- which(cumulative_variance >= variance_threshold)[1]

  if (is.na(k) || k == 0) {
    message("Error: Could not select any components for the given variance threshold.")
    return(0.0)
  }
  
  message(sprintf("Selected k=%d components to explain %.2f%% of the variance.", k, cumulative_variance[k] * 100))
  
  # The eigenvectors are in pca_result$rotation. These are the principal axes.
  selected_eigenvectors <- pca_result$rotation[, 1:k, drop = FALSE]
  
  # The dimensionality of the ambient space is the number of original variables
  ambient_dim <- nrow(selected_eigenvectors)
  
  message(sprintf("The %d selected eigenvectors exist in a %d-dimensional ambient space.", k, ambient_dim))
  
  # --- 2. Geometric Construction ---
  # As per the document, we create a polytope from the eigenvectors and the origin.
  # This forms a k-simplex, a shape with a well-defined non-zero volume in its subspace.
  origin <- matrix(0, nrow = ambient_dim, ncol = 1)
  vertices_matrix <- cbind(selected_eigenvectors, origin)
  
  # The Vpolytope is defined by k+1 vertices in an ambient_dim space.
  P_eigen <- Vpolytope(V = vertices_matrix)
  
  # --- 3. Hypervolume Calculation ---
  high_accuracy_settings <- list(
    # algorithm = "CB",
    # random_walk = "brdhr",
    # error = error_tolerance,
    # walk_length = 5,
    seed = seed
  )
  
  message(paste0("Computing convex hull volume of the EIGENVECTORS in a ", ambient_dim, "-dimensional space..."))
  
  vol <- volume(
    P = P_eigen,
    settings = high_accuracy_settings,
    rounding = TRUE
  )
  
  message("Successfully computed hypervolume of the eigenvectors.")
  return(vol)
}



# --- Script Mode: Accept input and output file paths as arguments ---
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript calculate_pca_hypervolume.R <input_json_path> <output_json_path>\n")
  quit(status = 1)
}
input_json_path <- args[1]
output_json_path <- args[2]

start_time <- Sys.time()

if (file.exists(input_json_path)) {
  cat(paste0("--- Loading data from ", input_json_path, " ---\n"))
  data <- jsonlite::fromJSON(input_json_path, simplifyVector = FALSE)
  dicts <- lapply(data, function(d) d$control_points)
  cat(paste0("Loaded ", length(dicts), " supervoxel dicts from JSON.\n"))

  # Create the data matrix
  data_matrix <- get_data_matrix_from_supervoxel_lists(dicts)
  result <- list()
  if (!is.null(data_matrix)) {
    cat(paste0("\n--- Calculating Volume of PCA Eigenvectors ---\n"))
    eigenvector_volume <- calculate_eigenvector_hypervolume(data_matrix, seed=42)
    cat(paste("\nFinal Volume of Eigenvectors:", eigenvector_volume, "\n"))
    result$eigenvector_volume <- eigenvector_volume
    cat("\n--- Calculation Finished ---\n")
  } else {
    cat("--- Calculation Aborted: Could not create data matrix. ---\n")
    result$error <- "Could not create data matrix."
    result$eigenvector_volume <- 0.0
  }
  end_time <- Sys.time()
  elapsed <- as.numeric(difftime(end_time, start_time, units = "secs"))
  cat(sprintf("Total execution time: %.3f seconds\n", elapsed))
  result$execution_time_seconds <- elapsed
  jsonlite::write_json(result, output_json_path, auto_unbox = TRUE, pretty = TRUE)
  cat(paste0("Results written to ", output_json_path, "\n"))
} else {
  cat(paste("Error: JSON file not found at path:", input_json_path, "\n"))
  cat("Please provide a valid input JSON file.\n")
  quit(status = 2)
}
#Rscript superVoxelJuliaCode/src/tests/points_from_weights/constrained_vllm_python/calculate_pca_hypervolume.R /workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_viz/sv333_debug_synth.json /workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_viz/sv333_debug_out.json