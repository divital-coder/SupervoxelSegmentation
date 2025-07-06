cd("/workspaces/superVoxelJuliaCode_lin_sampl")


# In the Julia REPL:
using Pkg
Pkg.activate(".")  # Activate the current directory as project

using Revise, CUDA, HDF5
using Lux, Optimisers, Random, Zygote
using Meshes, LinearAlgebra, GLMakie, GLFW
using Statistics, ChainRulesCore, Logging
using ModernGL, TensorBoardLogger, DelimitedFiles
using PyCall, Wavelets
using UUIDs


# Import Python libraries for NIfTI saving
sitk = pyimport_conda("SimpleITK", "simpleitk")
np = pyimport_conda("numpy", "numpy")

# Define required global constants
global const len_get_random_point_in_tetrs_kern = Int(floor(256/3))
global const closeness_to_discr = 1000000
global const is_distributed = false
const is_point_per_triangle = true
global const num_additional_oblique_points_per_side=2

# Include necessary files
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/Lux_model.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/prepare_polygons.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/render_grey_poligons.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/OpenGLUtils.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/render_for_tensor_board.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/main_loop.jl")

"""
Create a simplified model for supervoxel segmentation.
This follows the blueprint from get_model_consts in Lux_model.jl

Points_weights_str if debug_with_source_arr is true will accept source_arr,weights and return ((sv_centers_out, control_points_out), source_arr)

"""
function create_simplified_model(image_shape, radiuss, spacing, batch_size, is_point_per_triangle)
    pad_voxels = 1
    zero_der_beg = false
    num_base_samp_points = 2
    num_additional_samp_points = 1
    

    


    model = Lux.Chain(
        Points_weights_str(radiuss, batch_size, pad_voxels, 
        (image_shape[1], image_shape[2], image_shape[3]), is_point_per_triangle, zero_der_beg,true),
    Set_tetr_dat_str(radiuss, spacing, batch_size, 0, 
        (image_shape[1], image_shape[2], image_shape[3]), true, false),
    Point_info_kern_str(radiuss, spacing, batch_size, 
        (image_shape[1], image_shape[2], image_shape[3]), 
        num_base_samp_points, num_additional_samp_points, true, false)
    )
    
    return model 
end




"""
Helper function to transfer to CPU
"""
function cpu_device()
    return Array
end


function get_data_for_iter(imm_val_loc_a, tetr_dat, sv_means, axis, plane_dist, is_point_per_triangle)
    println("Rendering slice $(plane_dist)")
    imm_val_loc = Array(copy(imm_val_loc_a))
    tetr_num = get_num_tetr_in_sv(is_point_per_triangle)
    image_size = size(imm_val_loc)
    
    # Get data for display
    triang_vec, sv_i, line_vertices, line_indices, imm = get_data_to_display(
        tetr_dat, tetr_num, axis, plane_dist, image_size, imm_val_loc)

    # Generate random colors for each supervoxel
    colorss = Float32.(sv_means[Int.(sv_i)])
    return (triang_vec, colorss, imm)
end


function parallel_process_slices(imm_val_loc_a, tetr_dat, sv_means, axis, dims, is_point_per_triangle, window, windowWidth, windowHeight)
    # Create container for results
    im_gp_whole = []
    
    # Determine number of available threads
    num_threads = Threads.nthreads()
    println("Using $(num_threads) threads for data preparation")
    
    # Create chunks of plane distances based on thread count
    plane_dists = 1:dims[3]
    # chunk_size = max(1, ceil(Int, dims[3] / num_threads))
    chunks = Base.Iterators.partition(plane_dists, num_threads)
    
    # Process each chunk
    for chunk in chunks
        print("\n cccc chunk $(size(chunk)) \n")

        # Pre-allocate results array for this chunk
        chunk_results = Vector{Any}(undef, length(chunk))
        
        # Parallelize the data preparation
        Threads.@threads for i in eachindex(chunk)
            plane_dist = chunk[i]
            try
                # This part runs in parallel
                chunk_results[i] = get_data_for_iter(imm_val_loc_a, tetr_dat, sv_means, axis, plane_dist, is_point_per_triangle)
            catch e
                println("Error in data preparation for slice $(plane_dist): ", e)
                chunk_results[i] = nothing
            end
        end
        
        # Process the rendering sequentially
        for i in eachindex(chunk)
            plane_dist = chunk[i]
            if isnothing(chunk_results[i])
                # Handle error case
                push!(im_gp_whole, zeros(Float32, 1, dims[1], dims[2]))
                continue
            end
            
            try
                # Sequential rendering on main thread
                triang_vec, colorss, imm = chunk_results[i]
                
                # Render greyscale polygons
                im_gp = render_grey_polygons_loc(
                    triang_vec, colorss, window, windowWidth, windowHeight, "synth_data", plane_dist)
                
                GLFW.PollEvents()
                push!(im_gp_whole, sum(im_gp, dims=1)[:,:,:])
                println("Processed slice $(plane_dist), size: $(size(sum(im_gp, dims=1)[:,:,:]))")
            catch e
                println("Error rendering slice $(plane_dist): ", e)
                push!(im_gp_whole, zeros(Float32, 1, dims[1], dims[2]))
            end
        end
    end
    
    return im_gp_whole
end



"""
Generate synthetic data with supervoxel visualization.
This uses CPU for all operations to avoid CUDA issues.
"""
function generate_synth_data_cpu(dims, 
                            radiuss, 
                            spacing,
                            batch_size,
                            seed,
                            output_dir,
                            is_point_per_triangle,
                            set_zero_weights
                            ,window
                            )
    
    # Set random seed
    rng = Random.MersenneTwister(seed)
    
    # Setup image shape
    image_shape = (dims[1], dims[2], dims[3], 2, batch_size)
    
    println("Creating model...")
    # Create the model
    model = create_simplified_model(image_shape, radiuss, spacing, batch_size, is_point_per_triangle)
    
    # Initialize control points for weights
    # rrrrrrrrrrrrrrrrrrrrrrrrrr image_shape (128, 128, 128, 1, 4) radius (3.5f0, 3.5f0, 3.5f0)  
    set_of_svs = initialize_centers_and_control_points(image_shape, radiuss, is_point_per_triangle)
    
    if is_point_per_triangle
        sv_centers, control_points, tetrs, dims_sv, plan_tensor = set_of_svs
    else
        sv_centers, control_points, tetrs, dims_sv = set_of_svs
    end
    
    # Create random weights with the required dimensions
    control_points_weights = randn( Float32, 
        size(control_points, 1) + 2, 
        size(control_points, 2) + 2, 
        size(control_points, 3) + 2, 
        24 + 24*6, 1, 1)
    # Create a tensor of ones with the same dimensions as control_points_weights
    if(set_zero_weights)
        control_points_weights = control_points_weights./100000
    end

    # Setup model parameters
    dev = gpu_device() 

    ps, st = Lux.setup(rng, model) |> dev
    # Create a synthetic image (random data)

    imm_val = randn(rng, Float32, image_shape)|> dev
    opt= Optimisers.Adam(0.01)
    st_opt = Optimisers.setup(opt, st) |> dev
    tstate_glob = Lux.Training.TrainState(nothing, nothing, model, ps, st, opt, st_opt, 0)    
    control_points_weights = control_points_weights |> dev
    imm_val = imm_val |> dev
   
    out_sampled_points, tetr_dat,b_loss= infer_model(tstate_glob, model, (imm_val,control_points_weights)) #(out_sampled_points, tetr_dat,(b_loss+b_loss_b)), st
    tetr_dat=Array(tetr_dat)   
    
    # Generate data for visualization
    imm_val_loc_a = imm_val[:,:,:,1,1]  # Get first channel and batch
    
    println("Setting up visualization...")
    # Setup visualization dimensions
    windowWidth, windowHeight = dims[1], dims[2]
    texture_width, texture_height = dims[1], dims[2]
    
    # Close existing GLFW windows and initialize a new one
    
    println("Rendering slices...")
    # Generate images for each slice
    axis = 3  # Z-axis
    im_gp_whole = []
    
    # Create random values for supervoxel coloring
    max_sv_id = Int(ceil(dims[1] * dims[2] * dims[3] / get_num_tetr_in_sv(is_point_per_triangle)))
    big_mean_weighted = get_sv_mean(out_sampled_points,is_point_per_triangle)
    sv_means=big_mean_weighted[:,1,1]
    sv_means = rand(Float32,size(sv_means,1))

    
    # Initialize OpenGL resources
    window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program, 
    line_shader_program, textUreId = initialize_window_etc(
        windowWidth, windowHeight, texture_width, texture_height, window)


    
    im_gp_whole=parallel_process_slices(imm_val_loc_a, tetr_dat, sv_means, axis, dims, is_point_per_triangle, window, windowWidth, windowHeight)
    # for plane_dist in 1:(dims[3])
    #     try

    #         triang_vec, colorss, imm=get_data_for_iter(imm_val_loc_a, tetr_dat, sv_means, axis, plane_dist, is_point_per_triangle)
            
    #         # Render greyscale polygons
    #         im_gp = render_grey_polygons_loc(
    #             triang_vec, colorss, window, windowWidth, windowHeight, "synth_data", plane_dist)
            
    #         GLFW.PollEvents()
    #         push!(im_gp_whole, sum(im_gp, dims=1)[:,:,:])

    #     catch e
    #         println("Error rendering slice $(plane_dist): ", e)
    #         push!(im_gp_whole, zeros(Float32,1, dims[1], dims[2]))

    #     end
    #     # Store slice
    # end
    
    println("Creating output directories...")
    # Create output directory if it doesn't exist
    isdir(output_dir) || mkpath(output_dir)
    
    # Combine all images into a single volume
    im_gp_whole_im = vcat(im_gp_whole...)
    
    # Replace NaN and Inf values with 0.0
    replace!(im_gp_whole_im, NaN => 0.0)
    replace!(im_gp_whole_im, Inf => 0.0)
    replace!(im_gp_whole_im, -Inf => 0.0)

    # Save as NIfTI
    uuid = string(UUIDs.uuid4())
    p_out_im_gp = joinpath(output_dir, "synth_supervoxel_data$(uuid).nii.gz")
    println("\n Saving NIfTI to: $(p_out_im_gp)  size $(size(im_gp_whole_im)) sum $(sum(im_gp_whole_im)) \n  ")
    
    # Convert for SimpleITK
    im_sitk = sitk.GetImageFromArray(im_gp_whole_im)
    sitk.WriteImage(im_sitk, p_out_im_gp)
    
    # Save as HDF5 with both image and weights
    h5_path = joinpath(output_dir, "synth_supervoxel_data.h5")
    println("Saving HDF5 to: $(h5_path)")
    file_mode = isfile(h5_path) ? "r+" : "w" 
    h5open(h5_path, file_mode) do file
        group = create_group(file,"$(uuid)")
        group["image"] = im_gp_whole_im
        group["weights"] = Array(control_points_weights)
        group["seed"] = [seed]
        group["radiuss"] = [radiuss[1],radiuss[2],radiuss[3]]
        group["spacing"] = [spacing[1],spacing[2],spacing[3]]
        group["sv_means"] = Array(sv_means)
        # file["image"] = im_gp_whole_im
        # file["weights"] = Array(control_points_weights)
        # file["seed"] =[seed]
        # file["radiuss"] = Array(radiuss)
        # file["spacing"] = Array(spacing)
        # file["sv_means"] = Array(sv_means)
    end
    
    
    println("\n Done! Synthetic supervoxel data generated.  size $(size(im_gp_whole_im)) \n ")
    # return p_out_im_gp
end

# Call the function with CPU implementation to avoid CUDA issues
# Using very small dimensions for faster testing

main_fold="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_viz/256"
GLFW.Terminate()
dims=(256, 256, 256)
window = initializeWindow(dims[1], dims[2])

for i in 1:1000
set_zero_weights=false
generate_synth_data_cpu(
    dims,  # Small dims for testing
    (Float32(3.5), Float32(3.5), Float32(3.5)),
    (Float32(1.0), Float32(1.0), Float32(1.0)),
    1,  # batch_size
    i,  # seed
    main_fold,
    true,  # is_point_per_triangle set to true as required
    set_zero_weights,
    window
)
GC.gc()
end