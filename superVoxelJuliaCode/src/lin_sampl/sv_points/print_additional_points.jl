# Define the helper functions (assuming they are defined elsewhere or provided)
# These functions generate CUDA code strings for interpolation.

"""
interpolate between two points save in control points
x_add, y_add, z_add - coordinates of the point in the first control point
x_add_b, y_add_b, z_add_b - coordinates of the point in the second control point
weights_channel - channel of the weights
cp1_channel - channel of the first control point
cp2_channel - channel of the second control point
out_channel - channel to save the result in shared memory
coord - coordinate 1 for x 2 for y and 3 for z on which we are currently working
"""
function save_point_for_cp_to_cp(out_channel, x_add, y_add, z_add, x_add_b, y_add_b, z_add_b, weights_channel, cp1_channel, cp2_channel, coord)
    # Assuming num_blocks_z_pure is defined in the scope where the generated code runs
    return """
        shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, $out_channel] =
        (control_points[x+($x_add), y+($y_add), z+($z_add), $cp1_channel, $coord, batch_channel_index]
                                                                                    +
    ((control_points[x+($x_add_b), y+($y_add_b), z+($z_add_b), $cp2_channel, $coord, batch_channel_index]
    - (control_points[x+($x_add), y+($y_add), z+($z_add), $cp1_channel, $coord, batch_channel_index]))
    * weights[x, y, z, ($weights_channel), batch_channel_index]))
    """
end

"""
interpolate between two points one in control points other in shared memory
x_add_b, y_add_b, z_add_b - coordinates of the point in the second control point (target)
weights_channel - channel of the weights
in_channel_1 - channel of the first control point from shared memory (source)
cp2_channel - channel of the second control point (target)
out_channel - channel to save the result in shared memory
coord - coordinate 1 for x 2 for y and 3 for z on which we are currently working
"""
function save_point_for_shared_to_cp(out_channel, x_add_b, y_add_b, z_add_b, weights_channel, in_channel_1, cp2_channel, coord)
    # Assuming num_blocks_z_pure is defined in the scope where the generated code runs
    return """
        shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, $out_channel] =
        (shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, $in_channel_1]
                                                                                    +
    ((control_points[x+($x_add_b), y+($y_add_b), z+($z_add_b), $cp2_channel, $coord, batch_channel_index]
    - (shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, $in_channel_1]))
    * weights[x, y, z, ($weights_channel), batch_channel_index]))
    """
end

"""
Generates CUDA code for a single coordinate interpolation step.
"""
function generate_interpolation_code(
    iter::Int,
    sh_chan_out::Int,
    input_sh_chan::Int,
    start_offset::Tuple{Int,Int,Int},
    end_offset::Tuple{Int,Int,Int},
    weight_idx::Int,
    cp1_chan::Int,
    cp2_chan::Int,
    coord_idx::Int
)
    code = ""
    if iter == 1
        # Interpolate between two control points for the first iteration
        code *= save_point_for_cp_to_cp(1,
            start_offset[1], start_offset[2], start_offset[3],
            end_offset[1], end_offset[2], end_offset[3],
            weight_idx, cp1_chan, cp2_chan, coord_idx)
    else
        # Interpolate from the previous point (in shared memory) towards the target control point
        code *= save_point_for_shared_to_cp(2,
            end_offset[1], end_offset[2], end_offset[3], # Target offset relative to grid point (x,y,z)
            weight_idx, 1, cp2_chan, coord_idx) # Target CP channel

    end
    code *= "\n        # Weight used (coord $coord_idx): $weight_idx\n" # Add comment indicating weight used
    return code
end

"""
Generates CUDA code fragment for calculating additional oblique points for a single axis.
"""
function generate_axis_points_code(
    axis_name::String,
    axis_coord_idx::Int,
    main_coord_cp1_start::Int, main_coord_cp2_start::Int, # CP channels for main coord, start->end
    other_coord_cp1_start::Int, other_coord_cp2_start::Int, # CP channels for other coords, start->end
    main_coord_start_offset::Tuple{Int,Int,Int}, # Offset for start point (usually (0,0,0))
    main_coord_end_offset::Tuple{Int,Int,Int},   # Offset for end point (e.g., (1,0,0) for X)
    other_coord_start_offset::Tuple{Int,Int,Int}, # Offset for other coords start (usually (0,0,0))
    other_coord_end_offset::Tuple{Int,Int,Int},   # Offset for other coords end (usually (0,0,0))
    num_additional_oblique_points_per_side::Int,
    current_weight_index::Int,
    current_cp_chan_out::Int
)
    res = """

        # --- Oblique $(axis_name) Calculations ---
"""
    weight_index = current_weight_index
    cp_chan_out = current_cp_chan_out

    # Determine the other two coordinate indices
    other_axes = filter(i -> i != axis_coord_idx, [1, 2, 3])
    other_axis_1_idx = other_axes[1]
    other_axis_2_idx = other_axes[2]

    # Store the output channel used in the previous iteration for start->end

    # Loop num_additional_oblique_points_per_side * 2 times for start->end direction only
    for iter in 1:(num_additional_oblique_points_per_side*2)
        res *= """ 
        
        #iter $iter
        
        """
        # --- Point 'iter' from Start towards End ---
        res *= """
        # Oblique $(axis_name): Iteration $iter / $(num_additional_oblique_points_per_side * 2) (Start -> End)
"""
        # Determine shared memory output channel based on iteration (ping-pong between 1 and 2)
        sh_chan_out_start_end = 1
        input_sh_chan_start_end=1

        # Generate interpolation code based on the current coordinate being processed
        res *= """
        if (current_coord == $axis_coord_idx) # Calculate main coordinate ($(axis_name))
"""
        # Call helper for the main coordinate
        res *= generate_interpolation_code(iter, sh_chan_out_start_end, input_sh_chan_start_end,
            main_coord_start_offset, main_coord_end_offset,
            weight_index, main_coord_cp1_start, main_coord_cp2_start, axis_coord_idx)

        res *= """
        elseif (current_coord == $other_axis_1_idx) # Calculate other coordinate $other_axis_1_idx
"""
        # Call helper for the first other coordinate
        res *= generate_interpolation_code(iter, sh_chan_out_start_end, input_sh_chan_start_end,
            other_coord_start_offset, other_coord_end_offset,
            weight_index+1 , other_coord_cp1_start, other_coord_cp2_start, other_axis_1_idx) 

        res *= """
        else # Calculate other coordinate $other_axis_2_idx (current_coord == $other_axis_2_idx)
"""
        # Call helper for the second other coordinate
        res *= generate_interpolation_code(iter, sh_chan_out_start_end, input_sh_chan_start_end,
            other_coord_start_offset, other_coord_end_offset,
            weight_index+2 , other_coord_cp1_start, other_coord_cp2_start, other_axis_2_idx)

        res *= """
        end # End coordinate check
"""
        # Increment weight index AFTER all 3 coordinate calculations for this iteration step
        weight_index += 3

        if(iter==1)
            sh_chan_out_start_end=1
        else
            sh_chan_out_start_end=2
            res *= """
            shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 1] = shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 2]
            """    
        end
        # Save the calculated coordinate value from shared memory to the output control points array
        res *= """
        # Save Oblique $(axis_name) point (Start -> End, Iter $iter) for coord = current_coord
        control_points_out[x, y, z, $cp_chan_out, current_coord, batch_channel_index] = shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, $sh_chan_out_start_end]
        # Output CP channel: $cp_chan_out
"""
        cp_chan_out += 1 # Increment output control point channel for the next point

        # Synchronize threads after each point calculation (across all 3 coordinates)
        res *= """
        sync_threads() # Sync after completing iteration $iter for Oblique $(axis_name)
"""

    end # End loop for iterations

    return res, weight_index, cp_chan_out
end

# Main code generation function
function generate_additional_oblique_code(num_additional_oblique_points_per_side::Int)
    res = """
function apply_weights_to_locs_kern_add_a(sv_centers_mod, control_points, control_points_out, weights, cp_x::UInt32, cp_y::UInt32, cp_z::UInt32, num_blocks_z_pure, num_blocks_y_pure)

    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))
    y = ((threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) % cp_y) + 1
    z = ((threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) % cp_z) + 1
    shared_arr = CuStaticSharedArray(Float64, (8, 8, 4, 3))

    # CUDA Kernel Code Fragment for Calculating Additional Oblique Control Points
    # This code is generated by Julia and intended to be part of a larger CUDA kernel.

    # Check if the current thread is within the valid processing bounds
    # Assuming cp_x, cp_y, cp_z are defined (max grid dimensions)
    if (x <= cp_x && y <= cp_y && z <= cp_z && x > 0 && y > 0 && z > 0)

        # Initialize indices
        current_coord = Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)) # 1 for X, 2 for Y, 3 for Z coordinate
        batch_channel_index = Int(ceil(blockIdx().z / num_blocks_z_pure)) # Index for batch/channel dimension

        # Shared memory channels:
        # 1: Output for cp_to_cp (first point in each direction)
        # 2: Output for shared_to_cp (subsequent points)
        # Input for shared_to_cp is the output channel of the previous step (1 or 2).

        # --- Start of Code Generation ---
"""

    weight_index = 10 # Start weights from channel 10
    cp_chan_out = 5   # Start output control points from channel 5

    # Define common offsets
    offset_zero = (0, 0, 0)
    offset_x1 = (1, 0, 0)
    offset_y1 = (0, 1, 0)
    offset_z1 = (0, 0, 1)

    # ==================================
    # === Oblique X (Axis X) Points ===
    # ==================================
    axis_code_x, weight_index, cp_chan_out = generate_axis_points_code(
        "X", 1,                     # Axis name and coordinate index
        4, 4,                       # Main coord CP channels (start->end): Oblique(0,0,0) -> Oblique(1,0,0)
        3, 2,                       # Other coord CP channels (start->end): LinZ(0,0,0) -> LinY(0,0,0)
        offset_zero, offset_x1,     # Main coord offsets
        offset_zero, offset_zero,   # Other coord offsets
        num_additional_oblique_points_per_side,
        weight_index, cp_chan_out
    )
    res *= axis_code_x

    # ==================================
    # === Oblique Y (Axis Y) Points ===
    # ==================================
    axis_code_y, weight_index, cp_chan_out = generate_axis_points_code(
        "Y", 2,                     # Axis name and coordinate index
        4, 4,                       # Main coord CP channels (start->end): Oblique(0,0,0) -> Oblique(0,1,0)
        3, 1,                       # Other coord CP channels (start->end): LinZ(0,0,0) -> LinX(0,0,0)
        offset_zero, offset_y1,     # Main coord offsets
        offset_zero, offset_zero,   # Other coord offsets
        num_additional_oblique_points_per_side,
        weight_index, cp_chan_out
    )
    res *= axis_code_y

    # ==================================
    # === Oblique Z (Axis Z) Points ===
    # ==================================
    axis_code_z, weight_index, cp_chan_out = generate_axis_points_code(
        "Z", 3,                     # Axis name and coordinate index
        4, 4,                       # Main coord CP channels (start->end): Oblique(0,0,0) -> Oblique(0,0,1)
        2, 1,                       # Other coord CP channels (start->end): LinY(0,0,0) -> LinX(0,0,0)
        offset_zero, offset_z1,     # Main coord offsets
        offset_zero, offset_zero,   # Other coord offsets
        num_additional_oblique_points_per_side,
        weight_index, cp_chan_out
    )
    res *= axis_code_z

    # --- Finalization ---
    # Total points = (points per side * 2 directions) * 3 axes
    total_points_generated = num_additional_oblique_points_per_side * 2 * 3
    # Total weights = (3 weights per coord per iter) * (points per side * 2 directions) * 3 axes
    total_weights_used = 3 * num_additional_oblique_points_per_side * 2 * 3

    res *= """

        # --- End of Additional Oblique Point Calculations ---
        # Total points generated per (x,y,z): $total_points_generated
        # Total weights consumed per (x,y,z): $total_weights_used (starting from index 10)
        # Final output CP channel used: $(cp_chan_out - 1)

    end # End main bounds check if

        return nothing

end #apply_weights_to_locs

"""

    return res
end

# Example Usage:
num_points_per_side = 2 # Generate 2 points from start->end and 2 from end->start per axis
generated_cuda_code = generate_additional_oblique_code(num_points_per_side)

# Save or print the generated code
filepath = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights_additional_unrolled_kernel.jl"
open(filepath, "w") do file
    write(file, generated_cuda_code)
end

println("Generated CUDA code saved to: $filepath")
# println(generated_cuda_code) # Optional: print to console


# main idea now is what we need is to adapt the angle of the current point up to the next it
# so we are putting a limit to maximum change on the angles on both planes and it should in principle lead to regular shapes
# problem how far one can travbel in the direction of the angle - probably we can deal with the problem like
# on main axis for example x limit it to besmaller than x of next main oblique point
#     on y,z we can most ptobably define the plane using linears and sv centers - to rethink 

#     Maybe would be good to define a plane from the current supervoxel center and neighour in x and get a plane by rotating a plane around this line 
#     similarly with other exes the intersection of each 2 planes should give a line - maybe a triangle inside this lines will be correct spot ?
    