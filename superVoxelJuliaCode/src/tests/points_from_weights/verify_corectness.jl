cd("/workspaces/superVoxelJuliaCode_lin_sampl")


# In the Julia REPL:
using Pkg

Pkg.activate(".")  # Activate the current directory as project
# Pkg.add("Unitful")
# Pkg.instantiate()

using Revise
using Meshes
using LinearAlgebra
using GLMakie
using Combinatorics
using SplitApplyCombine
using CUDA
using Combinatorics
using Random
using Statistics
using ChainRulesCore
using Test
using Unitful # Often used in Meshes.jl, good practice to include
using LinearAlgebra # For norm, though Meshes.measure is preferred
using GeometryBasics # Import GeometryBasics to access coordinates function


using Logging
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")


includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern _old.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/sv_centr_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/additional_points_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/tests/points_from_weights/check_is_star.jl")

GLMakie.activate!()

Random.seed!(29)#3; 27
is_point_per_triangle = true
const global num_additional_oblique_points_per_side = 2


#radius of supervoxel
radiuss = (Float32(3.1), Float32(4.3), Float32(4.7))
#size of each voxel - here we use isovolumetric voxels
spacing = (Float32(1.0), Float32(1.0), Float32(1.0))
#size of the image tensor including batch size; x,y,z dimensions are set to be a
batch_size = 2
a = 71
image_shape = (a, a, a, 2)
#we are setting weights shape - here making it too big for needs for convinience
weights_shape = Int.(round.((a / 2, a / 2, a / 2, 100)))

#we are initializing the grid of points (the regular ones) in sv_centers
#control_points - is place holder of control points for the SVs that are inferred from weights
#tetrs - is indicating indicating the division of supervoxels into tetrahedrons using the control points and sv_centers
example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
sv_centers, control_points, tetrs, dims = example_set_of_svs

#we copy it to get the same control points for all the batches (for testing)
control_points = repeat(control_points, inner=(1, 1, 1, 1, 1, batch_size))
#we are initializing the weights of the SVs a bit too big for convinience
weights_shape = Int.(round.((a / 2, a / 2, a / 2, 110, batch_size)))
rng = MersenneTwister(78)#46  # Using seed 42 for reproducible results

weights = rand(rng, weights_shape...) #ones(weights_shape).-0.5 #
#we initialize random image - in this case it do not influence the outcome as weights are not learned
image_shape = (a, a, a, 2, batch_size)
# Initialize a Mersenne Twister RNG with a specific seed for reproducibility
source_arr = rand(rng, image_shape...)
#cast to float32
weights = Float32.(weights)
control_points = Float32.(control_points)
source_arr = Float32.(source_arr)
sv_centers = Float32.(sv_centers)


#Here we are getting the constant values for the kernel  - like thread blocks etc.
threads_apply_w, blocks_apply_w, num_blocks_z_pure_sv, num_blocks_y_pure_sv = prepare_for_apply_weights_to_locs_kern(size(sv_centers), weights_shape, batch_size)
#casting data to CUDA and executing functions that changes the location of the sv centers not more than 0.75 times radius in each direction
sv_centers_out = call_apply_weights_sv(CuArray(sv_centers), CuArray(weights), radiuss, threads_apply_w, blocks_apply_w, num_blocks_z_pure_sv, batch_size, num_blocks_y_pure_sv, [])
CUDA.synchronize()


#first we get the control constants like number of threadblocks etc for executing main control points kernel and then we execute it 

threads_apply_w, blocks_apply_w, num_blocks_z_pure, num_blocks_y_pure_w = prepare_for_apply_weights_to_locs_kern(size(control_points), weights_shape, batch_size)
# control_points=Float32.(control_points)
control_points_out = call_apply_weights_to_locs_kern(sv_centers_out, size(control_points), CuArray(weights), threads_apply_w, blocks_apply_w, num_blocks_z_pure, num_blocks_y_pure_w, [])
CUDA.synchronize()


#first we execute code to augment points with additional oblique points 

control_points_out = call_apply_weights_to_locs_kern_add_a(sv_centers_out, control_points_out, CuArray(weights), threads_apply_w, blocks_apply_w, num_blocks_z_pure, num_blocks_y_pure_w, [])
CUDA.synchronize()


threads_tetr_set, blocks_tetr_set = prepare_for_set_tetr_dat(image_shape, size(tetrs), batch_size)
tetr_dat_out = call_set_tetr_dat_kern(CuArray(tetrs), CuArray(source_arr), control_points_out, sv_centers_out, threads_tetr_set, blocks_tetr_set, spacing, batch_size)
CUDA.synchronize()


check_star_shape(Array(tetr_dat_out))



main_index = 0

max_index = (get_num_tetr_in_sv(false) * main_index) + 11
min_index = (get_num_tetr_in_sv(false) * main_index) + 1
sv_center_index = tetrs[min_index, 1, 1:4]
sv_centr = Array(sv_centers_out)[Int(sv_center_index[1]), Int(sv_center_index[2]), Int(sv_center_index[3]), :, 1]
sv_centr_p_x = Array(sv_centers_out)[Int(sv_center_index[1]), Int(sv_center_index[2] - 1), Int(sv_center_index[3] - 1), :, 1]
sv_centr_p_xy = Array(sv_centers_out)[Int(sv_center_index[1]), Int(sv_center_index[2]), Int(sv_center_index[3] - 1), :, 1]
sv_centr_p_y = Array(sv_centers_out)[Int(sv_center_index[1]), Int(sv_center_index[2])-1, Int(sv_center_index[3]), :, 1]


barycenters_indicies = tetrs[min_index:max_index, 5, 1:4]
triangles_indicies = tetrs[min_index:max_index, 2:4, :]

base_index = tetrs[min_index, 2, 1:4]
min_index








using Meshes
using GLMakie
using LinearAlgebra # Ensure LinearAlgebra is loaded if not already
# using Makie: LabelElement # Removed this import

"""
    visualize_control_points_at_base(triangles_indicies, control_points_out, base_index, sv_centr, sv_centr_p_x, sv_centr_p_xy, sv_centr_p_y; batch_idx=1)

Extracts triangle definitions, constructs Meshes.Triangle objects,
and visualizes specific control point types (1-8) located at the `base_index`
within the `control_points_out` tensor. Also visualizes the triangles themselves,
the four provided supervoxel center points, and lines from the main SV center (`sv_centr`)
to the barycenter of each triangle.
The color of each triangle corresponds to its index in the input `triangles_indicies`.
A legend identifies the control point types and SV centers, and a label explains the triangle coloring.

Args:
    triangles_indicies (Array{<:Real, 3}): Tensor of size (num_triangles, 3, 4).
        Each [i, j, :] contains [ix, iy, iz, point_type] indices into `control_points_out`
        for the j-th vertex of the i-th triangle.
    control_points_out (Array{<:Real, 6}): Tensor containing control point coordinates.
        Expected dimensions: (nx, ny, nz, num_point_types, 3, num_batches).
    base_index (AbstractVector{<:Integer}): A vector/tuple of 3 integers [bx, by, bz]
        specifying the base location in `control_points_out` to visualize points 1-8 from.
    sv_centr (AbstractVector{<:Real}): Coordinates [x, y, z] of the main SV center.
    sv_centr_p_x (AbstractVector{<:Real}): Coordinates [x, y, z] of the SV center +x neighbor.
    sv_centr_p_xy (AbstractVector{<:Real}): Coordinates [x, y, z] of the SV center +x+y neighbor.
    sv_centr_p_y (AbstractVector{<:Real}): Coordinates [x, y, z] of the SV center +y neighbor.
    batch_idx (Int): The batch index to use (default: 1).

Returns:
    Tuple: (fig, ax, triangles, legend_points)
        - fig: The GLMakie Figure object.
        - ax: The GLMakie Axis3 object.
        - triangles: A vector of Meshes.Triangle objects.
        - legend_points: A dictionary mapping point names to Meshes.Point objects.
"""
function visualize_control_points_at_base(triangles_indicies, control_points_out, base_index, sv_centr, sv_centr_p_x, sv_centr_p_xy, sv_centr_p_y; batch_idx=1)
    println("Step 1: Defining point types and colors...")
    # Define mapping for point types 1-8 and the additional main_oblique_x
    point_type_map = Dict(
        1 => (name="lin_x", color=:blue),
        2 => (name="lin_y", color=:cyan),
        3 => (name="lin_z", color=:purple),
        4 => (name="main_oblique", color=:orange),
        5 => (name="oblique_x_a", color=:red),
        6 => (name="oblique_x_b", color=:darkred),
        7 => (name="oblique_x_c", color=:pink),
        8 => (name="oblique_x_d", color=:magenta),
        9 => (name="main_oblique_x", color=:darkorange4) # Added entry for main_oblique_x (using type 9 conceptually)
    )
    println("Point type map defined.")
    # --- Code from Step 1 ---
    println("\nStep 2: Extracting triangle vertex coordinates...")
    num_triangles = size(triangles_indicies, 1)
    triangle_vertex_coords_list = Vector{Vector{Vector{Float64}}}(undef, num_triangles) # Preallocate outer vector

    for i in 1:num_triangles
        current_triangle_points = Vector{Vector{Float64}}(undef, 3) # Preallocate inner vector
        for j in 1:3 # Iterate through 3 vertices of the triangle
            # Get indices [ix, iy, iz, point_type]
            idx_info = triangles_indicies[i, j, :]
            ix, iy, iz, point_type = Int.(round.(idx_info)) # Ensure integer indices

            # Basic bounds check (optional but recommended)
            if checkbounds(Bool, control_points_out, ix, iy, iz, point_type, :, batch_idx)
                # Extract coordinates [x, y, z] for the vertex
                coords = control_points_out[ix, iy, iz, point_type, :, batch_idx]
                current_triangle_points[j] = Float64.(coords) # Store as Float64
            else
                println("Warning: Indices out of bounds for triangle $i, vertex $j: [$ix, $iy, $iz, $point_type]")
                current_triangle_points[j] = [NaN, NaN, NaN] # Placeholder for invalid points
            end
        end
        triangle_vertex_coords_list[i] = current_triangle_points
    end

    # --- Added Step 2.5 ---
    colors_by_original_index = Vector{Symbol}(undef, num_triangles)
    for i in 1:num_triangles
        has_lin_z = false
        has_lin_y = false
        for j in 1:3
            # Check bounds before accessing index - use NaN or skip if out of bounds
            if checkbounds(Bool, triangles_indicies, i, j, 4)
                point_type = Int(round(triangles_indicies[i, j, 4]))
                if point_type == 3 # lin_z
                    has_lin_z = true
                    break # Prioritize green
                elseif point_type == 2 # lin_y
                    has_lin_y = true
                end
            else
                println("Warning: Index out of bounds when checking point type for triangle $i, vertex $j.")
                # Decide default color or handle error - using gray here
                has_lin_z = false
                has_lin_y = false
                break
            end
        end

        if has_lin_z
            colors_by_original_index[i] = :green
        elseif has_lin_y
            colors_by_original_index[i] = :red
        else
            colors_by_original_index[i] = :gray # Default color
        end
    end


    # --- Modified Step 3: Build triangles and barycenters from raw data ---
    triangles = Meshes.Triangle[] # For visualization
    final_triangle_colors = Symbol[] # For visualization
    barycenters_xyz = Vector{NTuple{3,Float64}}() # Store barycenter coordinates as tuples

    for i in 1:num_triangles
        p_coords = triangle_vertex_coords_list[i]
        # Check for NaN coordinates introduced by bounds errors
        if any(isnan, vcat(p_coords...))
            continue
        end
        # p_coords is a Vector of 3 points, each is a Vector{Float64} of length 3
        # Compute barycenter manually from raw numbers
        x = (p_coords[1][1] + p_coords[2][1] + p_coords[3][1]) / 3
        y = (p_coords[1][2] + p_coords[2][2] + p_coords[3][2]) / 3
        z = (p_coords[1][3] + p_coords[2][3] + p_coords[3][3]) / 3
        push!(barycenters_xyz, (x, y, z))
        # For visualization, create Meshes.Point and Meshes.Triangle
        p1 = Meshes.Point(p_coords[1]...)
        p2 = Meshes.Point(p_coords[2]...)
        p3 = Meshes.Point(p_coords[3]...)
        triangle = Meshes.Triangle(p1, p2, p3)
        push!(triangles, triangle)
        push!(final_triangle_colors, colors_by_original_index[i])
    end

    legend_points = Dict{String,Meshes.Point}()
    bx, by, bz = Int.(round.(base_index)) # Ensure integer indices

    max_point_type_to_visualize = 8 # Only visualize types 1 through 8 initially from the map
    for point_type in 1:max_point_type_to_visualize
        if haskey(point_type_map, point_type)
            name = point_type_map[point_type].name
            # Basic bounds check
            if checkbounds(Bool, control_points_out, bx, by, bz, point_type, :, batch_idx)
                coords = control_points_out[bx, by, bz, point_type, :, batch_idx]
                p = Meshes.Point(Float64.(coords)...)
                legend_points[name] = p
            else
                println("Warning: Indices out of bounds for legend point type $point_type at base_index: [$bx, $by, $bz]")
            end
        else
            println("Warning: No definition found in point_type_map for point type $point_type.")
        end
    end

    # Extract the additional "main_oblique_x" point
    main_oblique_x_idx = (bx + 1, by, bz) # Index for the next main oblique along x
    main_oblique_type = 4 # Point type for main oblique
    main_oblique_x_name = "main_oblique_x"
    if checkbounds(Bool, control_points_out, main_oblique_x_idx..., main_oblique_type, :, batch_idx)
        coords_x = control_points_out[main_oblique_x_idx..., main_oblique_type, :, batch_idx]
        p_x = Meshes.Point(Float64.(coords_x)...)
        legend_points[main_oblique_x_name] = p_x
    else
        println("Warning: Indices out of bounds for $main_oblique_x_name at index: $main_oblique_x_idx")
    end

    println("Finished extracting legend points.")

    # --- Add SV Center Points ---
    sv_centr_point = Meshes.Point(Float64.(sv_centr)...) # Convert main center to Point
    sv_center_points = Dict(
        "sv_centr" => (point=sv_centr_point, color=:green),
        "sv_centr_p_x" => (point=Meshes.Point(Float64.(sv_centr_p_x)...), color=:lightgreen),
        "sv_centr_p_xy" => (point=Meshes.Point(Float64.(sv_centr_p_xy)...), color=:darkgreen),
        "sv_centr_p_y" => (point=Meshes.Point(Float64.(sv_centr_p_y)...), color=:springgreen)
    )
    # Add SV centers to the legend_points dictionary for potential later use
    for (name, data) in sv_center_points
        legend_points[name] = data.point
        println("  Added SV center point: $name")
    end
    println("Finished adding SV Center points.")

    # --- Modified Step 5 ---
    println("\nStep 5: Visualizing triangles, points, and segments...")
    GLMakie.activate!() # Ensure GLMakie is active
    fig = Figure(size=(1000, 700)) # Adjust figure size if needed
    # Modify the title - keep it concise
    ax = Axis3(fig[1, 1], aspect=:data, title="Control Points at Base $base_index & SV Centers") # Updated title

    # Visualize the triangles themselves
    if !isempty(triangles)
        # Color triangles based on the determined colors
        viz!(ax, triangles, color=final_triangle_colors, alpha=0.7)
        println("  Added triangles to plot with specific colors.")
    end

    # --- Visualize lines from SV center to barycenters ---
    println("  Creating and visualizing lines to barycenters...")
    barycenter_lines = Meshes.Segment[]
    for bary_xyz in barycenters_xyz
        barycenter_point = Meshes.Point(bary_xyz...)
        push!(barycenter_lines, Meshes.Segment(sv_centr_point, barycenter_point))
    end
    if !isempty(barycenter_lines)
        viz!(ax, barycenter_lines, color=:gray, linestyle=:dash, linewidth=1.5)
        println("    Added $(length(barycenter_lines)) lines from SV center to triangle barycenters.")
    end

    # Visualize the legend points and collect legend entries
    legend_elements = [] # Renamed from legend_entries for clarity
    legend_labels = []   # Separate list for labels

    # Visualize Control Points (1-8 and main_oblique_x)
    if !isempty(legend_points)
        # Iterate through defined point types including the conceptual type 9 for main_oblique_x
        for point_type in vcat(1:max_point_type_to_visualize, 9) # Include type 9
            if haskey(point_type_map, point_type)
                name = point_type_map[point_type].name
                color = point_type_map[point_type].color
                if haskey(legend_points, name)
                    point_to_plot = legend_points[name]
                    # Plot the point
                    viz!(ax, point_to_plot, color=color, pointsize=15) # No label needed here
                    # Create a MarkerElement for the legend
                    push!(legend_elements, MarkerElement(color=color, marker=:circle, markersize=15, markerstrokecolor=:black))
                    push!(legend_labels, name) # Store the label string

                    println("  Plotting control point: $name with color $color")
                end
            end
        end
    end

    # Visualize SV Center Points
    println("  Plotting SV Center points...")
    for (name, data) in sv_center_points
        viz!(ax, data.point, color=data.color, pointsize=20) # Larger size for SV centers
        push!(legend_elements, MarkerElement(color=data.color, marker=:dtriangle, markersize=20, markerstrokecolor=:black)) # Use triangle marker
        push!(legend_labels, name)
        println("    Plotting SV center: $name with color $(data.color)")
    end

    # Create the Legend manually in the layout
    if !isempty(legend_elements)
        # Create Legend with elements and labels directly (removed nesting)
        Legend(fig[1, 2], legend_elements, legend_labels, "Points Legend", nbanks=2) # Updated title, Re-added nbanks=2
        println("  Added point legend to layout.")
    else
        println("  No legend points found or extracted to visualize.")
    end

    # Create and visualize the segments
    println("  Creating and visualizing segments...")
    segment_points = ["main_oblique", "oblique_x_a", "oblique_x_b", "oblique_x_c", "oblique_x_d", "main_oblique_x"]
    segments_to_draw = []
    all_segment_points_found = true
    for i in 1:(length(segment_points)-1)
        p1_name = segment_points[i]
        p2_name = segment_points[i+1]
        if haskey(legend_points, p1_name) && haskey(legend_points, p2_name)
            p1 = legend_points[p1_name]
            p2 = legend_points[p2_name]
            push!(segments_to_draw, Meshes.Segment(p1, p2))
            println("    Created segment: $p1_name -> $p2_name")
        else
            println("Warning: Could not create segment $p1_name -> $p2_name. Point(s) missing.")
            all_segment_points_found = false
        end
    end

    if !isempty(segments_to_draw)
        viz!(ax, segments_to_draw, color=:black, linewidth=3) # Use linewidth for thickness
        println("  Added segments to plot.")
    end


    # Add a separate label explaining triangle colors below the legend
    Label(fig[2, 1:2], "Triangle color: Green (contains lin_z), Red (contains lin_y, no lin_z), Gray (otherwise).", tellwidth=false, halign=:center)
    println("  Added triangle color explanation label.")


    display(fig)
    println("Visualization complete.")

    return fig, ax, triangles, legend_points
end

# Make sure the input arrays are on the CPU
control_points_out_cpu = Array(control_points_out) # Assuming control_points_out might be a CuArray
triangles_indicies_cpu = Array(triangles_indicies) # Assuming triangles_indicies might be from tetrs (CPU)
base_index_cpu = Array(base_index) # Ensure base_index is a CPU array/vector
# Ensure SV centers are also CPU arrays
sv_centr_cpu = Array(sv_centr)
sv_centr_p_x_cpu = Array(sv_centr_p_x)
sv_centr_p_xy_cpu = Array(sv_centr_p_xy)
sv_centr_p_y_cpu = Array(sv_centr_p_y)


# Define the batch index you want to visualize
batch_to_visualize = 1

# Call the function with the new arguments
fig_legend, ax_legend, extracted_triangles, extracted_legend_points = visualize_control_points_at_base(
    triangles_indicies_cpu,
    control_points_out_cpu,
    base_index_cpu,
    sv_centr_cpu,          # Pass the SV center points
    sv_centr_p_x_cpu,
    sv_centr_p_xy_cpu,
    sv_centr_p_y_cpu,
    batch_idx=batch_to_visualize
)


using Logging # Add logging if not already present

"""
    print_triangle_point_names(triangles_indicies)

Prints the human-readable names of the point types forming each triangle.

Args:
    triangles_indicies (Array{<:Real, 3}): Tensor of size (num_triangles, 3, 4).
        Each [i, j, :] contains [ix, iy, iz, point_type] indices.
        The function uses only the point_type (the 4th element).
"""
function print_triangle_point_names(triangles_indicies)
    # Define mapping from point type index to human-readable name
    point_type_map = Dict(
        1 => "lin_x",
        2 => "lin_y",
        3 => "lin_z",
        4 => "main_oblique",
        5 => "oblique_x_a",
        6 => "oblique_x_b",
        7 => "oblique_x_c",
        8 => "oblique_x_d"
        # Add more mappings if other point types are possible
    )

    num_triangles = size(triangles_indicies, 1)
    println("Analyzing $num_triangles triangles:")

    for i in 1:num_triangles
        vertex_names = String[] # Store names for the current triangle
        valid_triangle = true
        for j in 1:3 # Iterate through the 3 vertices of the triangle
            # Extract the point type (the 4th element in the last dimension)
            # Ensure it's treated as an integer for dictionary lookup
            point_type = Int(round(triangles_indicies[i, j, 4]))

            # Look up the name in the map
            point_name = get(point_type_map, point_type, "Unknown Type ($point_type)")
            if point_name == "Unknown Type ($point_type)"
                @warn "Triangle $i, Vertex $j: Found unknown point type $point_type"
                valid_triangle = false # Mark triangle as potentially invalid if type is unknown
            end
            push!(vertex_names, point_name)
        end

        # Print the result for the current triangle
        println("Triangle $i: [$(join(vertex_names, ", "))]")
        # Optionally add more details if needed, like the raw indices
        # println("  Raw indices: $(triangles_indicies[i, :, :])")
    end
    println("Finished analyzing triangles.")
end

# Example Usage:
# Assuming 'triangles_indicies' is already defined and populated in your environment
# Make sure it's on the CPU if it came from GPU operations
triangles_indicies_cpu = Array(triangles_indicies)


# You can inspect the results
for (name, point) in extracted_legend_points
    println("- $name: $point")
end
print_triangle_point_names(triangles_indicies_cpu)



# control_points_out_cpu[2, 3, 2, 5:9, :,1]