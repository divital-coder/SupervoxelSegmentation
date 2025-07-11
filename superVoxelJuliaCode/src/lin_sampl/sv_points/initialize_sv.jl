
using SplitApplyCombine, KernelAbstractions

"""
get 4 dimensional array of cartesian indicies of a 3 dimensional array
thats size is passed as an argument dims
"""
function get_base_indicies_arr(dims)
    indices = CartesianIndices(dims)
    # indices=collect.(Tuple.(collect(indices)))
    indices = Tuple.(collect(indices))
    indices = collect(Iterators.flatten(indices))
    indices = reshape(indices, (3, dims[1], dims[2], dims[3]))
    indices = permutedims(indices, (2, 3, 4, 1))
    return indices
end#get_base_indicies_arr


function get_corrected_dim(ax, radius, image_shape)
    return Int(floor((image_shape[ax]) / (radius[ax] * 2)))
end

function get_dif(ax, image_shape, dims, radius, pad)
    # return max(floor((image_shape[ax]-((dims[ax]+1).*diam))/2),2.0)+pad
    # return floor((image_shape[ax]-((dims[ax]).*(radius[ax]))))+((radius[ax]))
    return floor((image_shape[ax] - ((dims[ax]) .* (radius[ax] * 2))) / 2) + (radius[ax])

end

"""
initialize sv centers coordinates  we need sv centers that is in each axis 1 bigger than control points
"""
function get_sv_centers(radius, image_shape, pad=0.0)
    dims = (get_corrected_dim(1, radius, image_shape), get_corrected_dim(2, radius, image_shape), get_corrected_dim(3, radius, image_shape))
    diffs = (get_dif(1, image_shape, dims, radius, pad), get_dif(2, image_shape, dims, radius, pad), get_dif(3, image_shape, dims, radius, pad))

    #print("\n dddd dims $(dims) diffs $(diffs) radius $(radius);  $(radius[1])\n")

    res = (get_base_indicies_arr(dims) .- 1)#*diam
    res = Float32.(res)

    res[:, :, :, 1] = (res[:, :, :, 1] .* (radius[1] * 2)) .+ diffs[1]
    res[:, :, :, 2] = (res[:, :, :, 2] .* (radius[2] * 2)) .+ diffs[2]
    res[:, :, :, 3] = (res[:, :, :, 3] .* (radius[3] * 2)) .+ diffs[3]
    return res, dims, diffs

end


"""
flips the value of the index of the tuple at the position ind needed for get_linear_between function
"""
function flip_num(base_ind, tupl, ind)
    arr = collect(tupl)
    # arr=append!(arr,[4])
    if (arr[ind] == base_ind[ind])
        arr[ind] = base_ind[ind] + 1
    else
        arr[ind] = base_ind[ind]
    end
    return arr
end

"""
we can identify the line between two corners that go obliquely through the wall of the cube
it connects points that has 2 coordinates diffrent and one the same 
we can also find a point in the middle so it will be in lin_x if this common index is 1 and in lin_y if it is 2 and lin_z if 3
next if we have 1 it is pre and if 2 post
    control_points first dimension is lin_x, lin_y, lin_z, oblique
"""
function get_linear_between(base_ind, ind_1, ind_2)
    if (ind_1[1] == ind_2[1])
        return [ind_1[1], base_ind[2], base_ind[3], 1]
    end
    if (ind_1[2] == ind_2[2])
        return [base_ind[1], ind_1[2], base_ind[3], 2]
    end

    return [base_ind[1], base_ind[2], ind_1[3], 3]
end


"""
helper function to set values to the all_surf_triangles array in the appropriate index
"""
function set_to_index(all_surf_triangles, add_ind, res_main_ind, el1, el2, el3, el4, el5)
    all_surf_triangles[res_main_ind+add_ind, 1, :] = el1
    all_surf_triangles[res_main_ind+add_ind, 2, :] = el2
    all_surf_triangles[res_main_ind+add_ind, 3, :] = el3
    all_surf_triangles[res_main_ind+add_ind, 4, :] = el4
    all_surf_triangles[res_main_ind+add_ind, 5, :] = el5

end


"""
get a flattened array of all surface triangles of all supervoxels
in first dimension every get_num_tetr_in_sv() elements are a single supervoxel
second dimension is size 5 and is in order sv_center, point a,point b,point c,centroid of the base
    where centroid is a placeholder for centroid of the triangle a,b,c
in last dimension we have x,y,z coordinates of the point
currently we have just indicies to the appropriate arrays -> it need to be populated after weights get applied        
"""
function get_tetr_triangles_in_corner_on_kern(indices, corner_add, all_surf_triangles, index, corn_num, num_additional_oblique_points_per_side)
    

    
    # --- Configuration ---
    N = num_additional_oblique_points_per_side * 2 # Number of additional points per axis edge
    num_tetr_per_axis_subdivision = N + 1 # Tetrahedrons generated per face subdivision along one edge
    num_tetr_per_corner = 6 * num_tetr_per_axis_subdivision # Total tetrahedrons for this corner (6 faces subdivided)
    num_tetr_per_sv = 4 * num_tetr_per_corner # Total tetrahedrons for the supervoxel

    # --- Base Indices and Points ---
    base_ind = indices[index[1], :] # Base index (x,y,z) for the current supervoxel corner
    # Main oblique point (corner of the cube subdivision)
    corner = (base_ind[1] + corner_add[1], base_ind[2] + corner_add[2], base_ind[3] + corner_add[3])
    corner = Float32.(append!(collect(corner), [4])) # Channel 4 for main oblique

    # Supervoxel center (common apex for all tetrahedrons)
    sv_center = Float32.([base_ind[1], base_ind[2], base_ind[3], -1.0]) # Channel -1 for sv_center
    sv_center = (sv_center .+ 1) # Adjust index for sv_centers array

    # Adjacent main oblique points (relative to corner)
    p_a = Float32.(flip_num(base_ind, corner, 1)) # Point along X axis from corner
    p_b = Float32.(flip_num(base_ind, corner, 2)) # Point along Y axis from corner
    p_c = Float32.(flip_num(base_ind, corner, 3)) # Point along Z axis from corner

    # Linear interpolation points on faces (between adjacent main obliques)
    p_ab = Float32.(get_linear_between(base_ind, p_a, p_b)) # Lin point on XY face (Channel 1/2/3)
    p_ac = Float32.(get_linear_between(base_ind, p_a, p_c)) # Lin point on XZ face
    p_bc = Float32.(get_linear_between(base_ind, p_b, p_c)) # Lin point on YZ face

    # --- Calculate Indices for Additional Oblique Points ---
    start_channel_x = 5
    start_channel_y = start_channel_x + N
    start_channel_z = start_channel_y + N

    # Helper function to create an oblique point index vector
    function create_oblique_point(base_corner, axis_index, channel, corner_offset)
        p = copy(base_corner)
        p[4] = channel
        # Adjust coordinate based on corner offset ONLY for the specified axis
        p[axis_index] -= corner_offset[axis_index]
        return p
    end

    # Generate N points for each axis in the correct order (corner -> other end)
    oblique_points_x = Vector{Vector{Float32}}(undef, N)
    oblique_points_y = Vector{Vector{Float32}}(undef, N)
    oblique_points_z = Vector{Vector{Float32}}(undef, N)

    # half_N = num_additional_oblique_points_per_side # = N/2

    # X-axis points (along edge corner -> p_a)
    for i in 1:N
        oblique_points_x[i] = create_oblique_point(corner, 1, start_channel_x + i - 1, corner_add)
    end

    # Y-axis points (along edge corner -> p_b)
    for i in 1:N
        oblique_points_y[i] = create_oblique_point(corner, 2, start_channel_y + i - 1, corner_add)

    end

    # Z-axis points (along edge corner -> p_c)
    for i in 1:N
        oblique_points_z[i] = create_oblique_point(corner, 3, start_channel_z + i - 1, corner_add)

    end

    # --- Set Tetrahedron Indices ---
    dummy = Float32.([-1.0, -1.0, -1.0, -1.0]) # Placeholder for centroid
    # Calculate base index for this corner's tetrahedrons in the flattened array
    res_main_ind = (index[1] - 1) * num_tetr_per_sv + (corn_num - 1) * num_tetr_per_corner
    add_ind_counter = 1 # Counter for the tetrahedron index within this corner (1 to num_tetr_per_corner)

    # --- X-Axis Subdivisions ---
    # Subdivide faces adjacent to the corner-p_a edge: (corner, p_ab, p_a) and (corner, p_ac, p_a)
    # using oblique_points_x which lie on the corner-p_a edge.
    for i in 0:N
        # Define the segment start/end points along the corner-p_a edge for this subdivision
        p_start_edge = (i == 0) ? corner : oblique_points_x[i]
        p_end_edge   = (i == N) ? p_a    : oblique_points_x[i+1]

        # Tetrahedron 1: Base (p_start_edge, p_ab, p_end_edge)
        set_to_index(all_surf_triangles, add_ind_counter, res_main_ind, sv_center, p_start_edge, p_ab, p_end_edge, dummy)
        add_ind_counter += 1
        # Tetrahedron 2: Base (p_start_edge, p_ac, p_end_edge)
        set_to_index(all_surf_triangles, add_ind_counter, res_main_ind, sv_center, p_start_edge, p_ac, p_end_edge, dummy)
        add_ind_counter += 1
    end

    # --- Y-Axis Subdivisions ---
    # Subdivide faces adjacent to the corner-p_b edge: (corner, p_ab, p_b) and (corner, p_bc, p_b)
    # using oblique_points_y which lie on the corner-p_b edge.
     for i in 0:N
        # Define the segment start/end points along the corner-p_b edge
        p_start_edge = (i == 0) ? corner : oblique_points_y[i]
        p_end_edge   = (i == N) ? p_b    : oblique_points_y[i+1]

        # Tetrahedron 1: Base (p_start_edge, p_ab, p_end_edge)
        set_to_index(all_surf_triangles, add_ind_counter, res_main_ind, sv_center, p_start_edge, p_ab, p_end_edge, dummy)
        add_ind_counter += 1
        # Tetrahedron 2: Base (p_start_edge, p_bc, p_end_edge)
        set_to_index(all_surf_triangles, add_ind_counter, res_main_ind, sv_center, p_start_edge, p_bc, p_end_edge, dummy)
        add_ind_counter += 1
    end

    # --- Z-Axis Subdivisions ---
    # Subdivide faces adjacent to the corner-p_c edge: (corner, p_ac, p_c) and (corner, p_bc, p_c)
    # using oblique_points_z which lie on the corner-p_c edge.
     for i in 0:N
        # Define the segment start/end points along the corner-p_c edge
        p_start_edge = (i == 0) ? corner : oblique_points_z[i]
        p_end_edge   = (i == N) ? p_c    : oblique_points_z[i+1]

        # Tetrahedron 1: Base (p_start_edge, p_ac, p_end_edge)
        set_to_index(all_surf_triangles, add_ind_counter, res_main_ind, sv_center, p_start_edge, p_ac, p_end_edge, dummy)
        add_ind_counter += 1
        # Tetrahedron 2: Base (p_start_edge, p_bc, p_end_edge)
        set_to_index(all_surf_triangles, add_ind_counter, res_main_ind, sv_center, p_start_edge, p_bc, p_end_edge, dummy)
        add_ind_counter += 1
    end

    # Verification (optional): Check if add_ind_counter - 1 == num_tetr_per_corner
    # @assert add_ind_counter - 1 == num_tetr_per_corner "Mismatch in expected number of tetrahedrons per corner. Expected $(num_tetr_per_corner), Got $(add_ind_counter-1)"

end

@kernel function set_triangles_kern(@Const(indices), all_surf_triangles, num_additional_oblique_points_per_side)

    index = @index(Global, Cartesian) # index is (sv_index, corner_index)

    # Determine which corner this thread is processing based on index[2] (corn_num)
    # and calculate the corresponding corner_add offset
    corner_add = (0.0, 0.0, 0.0) # Default for corner 1 (corn_num=1)
    corn_num = index[2]
    if (corn_num == 2)
        corner_add = (1.0, 1.0, 0.0)
    elseif (corn_num == 3)
        corner_add = (0.0, 1.0, 1.0)
    elseif (corn_num == 4)
        corner_add = (1.0, 0.0, 1.0)
    end

    # Call the function to set triangles for the assigned corner
    get_tetr_triangles_in_corner_on_kern(indices, corner_add, all_surf_triangles, index, corn_num, num_additional_oblique_points_per_side)

end


"""
return number of tetrahedrons in single supervoxel
"""
function get_num_tetr_in_sv(is_point_per_triangle=false)
    # This function needs to decide based on is_point_per_triangle OR the new logic.
    # Assuming the new logic replaces the old is_point_per_triangle subdivision for now.
    if (is_point_per_triangle)
         # The augmentation logic multiplies the number of tetrs by 3
         # So we start with the new base number and multiply by 3
         N = num_additional_oblique_points_per_side * 2
         num_tetr_per_corner = 6 * (N + 1)
         base_num_tetr = 4 * num_tetr_per_corner
         return base_num_tetr * 3
     else
        N = num_additional_oblique_points_per_side * 2
        num_tetr_per_corner = 6 * (N + 1)
        return 4 * num_tetr_per_corner # 4 corners processed per supervoxel
     end
end

"""
calculate shape of the tetr_dat array - array with tetrahedrons that are created by the center of the supervoxel
"""
function get_tetr_dat_shape(dims)
    num_sv = (dims[1]-2) * (dims[2]-2) * (dims[3]-2) # Number of supervoxels
    num_tetr = get_num_tetr_in_sv()
    return (num_sv * num_tetr, 5, 4)
end

"""
get a flattened array of all surface triangles of all supervoxels
in first dimension every get_num_tetr_in_sv() elements are a single supervoxel
second dimension is size 5 and is in orde sv_center, point a,point b,point c,centroid 
    where centroid is a placeholder for centroid of the triangle a,b,c
in last dimension we have x,y,z coordinates of the point
currently we have just indicies to the appropriate arrays -> it need to be populated after weights get applied        
"""
function get_flattened_triangle_data(dims, num_additional_oblique_points_per_side, is_point_per_triangle=false)
    dims_sv = dims .- 2 # Dimensions of the grid of supervoxels
    num_sv = dims_sv[1] * dims_sv[2] * dims_sv[3]

    # Create CartesianIndices for supervoxels (not control points)
    sv_indices_cart = CartesianIndices(dims_sv)
    # Convert to list of tuples and reshape for indexing
    sv_indices_tuples = Tuple.(collect(sv_indices_cart))
    sv_indices_flat = collect(Iterators.flatten(sv_indices_tuples))
    # Reshape to (num_sv, 3) - each row is (x,y,z) of a supervoxel corner relative to origin (0,0,0)
    indices_for_kernel = reshape(sv_indices_flat, (3, num_sv))
    indices_for_kernel = permutedims(indices_for_kernel, (2, 1))

    # Allocate space for the output triangle data
    # Use the potentially modified shape if is_point_per_triangle is true later
    shape_for_alloc = get_tetr_dat_shape(dims) # Allocate for base tetrs first
    all_surf_triangles = zeros(Float32, shape_for_alloc)

    dev = get_backend(all_surf_triangles) # Assuming get_backend is defined elsewhere
    # Launch kernel: grid is (num_supervoxels, 4_corners_per_supervoxel)
    set_triangles_kern(dev, 19)(Float32.(indices_for_kernel), all_surf_triangles, num_additional_oblique_points_per_side, ndrange=(num_sv, 4))
    KernelAbstractions.synchronize(dev)

    # Augmentation step if needed (outside the kernel)
    if is_point_per_triangle
        # This part still needs the plan tensor logic adapted for the new tetrahedron structure
        # The old plan tensor logic based on 48 tetrs/sv won't work directly.
        # For now, just returning the non-augmented triangles.
        @warn "is_point_per_triangle=true: Augmentation logic needs update for new tetrahedron structure. Returning non-augmented triangles."
        # Example placeholder for future update:
        # plan_tensor = get_plan_tensor_for_new_structure(...)
        # all_surf_triangles = augment_flattened_triangles_new(all_surf_triangles, plan_tensor, ...)
    end


    return all_surf_triangles
end

"""
initializing control points - to be modified later based on learnable weights
"""
function initialize_control_points(image_shape, radius, num_additional_oblique_points_per_side, is_points_per_triangle=false)
    pad = 0.0
    dims = (get_corrected_dim(1, radius, image_shape), get_corrected_dim(2, radius, image_shape), get_corrected_dim(3, radius, image_shape))
    # Control points grid is smaller than sv_centers grid
    dims_cp = dims .- 1
    icp = get_base_indicies_arr(dims_cp) # Base indices for control points

    # Calculate total number of channels needed
    # 1-3: lin_x, lin_y, lin_z
    # 4: oblique_main
    # 5 onwards: additional oblique points (3 axes * N points/axis)
    N = num_additional_oblique_points_per_side * 2
    num_base_channels = 4 + 3 * N

    # Additional channels if is_points_per_triangle is true
    # The old logic added 24 channels. The new logic needs careful consideration
    # based on how the plan tensor and augmentation are adapted.
    # Assuming for now it adds channels based on the *new* number of boundary triangles.
    num_extra_channels = 0
    if is_points_per_triangle
        # Placeholder: Needs recalculation based on the new structure's boundary triangles
        num_boundary_triangles_per_sv = 24 * (N+1) # Rough estimate, needs verification
        num_extra_channels = num_boundary_triangles_per_sv ÷ 2 # Assuming one point per pair
        @warn "is_point_per_triangle=true: Number of extra channels ($num_extra_channels) is a placeholder and needs verification based on updated augmentation logic."

    end
    num_channels = num_base_channels + num_extra_channels

    # Create the control points array initialized with base indices
    res = combinedims(map(a -> copy(icp), 1:num_channels), 4)

    return res
end#initialize_control_points


function count_zeros(arr, name::String)
    num_zeros = count(x -> x == 0.0, arr)
    num_entries = length(arr)
    percentt = (num_zeros / num_entries) * 100
    println("percent of zeros in $name: $percentt % sum: $(sum(arr))  ")
end


################
#initialize for point per triangle unrolled

############



"""
given 3 by 3 matrix where first dimension is the point index and second is the 
coordinate index we will check weather the triangles are the same - so the a set of points in both cases are the same the order of 
triangle verticies in botsh cases is not important
"""
function is_equal_point(point, points)
    for i in 1:3

        a = point[1] ≈ points[i][1]
        b = point[2] ≈ points[i][2]
        c = point[3] ≈ points[i][3]
        d = point[4] ≈ points[i][4]
        if a && b && c && d
            return true
        end
    end
    return false
end




"""
looks weather given points are present in both arrays - order of points do not matter
"""
function are_triangles_equal(triangle1, triangle2)
    # Extract points from the matrices
    points1 = [triangle1[i, :] for i in 1:3]
    points2 = [triangle2[i, :] for i in 1:3]
    a = is_equal_point(points1[1], points2)
    b = is_equal_point(points1[2], points2)
    c = is_equal_point(points1[3], points2)
    # if(a || b || c)
    #     print("** $a $b $c **")
    # end
    return (a && b && c)
end




function check_triangles(middle_coords, tetr_3d, middle_tetr, full_data=false)
    # Generate all possible (xd, yd, zd) combinations
    coords = [(xd, yd, zd) for xd in (middle_coords[1]-1):(middle_coords[1]+1),
              yd in (middle_coords[2]-1):(middle_coords[2]+1),
              zd in (middle_coords[3]-1):(middle_coords[3]+1)
              if !(xd == middle_coords[1] && yd == middle_coords[2] && zd == middle_coords[3])]

    # Function to check triangles and return results
    function check_and_collect(xd, yd, zd)
        results = []
        for i in 1:48
            for j in 1:48

                rel = (xd - middle_coords[1], yd - middle_coords[2], zd - middle_coords[3])
                submatrix_tetr_3d = tetr_3d[xd, yd, zd, i, 2:4, 1:4]
                submatrix_middle_tetr = middle_tetr[j, :, :]
                are_equal = are_triangles_equal(submatrix_tetr_3d, submatrix_middle_tetr)

                triangle_points_middle = [submatrix_middle_tetr[ii, :] for ii in 1:3]
                apex1_middle = tetr_3d[middle_coords[1], middle_coords[2], middle_coords[3], j, 1, 1:4]
                apex2_middle = tetr_3d[xd, yd, zd, j, 1, 1:4]
                triangle_points = map(tp -> [(tp[1] - middle_coords[1]), (tp[2] - middle_coords[2]), (tp[3] - middle_coords[3]), tp[4]], triangle_points_middle)
                apex1 = [(apex1_middle[1] - middle_coords[1]), (apex1_middle[2] - middle_coords[2]), (apex1_middle[3] - middle_coords[3])]
                apex2 = [(apex2_middle[1] - middle_coords[1]), (apex2_middle[2] - middle_coords[2]), (apex2_middle[3] - middle_coords[3])]

                if are_equal
                    if (((xd - middle_coords[1]) < 1 && (yd - middle_coords[2]) < 1 && (zd - middle_coords[3]) < 1) || (full_data))
                        push!(results, Dict("rel_coord" => rel,
                            # "triangle_points"=>[[rel...,submatrix_middle_tetr[ii, 4]] for ii in 1:3],
                            "triangle_points" => triangle_points,
                            "triangle_points_middle" => triangle_points_middle,
                            "apex1_middle" => tetr_3d[middle_coords[1], middle_coords[2], middle_coords[3], j, 1, 1:4],#current sv center
                            "apex2_middle" => tetr_3d[xd, yd, zd, j, 1, 1:4],#neighbour sv center
                            "apex1" => apex1,#current sv center
                            "apex2" => apex2,#neighbouring sv center
                            "num_tetr_neigh" => i, "current_tetr" => j))
                    end
                end
            end
        end
        return results
    end

    # Map over the coordinates and collect results
    res = map(coords) do (xd, yd, zd)
        check_and_collect(xd, yd, zd)
    end

    # Flatten the list of results
    return vcat(res...)
end


function initialize_for_tetr_dat(image_shape, radius, num_additional_oblique_points_per_side, is_point_per_triangle=false, pad=0)
    sv_centers, dims, diffs = get_sv_centers(radius, image_shape, pad)
    # Pass num_additional_oblique_points_per_side to the data generation function
    return get_flattened_triangle_data(dims, num_additional_oblique_points_per_side, is_point_per_triangle)
end#initialize_for_tetr_dat


"""
based on output from check_triangles will create a plan how to access the information
about tetrahedrons that share base - we are looking just back axis so those that are before in x,y,z axis 
in order to avoid duplication first dimension will be index of a new point 
second will be indicating where to find in order 
    1)triangle_point1
    2)triangle_point2
    3)triangle_point3
    4)apex1
    5)apex2
third wil have 4 values where first 3 will indicate the x,y,z of control points and fourth the channel in control points 
where to find the coordinates of the control points used now ; Hovewer the coordinates would be relatinve to the current index 
so we will need to add the current index to the values from last dimension to get actual spot in the control points
"""
function get_plan_tensor_for_points_per_triangle(plan_tuples)
    #initialize the plan tensor
    plan_tensor = zeros(Int64, (length(plan_tuples), 5, 4))
    plan_tuples = sort!(plan_tuples, by=x -> x["current_tetr"])

    for i in 1:length(plan_tuples)
        plan = plan_tuples[i]
        plan_tensor[i, 1, :] = plan["triangle_points"][1]
        plan_tensor[i, 2, :] = plan["triangle_points"][2]
        plan_tensor[i, 3, :] = plan["triangle_points"][3]
        plan_tensor[i, 4, :] = [plan["apex1"]..., -1]# sv center of current supervoxel
        plan_tensor[i, 5, :] = [plan["apex2"]..., -1]# sv center of neighbour supervoxel
    end
    return plan_tensor
end


###### main

"""
    augment_flattened_triangles(flattened_triangles, plan_tensor, sv_centers)
Augments the `flattened_triangles` with information from `plan_tensor` and divides each tetrahedron into six new tetrahedrons.
# Arguments
- `flattened_triangles::Array{Int64, 4}`: 3D tensor where the first dimension is the index of the tetrahedron, the second is the index of the point (sv center is the first dimension, next 3 are indices of a base, and the last is additional space filled with -1), and the last dimension is x, y, z coordinates.
- `plan_tensor::Array{Int64, 3}`: 3D tensor where the first index is the plan_index, the next is the index of a point (first 3 indices indicate the points for the tetrahedron base, and the last has length 4 where the first 3 are relative x, y, z coordinates and the fourth is the channel that is not relative).
- `sv_centers::Vector{Int}`: List of supervoxel centers.
# Returns
- `flattened_triangles_augmented::Array{Int64, 4}`: Augmented list of tetrahedrons.
!!! requirees full plan_tensor - so with 48 entries

basically the sv center is base ind .+1
"""
function augment_flattened_triangles(tetrs, tetr_3d, plan_tuples_full, plan_tuples_sorted)

    # Initialize an empty dictionary to find dictionary based on tetr index
    dict_of_dicts = Dict{Int,Dict}()
    # Iterate over each dictionary in the list
    for subdict in plan_tuples_full
        # Use the value under "current_tetr" as the key
        key = subdict["current_tetr"]
        # Assign the entire subdictionary to this key
        dict_of_dicts[key] = subdict
    end

    # Initialize an empty dictionary to find position (channel in control points out) based on tetr index 
    pos_index_dict = Dict{Any,Int}()

    # Iterate over each dictionary in the list with an index
    for (index, subdict) in enumerate(plan_tuples_sorted)
        # Add entries for both "current_tetr" and "num_tetr_neigh"
        pos_index_dict[subdict["current_tetr"]] = index
        pos_index_dict[subdict["num_tetr_neigh"]] = index
    end

    #get a list of primary indicies - if index is not here we need to reach out to the neighbour for appropriate point
    prim_indicies = map(el -> el["current_tetr"], plan_tuples_sorted)

    tetrs_size = size(tetrs)
    new_tetr_size = (tetrs_size[1] * 3, tetrs_size[2], tetrs_size[3])
    new_flat_tetrs = zeros(new_tetr_size)
    #iterating over first dimension of the tetrs



    Threads.@threads for ind_prim in 1:tetrs_size[1]
        # print("* $ind_prim *")
        #getting which tetrahedron in sv it is
        ind_tetr = ((ind_prim - 1) % 48) + 1
        channel_control_points = pos_index_dict[ind_tetr]
        curr_dict = dict_of_dicts[ind_tetr]
        t1, t2, t3 = curr_dict["triangle_points"]

        new_ind_tetr_base = ((ind_prim - 1) * 3) + 1
        tetr_curr = tetrs[ind_prim, :, :]
        base_ind = tetr_curr[1, 1:3] .- 1


        sv_center = tetr_curr[1, :]
        triang_1 = [t1[1] + base_ind[1], t1[2] + base_ind[2], t1[3] + base_ind[3], t1[4]]
        triang_2 = [t2[1] + base_ind[1], t2[2] + base_ind[2], t2[3] + base_ind[3], t2[4]]
        triang_3 = [t3[1] + base_ind[1], t3[2] + base_ind[2], t3[3] + base_ind[3], t3[4]]



        dummy = tetr_curr[5, :]
        #weather it is base ind or not depends on weather we are looking on prev or next in axis
        base_ind_p = base_ind
        if (!(ind_tetr in prim_indicies))
            base_ind_p = base_ind + (curr_dict["apex2"] .- 1)
        end
        #new point we created using get_random_point_in_tetrs_kern
        new_point = [base_ind_p[1], base_ind_p[2], base_ind_p[3], channel_control_points + 7]
        # new_point=[1,1,1,2]


        #populating with new data - we will always have the same sv center 
        #and the same 2 old triangle points and a new one 
        #we start from new_ind_tetr_base and we will add 1,2,3

        ###1
        to_add = 0
        new_flat_tetrs[new_ind_tetr_base+to_add, 1, :] = sv_center
        new_flat_tetrs[new_ind_tetr_base+to_add, 2, :] = triang_1
        new_flat_tetrs[new_ind_tetr_base+to_add, 3, :] = triang_2
        new_flat_tetrs[new_ind_tetr_base+to_add, 4, :] = new_point
        new_flat_tetrs[new_ind_tetr_base+to_add, 5, :] = dummy

        to_add = 1
        new_flat_tetrs[new_ind_tetr_base+to_add, 1, :] = sv_center
        new_flat_tetrs[new_ind_tetr_base+to_add, 2, :] = triang_1
        new_flat_tetrs[new_ind_tetr_base+to_add, 3, :] = triang_3
        new_flat_tetrs[new_ind_tetr_base+to_add, 4, :] = new_point
        new_flat_tetrs[new_ind_tetr_base+to_add, 5, :] = dummy

        to_add = 2
        new_flat_tetrs[new_ind_tetr_base+to_add, 1, :] = sv_center
        new_flat_tetrs[new_ind_tetr_base+to_add, 2, :] = triang_3
        new_flat_tetrs[new_ind_tetr_base+to_add, 3, :] = triang_2
        new_flat_tetrs[new_ind_tetr_base+to_add, 4, :] = new_point
        new_flat_tetrs[new_ind_tetr_base+to_add, 5, :] = dummy

    end

    return new_flat_tetrs
end


"""
given the size of the x,y,z dimension of control weights (what in basic architecture get as output of convolutions)
and the radius of supervoxels will return the grid of points that will be used as centers of supervoxels 
and the intialize positions of the control points
is_points_per_triangle- indicates weather we additionally supply the plan tensor for the points per triangle unrolled
and weather we are going to use the points per triangle unrolled - need to include them in the flattened_triangles data 
"""
function initialize_centers_and_control_points(image_shape, radius, is_points_per_triangle=false)

    sv_centers, dims, diffs = get_sv_centers(radius, image_shape)
    # Get potentially augmented triangles
    flattened_triangles = get_flattened_triangle_data(dims, num_additional_oblique_points_per_side, is_points_per_triangle)
    control_points = initialize_control_points(image_shape, radius, num_additional_oblique_points_per_side, is_points_per_triangle)

    if (is_points_per_triangle)
        # The plan tensor logic needs to be fully adapted. Returning placeholder.
        @warn "is_point_per_triangle=true: Plan tensor generation needs update for new tetrahedron structure. Returning dummy plan tensor."
        # Placeholder: Determine the correct size based on the new structure
        num_boundary_triangles_per_sv = 24 * (num_additional_oblique_points_per_side*2 + 1) # Rough estimate
        plan_tensor_size = (num_boundary_triangles_per_sv ÷ 2, 5, 4) # Placeholder size
        plan_tensor = zeros(Int64, plan_tensor_size) # Dummy tensor
        return sv_centers, control_points, flattened_triangles, dims, plan_tensor
    else
        return sv_centers, control_points, flattened_triangles, dims
    end
end#initialize_centeris_and_control_points

"""
return a function that casts to proper device
    it requires const gpu_i to be defined in the scope
"""
# function CuArray

#     return gpu_device(1)
#     # return gpu_device(1) gpu_device((Int(Threads.threadid()%2==0)+1))

# end

function cast_dev_zeros(args...)
    return CuArray(zeros(args...))
end

