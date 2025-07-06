using LinearAlgebra
# Ensure the path is correct or manage dependencies via Project.toml
include("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
using Meshes # Using Meshes for visualization and geometric types
using GLMakie # Using Makie for visualization
using Unitful # Need this for ustrip

using Revise
# using Combinatorics # Not used
# using SplitApplyCombine # Not used
# using CUDA # Not used in this script
using Random
using Statistics
# using ChainRulesCore # Not used
using Test
# using Unitful # Not used currently, can be removed if not needed

"""
    intersect_segment_triangle(seg_p1, seg_p2, tri_v0, tri_v1, tri_v2; ϵ=1e-8)

Checks if a line segment intersects a triangle using the Möller-Trumbore algorithm.
Returns true if an intersection occurs strictly between the segment endpoints and within the triangle, false otherwise.
`ϵ` is a tolerance for floating-point comparisons.
"""
function intersect_segment_triangle(seg_p1::AbstractVector{T}, seg_p2::AbstractVector{T},
                                  tri_v0::AbstractVector{T}, tri_v1::AbstractVector{T}, tri_v2::AbstractVector{T};
                                                         ϵ::T=T(1e-8)) where {T<:AbstractFloat}

    edge1 = tri_v1 - tri_v0
    edge2 = tri_v2 - tri_v0
    seg_dir = seg_p2 - seg_p1
    h = cross(seg_dir, edge2)
    a = dot(edge1, h)

    # Check if segment is parallel to the triangle plane
    if -ϵ < a < ϵ
        return false
    end

    f = T(1.0) / a
    s = seg_p1 - tri_v0
    u = f * dot(s, h)

    # Check barycentric coordinate u
    if u < T(0.0) || u > T(1.0)
        return false
    end

    q = cross(s, edge1)
    v = f * dot(seg_dir, q)

    # Check barycentric coordinate v
    if v < T(0.0) || u + v > T(1.0)
        return false
    end

    # Calculate t, the parameter along the segment where intersection occurs
    t = f * dot(edge2, q)

    # Check if intersection point is strictly within the segment (not at endpoints)
    # and not behind the segment start
    if t > ϵ && t < (T(1.0) - ϵ)
        return true # Intersection within the segment and triangle
    else
        return false # Intersection is outside segment bounds or at an endpoint
    end
end


"""
    get_triangle_points(tetr_dat_out, triangle_idx, batch_idx)

Extracts the 3 vertices (p1, p2, p3) of a triangle from tetr_dat_out.
Ensures the output is Vector{T}.
"""
function get_triangle_points(tetr_dat_out::AbstractArray{T, 4}, triangle_idx::Int, batch_idx::Int) where {T}
    # tetr_dat_out dimensions: [triangle_idx, point_idx (1=sv, 2-4=verts, 5=bary), coord (x,y,z), batch_idx]
    # Extract coordinates using views or direct indexing
    # Ensure indices are within bounds
    @assert size(tetr_dat_out, 2) >= 4 "Dimension 2 of tetr_dat_out too small"
    @assert size(tetr_dat_out, 3) >= 3 "Dimension 3 of tetr_dat_out too small"
    p1 = @view tetr_dat_out[triangle_idx, 2, 1:3, batch_idx]
    p2 = @view tetr_dat_out[triangle_idx, 3, 1:3, batch_idx]
    p3 = @view tetr_dat_out[triangle_idx, 4, 1:3, batch_idx]
    # Ensure they are concrete vectors of type T
    return Vector{T}(p1), Vector{T}(p2), Vector{T}(p3)
end

"""
    get_sv_center(tetr_dat_out, triangle_idx, batch_idx)

Extracts the supervoxel center associated with a triangle.
Ensures the output is Vector{T}.
"""
function get_sv_center(tetr_dat_out::AbstractArray{T, 4}, triangle_idx::Int, batch_idx::Int) where {T}
    # Assuming sv_center is at point_idx 1
    @assert size(tetr_dat_out, 2) >= 1 "Dimension 2 of tetr_dat_out too small"
    @assert size(tetr_dat_out, 3) >= 3 "Dimension 3 of tetr_dat_out too small"
    sv_center = @view tetr_dat_out[triangle_idx, 1, 1:3, batch_idx]
    return Vector{T}(sv_center)
end

"""
    get_barycenter(tetr_dat_out, triangle_idx, batch_idx)

Extracts the barycenter of a triangle's base.
Ensures the output is Vector{T}.
"""
function get_barycenter(tetr_dat_out::AbstractArray{T, 4}, triangle_idx::Int, batch_idx::Int) where {T}
    # Assuming barycenter is at point_idx 5
    @assert size(tetr_dat_out, 2) >= 5 "Dimension 2 of tetr_dat_out too small"
    @assert size(tetr_dat_out, 3) >= 3 "Dimension 3 of tetr_dat_out too small"
    barycenter = @view tetr_dat_out[triangle_idx, 5, 1:3, batch_idx]
    return Vector{T}(barycenter)
end


"""
    check_star_shape(tetr_dat_out::AbstractArray{T, 4}; batch_idx::Int = 1) where {T<:AbstractFloat}

Checks if all supervoxels defined in `tetr_dat_out` for a given `batch_idx`
are star-shaped with respect to their centers (segments to barycenters).

Args:
    tetr_dat_out: 4D tensor containing triangle data for supervoxels.
                  Dims: [triangle_idx, point_idx(1=sv, 2-4=verts, 5=bary), coord(x,y,z), batch]
    batch_idx: The batch index to check (relative to the 4th dimension of the input array).

Returns:
    `true` if all checked supervoxels are star-shaped, `false` otherwise.
    Prints an error message and visualizes the issue upon finding the first violation.
"""
function check_star_shape(tetr_dat_out::AbstractArray{T, 4}; batch_idx::Int = 1) where {T<:AbstractFloat}

    num_total_triangles = size(tetr_dat_out, 1)
    triangles_per_sv = try
        get_num_tetr_in_sv(false)
    catch e
        println("Error calling get_num_tetr_in_sv(false): $e")
        return false # Cannot proceed without this value
    end

    if triangles_per_sv <= 0
        println("Error: get_num_tetr_in_sv(false) returned non-positive value: $(triangles_per_sv)")
        return false
    end
    if num_total_triangles == 0
        println("Warning: tetr_dat_out has no triangles.")
        return true # Technically, vacuously true
    end

    num_supervoxels = num_total_triangles ÷ triangles_per_sv
    if num_total_triangles % triangles_per_sv != 0
        println("Warning: Total number of triangles ($(num_total_triangles)) is not a multiple of triangles per supervoxel ($(triangles_per_sv)). Checking only full supervoxels.")
    end
    if num_supervoxels == 0
        println("Warning: Not enough triangles to form a single supervoxel.")
        return true # No full supervoxels to check
    end

    println("Checking $(num_supervoxels) supervoxels for star-shape property (batch index $batch_idx)...")

    epsilon = T(1e-8) # Tolerance for float comparisons, use type T
    zero_len_threshold = T(1e-8) # Threshold for zero-length segments

    for sv_idx in 1:num_supervoxels
        start_idx = (sv_idx - 1) * triangles_per_sv + 1
        end_idx = sv_idx * triangles_per_sv

        sv_center_vec = get_sv_center(tetr_dat_out, start_idx, batch_idx)

        barycenters = Dict{Int, Vector{T}}()
        local_triangle_indices = start_idx:end_idx

        for current_triangle_idx in local_triangle_indices
            current_sv_center_check = get_sv_center(tetr_dat_out, current_triangle_idx, batch_idx)
            if norm(current_sv_center_check - sv_center_vec) > epsilon
                println("Error: Inconsistent SV center within supervoxel $(sv_idx) at triangle $(current_triangle_idx).")
                println("  Expected: $(sv_center_vec), Found: $(current_sv_center_check)")
                return false
            end
            barycenter_vec = get_barycenter(tetr_dat_out, current_triangle_idx, batch_idx)
            barycenters[current_triangle_idx] = barycenter_vec
        end

        for (source_triangle_idx, barycenter_vec) in barycenters
            segment_start = sv_center_vec
            segment_end = barycenter_vec

            if norm(segment_end - segment_start) < zero_len_threshold
                continue
            end

            for other_triangle_idx in local_triangle_indices
                if other_triangle_idx == source_triangle_idx
                    continue
                end

                tri_v0_vec, tri_v1_vec, tri_v2_vec = get_triangle_points(tetr_dat_out, other_triangle_idx, batch_idx)

                if intersect_segment_triangle(segment_start, segment_end, tri_v0_vec, tri_v1_vec, tri_v2_vec; ϵ=epsilon)
                    println("-------------------------------------------")
                    println("Error: Supervoxel $(sv_idx) is not star-shaped!")
                    println("  Segment from SV Center $(segment_start) to Barycenter $(segment_end)")
                    println("  (associated with triangle $(source_triangle_idx))")
                    println("  intersects triangle $(other_triangle_idx) with vertices:")
                    println("    v0: $(tri_v0_vec)")
                    println("    v1: $(tri_v1_vec)")
                    println("    v2: $(tri_v2_vec)")
                    println("-------------------------------------------")

                    GLMakie.activate!()
                    fig = Figure(size = (1000, 600))
                    ax = Axis3(fig[1, 1], aspect = :data, title="Star Shape Violation - SV $(sv_idx)")

                    sv_triangles_list = Meshes.Triangle[]
                    sv_points_coords = Set{NTuple{3, Float64}}()

                    # for vis_idx in local_triangle_indices
                    #     v0_vis, v1_vis, v2_vis = get_triangle_points(tetr_dat_out, vis_idx, batch_idx)
                    #     p0 = Meshes.Point(Float64.(v0_vis)...)
                    #     p1 = Meshes.Point(Float64.(v1_vis)...)
                    #     p2 = Meshes.Point(Float64.(v2_vis)...)
                    #     push!(sv_triangles_list, Meshes.Triangle(p0, p1, p2))
                    #     push!(sv_points_coords, ustrip.(Meshes.to(p0).coords))
                    #     push!(sv_points_coords, ustrip.(Meshes.to(p1).coords))
                    #     push!(sv_points_coords, ustrip.(Meshes.to(p2).coords))
                    # end

                    # viz!(ax, sv_triangles_list, color=:lightgray, alpha=0.05, shading=false)

                    intersecting_p0 = Meshes.Point(Float64.(tri_v0_vec)...)
                    intersecting_p1 = Meshes.Point(Float64.(tri_v1_vec)...)
                    intersecting_p2 = Meshes.Point(Float64.(tri_v2_vec)...)
                    intersecting_triangle = Meshes.Triangle(intersecting_p0, intersecting_p1, intersecting_p2)
                    viz!(ax, intersecting_triangle, color=:blue, alpha=0.7)
                    viz!(ax, Meshes.Segment(intersecting_p0, intersecting_p1), color=:darkblue, linewidth=2.0)
                    viz!(ax, Meshes.Segment(intersecting_p1, intersecting_p2), color=:darkblue, linewidth=2.0)
                    viz!(ax, Meshes.Segment(intersecting_p2, intersecting_p0), color=:darkblue, linewidth=2.0)

                    source_v0_vec, source_v1_vec, source_v2_vec = get_triangle_points(tetr_dat_out, source_triangle_idx, batch_idx)
                    source_p0 = Meshes.Point(Float64.(source_v0_vec)...)
                    source_p1 = Meshes.Point(Float64.(source_v1_vec)...)
                    source_p2 = Meshes.Point(Float64.(source_v2_vec)...)
                    source_triangle = Meshes.Triangle(source_p0, source_p1, source_p2)
                    viz!(ax, source_triangle, color=:green, alpha=0.7)
                    viz!(ax, Meshes.Segment(source_p0, source_p1), color=:darkgreen, linewidth=2.0)
                    viz!(ax, Meshes.Segment(source_p1, source_p2), color=:darkgreen, linewidth=2.0)
                    viz!(ax, Meshes.Segment(source_p2, source_p0), color=:darkgreen, linewidth=2.0)

                    segment_start_p = Meshes.Point(Float64.(segment_start)...)
                    segment_end_p = Meshes.Point(Float64.(segment_end)...)
                    problem_segment = Meshes.Segment(segment_start_p, segment_end_p)
                    viz!(ax, problem_segment, color=:red, linewidth=4)

                    start_coords = Tuple(ustrip.(Meshes.to(segment_start_p).coords))
                    end_coords = Tuple(ustrip.(Meshes.to(segment_end_p).coords))
                    
                    scatter!(ax, [start_coords], color=:cyan, markersize=20, label="SV Center")
                    scatter!(ax, [end_coords], color=:magenta, markersize=20, label="Barycenter")

                    # Label(fig[1, 2, Top()], "Legend", fontsize = 16, font = :bold, padding = (0, 0, 5, 0))
                    # text_box = Textbox(fig[1, 2],
                    #     placeholder = """
                    #     GREEN: Source Triangle (Triangle $source_triangle_idx)
                    #     BLUE: Intersecting Triangle (Triangle $other_triangle_idx)
                    #     RED: Problem Segment (Center to Barycenter)
                    #     CYAN: Supervoxel Center
                    #     MAGENTA: Barycenter of Source Triangle
                    #     """,
                    #     fontsize = 14,
                    #     bordercolor = :black,
                    #     borderwidth = 1
                    # )
                    # colsize!(fig.layout, 2, Relative(0.3))

                    println("Displaying visualization of the error...")
                    display(fig)
                    println("Visualization displayed. Check the plot window.")
                    sleep(2)
                    return false
                end
            end
        end

        if sv_idx % 100 == 0
            println("Checked $(sv_idx)/$(num_supervoxels) supervoxels...")
        end
    end

    println("Success: All $(num_supervoxels) checked supervoxels appear to be star-shaped with respect to segments to barycenters.")
    return true
end

"""
    check_star_shape(tetr_dat_out::AbstractArray{T, 5}; batch_idx::Int = 1) where {T<:AbstractFloat}

Wrapper for `check_star_shape` that accepts a 5D tensor. Extracts the specified
batch slice and calls the 4D version.

Args:
    tetr_dat_out: 5D tensor. Dims: [triangle, point, coord, channel/feature, batch]
                  OR [triangle, point, coord, batch, channel/feature].
                  Assuming 5th dim is batch based on original code.
    batch_idx: The index of the batch in the 5th dimension to check.

Returns:
    Result from the 4D `check_star_shape` function.
"""
function check_star_shape(tetr_dat_out::AbstractArray{T, 5}; batch_idx::Int = 1) where {T<:AbstractFloat}
     batch_dim_size = size(tetr_dat_out, 5)
     if !(1 <= batch_idx <= batch_dim_size)
         println("Error: batch_idx $(batch_idx) is out of bounds for the 5th dimension (size: $(batch_dim_size)).")
         return false
     end
     tetr_dat_out_4d = @view tetr_dat_out[:,:,:,:,batch_idx]
     println("Checking 5D tensor, extracting batch $(batch_idx) from dimension 5.")
     return check_star_shape(tetr_dat_out_4d; batch_idx=1)
end
