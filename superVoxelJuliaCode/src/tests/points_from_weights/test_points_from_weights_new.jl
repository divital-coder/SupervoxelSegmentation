cd("/workspaces/superVoxelJuliaCode_lin_sampl")


# In the Julia REPL:
using Pkg
Pkg.activate(".")  # Activate the current directory as project



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



using Logging
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")

# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_a.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_b.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern _old.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/sv_centr_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/additional_points_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")


Random.seed!(3)
is_point_per_triangle=true
global const num_additional_oblique_points_per_side=2


#1) check minimal values of control points and max so we know it do not go outside image_from_points
# global const is_point_per_triangle=false

function fill_tetrahedron_data(tetr_dat, sv_centers, control_points, index)
    center = map(axis -> sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), axis], [1, 2, 3])
    corners = map(corner_num ->
            map(axis -> control_points[Int(tetr_dat[index, corner_num, 1]), Int(tetr_dat[index, corner_num, 2]), Int(tetr_dat[index, corner_num, 3]), Int(tetr_dat[index, corner_num, 4]), axis], [1, 2, 3]), [2, 3, 4])
    corners = [center, corners...]
    return corners
end

function get_tetrahedrons_from_corners(corners)
    points = map(el -> Meshes.Point((el[1], el[2], el[3])), corners)
    return Meshes.Tetrahedron(points...)
end
function get_base_triangles_from_corners(corners)
    points = map(el -> Meshes.Point((el[1], el[2], el[3])), corners)
    return Meshes.Triangle(points[2:end]...)
end

# radiuss = (Float32(3.2),Float32(4.1),Float32(4.25))

function test_set_points_from_weights(is_noise_both_batch_same=true)
    radiuss = (Float32(3.1), Float32(4.3), Float32(4.7))
    num_weights_per_point = 8
    spacing = (Float32(1.0), Float32(1.0), Float32(1.0))
    batch_size = 2
    a = 71
    image_shape = (a, a, a, 2)

    weights_shape = Int.(round.((a / 2, a / 2, a / 2, 100)))

    example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
    sv_centers, control_points, tetrs, dims = example_set_of_svs



    # weights = weights .+ 0.5
    # source_arr = rand(image_shape...)

    #batching
    control_points = repeat(control_points, inner=(1, 1, 1, 1, 1, batch_size))
    # sv_centers = repeat(sv_centers, inner=(1, 1, 1, 1, batch_size))
    # tetrs = repeat(tetrs, inner=(1, 1, 1, batch_size))


    if (is_noise_both_batch_same)
        weights_shape = Int.(round.((a / 2, a / 2, a / 2, 110)))
        weights = rand(weights_shape...)
        weights = repeat(weights, inner=(1, 1, 1, 1, batch_size))
        image_shape = (a, a, a, 2)
        source_arr = rand(image_shape...)
        source_arr = repeat(source_arr, inner=(1, 1, 1, 1, batch_size))
    else
        weights_shape = Int.(round.((a / 2, a / 2, a / 2, 110, batch_size)))
        weights = rand(weights_shape...)
        image_shape = (a, a, a, 2, batch_size)
        source_arr = rand(image_shape...)

    end


    weights = Float32.(weights)
    control_points = Float32.(control_points)
    source_arr = Float32.(source_arr)
    sv_centers = Float32.(sv_centers)

    print("\n  control_points $(size(control_points)) \n ")
    print("\n  sv_centers $(size(sv_centers)) \n ")
    print("\n  tetrs $(size(tetrs)) \n ")
    print("\n  source_arr $(size(source_arr)) \n ")
    print("\n  weights $(size(weights)) \n ")


    threads_apply_w, blocks_apply_w, num_blocks_z_pure_sv, num_blocks_y_pure_sv = prepare_for_apply_weights_to_locs_kern(size(sv_centers), weights_shape, batch_size)

    sv_centers_out = call_apply_weights_sv(CuArray(sv_centers), CuArray(weights), radiuss, threads_apply_w, blocks_apply_w, num_blocks_z_pure_sv, batch_size, num_blocks_y_pure_sv,[])

    CUDA.synchronize()

    threads_apply_w, blocks_apply_w, num_blocks_z_pure, num_blocks_y_pure_w = prepare_for_apply_weights_to_locs_kern(size(control_points), weights_shape, batch_size)
    # control_points=Float32.(control_points)
    control_points_out = call_apply_weights_to_locs_kern(sv_centers_out, size(control_points), CuArray(weights), threads_apply_w, blocks_apply_w, num_blocks_z_pure, num_blocks_y_pure_w,[])
    CUDA.synchronize()


    control_points_out = call_apply_weights_to_locs_kern_add_a(sv_centers_out, control_points_out, CuArray(weights), threads_apply_w, blocks_apply_w, num_blocks_z_pure, num_blocks_y_pure_w,[])
    CUDA.synchronize()


    threads_tetr_set, blocks_tetr_set = prepare_for_set_tetr_dat(image_shape, size(tetrs), batch_size)
    CUDA.synchronize()



    tetr_dat_out = call_set_tetr_dat_kern(CuArray(tetrs), CuArray(source_arr), control_points_out, sv_centers_out, threads_tetr_set, blocks_tetr_set, spacing, batch_size)

    return tetr_dat_out, sv_centers_out, control_points_out, tetrs,sv_centers
end




# first_sv_tetrs = map(index -> fill_tetrahedron_data(tetrs, Array(sv_centers_out), Array(control_points_out), index), 1:4)
# first_sv_tetrs = map(index -> fill_tetrahedron_data(tetrs, Array(sv_centers_out), Array(control_points_out), index), 1:get_num_tetr_in_sv(is_point_per_triangle))

# viz(curr,color=1:length(curr))#,alpha=0.5


function get_tetr_for_vis(curr_batch, n, tetr_dat_out, sv_centers_out, control_points_out, tetrs)
    size(sv_centers_out)
    size(sv_centers_out)
    size(control_points_out)
    # tetrs=tetrs[:,:,:,curr_batch]
    sv_centers_out = sv_centers_out[:, :, :, :, curr_batch]
    control_points_out = control_points_out[:, :, :, :, :, curr_batch]
    first_sv_tetrs_a = map(index -> fill_tetrahedron_data(tetrs, Array(sv_centers_out), Array(control_points_out), index), ((get_num_tetr_in_sv(is_point_per_triangle)*n)+1):get_num_tetr_in_sv(is_point_per_triangle)*(n+2))
    first_sv_tetrs = map(get_tetrahedrons_from_corners, first_sv_tetrs_a)
    first_sv_tetrs_base_triangles = map(get_base_triangles_from_corners, first_sv_tetrs_a)

    # curr=first_sv_tetrs_base_triangles[24:30]
    curr = first_sv_tetrs_base_triangles[1:get_num_tetr_in_sv(is_point_per_triangle)]
    return curr
end

tetr_dat_out, sv_centers_out, control_points_out, tetrs ,sv_centers = test_set_points_from_weights(true)



for i in 1:7
    a = isapprox(control_points_out[:, :, :, i, :, 1], control_points_out[:, :, :, i, :, 2])
    println(" \n $i $a \n")
end
sv_centers_out[:, :, :, :, 1] == sv_centers_out[:, :, :, :, 2]
for i in 1:4
    a = isapprox(tetr_dat_out[:, :, i, 1], tetr_dat_out[:, :, i, 2], atol=0.1)
    println(" \n $i $a \n")
end


curr = get_tetr_for_vis(2, 1, tetr_dat_out, sv_centers_out, control_points_out, tetrs)

curr
# curr=first_sv_tetrs_base_triangles[1:12]
# curr=first_sv_tetrs_base_triangles[37:48]
# curr=first_sv_tetrs_base_triangles
# viz(curr,color=[2,2,2,2,ones(8)...,3,3,3,3,ones(8)...] )#,alpha=0.5
# viz(curr,color=[2,2,2,2,ones(length(curr)-4)...] )#,alpha=0.5
curr_b = []

for i in 1:length(curr)
    try
        viz(curr[i])
        push!(curr_b, curr[i])
    catch e
        println("Error visualizing curr[$i]: $e")
    end
end

viz(curr_b, color=1:length(curr_b))#,alpha=0.5





