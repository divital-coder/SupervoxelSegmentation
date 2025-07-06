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


radiuss = (Float32(3.1), Float32(4.3), Float32(4.7))
diam = radiuss * 2
num_weights_per_point = 6
a = 64
batch_size = 2

image_shape = (a, a, a, 2, batch_size)
example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
sv_centers, control_points, tetrs, dims = example_set_of_svs

size(sv_centers)

size(control_points)

dims_plus = (dims[1] + 1, dims[2] + 1, dims[3] + 1)
# control_points first dimension is lin_x, lin_y, lin_z, oblique
weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)
# # weights = rand(image_shape...)
# weights = weights .- 0.50001
# weights = (weights) .* 100
# weights = Float32.(tanh.(weights * 0.02))



weights = ones(size(weights)...) .* 0.5

control_points_non_modified = copy(control_points)

control_points_size = size(control_points)

threads_apply_w, blocks_apply_w = prepare_for_apply_weights_to_locs_kern(control_points_size, size(weights))

# control_points = call_apply_weights_to_locs_kern(CuArray(control_points), CUDA.zeros(size(control_points)...), CuArray(weights), radiuss, threads_apply_w, blocks_apply_w)
control_points = call_apply_weights_to_locs_kern(CuArray(control_points), CuArray(weights), radiuss, threads_apply_w, blocks_apply_w)

# """
# this function test single point in the control_points and check weather change in value is correct generally if weight is 
# changed the control point should change as well if it is for example positive 0.5 then the control point should be moved in the direction of the axis of change
#     0.5*radius a
# """
@testset "Control Point Tests" begin

    function test_control_point(cart_index, axes_of_change, point_ind, weight, radius, control_points, control_points_non_modified)
        old_p = control_points_non_modified[cart_index[1], cart_index[2], cart_index[3], point_ind, :, :]
        new_p = control_points[cart_index[1], cart_index[2], cart_index[3], point_ind, :, :]
        changes = [0.0, 0.0, 0.0]
        for ax in axes_of_change
            changes[ax] = weight * radius
        end
        old_p = old_p .+ changes
        return @test old_p == new_p
    end

    for cart in [(1, 1, 1), (2, 2, 2)]
        test_control_point((1, 1, 1), (1), 1, 0.5, radiuss, Array(control_points), Array(control_points_non_modified))
        test_control_point((1, 1, 1), (2), 2, 0.5, radiuss, Array(control_points), Array(control_points_non_modified))
        test_control_point((1, 1, 1), (3), 3, 0.5, radiuss, Array(control_points), Array(control_points_non_modified))
        test_control_point((1, 1, 1), (1, 2, 3), 4, 0.5, radiuss, Array(control_points), Array(control_points_non_modified))
    end
end





# weights=CuArray(Float32.(weights))
# control_points=CuArray(Float32.(control_points))
# control_points_out = CuArray(copy(Float32.(control_points)))

