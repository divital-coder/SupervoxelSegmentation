using Revise, CUDA
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
using Interpolations

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern_unrolled.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_a.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_b.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern _old.jl")


includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/Lux_model.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")

const is_point_per_triangle=false


"""
interpolation check - chek weather the value we got from interpolation make sense (we are meaking separete kernel just for this tests)
!!!! currently we simplify and just do trilinear interpolation but points 1-4 will be usefull for testing more elaborate plans that will
    take sapcing into account
1) create arrays that will be used for interpolation check in such a way that we will create a separate 3d array for each axis and to the half of this axis it will have value 0 
   and the other half value 1 and then we will interpolate the value in the middle of the axis and check weather the value is 0.5
2) get arrays from above and mutate one zero close to queried point into 1 and check weather the value in the middle of the axis is above 0.5 but below 1
3) get arrays from above and mutate one one close to queried point into 0 and check weather the value in the middle of the axis is below 0.5 but above 1
4) get a array with various values (arrange consecutive integers) get a point arbitrary to one of the points and check 
    weather value is approximately equal to the value of the point
"""

# function scale(itp::AbstractInterpolation{T,N,IT}, ranges::Vararg{AbstractRange,N}) where {T,N,IT}
#     # overwriting this function becouse check_ranges giving error
#     # check_ranges(itpflag(itp), axes(itp), ranges)
#     ScaledInterpolation{T,N,typeof(itp),IT,typeof(ranges)}(itp, ranges)
# end


function ceil_or_floor(code, num)
    if (code == "c")
        return ceil(num)
    else
        return floor(num)
    end
end






"""
given point will look around for each ceil or floored coordinate and will get the values of them inversly proportianal to the distance
    of ceiled/floored coordineate to the point; distance take into account spacing
"""
function trilinear_interpolation_or_var_kernel_cpu(point, input_array, spacing, meann)
    res = 0.0
    dists = 0.0
    for xs in ["c", "f"], ys in ["c", "f"], zs in ["c", "f"]
        x = ceil_or_floor(xs, point[1])
        y = ceil_or_floor(ys, point[2])
        z = ceil_or_floor(zs, point[3])
        dist = (1 / (((((point[1] - x) * spacing[1])^2) + (((point[2] - y) * spacing[2])^2) + (((point[3] - z) * spacing[3])^2) + 0.00000000001)))
        # dist=(1/(sqrt(( ((point[1] - x)*spacing[1])^2) + (((point[2] - y) * spacing[2])^2) + (((point[3] - z)*spacing[3])^2) +0.00000000001   )) )
        dists = dist + dists
        if meann != 0.0
            curr_res = (((input_array[Int(x), Int(y), Int(z)] - meann)^2) * dist)
        else
            curr_res = (input_array[Int(x), Int(y), Int(z)] * dist)
        end
        res = res + curr_res

    end
    # print("\n res $res dist $dists \n")
    return res / dists
end

function trilinear_variance_kernel_cpu(input_array, point, spacing)
    meann = 0.0
    interp = trilinear_interpolation_or_var_kernel_cpu(point, input_array, spacing, meann)
    return trilinear_interpolation_or_var_kernel_cpu(point, input_array, spacing, interp)
end

function trilinear_interpolation_kernel_cpu(point, input_array)
    return trilinear_interpolation_or_var_kernel_cpu(point, input_array, (1.0, 1.0, 1.0), 0.0)
end

# input_array=zeros(Float32, (10,10,10))
# input_array[7:end,:,:].=1.0
# point=(5.5,2.0,2.0)

# trilinear_interpolation_kernel_cpu(point, input_array)


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


# Test functions


function prepare_for_point_info_kern(tetr_dat_shape)
    bytes_per_thread = 6
    # blocks,threads,maxBlocks=computeBlocksFromOccupancy(point_info_kern,(tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points), bytes_per_thread)
    # threads=256
    threads = 128

    # total_num=control_points_shape[1]*control_points_shape[2]*control_points_shape[3]
    # needed_blocks=ceil(total_num / threads_apply_w)
    needed_blocks = ceil(Int, tetr_dat_shape[1] / threads)
    to_pad = (threads * needed_blocks) - tetr_dat_shape[1]

    return threads, needed_blocks, to_pad
end



function get_on_a_line(point_a, point_b, rel_distance)
    return [point_a[1] + (point_b[1] - point_a[1]) * rel_distance,
        point_a[2] + (point_b[2] - point_a[2]) * rel_distance,
        point_a[3] + (point_b[3] - point_a[3]) * rel_distance]
end #get_on_a_line    

function get_line_diff(point_a, point_b, rel_distance)
    return [(point_a[1] - point_b[1]) * rel_distance,
        (point_a[2] - point_b[2]) * rel_distance,
        (point_a[3] - point_b[3]) * rel_distance], rel_distance
end #get_on_a_line    

# function get_on_a_line_b(point_a,point_b,rel_distance)
#     return [point_a[1]+(point_b[1]-point_a[1])*rel_distance,
#     point_a[2]+(point_b[2]-point_a[2])*rel_distance,
#     point_a[3]+(point_b[3]-point_a[3])*rel_distance]
# end #get_on_a_line    

function cross_product(a, b)
    return [a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]]
end

function normm(a)
    return sqrt(sum([i^2 for i in a]))
end

function distance_point_to_line_gold(p, l1, l2)
    return normm(cross_product(l2 - l1, l1 - p)) / norm(l2 - l1)
end




@testset "Variance Tests cpu " begin
    spacing = (1.0, 1.0, 1.0)
    data = ones(10, 10, 10)
    @test trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5), spacing) == 0

    data[5, 5, 5] = 2
    var1 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5), spacing)
    @test var1 > 0

    data[6, 5, 5] = 3
    var2 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5), spacing)
    @test var2 > var1

    data[5, 6, 5] = 4
    var3 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5), spacing)
    @test var3 > var2

    data[5, 5, 6] = 5
    var4 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5), spacing)
    @test var4 > var3

    data[6, 5, 6] = 6
    var5 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5), spacing)
    @test var5 > var4

    data[5, 6, 6] = 7
    var6 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5), spacing)
    @test var6 > var5

    data[6, 6, 5] = 8
    var7 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5), spacing)
    @test var7 > var6

    data[6, 6, 6] = 11
    var8 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5), spacing)
    @test var8 > var7


    data = ones(10, 10, 10)
    data[1:5, :, :] = rand(5, 10, 10)
    var_a = trilinear_variance_kernel_cpu(data, (5.1, 5.5, 5.5), spacing)
    var_b = trilinear_variance_kernel_cpu(data, (5.9, 5.5, 5.5), spacing)
    @test var_a > var_b

    data = ones(10, 10, 10)
    data[:, 1:5, :] = rand(10, 5, 10)
    var_a = trilinear_variance_kernel_cpu(data, (5.5, 5.1, 5.5), spacing)
    var_b = trilinear_variance_kernel_cpu(data, (5.5, 5.9, 5.5), spacing)
    @test var_a > var_b

    data = ones(10, 10, 10)
    data[:, :, 1:5] = rand(10, 10, 5)
    var_a = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.1), spacing)
    var_b = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.9), spacing)
    @test var_a > var_b



end







using Random
Random.seed!(12)

radiuss = (Float32(4.0), Float32(4.0), Float32(4.0))
# diam = radiuss * 2
num_weights_per_point = 6
spacing = (Float32(1.0), Float32(1.0), Float32(1.0))
batch_size = 2

a = 48
image_shape = (a, a, a, 2)
num_convs_per_dim = (3, 3, 3)
num_base_samp_points, num_additional_samp_points = 2, 1
pad_voxels = 2

example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
sv_centers, control_points, tetrs, dims = example_set_of_svs

# weights=Float32.(weights)
control_points = Float32.(control_points)
control_points = repeat(control_points, inner=(1, 1, 1, 1, 1, batch_size))
sv_centers = Float32.(sv_centers)

#here we get all tetrahedrons mapped to non modified locations
source_arr = rand(Float32, image_shape)
source_arr = repeat(source_arr, inner=(1, 1, 1, 1, batch_size))
#get dummy model 
rng = Random.default_rng()


sv_centers = Float32.(sv_centers)

weights_shape = Int.(round.((a / 2, a / 2, a / 2, 24, 2)))
weights = rand(Float32, weights_shape)
threads_apply_w, blocks_apply_w, num_blocks_z_pure_sv, pure_y_sv = prepare_for_apply_weights_to_locs_kern(size(sv_centers), weights_shape, batch_size)

sv_centers_out = call_apply_weights_sv(CuArray(sv_centers), CuArray(weights), radiuss, threads_apply_w, blocks_apply_w, num_blocks_z_pure_sv, batch_size, pure_y_sv)

CUDA.synchronize()

threads_apply_w, blocks_apply_w, num_blocks_z_pure, pure_y_w = prepare_for_apply_weights_to_locs_kern(size(control_points), weights_shape, batch_size)
# control_points=Float32.(control_points)
control_points_out = call_apply_weights_to_locs_kern(sv_centers_out, size(control_points), CuArray(weights), threads_apply_w, blocks_apply_w, num_blocks_z_pure, pure_y_w)
CUDA.synchronize()


control_points_out = call_apply_weights_to_locs_kern_add_a(sv_centers_out, control_points_out, CuArray(weights), threads_apply_w, blocks_apply_w, num_blocks_z_pure, pure_y_w)
CUDA.synchronize()

conv2 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.tanh, stride=2, pad=Lux.SamePad(), init_weight=zeros32, init_bias=zeros32)


threads_tetr_set, blocks_tetr_set = prepare_for_set_tetr_dat(image_shape, size(tetrs), batch_size)
CUDA.synchronize()



tetr_dat_out = call_set_tetr_dat_kern(CuArray(tetrs), CuArray(source_arr), control_points_out, sv_centers_out, threads_tetr_set, blocks_tetr_set, spacing, batch_size)


threads_point_info, blocks_point_info = prepare_point_info_kern(image_shape, get_tetr_dat_shape(image_shape), batch_size)

out_sampled_points = call_point_info_kern(tetr_dat_out, CuArray(source_arr), num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info, batch_size, spacing)

# ###tetr dat model
# conv_part = Lux.Chain(conv2(2, 6), conv2(6, 6), conv2(6, 24))
# before_point_kerns = SkipConnection(Lux.Chain(conv_part
# , Points_weights_str(radiuss,batch_size, pad_voxels, (image_shape[1], image_shape[2], image_shape[3]), num_convs_per_dim)), connection_before_set_tetr_dat_kern)
# model = Lux.Chain(before_point_kerns, Set_tetr_dat_str(radiuss,spacing,batch_size, pad_voxels, (image_shape[1], image_shape[2], image_shape[3]))
#                         , Point_info_kern_str(radiuss,spacing,batch_size, (image_shape[1], image_shape[2], image_shape[3]), num_base_samp_points, num_additional_samp_points)
# )




# ps, st = Lux.setup(rng, model)
# st = cu(st)
# ps = cu(ps)
# y_pred, st = Lux.apply(model, CuArray(source_arr), ps, st)
# out_sampled_points, tetr_dat_out = y_pred
# a=1

# size(out_sampled_points)
# for i in 1:5
#     print("\n $(out_sampled_points[:,:,i,1]==out_sampled_points[:,:,i,2]) \n")
# end

# threads_point_info, blocks_point_info, pad_point_info = prepare_for_set_tetr_dat(size(tetrs))
# tetr_dat_out = call_set_tetr_dat_kern_test_unrolled(tetrs, source_arr, control_points, sv_centers, threads_point_info, blocks_point_info, pad_point_info)


# sv_tetrs[1][1]
# tetr_dat_out[1,1,:][1:3]
# @testset "is tetr dat out populated correctly" begin


tetr_dat_out = Array(tetr_dat_out)[:, :, :, 1]
source_arr = Array(source_arr)[:, :, :, 1, 1]
source_arr[5, 5, 3]
source_arr[5, 5, 4]

Array(sv_centers_out)[2, 2, 2, 1, 1]
Array(sv_centers)[1, 1, 1, 1]

sv_centers_out = Array(sv_centers_out)[:, :, :, :, 1]
sv_tetrs = map(index -> fill_tetrahedron_data(tetrs, sv_centers_out, Array(control_points_out)[:, :, :, :, :, 1], index), 1:(size(tetrs)[1]))

out_sampled_points = Array(out_sampled_points)[:, :, :, 1]


source_arr[9, 9, 9]

sv_tetrs[1]

for v in eachindex(sv_tetrs)
    sum_x = 0.0
    sum_y = 0.0
    sum_z = 0.0
    # for p in eachindex(sv_tetrs[v])
    for p in eachindex(sv_tetrs[v])
        ## test is the location of the tetrahedron points was updated correctly
        @test sv_tetrs[v][p] == tetr_dat_out[v, p, :][1:3]
        ## check is interpolation of sv cenetr is correctly written
        if (p == 1)
            @test tetr_dat_out[v, p, 4] ≈ trilinear_interpolation_kernel_cpu(sv_tetrs[v][p], source_arr)
        end
        ## check is variance of other points is correctly written   
        if (p > 1)
            @test tetr_dat_out[v, p, 4] ≈ trilinear_variance_kernel_cpu(source_arr, sv_tetrs[v][p], spacing)
            sum_x += sv_tetrs[v][p][1]
            sum_y += sv_tetrs[v][p][2]
            sum_z += sv_tetrs[v][p][3]
        end
    end
    ## check is centroid of the tetrahedron base is in the middle of the points of a tetrahedron base    
    @test tetr_dat_out[v, 5, 1] ≈ (sum_x / 3)
    @test tetr_dat_out[v, 5, 2] ≈ (sum_y / 3)
    @test tetr_dat_out[v, 5, 3] ≈ (sum_z / 3)
    @test tetr_dat_out[v, 5, 4] ≈ trilinear_variance_kernel_cpu(source_arr, ((sum_x / 3), (sum_y / 3), (sum_z / 3)), spacing)
end

########### testing point info kernel
for index in 1:size(tetr_dat_out)[1]
    pp = []
    ### testing base sample points
    for num_point in 1:num_base_samp_points
        pp = get_on_a_line(tetr_dat_out[index, 1, :], tetr_dat_out[index, 5, :], num_point / (num_base_samp_points + 1))
        @test out_sampled_points[index, num_point, :][3:5] ≈ pp
        @test isapprox(out_sampled_points[index, num_point, 1], trilinear_interpolation_kernel_cpu(pp, source_arr), atol=0.001)
        if (num_point == num_base_samp_points)


            @test isapprox(((((norm((tetr_dat_out[index, 1, 1:3] - (tetr_dat_out[index, 5, 1:3])) ./ (num_base_samp_points + 1)) * 2) +
                              (norm((pp - (tetr_dat_out[index, 2, 1:3])) .* (1 / (num_additional_samp_points + 1)))) +
                              (norm((pp - (tetr_dat_out[index, 3, 1:3])) .* (1 / (num_additional_samp_points + 1)))) +
                              (norm((pp - (tetr_dat_out[index, 4, 1:3])) .* (1 / (num_additional_samp_points + 1))))
                ) / 5)^3), out_sampled_points[index, num_point, 2], atol=0.01)
        else


            @test isapprox(((((norm((tetr_dat_out[index, 1, 1:3] - (tetr_dat_out[index, 5, 1:3])) ./ (num_base_samp_points + 1)) * 2) +
                              distance_point_to_line_gold(pp, tetr_dat_out[index, 1, 1:3], tetr_dat_out[index, 2, 1:3]) +
                              distance_point_to_line_gold(pp, tetr_dat_out[index, 1, 1:3], tetr_dat_out[index, 3, 1:3]) +
                              distance_point_to_line_gold(pp, tetr_dat_out[index, 1, 1:3], tetr_dat_out[index, 4, 1:3])
                ) / 5)^3), out_sampled_points[index, num_point, 2], atol=0.01)


        end


    end
    ##pp is last base sample point
    ### testing additional sample points
    for num_point in 1:num_additional_samp_points
        for triangle_corner_num in UInt8(1):UInt8(3)
            pp2 = get_on_a_line(pp, tetr_dat_out[index, triangle_corner_num+1, :], num_point / (num_additional_samp_points + 1))
            # distst=get_line_diff(tetr_dat_out[index,triangle_corner_num+1,:],pp,num_point/(num_additional_samp_points+1))
            @test out_sampled_points[index, (num_base_samp_points+triangle_corner_num)+((num_point-1)*3), :][3:5] ≈ pp2
            @test isapprox(Float64(out_sampled_points[index, (num_base_samp_points+triangle_corner_num)+((num_point-1)*3), 1]), trilinear_interpolation_kernel_cpu(pp2, source_arr), rtol=0.1)

            aa = [1, 2, 3, 4]
            aa = deleteat!(aa, triangle_corner_num + 1)

            rrr = norm((tetr_dat_out[index, triangle_corner_num+1, 1:3] - pp) ./ (num_additional_samp_points + 1)) * 2
            for ii in aa
                rrr += distance_point_to_line_gold(pp2, tetr_dat_out[index, ii, 1:3], tetr_dat_out[index, triangle_corner_num+1, 1:3])
            end

            rr = (((rrr) / 5)^3)

            @test isapprox(rr, out_sampled_points[index, (num_base_samp_points+triangle_corner_num)+((num_point-1)*3), 2], atol=0.001)

        end
    end

end



"""
visualization
visualize the points with weights as balls and the line between the center of the triangle and the center of the super voxel plus lines and balls for base and additional sample points
"""


"""
now we want to visualize the points that were selected for sampling and their weights
    we will display their weights by the spheres of the radius equal to weight
"""

function radius_from_volume(volume)
    return (volume^(1 / 3))
end
index = 1
from_sampled_points = out_sampled_points[index, :, :]
terr_dat_curr = tetr_dat_out[index, :, :]
terr_dat_curr = terr_dat_curr[:, 1:3]
to_disp_points = false
to_disp_tetr = true
to_disp_tetr_tetr_only = false

to_points = invert(splitdims(from_sampled_points))
points_mesh_a = map(v -> Meshes.Point(v[3], v[4], v[5]), to_points)
points_mesh_b = map(v -> Meshes.Point(v...), invert(splitdims(terr_dat_curr)))
points_mesh = [points_mesh_a; points_mesh_b]

spheres = map(v -> Meshes.Sphere((v[3], v[4], v[5]), radius_from_volume(v[2]) / 2), to_points)
# spheres=map(i->Meshes.Sphere((out_sampled_points[i,3],out_sampled_points[i,4],out_sampled_points[i,5]),out_sampled_points[i,2]/2),1:6)
if (to_disp_points)
    spheres = [spheres; points_mesh_b]
    viz(spheres, color=1:length(spheres), alpha=collect(1:length(spheres)) .* 0.9)
end

# if (to_disp_tetr)
first_sv_tetrs = sv_tetrs[1:get_num_tetr_in_sv()]
first_sv_tetrs = map(get_tetrahedrons_from_corners, first_sv_tetrs)
spheres = [spheres; [first_sv_tetrs[1]]]
viz(spheres, color=1:length(spheres), alpha=collect(1:length(spheres)) .* 0.9)
# end

if (to_disp_tetr_tetr_only)
    first_sv_tetrs = sv_tetrs[1:get_num_tetr_in_sv()]
    first_sv_tetrs = map(get_tetrahedrons_from_corners, first_sv_tetrs)

    viz(first_sv_tetrs, color=1:length(first_sv_tetrs), alpha=collect(1:length(first_sv_tetrs)) .* 0.9)
end




































# # out_sampled_points, tetr_dat_out = y_pred



# # function trilinear_interpolation_kernel_cpu(point, input_array)
# #     # index = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))
# #     c = (((
# #         input_array[floor(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])] * (1 - (point[1] - floor(Int, point[1]))) +
# #         input_array[ceil(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])] * (point[1] - floor(Int, point[1]))
# #     )
# #           *
# #           (1 - (point[2] - floor(Int, point[2]))) +
# #           (input_array[floor(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])] * (1 - (point[1] - floor(Int, point[1])))
# #            +
# #            input_array[ceil(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])] * (point[1] - floor(Int, point[1])))
# #           *
# #           (point[2] - floor(Int, point[2])))
# #          *
# #          (1 - (point[3] - floor(Int, point[3])))
# #          +
# #          ((input_array[floor(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])] * (1 - (point[1] - floor(Int, point[1])))
# #            +
# #            input_array[ceil(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])] * (point[1] - floor(Int, point[1])))
# #           *
# #           (1 - (point[2] - floor(Int, point[2]))) +
# #           (input_array[floor(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])] * (1 - (point[1] - floor(Int, point[1])))
# #            +
# #            input_array[ceil(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])] * (point[1] - floor(Int, point[1])))
# #           *
# #           (point[2] - floor(Int, point[2])))
# #          *
# #          (point[3] - floor(Int, point[3])))

# #     return c
# # end




# # """
# # function that will be used for testing interpolation
# # """
# # function interpolate_my(point, input_array, input_array_spacing)

# #     old_size = size(input_array)
# #     itp = interpolate(input_array, BSpline(Linear()))
# #     #we indicate on each axis the spacing from area we are samplingA
# #     A_x1 = 1:input_array_spacing[1]:(old_size[1])
# #     A_x2 = 1:input_array_spacing[2]:(old_size[2])
# #     A_x3 = 1:input_array_spacing[3]:(old_size[3])

# #     itp = extrapolate(itp, 0.0)
# #     itp = scale(itp, A_x1, A_x2, A_x3)
# #     return itp(point[1], point[2], point[3])
# # end#interpolate_my


# # ### testing base cpu inmplementation of interpolation
# # input_array = rand(10, 10, 10)
# # point = [5.5, 5.5, 5.5]
# # input_array_spacing = [1.0, 1.0, 1.0]
# # @test interpolate_my(point, input_array, input_array_spacing) ≈ trilinear_interpolation_kernel_cpu(point, input_array)




# """
# variance check - check weather the variance of the values that we get from interpolation make sense (we are meaking separete kernel just for this tests)
# 1) get array of constant values and check weather the variance is 0
# 2) mutate array from 1 and set single voxel close to point at 1 and check weather the variance has increased above 0
# 3) mutete second voxel with value 2 and check weather the variance has increased more then in case of 2 ...
#    do this with all surrounding voxels of the point and be sure that the values are increaing with each addition
# 4) get array for each axis where first half of the axis is random and second is ones and check if the point a bit closer to random part has higher variance than a point 
#     a bit closer to ones part
#    """

# # function trilinear_variance_kernel_cpu(input_array, point)

# #     mean = (((
# #         input_array[floor(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])]
# #         *
# #         (1 - (point[1] - floor(Int, point[1]))) +
# #         input_array[ceil(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])]
# #         *
# #         (point[1] - floor(Int, point[1]))
# #     )
# #              *
# #              (1 - (point[2] - floor(Int, point[2]))) +
# #              (input_array[floor(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])]
# #               *
# #               (1 - (point[1] - floor(Int, point[1])))
# #               +
# #               input_array[ceil(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])]
# #               *
# #               (point[1] - floor(Int, point[1])))
# #              *
# #              (point[2] - floor(Int, point[2])))
# #             *
# #             (1 - (point[3] - floor(Int, point[3])))
# #             +
# #             ((input_array[floor(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])]
# #               *
# #               (1 - (point[1] - floor(Int, point[1])))
# #               +
# #               input_array[ceil(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])]
# #               *
# #               (point[1] - floor(Int, point[1])))
# #              *
# #              (1 - (point[2] - floor(Int, point[2]))) +
# #              (input_array[floor(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])]
# #               *
# #               (1 - (point[1] - floor(Int, point[1])))
# #               +
# #               input_array[ceil(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])]
# #               *
# #               (point[1] - floor(Int, point[1])))
# #              *
# #              (point[2] - floor(Int, point[2])))
# #             *
# #             (point[3] - floor(Int, point[3])))
# #     ############ variance
# #     variance = (((
# #         ((input_array[floor(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])] - mean)^2)
# #         *
# #         (1 - (point[1] - floor(Int, point[1]))) +
# #         ((input_array[ceil(Int, point[1]), floor(Int, point[2]), floor(Int, point[3])] - mean)^2)
# #         *
# #         (point[1] - floor(Int, point[1]))
# #     )
# #                  *
# #                  (1 - (point[2] - floor(Int, point[2]))) +
# #                  (((input_array[floor(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])] - mean)^2)
# #                   *
# #                   (1 - (point[1] - floor(Int, point[1])))
# #                   +
# #                   ((input_array[ceil(Int, point[1]), ceil(Int, point[2]), floor(Int, point[3])] - mean)^2)
# #                   *
# #                   (point[1] - floor(Int, point[1])))
# #                  *
# #                  (point[2] - floor(Int, point[2])))
# #                 *
# #                 (1 - (point[3] - floor(Int, point[3])))
# #                 +
# #                 ((((input_array[floor(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])] - mean)^2)
# #                   *
# #                   (1 - (point[1] - floor(Int, point[1])))
# #                   +
# #                   ((input_array[ceil(Int, point[1]), floor(Int, point[2]), ceil(Int, point[3])] - mean)^2)
# #                   *
# #                   (point[1] - floor(Int, point[1])))
# #                  *
# #                  (1 - (point[2] - floor(Int, point[2]))) +
# #                  (((input_array[floor(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])] - mean)^2)
# #                   *
# #                   (1 - (point[1] - floor(Int, point[1])))
# #                   +
# #                   ((input_array[ceil(Int, point[1]), ceil(Int, point[2]), ceil(Int, point[3])] - mean)^2)
# #                   *
# #                   (point[1] - floor(Int, point[1])))
# #                  *
# #                  (point[2] - floor(Int, point[2])))
# #                 *
# #                 (point[3] - floor(Int, point[3])))



# #     # d_result[1] = variance

# #     return variance
# # end



# # @testset "Variance Tests cpu " begin
# #     spacing= (1.0,1.0,1.0)
# #     data = ones(10, 10, 10)
# #     @test trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5),spacing) == 0

# #     data[5, 5, 5] = 2
# #     var1 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5),spacing)
# #     @test var1 > 0

# #     data[6, 5, 5] = 3
# #     var2 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5),spacing)
# #     @test var2 > var1

# #     data[5, 6, 5] = 4
# #     var3 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5),spacing)
# #     @test var3 > var2

# #     data[5, 5, 6] = 5
# #     var4 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5),spacing)
# #     @test var4 > var3

# #     data[6, 5, 6] = 6
# #     var5 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5),spacing)
# #     @test var5 > var4

# #     data[5, 6, 6] = 7
# #     var6 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5),spacing)
# #     @test var6 > var5

# #     data[6, 6, 5] = 8
# #     var7 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5),spacing)
# #     @test var7 > var6

# #     data[6, 6, 6] = 11
# #     var8 = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.5),spacing)
# #     @test var8 > var7


# #     data = ones(10, 10, 10)
# #     data[1:5, :, :] = rand(5, 10, 10)
# #     var_a = trilinear_variance_kernel_cpu(data, (5.1, 5.5, 5.5),spacing)
# #     var_b = trilinear_variance_kernel_cpu(data, (5.9, 5.5, 5.5),spacing)
# #     @test var_a > var_b

# #     data = ones(10, 10, 10)
# #     data[:, 1:5, :] = rand(10, 5, 10)
# #     var_a = trilinear_variance_kernel_cpu(data, (5.5, 5.1, 5.5),spacing)
# #     var_b = trilinear_variance_kernel_cpu(data, (5.5, 5.9, 5.5),spacing)
# #     @test var_a > var_b

# #     data = ones(10, 10, 10)
# #     data[:, :, 1:5] = rand(10, 10, 5)
# #     var_a = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.1),spacing)
# #     var_b = trilinear_variance_kernel_cpu(data, (5.5, 5.5, 5.9),spacing)
# #     @test var_a > var_b



# # end




# """
# tetr dat check - we are testing the set_tetr_dat_kern kernel - tetr dat was updated correctly so 
#    a) first isolate inormation about some tetrahedron its indicies and their location in unmodified StatsBase
#    b) then check that the location of the tetrahedron was updated correctly
#    c) check weater the value that got associated with tetrahedron point makes sense (using interpolations tests)
#    d) establish if centroid of the tetrahedron  base is in the middle of the points of a tetrahedron base
#    e) check weather centroid interpolated correctly
#    f) check weather sv center interpolated correctly

#    testing point_info_kern function 
#     check base sample points
#     1) check weather the base sample points are on the line between the center of the triangle and the center of the super voxel
#     2) check weather the interpolation value of the sampled points agrees with interpolations checks
#     3) check is weight associated with sample point is proportional to the distance between them
#     check additional sample points
#     4) check weather they are on the line between the last base sample point and corners of the base of the tetrahedron
#     5) check weather the interpolation value of the sampled points agrees with interpolations checks
#     6) check is weight associated with sample point is proportional to the distance between them


# """


# function fill_tetrahedron_data(tetr_dat, sv_centers, control_points, index)
#     center = map(axis -> sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), axis], [1, 2, 3])
#     corners = map(corner_num ->
#             map(axis -> control_points[Int(tetr_dat[index, corner_num, 1]), Int(tetr_dat[index, corner_num, 2]), Int(tetr_dat[index, corner_num, 3]), Int(tetr_dat[index, corner_num, 4]), axis], [1, 2, 3]), [2, 3, 4])
#     corners = [center, corners...]
#     return corners
# end

# function get_tetrahedrons_from_corners(corners)
#     points = map(el -> Meshes.Point((el[1], el[2], el[3])), corners)
#     return Meshes.Tetrahedron(points...)
# end


# # Test functions


# function prepare_for_point_info_kern(tetr_dat_shape)
#     bytes_per_thread = 6
#     # blocks,threads,maxBlocks=computeBlocksFromOccupancy(point_info_kern,(tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points), bytes_per_thread)
#     # threads=256
#     threads = 128

#     # total_num=control_points_shape[1]*control_points_shape[2]*control_points_shape[3]
#     # needed_blocks=ceil(total_num / threads_apply_w)
#     needed_blocks = ceil(Int, tetr_dat_shape[1] / threads)
#     to_pad = (threads * needed_blocks) - tetr_dat_shape[1]

#     return threads, needed_blocks, to_pad
# end



# function get_on_a_line(point_a, point_b, rel_distance)
#     return [point_a[1] + (point_b[1] - point_a[1]) * rel_distance,
#         point_a[2] + (point_b[2] - point_a[2]) * rel_distance,
#         point_a[3] + (point_b[3] - point_a[3]) * rel_distance]
# end #get_on_a_line    

# function get_line_diff(point_a, point_b, rel_distance)
#     return [(point_a[1] - point_b[1]) * rel_distance,
#         (point_a[2] - point_b[2]) * rel_distance,
#         (point_a[3] - point_b[3]) * rel_distance], rel_distance
# end #get_on_a_line    

# # function get_on_a_line_b(point_a,point_b,rel_distance)
# #     return [point_a[1]+(point_b[1]-point_a[1])*rel_distance,
# #     point_a[2]+(point_b[2]-point_a[2])*rel_distance,
# #     point_a[3]+(point_b[3]-point_a[3])*rel_distance]
# # end #get_on_a_line    

# function cross_product(a, b)
#     return [a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]]
# end

# function normm(a)
#     return sqrt(sum([i^2 for i in a]))
# end

# function distance_point_to_line_gold(p, l1, l2)
#     return normm(cross_product(l2 - l1, l1 - p)) / norm(l2 - l1)
# end




# # """
# # we can check weather the approximate weights makes sense in the calculated points by ordering them from smallest to biggest
# # and checking is order consistent with objrctive order - objective order we will do by defining each sample point as a center of a sphere
# # we will the iterate and on each iteration will increase radius of a sphere by 0.1 and check weather the sphere we enlarged is intersecting
# # with any other sphere - if it is we stop growing this sphere ; we finish when all spheres intersected with some other 
# # we check weather spheres intersect by measuring the distance between their centers and checking if it is smaller than sum
# #      of their radii
# # """

# # points=out_sampled_points[1, :, 2:5]

# # # function process_points(points)
# #     # Initialize radii and volumes
# #     radii = zeros(size(points, 1))
# #     is_ready=zeros(Bool,size(points, 1))
# #     volumes = zeros(size(points, 1))

# #     # Grow spheres until they all intersect with another sphere
# #     while any(.!is_ready)
# #     for i in 1:size(points)[1]

# #             # Increase radius
# #             radii[i] += 0.05

# #             # Check for intersection with other spheres
# #             for j in 1:size(points)[1]
# #                 if i != j
# #                     dist = sqrt(sum((points[i, 2:end] - points[j, 2:end]).^2))
# #                     if dist < radii[i] + radii[j]
# #                         is_ready[i]=true
# #                         is_ready[j]=true
# #                     end
# #                 end
# #             end

# #             # Calculate volume
# #             volumes[i] = 4/3 * pi * radii[i]^3

# #             # If all spheres have intersected with another, break the loop
# #             if all(radii .> 0)
# #                 break
# #             end
# #         end
# #     end

# #     # Sort points by volume and check if the order is consistent with the order of the weights
# #     sorted_by_volume = sortperm(volumes)
# #     is_consistent = all(sorted_by_volume .== sortperm(points[:, 1]))

# #     # return is_consistent
# # # end

# # sorted_by_volume
# # sortperm(points[:, 1])

# # process_points(out_sampled_points[1, :, 2:5])


# #########################
# """
# testing unrolled versions of functions
# """
# function connection_before_set_tetr_dat_kern(x, y)
#     return (x, y)
# end

















# function get_test_data()
#     radiuss = (Float32(4.0),Float32(4.0),Float32(4.0))
#     # diam = radiuss * 2
#     num_weights_per_point = 6
#     spacing=(Float32(1.0),Float32(1.0),Float32(1.0))
#     batch_size=2

#     a = 48
#     image_shape = (a, a, a,2,batch_size)
#     num_convs_per_dim = (3, 3, 3)
#     num_base_samp_points, num_additional_samp_points = 3, 2
#     pad_voxels = 2

#     example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
#     sv_centers, control_points, tetrs, dims = example_set_of_svs

#     # weights=Float32.(weights)
#     control_points=Float32.(control_points)
#     sv_centers=Float32.(sv_centers)

#     #here we get all tetrahedrons mapped to non modified locations
#     sv_tetrs = map(index -> fill_tetrahedron_data(tetrs, sv_centers, control_points, index), 1:(size(tetrs)[1]))
#     source_arr = rand(Float32, image_shape)

#     #get dummy model 
#     rng = Random.default_rng()

#     conv2 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.tanh, stride=2, pad=Lux.SamePad(), init_weight=zeros32, init_bias=zeros32)

#     ###tetr dat model
#     conv_part = Lux.Chain(conv2(2, 6), conv2(6, 6), conv2(6, 6))
#     before_point_kerns = SkipConnection(Lux.Chain(conv_part
#     , Points_weights_str(radiuss,batch_size, pad_voxels, (image_shape[1], image_shape[2], image_shape[3]), num_convs_per_dim)), connection_before_set_tetr_dat_kern)
#     model = Lux.Chain(before_point_kerns, Set_tetr_dat_str(radiuss,spacing,batch_size, pad_voxels, (image_shape[1], image_shape[2], image_shape[3]))
#                             , Point_info_kern_str(radiuss,spacing,batch_size, (image_shape[1], image_shape[2], image_shape[3]), num_base_samp_points, num_additional_samp_points)
#     )
#     ps, st = Lux.setup(rng, model)
#     st = cu(st)
#     ps = cu(ps)
#     y_pred, st = Lux.apply(model, CuArray(source_arr), ps, st)
#     out_sampled_points, tetr_dat_out = y_pred
#     return Array(out_sampled_points), Array(tetr_dat_out), Array(sv_centers), Array(control_points), Array(tetrs), Array(sv_tetrs), Array(source_arr), num_base_samp_points, num_additional_samp_points

# end


# # @testset "set_tetr_dat_kern tests" begin
# function test_call_set_tetr_dat_kern_test_unrolled()
#     @testset "main unrolled kernels tests" begin
#         out_sampled_points, tetr_dat_out, sv_centers, control_points, tetrs, sv_tetrs, source_arr, num_base_samp_points, num_additional_samp_points = get_test_data()
#         spacing = (1.0, 1.0, 1.0)



#         # threads_point_info, blocks_point_info, pad_point_info = prepare_for_set_tetr_dat(size(tetrs))
#         # tetr_dat_out = call_set_tetr_dat_kern_test_unrolled(tetrs, source_arr, control_points, sv_centers, threads_point_info, blocks_point_info, pad_point_info)


#         # sv_tetrs[1][1]
#         # tetr_dat_out[1,1,:][1:3]
#         # @testset "is tetr dat out populated correctly" begin
#         tetr_dat_out = Array(tetr_dat_out)
#         out_sampled_points = Array(out_sampled_points)
#         for v in eachindex(sv_tetrs)
#             sum_x = 0.0
#             sum_y = 0.0
#             sum_z = 0.0
#             for p in eachindex(sv_tetrs[v])
#                 ## test is the location of the tetrahedron points was updated correctly
#                 # print("\n vvvv $(v)  $(p)\n ")
#                 @test sv_tetrs[v][p] == tetr_dat_out[v, p, :][1:3]
#                 ## check is interpolation of sv cenetr is correctly written
#                 if (p == 1)
#                     @test tetr_dat_out[v, p, 4] ≈ trilinear_interpolation_kernel_cpu(sv_tetrs[v][p], source_arr)
#                 end
#                 ## check is variance of other points is correctly written   
#                 if (p > 1)
#                     @test tetr_dat_out[v, p, 4] ≈ trilinear_variance_kernel_cpu(source_arr, sv_tetrs[v][p],spacing)
#                     sum_x += sv_tetrs[v][p][1]
#                     sum_y += sv_tetrs[v][p][2]
#                     sum_z += sv_tetrs[v][p][3]
#                 end
#             end
#             ## check is centroid of the tetrahedron base is in the middle of the points of a tetrahedron base    
#             @test tetr_dat_out[v, 5, 1] ≈ (sum_x / 3)
#             @test tetr_dat_out[v, 5, 2] ≈ (sum_y / 3)
#             @test tetr_dat_out[v, 5, 3] ≈ (sum_z / 3)
#             @test tetr_dat_out[v, 5, 4] ≈ trilinear_variance_kernel_cpu(source_arr, ((sum_x / 3), (sum_y / 3), (sum_z / 3)),spacing)
#         end

#         ########### testing point info kernel
#         for index in 1:size(tetr_dat_out)[1]
#             pp = []
#             ### testing base sample points
#             for num_point in 1:num_base_samp_points
#                 pp = get_on_a_line(tetr_dat_out[index, 1, :], tetr_dat_out[index, 5, :], num_point / (num_base_samp_points + 1))
#                 @test out_sampled_points[index, num_point, :][3:5] ≈ pp
#                 @test isapprox(out_sampled_points[index, num_point, 1], trilinear_interpolation_kernel_cpu(pp, source_arr), atol=0.001)
#                 if (num_point == num_base_samp_points)


#                     @test isapprox(((((norm((tetr_dat_out[index, 1, 1:3] - (tetr_dat_out[index, 5, 1:3])) ./ (num_base_samp_points + 1)) * 2) +
#                                       (norm((pp - (tetr_dat_out[index, 2, 1:3])) .* (1 / (num_additional_samp_points + 1)))) +
#                                       (norm((pp - (tetr_dat_out[index, 3, 1:3])) .* (1 / (num_additional_samp_points + 1)))) +
#                                       (norm((pp - (tetr_dat_out[index, 4, 1:3])) .* (1 / (num_additional_samp_points + 1))))
#                         ) / 5)^3), out_sampled_points[index, num_point, 2], atol=0.001)
#                 else

#                     @test isapprox(((((norm((tetr_dat_out[index, 1, 1:3] - (tetr_dat_out[index, 5, 1:3])) ./ (num_base_samp_points + 1)) * 2) +
#                                       distance_point_to_line_gold(pp, tetr_dat_out[index, 1, 1:3], tetr_dat_out[index, 2, 1:3]) +
#                                       distance_point_to_line_gold(pp, tetr_dat_out[index, 1, 1:3], tetr_dat_out[index, 3, 1:3]) +
#                                       distance_point_to_line_gold(pp, tetr_dat_out[index, 1, 1:3], tetr_dat_out[index, 4, 1:3])
#                         ) / 5)^3), out_sampled_points[index, num_point, 2], atol=0.001)


#                 end


#             end
#             ##pp is last base sample point
#             ### testing additional sample points
#             for num_point in 1:num_additional_samp_points
#                 for triangle_corner_num in UInt8(1):UInt8(3)
#                     pp2 = get_on_a_line(pp, tetr_dat_out[index, triangle_corner_num+1, :], num_point / (num_additional_samp_points + 1))
#                     # distst=get_line_diff(tetr_dat_out[index,triangle_corner_num+1,:],pp,num_point/(num_additional_samp_points+1))
#                     # print("\n pppp $pp2 get_line_diff $(distst) \n")
#                     @test out_sampled_points[index, (num_base_samp_points+triangle_corner_num)+((num_point-1)*3), :][3:5] ≈ pp2
#                     @test out_sampled_points[index, (num_base_samp_points+triangle_corner_num)+((num_point-1)*3), 1] ≈ trilinear_interpolation_kernel_cpu(pp2, source_arr)

#                     aa = [1, 2, 3, 4]
#                     aa = deleteat!(aa, triangle_corner_num + 1)

#                     rrr = norm((tetr_dat_out[index, triangle_corner_num+1, 1:3] - pp) ./ (num_additional_samp_points + 1)) * 2
#                     for ii in aa
#                         rrr += distance_point_to_line_gold(pp2, tetr_dat_out[index, ii, 1:3], tetr_dat_out[index, triangle_corner_num+1, 1:3])
#                     end

#                     rr = (((rrr) / 5)^3)

#                     @test isapprox(rr, out_sampled_points[index, (num_base_samp_points+triangle_corner_num)+((num_point-1)*3), 2], atol=0.001)

#                 end
#             end

#         end
#     end #main unrolled kernels tests set
# end

# test_call_set_tetr_dat_kern_test_unrolled()

# out_sampled_points, tetr_dat_out, sv_centers, control_points, tetrs, sv_tetrs, source_arr, num_base_samp_points, num_additional_samp_points = get_test_data()
# # # @testset "point_info_kern tests" begin



# """
# visualization
# visualize the points with weights as balls and the line between the center of the triangle and the center of the super voxel plus lines and balls for base and additional sample points
# """


# """
# now we want to visualize the points that were selected for sampling and their weights
#     we will display their weights by the spheres of the radius equal to weight
# """

# function radius_from_volume(volume)
#     return (volume^(1 / 3))
# end
# index = 1
# from_sampled_points = out_sampled_points[index, :, :]
# terr_dat_curr = tetr_dat_out[index, :, :]
# terr_dat_curr = terr_dat_curr[:, 1:3]
# to_disp_points = false
# to_disp_tetr = true
# to_disp_tetr_tetr_only = false

# to_points = invert(splitdims(from_sampled_points))
# points_mesh_a = map(v -> Meshes.Point(v[3], v[4], v[5]), to_points)
# points_mesh_b = map(v -> Meshes.Point(v...), invert(splitdims(terr_dat_curr)))
# points_mesh = [points_mesh_a; points_mesh_b]

# spheres = map(v -> Meshes.Sphere((v[3], v[4], v[5]), radius_from_volume(v[2]) / 2), to_points)
# # spheres=map(i->Meshes.Sphere((out_sampled_points[i,3],out_sampled_points[i,4],out_sampled_points[i,5]),out_sampled_points[i,2]/2),1:6)
# if (to_disp_points)
#     spheres = [spheres; points_mesh_b]
#     viz(spheres, color=1:length(spheres), alpha=collect(1:length(spheres)) .* 0.9)
# end

# # if (to_disp_tetr)
#     first_sv_tetrs = sv_tetrs[1:get_num_tetr_in_sv()]
#     first_sv_tetrs = map(get_tetrahedrons_from_corners, first_sv_tetrs)
#     spheres = [spheres; [first_sv_tetrs[1]]]
#     viz(spheres, color=1:length(spheres), alpha=collect(1:length(spheres)) .* 0.9)
# # end

# if (to_disp_tetr_tetr_only)
#     first_sv_tetrs = sv_tetrs[1:get_num_tetr_in_sv()]
#     first_sv_tetrs = map(get_tetrahedrons_from_corners, first_sv_tetrs)

#     viz(first_sv_tetrs, color=1:length(first_sv_tetrs), alpha=collect(1:length(first_sv_tetrs)) .* 0.9)
# end
































# viz(points_mesh, color = 1:length(points_mesh))




# p1=[2.5,5.7,1.8]
# p2=[1.4,5.1,2.3]
# p3=[2.2,3.4,5.1]

# sqrt(((((p2[2]-p3[2])*(p2[3]-p1[3]) - (p2[3]-p3[3])*(p2[2]-p1[2]))^2)+
#  (((p2[3]-p3[3])*(p2[1]-p1[1]) - (p2[1]-p3[1])*(p2[3]-p1[3]))^2)+
#  ((p2[1]-p3[1])*(p2[2]-p1[2]) - (p2[2]-p3[2])*(p2[1]-p1[1]))^2)) / sqrt((p3[2]-p2[2])^2+(p3[3]-p2[3])^2+(p3[1]-p2[1])^2)


# a=[(p2[2]-p3[2])*(p2[3]-p1[3]) - (p2[3]-p3[3])*(p2[2]-p1[2])
# , (p2[3]-p3[3])*(p2[1]-p1[1]) - (p2[1]-p3[1])*(p2[3]-p1[3])
# , (p2[1]-p3[1])*(p2[2]-p1[2]) - (p2[2]-p3[2])*(p2[1]-p1[1])]

# [i^2 for i in a]


# [(p2[2]-p3[2])*(p2[3]-p1[3]) - (p2[3]-p3[3])*(p2[2]-p1[2])
# , (p2[3]-p3[3])*(p2[1]-p1[1]) - (p2[1]-p3[1])*(p2[3]-p1[3])
# , (p2[1]-p3[1])*(p2[2]-p1[2]) - (p2[2]-p3[2])*(p2[1]-p1[1])]




# normm(cross_product(p2-p3, p2-p1))
# norm(cross(p2-p3, p2-p1))

# distance_point_to_line(p1,p2,p3)





#### getting data about first supervoxel (first 24 tetrahedrons in tetrs)


# function visualization()
#     radiuss = Float32(4.0)

#     a = 36
#     image_shape = (a, a, a)

#     example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
#     sv_centers, control_points, tetrs, dims = example_set_of_svs
#     size(tetrs)


#     first_sv_tetrs= map(index->fill_tetrahedron_data(tetrs, sv_centers,control_points,index),1:24)
#     first_sv_tetrs=map(get_tetrahedrons_from_corners,first_sv_tetrs)

#     viz(first_sv_tetrs, color=1:length(first_sv_tetrs))
# end



# using KernelAbstractions

# @kernel function mul2_kernel(A,@Const(B))
#     I = @index(Global)
#     tile = @localmem Float32 (@groupsize()[1], 4) 
#     # @print(I)
#     @print(@groupsize()[1])
#     A[I] = (3 * (A[I,1]+B[I,1]))
#   end

# dev = CPU()
# A = ones(2,3 )
# B = ones(5,6 )
# ev = mul2_kernel(dev, 32)(A,B, ndrange=(2))
# KernelAbstractions.synchronize(dev)
# A