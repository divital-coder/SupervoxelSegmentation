using Revise, CUDA, HDF5
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
using KernelAbstractions, Dates
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern_unrolled.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern _old.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")



"""
We want to create a synthethic image to detect the performance of the linear sampling kernel.
so we apply set of random weights 
"""

function prepare_for_kern(tetr_dat_shape)
    threads = 256

    needed_blocks = ceil(Int, tetr_dat_shape[1] / threads)
    to_pad = (threads * needed_blocks) - tetr_dat_shape[1]

    return threads, needed_blocks, to_pad
end



"""
given the coordinates of the sampled points and the image cartesian coordinates we want to interpolate the values of the sampled points to the image
will iterate through sampled points and each time will check weather after cail or floor operation on the coordinates of the sampled point we get the same
image coordinate as the coordinate analysed by this thread in kernel next we calculate distance from this sampled point to the image coordinate 
and divide value of sampled point by the distance and add it to the image coordinate value; we also store the sum of weights for all sampled points
we want to associate with given point in the image so after iterating through sample points we will divide the value of the image coordinate by the sum of weights
"""
@kernel function inverse_interp_kern(@Const(image_cartesian), @Const(coords), @Const(len_coords), res)
    index = @index(Global)
    res[index] = 0.0
    # dist_sum=0.0
    shared_arr = @localmem Float32 (128, 2)
    shared_arr[@index(Local, Linear), 1] = 0.0
    shared_arr[@index(Local, Linear), 2] = 0.0
    # shared_arr[@index(Local, Linear), 3] = 0.0
    # shared_arr[@index(Local, Linear), 4] = 0.0


    # we need to look over not cartesian indicies of new array and check just is a point in a given 
    # tetrahedron next we can additionally use it to make sure that sample popints are always inside sv
    shared_arr[@index(Local, Linear), 1] = 0.0

    for i in range(1, (len_coords - 1))
        # dist= sqrt(( ((image_cartesian[index,1] -shared_arr[@index(Local, Linear),1] ) ^2)+((image_cartesian[index,2] -shared_arr[@index(Local, Linear),2] ) ^2)+((image_cartesian[index,3] -shared_arr[@index(Local, Linear),3] ) ^2) ))
        shared_arr[@index(Local, Linear), 2] = sqrt(((image_cartesian[index, 1] - coords[1, i])^2) + ((image_cartesian[index, 2] - coords[2, i])^2) + ((image_cartesian[index, 3] - coords[3, i])^2))
        if (shared_arr[@index(Local, Linear), 2] < 4.5)
            shared_arr[@index(Local, Linear), 1] += shared_arr[@index(Local, Linear), 2]
            # @print("   ** $(dist)   **")
            # @print("   **i $i $(@index(Local, Linear))   **")
            res[index] += (coords[4, i] * shared_arr[@index(Local, Linear), 2])
        end
    end

    if (res[index] > 0.0)
        # @print("  $(res[index])  "  )
        res[index] = res[index] / shared_arr[@index(Local, Linear), 1]
    end

end#inverse_interp_kern

function threaded_map(f, max_index, out)
    # out = similar(arr)
    Threads.@threads for i in 1:max_index
        out[i] = f(i)
    end
    return out
end


"""
given radiuss of the supervoxel and the size of the image we want to create (cube with edge lenth a) 
it will return 3 dimansional array that is result of inversly interpolated values of the sampled points
where each supervoxel has associated consecutive integer value, the shapes of supervoxels are random (based on random weights)
"""
function create_synth_image_for_test(radiuss, a)
    diam = radiuss * 2
    num_weights_per_point = 6
    image_shape = (a, a, a)
    # initialization of data structures
    # print("\n uuuuuu 1 \n");current_time = get_current_time() 

    example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
    sv_centers, control_points, tetrs, dims = example_set_of_svs
    source_arr = rand(Float32, image_shape)
    num_base_samp_points, num_additional_samp_points = 3, 2
    # print("\n uuuuuu 2 min: $( Dates.value(get_current_time() - current_time)/ 60000.0) \n");current_time = get_current_time() 


    #get the number of threads and blocks needed for the kernel
    threads_point_info, blocks_point_info, pad_point_info = prepare_for_kern(size(tetrs))
    max_index = size(tetrs)[1]
    num_base_samp_points, num_additional_samp_points = 3, 2
    #put on GPU
    tetr_dat = CuArray(Float32.(tetrs))
    sv_centers = CuArray(Float32.(sv_centers))
    control_points = CuArray(Float32.(control_points))
    source_arr = CuArray(Float32.(source_arr))
    tetr_dat_out = CuArray(Float32.(tetrs))
    # print("\n uuuuuu 3 min: $( Dates.value(get_current_time() - current_time)/ 60000.0) \n");current_time = get_current_time() 

    out_sampled_points = CUDA.zeros((size(tetr_dat)[1], num_base_samp_points + (3 * num_additional_samp_points), 5))
    #initialize random weights
    dims_plus = (dims[1] + 1, dims[2] + 1, dims[3] + 1)
    weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)
    weights = weights .- 0.50001
    weights = (weights) .* 100
    weights = CuArray(Float32.(tanh.(weights * 0.1)))
    # weights = CuArray(Float32.(tanh.(weights)))
    # weights=CUDA.ones(size(weights)...).*0.5
    # print("\n uuuuuu 4 min: $( Dates.value(get_current_time() - current_time)/ 60000.0) \n");current_time = get_current_time() 

    # control_points_out = CuArray(Float32.(control_points))
    threads_apply_w, blocks_apply_w = prepare_for_apply_weights_to_locs_kern(size(control_points), size(weights))
    control_points_out = call_apply_weights_to_locs_kern(control_points, weights, radiuss, threads_apply_w, blocks_apply_w)
    # print("\n uuuuuu 5 min: $( Dates.value(get_current_time() - current_time)/ 60000.0) \n");current_time = get_current_time() 

    control_points = control_points_out
    ### execute kernel no autodiff
    @cuda threads = threads_point_info blocks = blocks_point_info set_tetr_dat_kern_forward(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers, max_index)
    tetr_dat = tetr_dat_out
    # print("\n uuuuuu 6 min: $( Dates.value(get_current_time() - current_time)/ 60000.0) \n");current_time = get_current_time() 
    CUDA.synchronize()
    @cuda threads = threads_point_info blocks = blocks_point_info point_info_kern_forward(tetr_dat_out, out_sampled_points, source_arr, num_base_samp_points, num_additional_samp_points, max_index)
    # print("\n uuuuuu 7 min: $( Dates.value(get_current_time() - current_time)/ 60000.0) \n");current_time = get_current_time() 
    CUDA.synchronize()
    out_sampled_points = Array(out_sampled_points)
    #getting coordinates of each sampled point and assigning them a value
    coords = threaded_map(index -> invert(splitdims(out_sampled_points[index, :, 3:5])), size(out_sampled_points)[1], [[] for _ in 1:size(out_sampled_points)[1]])
    #for now we assign value that is just integer index of which supervoxel they are from we divide by 24 to get the supervoxel index not tetrahedron index
    coords = threaded_map(index -> map(el -> [el..., Int(ceil(index / 24))], coords[index]), size(out_sampled_points)[1], [[] for _ in 1:size(out_sampled_points)[1]])

    coords = reduce(vcat, coords)
    coords = reduce(hcat, coords) #shape 4xn
    #now we get cartesian coordinates of the image 
    image_cartesian = get_base_indicies_arr(image_shape)
    sh = size(image_cartesian)
    image_cartesian = reshape(image_cartesian, (sh[1] * sh[2] * sh[3], 3))

    # print("\n uuuuuu 8 min: $( Dates.value(get_current_time() - current_time)/ 60000.0) \n");current_time = get_current_time() 

    # coords=coords[:,1:20]
    # image_cartesian=image_cartesian[1:9000,:] #
    len_coords = size(coords)[2]
    len_image_coords = size(image_cartesian)[1]
    res = CUDA.zeros(len_image_coords)
    dev = get_backend(res)
    inverse_interp_kern(dev, 128)(CuArray(image_cartesian), CuArray(coords), len_coords, res, ndrange=(len_image_coords))
    CUDA.synchronize()
    KernelAbstractions.synchronize(dev)
    # print("\n uuuuuu 9 min: $( Dates.value(get_current_time() - current_time)/ 60000.0) \n");current_time = get_current_time() 

    res_im = reshape(Array(res), (a, a, a))
    return res_im, Array(weights), Array(tetr_dat_out)
end

radiuss = Float32(4.0)
a = 128

function get_current_time()
    return Dates.now()
end






# current_time = get_current_time() 

# # res_im,weights=create_synth_image_for_test(radiuss,a)

# h5_path="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/synth_data.h5"
# f = h5open(h5_path, "w")

#     # Create a group named "i"
#     for i in 1:3
#         res_im,weights,tetr_dat_out=create_synth_image_for_test(radiuss,a)
#         print("\n iii $i $(sum(res_im))  res_im $(size(res_im)) weights $(size(weights))  tetr_dat_out $(size(tetr_dat_out))  \n")

#         group_name = string(i)
#         g = create_group(f,group_name)
#         dset = create_dataset(g, "image", res_im)
#         dsetb = create_dataset(g, "weights", weights)
#         dsetc = create_dataset(g, "tetr", tetr_dat_out)

#         write(f["$(i)/image"],res_im)
#         write(f["$(i)/weights"],weights)
#         write(f["$(i)/tetr"],tetr_dat_out)

#         print("\n iii $i $(sum(res_im)) dataset : $(sum(g["image"][:,:,:])) ;; res_im $(size(res_im)) weights $(size(weights))  tetr_dat_out $(size(tetr_dat_out))  \n")


#     end
# close(f)

# println("Time taken (minutes): ", Dates.value(get_current_time() - current_time)/ 60000.0) #15.297383333333334






# using PyCall
# sitk = pyimport_conda("SimpleITK", "simpleitk")
# # sitk.GetImageFromArray(res_im)
# sitk.WriteImage( sitk.GetImageFromArray(res_im),"/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/res_im.nii.gz")
# a
# # ##### viss


# function fill_tetrahedron_data(tetr_dat, sv_centers, control_points, index)
#     center = map(axis -> sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), axis], [1, 2, 3])
#     corners = map(corner_num ->
#             map(axis -> control_points[Int(tetr_dat[index, corner_num, 1]), Int(tetr_dat[index, corner_num, 2]), Int(tetr_dat[index, corner_num, 3]), Int(tetr_dat[index, corner_num, 4]), axis], [1, 2, 3]), [2, 3, 4])
#     corners = [center, corners...]
#     return corners
# end

# function get_tetrahedrons_from_corners(corners)
#     # print("ttttttttttttt $(corners)")
#     # points = map(el -> Meshes.Point((el[1], el[2], el[3])), corners)
#     points = map(el -> (el[1], el[2], el[3]), corners)[1:4]
#     return Meshes.Tetrahedron(points...)
# end


# index=1
# # first_sv_tetrs= map(index->fill_tetrahedron_data(Array(tetr_dat_out), Array(sv_centers),Array(control_points),index),1:24)
# t=Array(tetr_dat_out)
# first_sv_tetrs=t[1:24,:,:][:,:,1:3]
# first_sv_tetrs= map(index->get_tetrahedrons_from_corners( invert(splitdims(first_sv_tetrs[index,:,:]))),1:24)

# viz(first_sv_tetrs, color=1:length(first_sv_tetrs))

# function is_point_in_tetrahedron(point, v1, v2, v3, v4)
#     # Compute the barycentric coordinates of the point
#     mat = [v1 v2 v3 v4; 1 1 1 1]
#     bary_coords = mat \ [point; 1]

#     # Check if all barycentric coordinates are between 0 and 1
#     return all(0 .<= bary_coords .<= 1)
# end

# # Usage
# point = [1.0, 1.0, 1.0]
# v1 = [0.0, 0.0, 0.0]
# v2 = [2.0, 0.0, 0.0]
# v3 = [0.0, 2.0, 0.0]
# v4 = [0.0, 0.0, 2.0]

# println(is_point_in_tetrahedron(point, v1, v2, v3, v4))  # Should print true