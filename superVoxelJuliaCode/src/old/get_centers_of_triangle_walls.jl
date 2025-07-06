# """
# main idea is to check weather a line that goes from sv center to control point is inside the supervoxel
# In order to test it we need to define a mesh which surface is defined by the set of control points

# Plan is to define in meshes.jl the shape using a set of tetrahydras - we then need to check weather they intersect (they should not)
# we can then later also test sampling scheme and check if a sampled point is inside any of the tetrahedron 
# meshes additionally supply the visualization functionalities

# !! important we assume that weights are in the range between -1 and 1 (so basically after tanh)
# """

# using Revise
# using Meshes
# using LinearAlgebra
# using GLMakie
# using Combinatorics
# using SplitApplyCombine
# using CUDA
# using Combinatorics
# using Random
# using Statistics
# using ChainRulesCore




# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_a.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_b.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")


# radiuss = Float32(4.0)
# diam = radiuss * 2
# num_weights_per_point = 6
# a = 81
# image_shape = (a, a, a)

# example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
# sv_centers, control_points, tetrs, dims = example_set_of_svs

# dims_plus = (dims[1] + 1, dims[2] + 1, dims[3] + 1)
# # control_points first dimension is lin_x, lin_y, lin_z, oblique
# weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)
# # weights = rand(image_shape...)
# weights = weights .- 0.50001
# weights = (weights) .* 100
# weights = Float32.(tanh.(weights * 0.02))


# control_points_size = size(control_points)


# """
# check the optimal launch configuration for the kernel
# calculate the number of threads and blocks and how much padding to add if needed
# """
# function prepare_for_apply_weights_to_locs_kern(control_points_shape, weights_shape)
#     bytes_per_thread = 0
#     blocks_apply_w, threads_apply_w, maxBlocks = computeBlocksFromOccupancy(apply_weights_to_locs_kern, (cast_dev_zeros(control_points_shape...), cast_dev_zeros(control_points_shape...), cast_dev_zeros(weights_shape...), Float32(3), UInt32(control_points_shape[1]), UInt32(control_points_shape[2]), UInt32(control_points_shape[3]), UInt32(0), UInt32(0), UInt32(0)
#         ), bytes_per_thread)
#     # total_num=control_points_shape[1]*control_points_shape[2]*control_points_shape[3]
#     # needed_blocks=ceil(total_num / threads_apply_w)
#     # threads_res=(8,8,(floor(Int,threads_apply_w/64)))
#     threads_res = (8, 8, (floor(Int, threads_apply_w / 128)))
#     needed_blocks = (ceil(Int, control_points_shape[1] / threads_res[1]), ceil(Int, control_points_shape[2] / threads_res[2]), ceil(Int, control_points_shape[3] / threads_res[3]))

#     return threads_res, needed_blocks
# end


# threads_apply_w, blocks_apply_w = prepare_for_apply_weights_to_locs_kern(control_points_size, size(weights))

# control_points = call_apply_weights_to_locs_kern(CuArray(control_points), cast_dev_zeros(size(control_points)...), CuArray(weights), radiuss, threads_apply_w, blocks_apply_w)




# # control_points=Array(control_points)

# # #how many main sample points we want to have between each triangle center and sv center in each tetrahedron
# # num_base_samp_points=3
# # num_additional_samp_points=2

# # varr=10.0
# # meann=-0.8

# # # source_arr = meann .+ sqrt(varr) .* randn(Int.((dims.*(radiuss*2) ).+((radiuss*3))) )
# # # source_arr = meann .+ sqrt(varr) .* randn(Int.((dims.*(radiuss*2) )))
# # source_arr = meann .+ sqrt(varr) .* randn(image_shape...)

# # out_sampled_points=zeros((size(tetrs)[1],num_base_samp_points+(3*num_additional_samp_points),5))


# # #we get the threads constant as it uses constant shared memory


# # tetrs=CuArray(tetrs)
# # out_sampled_points=CuArray(out_sampled_points)
# # source_arr=CuArray(source_arr)
# # control_points=control_points.+radiuss
# # sv_centers=sv_centers.+radiuss
# # control_points=CuArray(control_points)
# # sv_centers=CuArray(sv_centers)



# # bytes_per_thread=6

# # function prepare_for_point_info(tetr_dat_shape)
# #     bytes_per_thread=6
# #     # blocks,threads,maxBlocks=computeBlocksFromOccupancy(point_info_kern,(tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points), bytes_per_thread)
# #     # threads=256
# #     threads=128

# #     # total_num=control_points_shape[1]*control_points_shape[2]*control_points_shape[3]
# #     # needed_blocks=ceil(total_num / threads_apply_w)
# #     needed_blocks=ceil(Int,tetr_dat_shape[1]/threads)
# #     to_pad=(threads*needed_blocks)-tetr_dat_shape[1]

# #     return threads,needed_blocks,to_pad
# # end

# # threads_point_info,blocks_point_info,pad_point_info=prepare_for_point_info(size(tetrs))

# # # print("**************      pre         ********************* tetrs $(size(tetrs))\n")
# # # for triangle_corner_num in 1:5
# # #     for axis in 1:3
# # #         print("triangle_corner_num $(triangle_corner_num) axis $(axis) maxxx ",maximum(tetrs[:,triangle_corner_num,axis]),"\n  ")
# # #         print("triangle_corner_num $(triangle_corner_num) axis $(axis)   minn ",minimum(tetrs[:,triangle_corner_num,axis]),"\n  ")

# # # end    
# # # end

# # tetrs=call_set_tetr_dat_kern(tetrs,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads_point_info,blocks_point_info,pad_point_info)
# # out_sampled_points=call_point_info_kern(tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads_point_info,blocks_point_info,pad_point_info)


# # # print("**************      post         ********************* \n")
# # # for triangle_corner_num in 1:5
# # #     for axis in 1:3
# # #         print("triangle_corner_num $(triangle_corner_num) axis $(axis) maxxx ",maximum(tetrs[:,triangle_corner_num,axis]),"\n  ")
# # #         print("triangle_corner_num $(triangle_corner_num) axis $(axis)   minn ",minimum(tetrs[:,triangle_corner_num,axis]),"\n  ")

# # # end    
# # # end
# # tetr_dat=tetrs

# # out_sampled_points=Float32.(out_sampled_points)
# # source_arr=Float32.(source_arr)
# # control_points=Float32.(control_points)
# # print("ccccccccccccccccc tetrs $(size(tetrs)) sum $(sum(tetrs))")
# # sv_centers

# # # a,ff=rrule(call_set_tetr_dat_kern,tetrs,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads_point_info,blocks_point_info,pad_point_info)
# # # # a,ff=rrule(call_set_tetr_dat_kern,tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads_point_info,blocks_point_info,pad_point_info)
# # # ff(tetr_dat)
# # # a,ff=rrule(call_point_info_kern,tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads_point_info,blocks_point_info,pad_point_info)
# # # a,ff=rrule(call_point_info_kern_add_b,tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads_point_info,blocks_point_info,pad_point_info)
# # a,ff=rrule(call_point_info_kern,tetrs,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads_point_info,blocks_point_info,pad_point_info)
# # ff(out_sampled_points)

# # maximum(a)
# # maximum(out_sampled_points)


# # maximum(tetr_dat[:,:,1])
# # # maximum(tetr_dat[:,:,2])
# # # maximum(tetr_dat[:,:,3])

# # # minimum(tetr_dat[:,:,1])
# # # minimum(tetr_dat[:,:,2])
# # # minimum(tetr_dat[:,:,3])

# # # control_points[1,1,1,:]

# # # maximum(control_points[:,:,:,1])
# # # maximum(control_points[:,:,:,2])
# # # maximum(control_points[:,:,:,3])

# # # size(out_sampled_points)

# # # size(source_arr)


# # """
# # now we want to visualize the points that were selected for sampling and their weights
# #     we will display their weights by the spheres of the radius equal to weight
# # """

# # # points_mesh_a=Meshes.Point3.(invert(splitdims(out_sampled_points[:,3:5])))
# # # tt=invert(splitdims(tetr_dat))
# # # # tt=tt[2:4]
# # # points_mesh_b=Meshes.Point3.(tt)
# # # points_mesh=[points_mesh_a;points_mesh_b]

# # # spheres=[spheres;points_mesh]
# # # viz(spheres, color = 1:length(spheres),alpha=collect(1:length(spheres)).*0.9)
# # # # viz(points_mesh, color = 1:length(points_mesh))
# # # viz(tetrs, color = 1:length(tetrs))


# # """
# # we want to check weather weighted sampling is working correctly by gettin base arr as the gaussian with known mean and variance
# # and checking weather our samples have similar weighted mean and variance
# # """

# # # using StatsBase

# # # m1=mean(ssamp[:,1])
# # # m2=sum(ssamp[:,1].*ssamp[:,2])/sum(ssamp[:,2])

# # # m1,m2
# # # v1=var(ssamp[:,1])
# # # v2=sum(ssamp[:,2].*(ssamp[:,1].-m2).^2)/sum(ssamp[:,2])
# # # v3=var(source_arr)
# # # v1,v2,v3

# # # ,var(ssamp[:,1])