# using Pkg
# using ChainRulesCore, Zygote, CUDA, Enzyme
# using KernelAbstractions
# using Zygote, Lux
# using Lux, Random
# import Optimisers, Plots, Random, Statistics, Zygote

# using LinearAlgebra

# using Revise

# Enzyme.API.strictAliasing!(false)# taken from here https://github.com/EnzymeAD/Enzyme.jl/issues/1159


# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")

# function point_info_kern_deff_add_a(tetr_dat, d_tetr_dat, out_sampled_points, d_out_sampled_points, source_arr, d_source_arr, control_points, d_control_points, sv_centers, d_sv_centers, num_base_samp_points, num_additional_samp_points
# )
#     # shared_arr = CuStaticSharedArray(Float32, (128,3))
#     # d_shared_arr = CuStaticSharedArray(Float32, (128,3))
#     Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(point_info_kern_add_a), Const
#         # , Duplicated(CuStaticSharedArray(Float32, (128,3)), CuStaticSharedArray(Float32, (128,3)))
#         , Duplicated(tetr_dat, d_tetr_dat), Duplicated(out_sampled_points, d_out_sampled_points), Duplicated(source_arr, d_source_arr), Duplicated(control_points, d_control_points), Duplicated(sv_centers, d_sv_centers), Const(num_base_samp_points), Const(num_additional_samp_points))


#     return nothing
# end


# function call_point_info_kern_add_a(tetr_dat, out_sampled_points, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points, threads, blocks, pad_point_info)
#     # shared_arr = CuStaticSharedArray(Float32, (threads[1],3))
#     # shared_arr = CuStaticSharedArray(Float32, (100,3))
#     # shared_arr = CuDynamicSharedArray(Float32, (threads[1],3))
#     #shmem is in bytes
#     tetr_shape = size(tetr_dat)
#     out_shape = size(out_sampled_points)
#     to_pad_tetr = CUDA.ones(pad_point_info, tetr_shape[2], tetr_shape[3]) * 2
#     tetr_dat = vcat(tetrs, to_pad_tetr)

#     to_pad_out = CUDA.ones(pad_point_info, out_shape[2], out_shape[3]) * 2
#     out_sampled_points = vcat(out_sampled_points, to_pad_out)


#     #@device_code_warntype  @cuda threads = threads blocks = blocks point_info_kern(CuStaticSharedArray(Float32, (128,3)),tetr_dat,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points)
#     @cuda threads = threads blocks = blocks point_info_kern_add_a(tetr_dat, out_sampled_points, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points)


#     tetr_dat = tetr_dat[1:tetr_shape[1], :, :]
#     out_sampled_points = out_sampled_points[1:out_shape[1], :, :]

#     # @device_code_warntype @cuda threads = threads blocks = blocks testKern( A, p,  Aout,Nx)
#     return out_sampled_points
# end



# # rrule for ChainRules.
# function ChainRulesCore.rrule(::typeof(call_point_info_kern_add_a), tetr_dat, out_sampled_points, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info, pad_point_info)


#     out_sampled_points = call_point_info_kern_add_a(tetr_dat, out_sampled_points, source_arr, control_points, sv_centers, num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info, pad_point_info)
#     d_tetr_dat = CUDA.ones(size(tetr_dat)...)
#     d_source_arr = CUDA.ones(size(source_arr)...)
#     d_control_points = CUDA.ones(size(control_points)...)
#     d_sv_centers = CUDA.ones(size(sv_centers)...)


#     #pad to avoid conditionals in kernel
#     tetr_shape = size(tetr_dat)
#     out_shape = size(out_sampled_points)
#     to_pad_tetr = CUDA.ones(pad_point_info, tetr_shape[2], tetr_shape[3])
#     tetr_dat = vcat(tetrs, to_pad_tetr)
#     d_tetr_dat = vcat(d_tetr_dat, to_pad_tetr)

#     to_pad_out = CUDA.ones(pad_point_info, out_shape[2], out_shape[3])
#     out_sampled_points = vcat(out_sampled_points, to_pad_out)


#     function call_test_kernel1_pullback(d_out_sampled_points)
#         #@device_code_warntype @cuda threads = threads blocks = blocks testKernDeff( A, dA, p, dp, Aout, CuArray(collect(dAout)),Nx)


#         d_out_sampled_points = vcat(CuArray(collect(d_out_sampled_points)), to_pad_out)

#         #@device_code_warntype 
#         # @device_code_warntype  @cuda threads = threads_point_info blocks = blocks_point_info Enzyme.autodiff_deferred(Enzyme.Reverse,point_info_kern, Const
#         #                                                                         , Duplicated(tetr_dat, d_tetr_dat)
#         #                                                                         , Duplicated(out_sampled_points, d_out_sampled_points)
#         #                                                                         , Duplicated(source_arr, d_source_arr)
#         #                                                                         , Duplicated(control_points, d_control_points)
#         #                                                                         , Duplicated(sv_centers, d_sv_centers)
#         #                                                                         ,Const(num_base_samp_points)
#         #                                                                         ,Const(num_additional_samp_points) 
#         # )  


#         @device_code_warntype @cuda threads = threads_point_info blocks = blocks_point_info point_info_kern_deff_add_a(tetr_dat, d_tetr_dat, out_sampled_points, d_out_sampled_points, source_arr, d_source_arr, control_points, d_control_points, sv_centers, d_sv_centers, num_base_samp_points, num_additional_samp_points
#         )
#         #reverse padding



#         d_tetr_dat = d_tetr_dat[1:tetr_shape[1], :, :]
#         d_out_sampled_points = d_out_sampled_points[1:out_shape[1], :, :]

#         return NoTangent(), d_tetr_dat, d_out_sampled_points, d_source_arr, d_control_points, d_sv_centers, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
#     end

#     # tetr_dat=tetr_dat[1:tetr_shape[1],:,:]
#     # out_sampled_points=out_sampled_points[1:out_shape[1],:,:]


#     return out_sampled_points, call_test_kernel1_pullback

# end
