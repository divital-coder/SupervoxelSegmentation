using Pkg
using ChainRulesCore, Zygote, CUDA, Enzyme
using KernelAbstractions
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote

using LinearAlgebra

using Revise
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern_unrolled.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern _old.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/main_loss.jl")
# Enzyme.API.strictAliasing!(false)# taken from here https://github.com/EnzymeAD/Enzyme.jl/issues/1159


# function set_point_info_kern_deff(tetr_dat,d_tetr_dat, out_sampled_points, d_out_sampled_points, source_arr,d_source_arr, num_base_samp_points,num_additional_samp_points,max_index)
function set_point_info_kern_deff(tetr, out, so, num_base_samp_points, num_additional_samp_points, max_index, spacing, dims)
    Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(point_info_kern_unrolled), Const, tetr, out, so, Const(num_base_samp_points)
    , Const(num_additional_samp_points), Const(max_index), Const(spacing), Const(dims))
    return nothing
end



function call_point_info_kern(tetr_dat, source_arr, num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info, batch_size, spacing)
    
    out_sampled_points = CUDA.zeros((size(tetr_dat)[1], num_base_samp_points + (3 * num_additional_samp_points), 5, batch_size))
    max_index = size(tetr_dat)[1]

    @cuda threads = threads_point_info blocks = blocks_point_info point_info_kern_unrolled(
        tetr_dat, out_sampled_points, source_arr, num_base_samp_points, num_additional_samp_points, max_index, spacing, size(source_arr))

    #@device_code_warntype  
    return out_sampled_points
end



# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_point_info_kern), tetr_dat, source_arr, num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info, batch_size, spacing)

    out_sampled_points = call_point_info_kern(tetr_dat, source_arr, num_base_samp_points, num_additional_samp_points, threads_point_info, blocks_point_info, batch_size, spacing)

    function call_test_kernel1_pullback(d_out_sampled_points)
        #@device_code_warntype 
        current_time = Dates.now()
        d_out_sampled_points = CuArray(Zygote.unthunk(d_out_sampled_points))

        max_index = size(tetr_dat)[1]

        d_tetr_dat = CUDA.zeros(size(tetr_dat)...)
        d_source_arr = CUDA.zeros(size(source_arr)...)
        so = Duplicated(source_arr, d_source_arr)
        tetr = Duplicated(tetr_dat, d_tetr_dat)
        out = Duplicated(out_sampled_points, d_out_sampled_points)


        @cuda threads = threads_point_info blocks = blocks_point_info set_point_info_kern_deff(tetr, out, so, num_base_samp_points, num_additional_samp_points, max_index, spacing, size(source_arr))
        # @cuda threads = threads_point_info blocks = blocks_point_info set_point_info_kern_deff(tetr_dat,d_tetr_dat, out_sampled_points
        # , d_out_sampled_points, source_arr,d_source_arr, num_base_samp_points,num_additional_samp_points,max_index)
        # dsss=sum(d_source_arr)
        # seconds_diff = Dates.value(Dates.now() - current_time)/ 1000
        # print("\n point loss kern backward d_source_arr sum $(dsss) ; $(seconds_diff) sec \n")
        # @info "point loss kern backward  " times=round(seconds_diff; digits = 2)
    


        return NoTangent(), d_tetr_dat, d_source_arr, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return out_sampled_points, call_test_kernel1_pullback

end
############## lux definitions
struct Point_info_kern_str <: Lux.AbstractLuxLayer
    radiuss::Tuple{Float32,Float32,Float32}
    spacing::Tuple{Float32,Float32,Float32}
    batch_size::Int
    image_shape::Tuple{Int64,Int64,Int64}
    num_base_samp_points::Int64
    num_additional_samp_points::Int64
    is_point_per_triangle::Bool
    first_pass::Bool
end



function Lux.initialparameters(rng::AbstractRNG, l::Point_info_kern_str)
    return ()
end
"""
check the optimal launch configuration for the kernel
calculate the number of threads and blocks and how much padding to add if needed
"""
function prepare_point_info_kern(image_shape, tetr_dat_shape, batch_size)
    # ,control_points_shape,sv_centers_shape)
    # bytes_per_thread=0
    # blocks_apply_w,threads_res,maxBlocks=set_tetr_dat_kern_unrolled(Cuda.zeros(tetr_dat_shape...)
    # , Cuda.zeros(tetr_dat_shape...)
    # , Cuda.zeros(image_shape...), control_points, sv_centers,max_index)
    threads_res = 256
    needed_blocks = ceil(Int, tetr_dat_shape[1] / threads_res)

    return threads_res, (needed_blocks, batch_size)
end
function Lux.initialstates(::AbstractRNG, l::Point_info_kern_str)::NamedTuple


    threads_point_info, blocks_point_info = prepare_point_info_kern(l.image_shape, get_tetr_dat_shape(l.image_shape), l.batch_size)
    return (first_pass=l.first_pass,  spacing=l.spacing, batch_size=l.batch_size, radiuss=l.radiuss, image_shape=l.image_shape, threads_point_info=threads_point_info, blocks_point_info=blocks_point_info, num_base_samp_points=l.num_base_samp_points, num_additional_samp_points=l.num_additional_samp_points)

end

function (l::Point_info_kern_str)(x, ps, st::NamedTuple)


    tetr_dat, source_arr, cp = x

    out_sampled_points = call_point_info_kern(tetr_dat, source_arr, st.num_base_samp_points, st.num_additional_samp_points, st.threads_point_info, st.blocks_point_info, st.batch_size, st.spacing)
    #reshaping to get the supervoxel index dimension
    sizz_out=size(out_sampled_points)
    out_sampled_points = reshape(out_sampled_points[:, :, :, :], (get_num_tetr_in_sv(l.is_point_per_triangle), Int(round(sizz_out[1] / get_num_tetr_in_sv(l.is_point_per_triangle))), sizz_out[2], 5, l.batch_size))
    out_sampled_points = permutedims(out_sampled_points, [2, 1, 3, 4, 5])
    
    b_loss=(get_sv_variance_loss(out_sampled_points,false)^2)/get_border_loss(tetr_dat)
    return (out_sampled_points, tetr_dat,(b_loss)), st

    # if(st.first_pass)
    #     tetr_dat, source_arr,cp = x

    #     out_sampled_points = call_point_info_kern(tetr_dat, source_arr, st.num_base_samp_points, st.num_additional_samp_points, st.threads_point_info, st.blocks_point_info, st.batch_size, st.spacing)
    #     #reshaping to get the supervoxel index dimension
    #     sizz_out=size(out_sampled_points)
    #     out_sampled_points = reshape(out_sampled_points[:, :, :, :], (get_num_tetr_in_sv(l.is_point_per_triangle), Int(round(sizz_out[1] / get_num_tetr_in_sv(l.is_point_per_triangle))), sizz_out[2], 5, l.batch_size))
    #     out_sampled_points = permutedims(out_sampled_points, [2, 1, 3, 4, 5])
        
    #     b_loss=(get_sv_variance_loss(out_sampled_points,false)^2)/get_border_loss(tetr_dat)
        
    #     return (b_loss, source_arr,cp), st

    # else
    #     tetr_dat, source_arr,b_loss = x
    #     out_sampled_points = call_point_info_kern(tetr_dat, source_arr, st.num_base_samp_points, st.num_additional_samp_points, st.threads_point_info, st.blocks_point_info, st.batch_size, st.spacing)
    #     #reshaping to get the supervoxel index dimension
    #     sizz_out=size(out_sampled_points)
    #     out_sampled_points = reshape(out_sampled_points[:, :, :, :], (get_num_tetr_in_sv(l.is_point_per_triangle), Int(round(sizz_out[1] / get_num_tetr_in_sv(l.is_point_per_triangle))), sizz_out[2], 5, st.batch_size))
    #     out_sampled_points = permutedims(out_sampled_points, [2, 1, 3, 4, 5])
        
    #     b_loss_b=(get_sv_variance_loss(out_sampled_points,false)^2)/get_border_loss(tetr_dat)


    #     return (out_sampled_points, tetr_dat,(b_loss+b_loss_b)), st
    # end

    
end
