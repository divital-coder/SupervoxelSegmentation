using Pkg
using ChainRulesCore, Zygote, CUDA, Enzyme
using KernelAbstractions
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote

using LinearAlgebra

using Revise

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern_unrolled.jl")


function set_tetr_dat_kern_deff(tetr_dat, d_tetr_dat, tetr_dat_out, d_tetr_dat_out, source_arr, d_source_arr, control_points, d_control_points, sv_centers, d_sv_centers, max_index, spacing, dims)
    Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(set_tetr_dat_kern_unrolled), Const, Duplicated(tetr_dat, d_tetr_dat), Duplicated(tetr_dat_out, d_tetr_dat_out), Duplicated(source_arr, d_source_arr), Duplicated(control_points, d_control_points), Duplicated(sv_centers, d_sv_centers), Const(max_index), Const(spacing), Const(dims))
    return nothing
end



function call_set_tetr_dat_kern(tetr_dat, source_arr, control_points, sv_centers, threads_tetr_set, blocks_tetr_set, spacing, batch_size)
    # tetr_dat_out = copy(tetr_dat)
    tetr_dat_out = CUDA.zeros(Float32,size(tetr_dat,1),size(tetr_dat,2),size(tetr_dat,2),batch_size)#repeat(Array(tetr_dat), inner=(1, 1, 1, batch_size)) 
    # tetr_dat_out = CuArray(tetr_dat_out)

    max_index = size(tetr_dat)[1]
    @cuda threads = threads_tetr_set blocks = blocks_tetr_set set_tetr_dat_kern_unrolled(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers, max_index, spacing, size(source_arr))
   
   
    return tetr_dat_out
end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_set_tetr_dat_kern), tetr_dat, source_arr, control_points, sv_centers, threads_tetr_set, blocks_tetr_set, spacing, batch_size)

    #we get here correct tetr dat out by mutation
    tetr_dat_out = call_set_tetr_dat_kern(tetr_dat, source_arr, control_points, sv_centers, threads_tetr_set, blocks_tetr_set, spacing, batch_size)


    function call_test_kernel1_pullback(d_tetr_dat_out)

        current_time = Dates.now()
        d_tetr_dat_out = CuArray(Zygote.unthunk(collect(d_tetr_dat_out)))
        d_tetr_dat = CUDA.zeros(size(tetr_dat)...)
        d_source_arr = CUDA.zeros(size(source_arr)...)
        d_control_points = CUDA.zeros(size(control_points)...)
        d_sv_centers = CUDA.zeros(size(sv_centers)...)
        max_index = size(tetr_dat_out)[1]

        @cuda threads = threads_tetr_set blocks = blocks_tetr_set set_tetr_dat_kern_deff(tetr_dat, d_tetr_dat, tetr_dat_out, d_tetr_dat_out, source_arr, d_source_arr, control_points, d_control_points, sv_centers, d_sv_centers, max_index, spacing, size(source_arr))

        # dsss=sum(d_source_arr)
        # seconds_diff = Dates.value(Dates.now() - current_time)/ 1000
        # print("\n tetr loss kern backward d_source_arr sum $(dsss) ; $(seconds_diff) sec \n")
        # @info "tetr loss kern backward  " times=round(seconds_diff; digits = 2)
    

        # return NoTangent(), d_tetr_dat, d_source_arr, d_control_points, d_sv_centers, NoTangent(), NoTangent(), NoTangent(), NoTangent()
        return NoTangent(), d_tetr_dat, d_source_arr, d_control_points, d_sv_centers, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    # tetr_dat_out=tetr_dat_out[1:tetr_shape[1],:,:]

    return tetr_dat_out, call_test_kernel1_pullback

end


############## lux definitions
struct Set_tetr_dat_str <: Lux.AbstractLuxLayer
    radiuss::Tuple{Float32,Float32,Float32}
    spacing::Tuple{Float32,Float32,Float32}
    batch_size::Int
    pad_voxels::Int
    image_shape::Tuple{Int,Int,Int}
    is_point_per_triangle::Bool
    first_pass::Bool
end

function Lux.initialparameters(rng::AbstractRNG, l::Set_tetr_dat_str)
    return ()
end
"""
check the optimal launch configuration for the kernel
calculate the number of threads and blocks and how much padding to add if needed
"""
function prepare_for_set_tetr_dat(image_shape, tetr_dat_shape, batch_size)
    # ,control_points_shape,sv_centers_shape)
    # bytes_per_thread=0
    # blocks_apply_w,threads_res,maxBlocks=set_tetr_dat_kern_unrolled(Cuda.zeros(tetr_dat_shape...)
    # , Cuda.zeros(tetr_dat_shape...)
    # , Cuda.zeros(image_shape...), control_points, sv_centers,max_index)
    threads_res = 256
    needed_blocks = ceil(Int, tetr_dat_shape[1] / threads_res)

    return threads_res, (needed_blocks, batch_size)
end
function Lux.initialstates(::AbstractRNG, l::Set_tetr_dat_str)::NamedTuple

    set_of_svs = initialize_centers_and_control_points(l.image_shape, l.radiuss,l.is_point_per_triangle)

    if(!l.is_point_per_triangle)
        sv_centers, control_points, tetrs, dims = set_of_svs
    else
        sv_centers,control_points,tetrs,dims,plan_tensors=set_of_svs
        siz_c=size(control_points)
        siz_plan=size(plan_tensors)
        x_blocks=Int(ceil((siz_c[1]*siz_c[2]*siz_c[3])/len_get_random_point_in_tetrs_kern))
        max_index_for_per_triangle=siz_c[1]*siz_c[2]*siz_c[3]
        blocks_per_triangle=(x_blocks,siz_plan[1],l.batch_size)
    end
    tetr_dat=tetrs
    threads_tetr_set, blocks_tetr_set = prepare_for_set_tetr_dat(l.image_shape, size(tetr_dat), l.batch_size)
    return (first_pass=l.first_pass,batch_size=l.batch_size, spacing=l.spacing, radiuss=l.radiuss, image_shape=l.image_shape, threads_tetr_set=threads_tetr_set, blocks_tetr_set=blocks_tetr_set, tetr_dat=Float32.(tetr_dat), pad_voxels=l.pad_voxels)

end

# function pad_source_arr(source_arr, out_arr, pad_voxels, image_shape)
#     source_arr = source_arr[:, :, :, 1, 1]
#     # p=pad_voxels*2
#     p_beg = pad_voxels + 1
#     # new_arr=zeros(image_shape[1]+p,image_shape[2]+p,image_shape[3]+p)
#     out_arr[p_beg:image_shape[1]+pad_voxels, p_beg:image_shape[2]+pad_voxels, p_beg:image_shape[3]+pad_voxels] = source_arr
#     return out_arr
# end





# function ChainRulesCore.rrule(::typeof(pad_source_arr), source_arr, out_arr, pad_voxels, image_shape)

#     #we get here correct tetr dat out by mutation
#     out_arr = pad_source_arr(source_arr, out_arr, pad_voxels, image_shape)


#     function call_test_kernel1_pullback(d_out_arr)
#         d_out_arr = CuArray(collect(d_out_arr))
#         d_source_arr = CUDA.zeros(size(source_arr)...)

#         Enzyme.autodiff(Enzyme.Reverse, f, Duplicated(source_arr, d_source_arr), Duplicated(out_arr, d_out_arr))

#         return NoTangent(), d_source_arr, d_out_arr, NoTangent(), NoTangent()
#     end

#     return out_arr, call_test_kernel1_pullback

# end


# function add_tetr(tetr_dat_out, out_arr, pad_voxels)

#     out_arr[:, :, 1:3] = (tetr_dat_out[:, :, 1:3] .+ pad_voxels)
#     out_arr[:, :, 4] = tetr_dat_out[:, :, 4]
#     return out_arr
# end


# function ChainRulesCore.rrule(::typeof(add_tetr), tetr_dat_out, out_arr, pad_voxels)

#     #we get here correct tetr dat out by mutation
#     out_arr = add_tetr(tetr_dat_out, out_arr, pad_voxels)


#     function call_test_kernel1_pullback(d_out_arr)
#         d_out_arr = CuArray(collect(d_out_arr))
#         d_tetr_dat_out = CUDA.zeros(size(tetr_dat_out)...)

#         Enzyme.autodiff(Enzyme.Reverse, f, Duplicated(tetr_dat_out, d_tetr_dat_out), Duplicated(out_arr, d_out_arr))

#         return NoTangent(), d_tetr_dat_out, d_out_arr, NoTangent()
#     end

#     return out_arr, call_test_kernel1_pullback

# end



function (l::Set_tetr_dat_str)(x, ps, st::NamedTuple)

    # if(st.first_pass)
    cp, source_arr = x
    source_arr=CuArray(source_arr)
    sv_centers_out, control_points_out = cp
    # print(control_points_out)
    tetr_dat_out = call_set_tetr_dat_kern(st.tetr_dat, source_arr, control_points_out, sv_centers_out, st.threads_tetr_set, st.blocks_tetr_set, st.spacing, st.batch_size)
    
    return (tetr_dat_out, source_arr,cp), st
    # else
    #     b_loss, source_arr,cp = x
    #     source_arr=CuArray(source_arr)
    #     sv_centers_out, control_points_out = cp
    #     # print(control_points_out)
    #     tetr_dat_out = call_set_tetr_dat_kern(st.tetr_dat, source_arr, control_points_out, sv_centers_out, st.threads_tetr_set, st.blocks_tetr_set, st.spacing, st.batch_size)
    #     return (tetr_dat_out, source_arr,b_loss), st
    
    # end

end
