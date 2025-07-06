using ChainRulesCore, Zygote, CUDA, Enzyme
using KernelAbstractions
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote
using LinearAlgebra
using Revise
using cuTENSOR, TensorOperations


# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights_kerns/get_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights_kerns/get_weights_heavy.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")

const global num_directions = 14
const global num_indicies_per_block = 7
const global num_shared_repr = 1
const global num_per_shared_mem_spot = 3
const global max_shared_get_weights = 200

"""
num_params_exec is telling about how many parameter sets we want to use in one execution
"""
struct GetWeightsFromDirections_str <: Lux.Lux.AbstractLuxLayer
    radiuss::Tuple{Float32,Float32,Float32}
    spacing::Tuple{Float32,Float32,Float32}
    batch_size::Int
    num_channels::Int
    num_params_exec::Int
    image_shape::Tuple{Int,Int,Int}
    is_to_mix_weights::Bool
    is_point_per_triangle::Bool
end

"""
initialize parameters for GetWeightsFromDirections
"""
function Lux.initialparameters(rng::AbstractRNG, l::GetWeightsFromDirections_str)::NamedTuple

    threads_mix_weights_kern = (10, 5, 5)
    threads_conv_weights = (10, 5, 5)

    param_set_length = l.num_params_exec

    @assert param_set_length % 4 == 0 "param_set_length must be divisible by 4"
    param_set_length_reduced = Int(param_set_length / 2)
    param_set_length_reduced_conv = Int(param_set_length / 2)

    num_channels = l.num_channels

    conv_kernels = glorot_normal(rng, Float32, 3, 3, 3, threads_conv_weights[1], threads_conv_weights[2], threads_conv_weights[3], num_channels, param_set_length_reduced_conv)  # Example convolution kernels

    param_mixing = glorot_normal(rng, Float32, 6, threads_mix_weights_kern[1], threads_mix_weights_kern[2], threads_mix_weights_kern[3], threads_mix_weights_kern[1], threads_mix_weights_kern[2], threads_mix_weights_kern[3], num_channels, param_set_length_reduced)
    param_reducing = glorot_normal(rng, Float32, 12, 1, 1, 1, threads_mix_weights_kern[1], threads_mix_weights_kern[2], threads_mix_weights_kern[3], num_channels, param_set_length_reduced)
    param_reducing_b = glorot_normal(rng, Float32, 12, 1, 1, 1, threads_mix_weights_kern[1], threads_mix_weights_kern[2] - 1, threads_mix_weights_kern[3], num_channels, param_set_length_reduced)
    param_reducing_c = glorot_normal(rng, Float32, 12, 1, 1, 1, threads_mix_weights_kern[1], threads_mix_weights_kern[2], threads_mix_weights_kern[3] - 1, num_channels, param_set_length_reduced)

    return (conv_kernels=conv_kernels, param_mixing=param_mixing, param_reducing=param_reducing, param_reducing_b=param_reducing_b, param_reducing_c=param_reducing_c)
end

# get block dim on x increased from 5 to 10
# get assertion that parma length is divisible by 2
# increase param mixing x dim to 10
# but decrease the param mixing dimension two times



function Lux.initialstates(::AbstractRNG, l::GetWeightsFromDirections_str)::NamedTuple
    radiuss = l.radiuss
    num_channels = l.num_channels
    image_shape = l.image_shape
    num_params_exec = l.num_params_exec
    param_set_length = l.num_params_exec
    is_to_mix_weights = l.is_to_mix_weights
    set_of_svs = initialize_centers_and_control_points(l.image_shape, l.radiuss, l.is_point_per_triangle)
    if (!l.is_point_per_triangle)
        sv_centers, control_points, tetrs, dims = set_of_svs
    else
        sv_centers, control_points, tetrs, dims, plan_tensors = set_of_svs
    end
    dims = size(sv_centers)
    threads_mix_weights_kern = (10, 5, 5)
    threads_conv_weights = (10, 5, 5)
    param_set_length_reduced = Int(param_set_length / 2)
    param_set_length_reduced_conv = Int(param_set_length / 2)

    flat_sv_centers = reshape(sv_centers, (size(sv_centers, 1) * size(sv_centers, 2) * size(sv_centers, 3), size(sv_centers, 4)))
    flat_sv_centers = round.(UInt32, flat_sv_centers)

    num_channels = l.num_channels
    num_params_exec = l.num_params_exec

    beg_axis_pad = 8
    blocks = (size(flat_sv_centers, 1), num_channels * param_set_length_reduced, Int(l.batch_size))
    blocks_conv = (size(flat_sv_centers, 1), num_channels * param_set_length_reduced_conv, Int(l.batch_size))
    num_blocks_y_true = num_channels
    l_relu_weight = 0.000001
    return (flat_sv_len=size(flat_sv_centers, 1), param_set_length=param_set_length,
        blocks_conv=blocks_conv, param_set_length_reduced_conv=param_set_length_reduced_conv, param_set_length_reduced=param_set_length_reduced, threads_conv_weights=threads_conv_weights, threads_mix_weights_kern=threads_mix_weights_kern, is_to_mix_weights=is_to_mix_weights, image_shape=l.image_shape, l_relu_weight=l_relu_weight, dims=dims, num_params_exec=num_params_exec, num_channels=num_channels, num_blocks_y_true=num_blocks_y_true, flat_sv_centers=flat_sv_centers, blocks=blocks, batch_size=l.batch_size, beg_axis_pad=beg_axis_pad)
end

# """
# call kernel to apply convolution on the image around the superVoxel centers
# """
# function call_kernel_analysis_conv(source_arr, flat_sv_centers,  beg_axis_pad, conv_kernels
#     ,num_blocks_y_true,param_set_length_reduced_conv,param_set_length, num_channels, batch_size,threads_conv_weights,blocks_conv,flat_sv_len)

#     conved=cast_dev_zeros(Float32, flat_sv_len  ,5,threads_conv_weights[2],threads_conv_weights[3], param_set_length, num_channels, batch_size)

#     @cuda threads=threads_conv_weights blocks=blocks_conv kernel_analysis_conv(source_arr, flat_sv_centers, conved, beg_axis_pad, conv_kernels
#     ,num_blocks_y_true,param_set_length_reduced_conv)

#     c=sum(conved)
#     return conved
# end


# function kernel_analysis_conv_deff(
#     source_arr,d_source_arr
#     , flat_sv_centers,d_flat_sv_centers
#     , conved,d_conved
#     , beg_axis_pad
#     , conv_kernels,d_conv_kernels
#     ,num_blocks_y_true
#     ,param_set_length_reduced_conv

# )


#     Enzyme.autodiff_deferred(
#             Enzyme.Reverse
#             ,Enzyme.Const(kernel_analysis_conv), Enzyme.Const
#             ,Enzyme.Duplicated(source_arr, d_source_arr)
#             ,Enzyme.Duplicated(flat_sv_centers, d_flat_sv_centers)
#             ,Enzyme.Duplicated(conved, d_conved)
#             ,Enzyme.Const(beg_axis_pad)
#             ,Enzyme.Duplicated(conv_kernels, d_conv_kernels)
#             ,Enzyme.Const(num_blocks_y_true)
#             ,Enzyme.Const(param_set_length_reduced_conv)
#             )
#     return nothing
# end





"""
after we initially prepared the image by performing convolution on it we can now apply the mixing and reducing kernel
"""
function call_get_weights_from_directions(source_arr, flat_sv_centers, conv_kernels, param_mixing, param_reducing, param_reducing_b
    , param_reducing_c, num_blocks_y_true, dims, half_num_params, beg_axis_pad, threads_mix_weights_kern, blocks, param_set_length)


    batch_size = size(source_arr)[end]
    result = CuArray(zeros(Float32, dims[1], dims[2], dims[3], 5, threads_mix_weights_kern[2], param_set_length, num_blocks_y_true, batch_size))

    # Check which GPU device source_arr is on
    # source_arr_device = CUDA.device(source_arr)
    # println("source_arr is on GPU device: $source_arr_device")

    # # Check which GPU device param_mixing is on
    # param_mixing_device = CUDA.device(param_mixing)
    # println("param_mixing is on GPU device: $param_mixing_device")
    # # Check which GPU device flat_sv_centers is on
    # flat_sv_centers_device = CUDA.device(flat_sv_centers)
    # println("flat_sv_centers_device is on GPU device: $flat_sv_centers_device")

    @cuda threads = threads_mix_weights_kern blocks = blocks kernel_analysis(source_arr, flat_sv_centers, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, result, num_blocks_y_true, dims, half_num_params, beg_axis_pad)
    c = sum(result)

    return result

end


function get_weights_from_directions_deff(
    source_arr, d_source_arr, flat_sv_centers, d_flat_sv_centers, conv_kernels, d_conv_kernels, param_mixing, d_param_mixing, param_reducing, d_param_reducing, param_reducing_b, d_param_reducing_b, param_reducing_c, d_param_reducing_c, result, d_result, num_blocks_y_true, dims, half_num_params, beg_axis_pad)

    Enzyme.autodiff_deferred(
        Enzyme.Reverse, Enzyme.Const(kernel_analysis), Enzyme.Const, Enzyme.Duplicated(source_arr, d_source_arr), Enzyme.Duplicated(flat_sv_centers, d_flat_sv_centers), Enzyme.Duplicated(conv_kernels, d_conv_kernels), Enzyme.Duplicated(param_mixing, d_param_mixing), Enzyme.Duplicated(param_reducing, d_param_reducing), Enzyme.Duplicated(param_reducing_b, d_param_reducing_b), Enzyme.Duplicated(param_reducing_c, d_param_reducing_c), Enzyme.Duplicated(result, d_result)
        , Enzyme.Const(num_blocks_y_true), Enzyme.Const(dims), Enzyme.Const(half_num_params), Enzyme.Const(beg_axis_pad))
    return nothing
end

# function ChainRulesCore.rrule(::typeof(call_kernel_analysis_conv),
#     source_arr, flat_sv_centers,  beg_axis_pad, conv_kernels
#     ,num_blocks_y_true,param_set_length_reduced_conv,param_set_length, num_channels, batch_size,threads_conv_weights,blocks_conv,flat_sv_len)


#     conved = call_kernel_analysis_conv(source_arr, flat_sv_centers,  beg_axis_pad, conv_kernels
#     ,num_blocks_y_true,param_set_length_reduced_conv,param_set_length
#     , num_channels, batch_size,threads_conv_weights,blocks_conv,flat_sv_len)

#     function pullback(d_conved)

#         current_time = Dates.now()

#         d_source_arr =       cast_dev_zeros(Float32, size(source_arr)...)
#         d_flat_sv_centers = cast_dev_zeros(UInt32, size(flat_sv_centers)...)
#         d_conved=CuArray(collect(d_conved))
#         d_conv_kernels =     cast_dev_zeros(Float32, size(conv_kernels)...)

#         # count_zeros(d_out_summarised,"d_out_summarised")
#         count_zeros(d_conved,"d_conved")


#         @cuda threads = threads_conv_weights blocks = blocks_conv kernel_analysis_conv_deff(
#             source_arr,d_source_arr
#             , flat_sv_centers,d_flat_sv_centers
#             , conved,d_conved
#             , beg_axis_pad
#             , conv_kernels,d_conv_kernels
#             ,num_blocks_y_true
#             ,param_set_length_reduced_conv

#         )


#         dsss=sum(d_source_arr)
#         seconds_diff = Dates.value(Dates.now() - current_time)/ 1000
#         print("\n get weights  conv  d_source_arr sum $(dsss) ; $(seconds_diff) sec \n")
#         count_zeros(d_source_arr,"d_source_arr")
#         count_zeros(d_conv_kernels,"d_conv_kernels")
#         # @info "get weights  backward  " times=round(seconds_diff; digits = 2)


#         return (NoTangent(),d_source_arr, d_flat_sv_centers,  NoTangent(), d_conv_kernels
#         ,NoTangent(),NoTangent(),NoTangent(), NoTangent()
#         , NoTangent(),NoTangent(),NoTangent(),NoTangent())



#     end

#     return conved, pullback
# end





function ChainRulesCore.rrule(::typeof(call_get_weights_from_directions),
    source_arr, flat_sv_centers, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, num_blocks_y_true
    , dims, half_num_params, beg_axis_pad, threads_mix_weights_kern, blocks, param_set_length)


    out_summarised = call_get_weights_from_directions(source_arr, flat_sv_centers, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, num_blocks_y_true, dims, half_num_params, beg_axis_pad, threads_mix_weights_kern, blocks, param_set_length)

    function pullback(d_out_summarised)

        # current_time = Dates.now()

        d_out_summarised = CuArray(Zygote.unthunk(d_out_summarised))
        d_source_arr = CuArray(zeros(Float32, size(source_arr)))
        d_flat_sv_centers = CuArray(zeros(UInt32, size(flat_sv_centers)))
        d_param_mixing = CuArray(zeros(Float32, size(param_mixing)))
        d_param_reducing = CuArray(zeros(Float32, size(param_reducing)))
        d_param_reducing_b = CuArray(zeros(Float32, size(param_reducing_b)))
        d_param_reducing_c = CuArray(zeros(Float32, size(param_reducing_c)))
        d_conv_kernels = CuArray(zeros(Float32, size(conv_kernels)))
        # count_zeros(d_out_summarised,"d_out_summarised")

        # count_zeros(d_param_mixing,"d_out_summarised")

        @cuda threads = threads_mix_weights_kern blocks = blocks get_weights_from_directions_deff(source_arr, d_source_arr, flat_sv_centers, d_flat_sv_centers, conv_kernels, d_conv_kernels, param_mixing, d_param_mixing, param_reducing, d_param_reducing, param_reducing_b, d_param_reducing_b, param_reducing_c, d_param_reducing_c, out_summarised, d_out_summarised, num_blocks_y_true, dims, half_num_params, beg_axis_pad)


        # dsss=sum(d_out_summarised)
        # seconds_diff = Dates.value(Dates.now() - current_time)/ 1000
        # print("\n get weights  backward d_out_summarised sum $(dsss) ; $(seconds_diff) sec \n")
        # # @info "get weights  backward  " times=round(seconds_diff; digits = 2)
        # # count_zeros(d_conved,"d_conved")
        # # count_zeros(d_flat_sv_centers,"d_flat_sv_centers")
        # count_zeros(d_source_arr,"d_source_arr")
        # count_zeros(d_conv_kernels,"d_conv_kernels")
        # count_zeros(d_param_mixing,"d_param_mixing")
        # count_zeros(d_param_reducing,"d_param_reducing")
        # count_zeros(d_param_reducing_b,"d_param_reducing_b")
        # count_zeros(d_param_reducing_c,"d_param_reducing_c")

        return (NoTangent(), d_source_arr, d_flat_sv_centers, d_conv_kernels, d_param_mixing, d_param_reducing, d_param_reducing_b, d_param_reducing_c, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent())

    end

    return out_summarised, pullback
end




function call_pad_image(beg_axis_pad, end_axis_pad, source_arr, num_channels, batch_size)

    cpu_source_arr = Array(source_arr)

    padded_image = cat(
        zeros(Float32, beg_axis_pad, size(cpu_source_arr, 2), size(cpu_source_arr, 3), num_channels, batch_size),
        cpu_source_arr,
        zeros(Float32, end_axis_pad, size(cpu_source_arr, 2), size(cpu_source_arr, 3), num_channels, batch_size);
        dims=1
    )

    padded_image = cat(
        zeros(Float32, size(padded_image, 1), beg_axis_pad, size(padded_image, 3), num_channels, batch_size),
        padded_image,
        zeros(Float32, size(padded_image, 1), end_axis_pad, size(padded_image, 3), num_channels, batch_size);
        dims=2
    )

    padded_image = cat(
        zeros(Float32, size(padded_image, 1), size(padded_image, 2), beg_axis_pad, num_channels, batch_size),
        padded_image,
        zeros(Float32, size(padded_image, 1), size(padded_image, 2), end_axis_pad, num_channels, batch_size);
        dims=3
    )
    return CuArray(padded_image)
end

function flatten_tensor(tensor)
    flattened_tensor = reshape(tensor, :)
    original_shape = size(tensor)
    return flattened_tensor, original_shape
end

function unflatten_tensor(flattened_tensor, original_shape)
    tensor = reshape(flattened_tensor, original_shape)
    return tensor
end




function (l::GetWeightsFromDirections_str)(source_arr, ps, st::NamedTuple)

    # current_time = Dates.now()

    image_shape = st.image_shape
    beg_axis_pad = st.beg_axis_pad
    batch_size = st.batch_size
    num_channels = st.num_channels
    param_set_length = st.num_params_exec


    # source_arr_device = CUDA.device(source_arr)
    # println("source_arr is on GPU device: $source_arr_device")

    curr_image = call_pad_image(beg_axis_pad, beg_axis_pad, source_arr, num_channels, batch_size)
    # print("curr_image $(size(curr_image)) orig source_arr $(size(source_arr)) num_channels $(num_channels) param_set_length $(param_set_length) st.blocks $(st.blocks) st.num_blocks_y_true $(st.num_blocks_y_true) \n")
    #we are applying tanh and multiply by 100 as otherwise if weights are too big infinities like to pop up

    # param_reducing,param_reducing_s=flatten_tensor(ps.param_reducing)
    # param_reducing_b,param_reducing_b_s=flatten_tensor(ps.param_reducing_b)
    # param_reducing_c,param_reducing_c_s=flatten_tensor(ps.param_reducing_c)
    # param_mixing,param_mixing_s=flatten_tensor(ps.param_mixing)

    # param_reducing = tanh.(param_reducing).*4
    # param_reducing_b = tanh.(param_reducing_b).*4
    # param_reducing_c = tanh.(param_reducing_c).*4
    # param_mixing = tanh.(param_mixing).*4

    # param_reducing=unflatten_tensor(param_reducing,param_reducing_s)
    # param_reducing_b=unflatten_tensor(param_reducing_b,param_reducing_b_s)
    # param_reducing_c=unflatten_tensor(param_reducing_c,param_reducing_c_s)
    # param_mixing=unflatten_tensor(param_mixing,param_mixing_s)

    out_summarised = call_get_weights_from_directions(curr_image, st.flat_sv_centers, ps.conv_kernels, ps.param_mixing, ps.param_reducing, ps.param_reducing_b, ps.param_reducing_c, st.num_blocks_y_true, st.dims, st.param_set_length_reduced, st.beg_axis_pad, st.threads_mix_weights_kern, st.blocks, st.param_set_length)

    out_summarised = reshape(out_summarised, st.dims[1], st.dims[2], st.dims[3], 5 * st.threads_mix_weights_kern[2] * param_set_length * st.num_channels, batch_size)
    # out_summarised=Float32.(out_summarised)



    # print("### $(sum(out_summarised)) ####")
    # out_summarised=tanh.(out_summarised).*0.0001
    # conved=call_kernel_analysis_conv(curr_image, st.flat_sv_centers,  st.beg_axis_pad, ps.conv_kernels
    # ,st.num_blocks_y_true,st.param_set_length_reduced_conv
    # ,st.param_set_length, st.num_channels, st.batch_size,st.threads_conv_weights,st.blocks_conv,st.flat_sv_len)


    # out_summarised = call_get_weights_from_directions(conved, ps.param_mixing
    # , ps.param_reducing,ps.param_reducing_b, ps.param_reducing_c
    # ,st.num_blocks_y_true,st.l_relu_weight,st.dims,st.is_to_mix_weights,st.param_set_length_reduced
    # ,st.threads_mix_weights_kern,st.blocks,st.param_set_length)

    # out_summarised = call_get_weights_from_directions(curr_image, st.flat_sv_centers,  st.beg_axis_pad, 
    # ps.conv_kernels
    # ,ps.param_mixing
    # ,ps.param_reducing
    # ,ps.param_reducing_b
    # ,ps.param_reducing_c
    # ,st.threads_mix_weights_kern,st.blocks, st.num_blocks_y_true,param_set_length, num_channels
    # , batch_size,st.l_relu_weight,st.dims,st.is_to_mix_weights)


    # dsss=sum(out_summarised)
    # seconds_diff = Dates.value(Dates.now() - current_time)/ 1000
    # print("\n get weights  forward out_summarised sum $(dsss) ; $(seconds_diff) sec \n")
    # @info "get weights  forward  " times=round(seconds_diff; digits = 2)


    return out_summarised, st
    # return (sv_centers_out, weights), st
end




