using ChainRulesCore, Zygote, CUDA, Enzyme
using KernelAbstractions
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote
using LinearAlgebra
using Revise
using cuTENSOR, TensorOperations


includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights_kerns/get_weights.jl")
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

    first_dense_out::Int
    second_dense_out::Int
    third_dense_out::Int
    primary_sv_repr::Int
    final_sv_repr::Int
end

"""
initialize parameters for GetWeightsFromDirections
"""
function Lux.initialparameters(rng::AbstractRNG, l::GetWeightsFromDirections_str)::NamedTuple

    num_channels = l.num_channels
    num_params_exec = l.num_params_exec

    directions_indicies, voxel_counts = get_directions_info(l.radiuss)
    voxel_counts = Int32.(voxel_counts)
    ds = size(directions_indicies)
    padded_dim = Int(ceil(ds[2] / num_indicies_per_block) * num_indicies_per_block)
    #parameters used for parametrarized aggregation of information from points in single direction
    param_matrix_a = rand(rng, Float32, num_params_exec, num_channels, num_directions, padded_dim + 1, num_shared_repr, num_per_shared_mem_spot)
    #parameters for convolutional kernels diffrent for each channel each parameter set and each direction
    conv_kernels = rand(rng, Float32, 3, 3, 3, num_channels, num_params_exec, num_directions)

    # scale=rand(rng,Float32,7)
    # shift=rand(rng,Float32,7)
    # P1=rand(rng,Float32,num_directions,l.first_dense_out)
    # P2=rand(rng,Float32,num_shared_repr*l.num_params_exec,l.second_dense_out) # (n,second_dense_out)
    # P3=rand(rng,Float32,l.third_dense_out,l.first_dense_out)
    # P3a=rand(rng,Float32,l.second_dense_out,l.third_dense_out,num_params_exec,num_directions)

    # P4=rand(rng,Float32,num_directions,l.primary_sv_repr,l.num_channels)#[d,e,c]  #- (second_dense_out,first_dense_out)
    # P5=rand(rng,Float32,num_params_exec,l.primary_sv_repr,l.final_sv_repr )




    return (param_matrix_a=param_matrix_a, conv_kernels=conv_kernels)
    # ,scale=scale,shift=shift,P1=P1,P2=P2,P3=P3,P4=P4,P5=P5,P3a=P3a)
end


"""
initialize state variables for GetWeightsFromDirections
"""
function get_state_variables_for_GetWeightsFromDirections(radiuss, num_channels, image_shape, num_params_exec)

    set_of_svs = initialize_centers_and_control_points(l.image_shape, l.radiuss,is_point_per_triangle)
    if(!is_point_per_triangle)
        sv_centers, control_points, tetrs, dims = set_of_svs
    else
        sv_centers,control_points,tetrs,dims,plan_tensors=set_of_svs
    end    

    #TODO calculate max_shared_get_weights through occupancy API 
    # blocks_apply_c, threads_apply_c, maxBlocks = computeBlocksFromOccupancy(apply_weights_to_locs_kern
    # , (CUDA.ones(control_points_shape...), CUDA.ones(control_points_shape...), CUDA.ones(weights_shape...), Float32(3), UInt32(control_points_shape[1]), UInt32(control_points_shape[2]), UInt32(control_points_shape[3])
    #     ), bytes_per_thread)


    flat_sv_centers = reshape(sv_centers, (size(sv_centers, 1) * size(sv_centers, 2) * size(sv_centers, 3), size(sv_centers, 4)))
    #direction indicies help in iteration in a cone that starts at sv center and go in a given direction
    directions_indicies, voxel_counts = get_directions_info(radiuss)
    voxel_counts = Int32.(voxel_counts)
    #we need to pad directions_indicies to avoid using if in kernel
    ds = size(directions_indicies)
    padded_dim = Int(ceil(ds[2] / num_indicies_per_block) * num_indicies_per_block)
    padded_ds = zeros(Int64, ds[1], padded_dim, ds[3])
    padded_ds[:, 1:ds[2], :] = directions_indicies
    directions_indicies = padded_ds
    directions_indicies = Int32.(directions_indicies)
    max_len_dir = size(directions_indicies, 2)


    #constants used to pad the image to avoid if in kernel
    unique_directions = sort(unique(directions_indicies))
    min_direction = unique_directions[2]
    max_direction = unique_directions[end]
    min_sv_centers = minimum(flat_sv_centers)
    max_sv_centers = maximum(flat_sv_centers)

    beg_axis_pad = (Int(ceil(abs(min_sv_centers + min_direction))) + 1) + 2
    end_axis_pad = (Int(ceil(abs(maximum(collect(image_shape)) - (max_sv_centers + max_direction)))) + 1) + 2
    pad_size = beg_axis_pad + end_axis_pad


    times_exec_per_dir_per_param = ceil(max_len_dir / num_indicies_per_block)
    threads_y = Int(ceil(times_exec_per_dir_per_param))
    threads_x = Int(ceil(max_shared_get_weights / threads_y))
    max_index = size(flat_sv_centers, 1)
    threads = (threads_x, threads_y)

    true_blocks_x = Int(ceil(max_index / threads[1]))
    blocks = (true_blocks_x, (num_directions * num_params_exec), batch_size * num_channels)

    return threads, blocks, flat_sv_centers, directions_indicies, voxel_counts, max_len_dir, pad_size, beg_axis_pad, end_axis_pad, max_index, true_blocks_x, num_indicies_per_block

end


function debug_reshape(tensor)
    required_size = 2 * 4 * 5832 * 8 * 14

    # Flatten the tensor and pad with zeros if necessary
    flattened_tensor = reshape(tensor, :)
    # sizzz=required_size - size(flattened_tensor)[1]
    # print("\n flattened_tensor $(size(flattened_tensor)) sizzz $(sizzz)\n")

    # padded_tensor = vcat(flattened_tensor, CuArray(zeros(Float32, (sizzz,))))

    # Step 3: Reshape the tensor to the desired shape
    reshaped_tensor = reshape(flattened_tensor[1:required_size], 2, 4, 5832, 8, 14)
    return reshaped_tensor
end


function call_get_weights_from_directions(batch_size, num_channels, flat_sv_centers, num_params_exec, num_directions, threads, blocks, curr_image, directions_indicies, param_matrix_a, conv_kernels, max_index, beg_axis_pad, num_indicies_per_block, voxel_counts)

    # out_summarised = debug_reshape(curr_image)
    # print("\n out_summarised $(size(out_summarised)) curr_image $(size(curr_image))  \n")
    # return out_summarised




    out_summarised = CuArray(zeros(Float32, batch_size, num_channels, size(flat_sv_centers, 1), num_params_exec, num_directions))
    # print("\n out_summarised $(size(out_summarised)) curr_image $(size(curr_image))  \n")

    @cuda threads = threads blocks = blocks get_weights_from_directions(
        flat_sv_centers,
        curr_image,
        directions_indicies,
        param_matrix_a,
        conv_kernels, num_channels,
        out_summarised, max_index, beg_axis_pad, num_indicies_per_block, num_directions, voxel_counts)



    return out_summarised

end



function ChainRulesCore.rrule(::typeof(call_get_weights_from_directions), batch_size, num_channels, flat_sv_centers, num_params_exec, num_directions, threads, blocks, curr_image, directions_indicies, param_matrix_a, conv_kernels, max_index, beg_axis_pad, num_indicies_per_block, voxel_counts)

    out_summarised = call_get_weights_from_directions(batch_size, num_channels, flat_sv_centers, num_params_exec, num_directions, threads, blocks, curr_image, directions_indicies, param_matrix_a, conv_kernels, max_index, beg_axis_pad, num_indicies_per_block, voxel_counts)

    function pullback(d_out_summarised)
        d_out_summarised = CuArray(Zygote.unthunk(d_out_summarised))
        d_flat_sv_centers = CUDA.zeros(Float32, size(flat_sv_centers)...)
        d_source_arr = CUDA.zeros(Float32, size(curr_image)...)
        d_directions_indicies = CUDA.zeros(Int32, size(directions_indicies)...)
        d_param_matrix_a = CUDA.zeros(Float32, size(param_matrix_a)...)
        d_conv_kernels = CUDA.zeros(Float32, size(conv_kernels)...)
        d_voxel_counts = CUDA.zeros(Int32, size(voxel_counts)...)

        @cuda threads = threads blocks = blocks get_weights_from_directions_deff(flat_sv_centers, d_flat_sv_centers,
            curr_image, d_source_arr,
            directions_indicies, d_directions_indicies,
            param_matrix_a, d_param_matrix_a,
            conv_kernels, d_conv_kernels, num_channels,
            out_summarised, d_out_summarised, max_index, beg_axis_pad, num_indicies_per_block, num_directions, voxel_counts, d_voxel_counts)



        return (NoTangent(), NoTangent(), NoTangent(), d_flat_sv_centers, NoTangent(), NoTangent(), NoTangent(), NoTangent(), d_source_arr, d_directions_indicies, d_param_matrix_a, d_conv_kernels, NoTangent(), NoTangent(), NoTangent(), d_voxel_counts)

    end

    return out_summarised, pullback
end



function Lux.initialstates(::AbstractRNG, l::GetWeightsFromDirections_str)::NamedTuple
    radiuss = l.radiuss
    num_channels = l.num_channels
    image_shape = l.image_shape
    num_params_exec = l.num_params_exec

    threads, blocks, flat_sv_centers, directions_indicies, voxel_counts, max_len_dir, pad_size, beg_axis_pad, end_axis_pad, max_index, true_blocks_x, num_indicies_per_block = get_state_variables_for_GetWeightsFromDirections(radiuss, num_channels, image_shape, num_params_exec)


    num_channels = l.num_channels
    num_params_exec = l.num_params_exec

    directions_indicies, voxel_counts = get_directions_info(l.radiuss)
    ds = size(directions_indicies)
    padded_dim = Int(ceil(ds[2] / num_indicies_per_block) * num_indicies_per_block)
    padded_ds = zeros(Int64, ds[1], padded_dim, ds[3])
    padded_ds[:, 1:ds[2], :] = directions_indicies
    directions_indicies = padded_ds
    directions_indicies = Int32.(directions_indicies)
    #parameters used for parametrarized aggregation of information from points in single direction
    param_matrix_a = rand(rng, Float32, num_params_exec, num_channels, num_directions, padded_dim + 1, num_shared_repr, num_per_shared_mem_spot)



    # Create a mask to set the appropriate entries to 0
    ps_mask = trues(size(param_matrix_a))

    for i in eachindex(Array(voxel_counts))
        ps_mask[:, :, i, (voxel_counts[i]+1):end, :, :] .= false
    end
    ps_mask = Float32.(ps_mask)

    return (ps_mask=ps_mask, image_shape=image_shape, num_params_exec=num_params_exec, num_channels=num_channels, num_indicies_per_block=num_indicies_per_block, true_blocks_x=true_blocks_x, max_index=max_index, end_axis_pad=end_axis_pad, beg_axis_pad=beg_axis_pad, pad_size=pad_size, max_len_dir=max_len_dir, voxel_counts=voxel_counts, directions_indicies=directions_indicies, flat_sv_centers=flat_sv_centers, blocks=blocks, threads=threads, batch_size=l.batch_size)
end


"""
    initial_mixing(out_summarised, P1, P2, P3, P4, P5, scale, shift)

Perform initial mixing of supervoxel representations through a series of dense layers, activations, and layer normalizations.

# Arguments
- `out_summarised`: Input tensor of shape (batch_size, num_channels, flat_sv_len, n, num_directions).
- `P1`: Weight tensor for the first dense layer.
- `P2`: Weight tensor for the second dense layer.
- `P3`: Weight tensor for the third dense layer.
- `P4`: Weight tensor for the fourth dense layer.
- `P5`: Weight tensor for the fifth dense layer.
- `scale`: Scaling factors for layer normalization.
- `shift`: Shifting factors for layer normalization.

# Returns
- `output`: Output tensor after mixing and normalization.
"""
# function initial_mixing(out_summarised, P1, P2, P3,P3a, P4, P5, scale, shift,output)
function call_initial_mixing(out_summarised, P1, P2, P3, P3a, P4, P5, scale, shift, batch_size, flat_sv_len, final_sv_repr)

    # scale=Array(scale)
    # shift=Array(shift)

    @tensor res[b, c, f, n, e] := out_summarised[b, c, f, n, d] * P1[d, e]
    # res=layer_norm_simpl(res,(4,5),1.0,0.0)
    res = swish.(res)#activation
    @tensor res[b, c, f, e, d] := res[b, c, f, n, d] * P2[n, e]
    # res=layer_norm_simpl(res,(4,5),1.0,0.0)
    res = swish.(res)#activation   
    @tensor res[b, c, f, n, e] := res[b, c, f, n, d] * P3[e, d]
    res = swish.(res)#activation
    @tensor res[b, c, f, n, d] := res[b, c, f, n1, d1] * P3a[n1, d1, n, d]
    res = res + out_summarised#residiual connection
    @tensor res[b, f, n, e] := res[b, c, f, n, d] * P4[d, e, c]
    res = swish.(res)#activation
    @tensor res[b, f, e] := res[b, f, n, d] * P5[n, d, e]
    res = swish.(res)#activation
    return res
    # out_summarised=res[:,1,:,:,:]
    # out_summarised=reshape(out_summarised, (batch_size,flat_sv_len, 8*17))
    # out_summarised=out_summarised[:,:,1:final_sv_repr]

    # #out_summarised (2, 4, 5832, 8, 14)   res (2, 5832, 64)
    # return out_summarised
    # output=output+res
    #Dense out_summarised (batch_size, num_channels, flat_sv_len, n, num_directions)
    # @tensor res[b,c,f,n,e] := out_summarised[b,c,f,n,d]*P1[d,e]   
    # res=swish.(res)#activation

    # #normalization
    # res=layer_norm_simpl(res,(4,5),1.0,0.0)

    # #Dense input (batch_size, num_channels, flat_sv_len, n, first_dense_out)
    # @tensor res[b,c,f,e,d] := res[b,c,f,n,d]*P2[n,e]   
    # res=swish.(res)#activation
    # # CUDA.@allowscalar res=layer_norm_simpl(res,(4,5),scale[2],shift[2])

    # #Dense input (batch_size, num_channels, flat_sv_len, second_dense_out, first_dense_out)
    # @tensor res[b,c,f,n,e] := res[b,c,f,n,d]*P3[e,d]
    # res=swish.(res)#activation
    # # CUDA.@allowscalar res=layer_norm_simpl(res,(4,5),scale[3],shift[3])


    # #Dense input (batch_size, num_channels, flat_sv_len, second_dense_out, third_dense_out)
    # @tensor res[b,c,f,n,d] := res[b,c,f,n1,d1]*P3a[n1,d1,n,d]
    # res=swish.(res)#activation
    # res=res+out_summarised#residiual connection


    # #Dense input (batch_size, num_channels, flat_sv_len, n, num_directions)
    # #Dense contracting channels
    # @tensor res[b,f,n,e] := res[b,c,f,n,d]*P4[d,e,c]
    # res=swish.(res)#activation
    # # CUDA.@allowscalar res=layer_norm_simpl(res,(3,4),scale[4],shift[4])

    # #Dense input (batch_size, flat_sv_len, second_dense_out, primary_sv_repr)
    # @tensor res[b,f,e] := res[b,f,n,d]*P5[n,d,e]
    # res=swish.(res)#activation
    # # CUDA.@allowscalar res=layer_norm_simpl(res,(3),scale[5],shift[5])    


    # return res
end



function call_pad_image(beg_axis_pad, end_axis_pad, source_arr, num_channels)
    padded_image = cat(
        CuArray(zeros(Float32, beg_axis_pad, size(source_arr, 2), size(source_arr, 3), num_channels, batch_size)),
        source_arr,
        CuArray(zeros(Float32, end_axis_pad, size(source_arr, 2), size(source_arr, 3), num_channels, batch_size));
        dims=1
    )

    padded_image = cat(
        CuArray(zeros(Float32, size(padded_image, 1), beg_axis_pad, size(padded_image, 3), num_channels, batch_size)),
        padded_image,
        CuArray(zeros(Float32, size(padded_image, 1), end_axis_pad, size(padded_image, 3), num_channels, batch_size));
        dims=2
    )

    padded_image = cat(
        CuArray(zeros(Float32, size(padded_image, 1), size(padded_image, 2), beg_axis_pad, num_channels, batch_size)),
        padded_image,
        CuArray(zeros(Float32, size(padded_image, 1), size(padded_image, 2), end_axis_pad, num_channels, batch_size));
        dims=3
    )
    return padded_image
end





function (l::GetWeightsFromDirections_str)(source_arr, ps, st::NamedTuple)


    image_shape = st.image_shape
    pad_size = st.pad_size
    beg_axis_pad = st.beg_axis_pad
    end_axis_pad = st.end_axis_pad
    voxel_counts = Int32.(st.voxel_counts)
    num_indicies_per_block = st.num_indicies_per_block
    batch_size = st.batch_size
    num_channels = st.num_channels
    flat_sv_len = size(st.flat_sv_centers, 1)
    num_params_exec = st.num_params_exec
    # curr_image = CuArray(ones(Float32, image_shape[1] + pad_size, image_shape[2] + pad_size, image_shape[3] + pad_size, num_channels, batch_size))
    curr_image = call_pad_image(beg_axis_pad, end_axis_pad, source_arr, num_channels)
    # curr_image[beg_axis_pad:end-(end_axis_pad+1), beg_axis_pad:end-(end_axis_pad+1), beg_axis_pad:end-(end_axis_pad+1), :, :] = source_arr


    # param_matrix_a=call_zero_params(ps.param_matrix_a,voxel_counts)

    param_matrix_a = ps.param_matrix_a .* (st.ps_mask)


    out_summarised = call_get_weights_from_directions(st.batch_size, st.num_channels, st.flat_sv_centers, st.num_params_exec, num_directions, st.threads, st.blocks, curr_image, st.directions_indicies, param_matrix_a, ps.conv_kernels, st.max_index, st.beg_axis_pad, num_indicies_per_block, voxel_counts)

    #batch_size, num_channels, flat_sv_len, num_shared_repr*num_params_exec, num_directions)
    # out_summarised= call_initial_mixing(out_summarised, ps.P1, ps.P2, ps.P3,ps.P3a,ps.P4, ps.P5, ps.scale, ps.shift, batch_size,flat_sv_len,final_sv_repr)


    return out_summarised, st
    # return (sv_centers_out, weights), st
end




