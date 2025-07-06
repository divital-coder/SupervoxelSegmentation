using cuTENSOR, TensorOperations
using Test, Revise, Lux
using PyCall
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
# 

using Logging
using cuTENSOR, TensorOperations


includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights_kerns/get_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/dif_get_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights_kerns/mix_weights_kern.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/util_layers.jl")


"""
we want to mix information between diffrent parameter sets then diffrent directions 
 those things can be done by tensor contractions and application of dense layers

 mixing between diffrent super voxels will be done via custom kernel 
 we will first use dense and contractions to get a representation of the supervoxel to be of length 64 then in kernel
 1) we load the supervoxel representation into shared memory
 2) we load the neigbouring sv representation do the multiplication by parameters and add the bias and nonlinearity and add to the current representation
 3) divide by 2 
 4) apply operation to the neighbours and add to the current representation do it for all neighbours that's distance 
    is less than d , where defoult value of d will be 5

 """



function adding_connection(x, y)
    return x + y
end #connection_before_set_tetr_dat_kern

conv5_a = (in, out) -> Lux.Conv((5, 5, 5), in => out, NNlib.swish, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
conv1_s = (in, out) -> Lux.Conv((1, 1, 1), in => out, NNlib.sigmoid, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)

function get_mix_weights_full_layers(image_shape, radiuss, spacing, batch_size, num_channels, num_params_exec, mixing_param_sets
    , num_weights_for_set_point_weights, is_to_mix_weights,is_point_per_triangle)

    threads = (5, 5, 5)
    param_set_length = num_params_exec

    set_of_svs = initialize_centers_and_control_points(image_shape, radiuss, is_point_per_triangle)
    if (!is_point_per_triangle)
        sv_centers, control_points, tetrs, dims = set_of_svs
    else
        sv_centers, control_points, tetrs, dims, plan_tensors = set_of_svs
    end
    dimss = size(sv_centers)

    #get directions
    get_w_from_dir = GetWeightsFromDirections_str(radiuss, spacing, batch_size, num_channels, num_params_exec, (image_shape[1], image_shape[2], image_shape[3]), is_to_mix_weights,is_point_per_triangle
    )

    return [get_w_from_dir,
    conv5_a(5 * 5 * param_set_length * num_channels, 256 * mixing_param_sets),
    
    Lux.BatchNorm(256 * mixing_param_sets),
    # First skip connection block
    Lux.SkipConnection(
        Lux.Chain(
            conv1_s(256 * mixing_param_sets, 256 * mixing_param_sets),
            Lux.BatchNorm(256 * mixing_param_sets)
        ),
        +  # Add the input to the output
    ),
    # Second skip connection block
    Lux.SkipConnection(
        Lux.Chain(
            conv1_s(256 * mixing_param_sets, 256 * mixing_param_sets),
            Lux.BatchNorm(256 * mixing_param_sets)
        ),
        +
    ),
    # Final layer without skip connection since output dims differ
    conv1_s(256 * mixing_param_sets, num_weights_for_set_point_weights),
]

end




######### Lux definitions

"""
num_params_exec is telling about how many parameter sets we want to use in one execution
"""
struct mix_sv_info_str <: Lux.Lux.AbstractLuxLayer
    radiuss::Tuple{Float32,Float32,Float32}
    image_shape
    mixing_param_sets::Int
    batch_size::Int
    is_point_per_triangle::Bool

end
"""
initialize parameters for mix_sv_info
"""
function Lux.initialparameters(rng::AbstractRNG, l::mix_sv_info_str)::NamedTuple

    mix_params = glorot_normal(rng, Float32, 6, 258, 256, l.mixing_param_sets)
    mix_params_b = glorot_normal(rng, Float32, 6, 258, 256, l.mixing_param_sets)
    mix_params_c = glorot_normal(rng, Float32, 6, 258, 256, l.mixing_param_sets)
    return (mix_params=mix_params, mix_params_b=mix_params_b,mix_params_c=mix_params_c)
end


function call_mix_sv_info(input, threads, blocks, mix_params)

    output = CUDA.zeros(Float32,size(input)...)
    @cuda threads = threads blocks = blocks mix_sv_info(input, mix_params, output)
    return output

end

function mix_sv_info_deff(input, d_input, mix_params, d_mix_params, output, d_output)
    Enzyme.autodiff_deferred(
        Enzyme.Reverse, Enzyme.Const(mix_sv_info), Enzyme.Const, Enzyme.Duplicated(input, d_input), Enzyme.Duplicated(mix_params, d_mix_params), Enzyme.Duplicated(output, d_output)
    )
    return nothing
end

function Lux.initialstates(::AbstractRNG, l::mix_sv_info_str)::NamedTuple
    set_of_svs = initialize_centers_and_control_points(l.image_shape, l.radiuss, l.is_point_per_triangle)
    if (!l.is_point_per_triangle)
        sv_centers, control_points, tetrs, dims = set_of_svs
    else
        sv_centers, control_points, tetrs, dims, plan_tensors = set_of_svs
    end
    threads = (256,)
    cp_x, cp_y, cp_z = size(sv_centers, 1), size(sv_centers, 2), size(sv_centers, 3)
    blocks = (cp_x * cp_y * cp_z, l.mixing_param_sets, l.batch_size)

    return (mixing_param_sets=l.mixing_param_sets, batch_size=l.batch_size, cp_z=cp_z, cp_y=cp_y, cp_x=cp_x, threads=threads, blocks=blocks)
end

"""
some inspiration also from here 
https://arxiv.org/html/2409.15161v1
"""
function (l::mix_sv_info_str)(Res, ps, st::NamedTuple)
    Res = reshape(Res, st.cp_x * st.cp_y * st.cp_z, st.mixing_param_sets, 256, st.batch_size)
    
    Res_a = call_mix_sv_info(Res, st.threads, st.blocks, ps.mix_params)
    
    #TODO establish weather to use the mix_params_b or mix_params_c
    # #mix in order to be able to mix between diffrent parameter sets also 
    # Res = reshape(Res_a, st.cp_x * st.cp_y * st.cp_z, st.mixing_param_sets, 64, 4, st.batch_size)
    # Res = permutedims(Res, [1, 2, 4, 3, 5])
    # Res = reshape(Res, st.cp_x * st.cp_y * st.cp_z, st.mixing_param_sets, 256, st.batch_size)
    # #mixed info do parameterized mixed one more time 
    # Res = call_mix_sv_info(Res, st.threads, st.blocks, ps.mix_params_b)
    # #mixed info do parameterized mixed one more time
    # Res = reshape(Res_a, st.cp_x * st.cp_y * st.cp_z, st.mixing_param_sets, 128, 2, st.batch_size)
    # Res = permutedims(Res, [1, 2, 4, 3, 5])
    # Res = reshape(Res, st.cp_x * st.cp_y * st.cp_z, st.mixing_param_sets, 256, st.batch_size)
    # #mixed info do parameterized mixed one more time 
    # Res = call_mix_sv_info(Res, st.threads, st.blocks, ps.mix_params_c)



    Res=Res+swish.(Res_a) #gated residiual connection
    Res = reshape(Res, st.cp_x, st.cp_y, st.cp_z, 256 * st.mixing_param_sets, st.batch_size)

    return Res, st
    # return (sv_centers_out, weights), st
end


function ChainRulesCore.rrule(::typeof(call_mix_sv_info), input, threads, blocks, mix_params)

    output = call_mix_sv_info(input, threads, blocks, mix_params)

    function pullback(d_output)
        d_output = CuArray(Zygote.unthunk(d_output))
        d_mix_params = CuArray(zeros(Float32, size(mix_params)...))
        d_input = CuArray(zeros(Float32, size(input)...))


        @cuda threads = threads blocks = blocks mix_sv_info_deff(input, d_input, mix_params, d_mix_params, output, d_output)





        return (NoTangent(), d_input, NoTangent(), NoTangent(), d_mix_params , NoTangent())

    end

    return output, pullback
end

