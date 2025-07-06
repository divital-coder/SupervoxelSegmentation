# """
# we have basic grid of points that will constitute the centers of supervoxels - this grid is constant
# basic control points will be also on lines between each of the sv_centers so we will have 3*sv_centers amount of control points
# plus the additional layer in each axis 

# next is a control point in oblique direction that is common for 6 neighbouting sv_centers where those sv centers create a cube
#     so in order to get any point in a cube we need to move on the 3 of the edges of the cube what will give us x y and z coordinates
#     for oblique control points we will just need to get a line between sv center bolow and above for x coordinate and so on for y and z

# important is to always be able to draw a line between supervoxel center and its control point without leaving supervoxel volume - as the later sampling will depend on it
#     we want to keep this star shaped polyhedra ... 

# """
using Pkg
using ChainRulesCore, Zygote, CUDA, Enzyme
# using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote

using LinearAlgebra
using Revise


"""
we want to use a list of weights to move sv centers no more then 0/75 times radius in some direction
weights should be between 0 and 1 
"""
function apply_weights_sv(sv_centers, sv_centers_out, weights, radius::Tuple{Float32,Float32,Float32}, cp_x::UInt32, cp_y::UInt32, cp_z::UInt32, num_blocks_z_pure, num_blocks_y_pure)

    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))
    # y = (threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y()))
    y = ((threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) % cp_y) + 1
    z = ((threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) % cp_z) + 1

    if (x <= (cp_x) && y <= (cp_y) && z <= (cp_z) && x > 0 && y > 0 && z > 0)
        #as we move weights so they will be from -.5 to .5 so we need to multiply them by 1.5*radius as we want it to 
        #to move in range -0.75radius to 0.75radius

        sv_centers_out[x, y, z, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))] = sv_centers[x, y, z, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure))] + (weights[x, y, z, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))] - 0.5) * radius[Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure))] * 1.5
    end
    return nothing

end #apply_weights_sv

############################# Enzyme differentiation


function apply_weights_sv_deff(sv_centers, d_sv_centers, sv_centers_out, d_sv_centers_out, weights, d_weights, radius::Tuple{Float32,Float32,Float32}, cp_x::UInt32, cp_y::UInt32, cp_z::UInt32, num_blocks_z_pure_sv, num_blocks_y_pure_sv)

    Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(apply_weights_sv), Const, Duplicated(sv_centers, d_sv_centers), Duplicated(sv_centers_out, d_sv_centers_out), Duplicated(weights, d_weights), Const(radius), Const(cp_x), Const(cp_y), Const(cp_z), Const(num_blocks_z_pure_sv), Const(num_blocks_y_pure_sv)
    )
    return nothing
end


function call_apply_weights_sv(sv_centers, weights, radius, threads, blocks, num_blocks_z_pure_sv, batch_size, num_blocks_y_pure_sv,zero_der_beg)
    sv_centers_out = Float64.(copy(sv_centers))
    sv_centers_out = repeat(sv_centers_out, inner=(1, 1, 1, 1, batch_size))
    @cuda threads = threads blocks = blocks apply_weights_sv(sv_centers, sv_centers_out, weights, radius, UInt32(size(sv_centers)[1]), UInt32(size(sv_centers)[2]), UInt32(size(sv_centers)[3]), num_blocks_z_pure_sv, num_blocks_y_pure_sv)
    return sv_centers_out
end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_apply_weights_sv), sv_centers, weights, radius, threads, blocks, num_blocks_z_pure_sv, batch_size, num_blocks_y_pure_sv,zero_der_beg)

    sv_centers_out = call_apply_weights_sv(sv_centers, weights, radius, threads, blocks, num_blocks_z_pure_sv, batch_size, num_blocks_y_pure_sv,zero_der_beg)

    function kernel1_pullback(d_sv_centers_out)
        # current_time = Dates.now()

        d_weights = CUDA.zeros(Float64,size(weights)...)
        sizz = size(sv_centers)
        d_sv_centers = CUDA.zeros(Float64,sizz...)
        sv_centers=Float64.(sv_centers)
        
        #@device_code_warntype
        @cuda threads = threads blocks = blocks apply_weights_sv_deff(sv_centers, d_sv_centers, sv_centers_out,
        CuArray(Zygote.unthunk(d_sv_centers_out)), weights, d_weights, radius, UInt32(sizz[1]), UInt32(sizz[2]), UInt32(sizz[3]), num_blocks_z_pure_sv, num_blocks_y_pure_sv)

        # dsss=sum(d_weights)
        # seconds_diff = Dates.value(Dates.now() - current_time)/ 1000
        # print("\n sv points kern backward d_weights sum $(dsss) ; $(seconds_diff) sec \n")
        # @info "sv points kern backward  " times=round(seconds_diff; digits = 2)


        # return NoTangent(),d_control_points_out,d_control_points_out_in,d_weights,NoTangent(),NoTangent(),NoTangent()
        
        if(zero_der_beg)
            return NoTangent(), d_sv_centers,  NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()
        end
        return NoTangent(), d_sv_centers, d_weights, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent()

    end


    return sv_centers_out, kernel1_pullback

end

