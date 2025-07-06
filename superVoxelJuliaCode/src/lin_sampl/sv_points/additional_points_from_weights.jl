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

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights_additional_unrolled_kernel.jl")
# Pkg.add(url="https://github.com/LuxDL/LuxCUDA.jl.git")
# Pkg.add(url="https://github.com/LuxDL/Lux.jl.git")
# Pkg.add(url="https://github.com/JuliaBinaryWrappers/Enzyme_jll.jl.git")
# Pkg.add(url="https://github.com/EnzymeAD/Enzyme.jl.git")
#] add  LuxCUDA,Lux,Enzyme_jll,Enzyme   MPI,BSON,CSV,DataFrames,DataStructures,DelaunayTriangulation,Setfield,MLDataDevices,TensorBoardLogger GLFW,GeometryTypes,ModernGL,NNlib,LLVMLoopInfo,ParameterSchedulers,JLD2,cuTENSOR,TensorOperations,VectorInterface,Wavelets,CUDA,MLUtils,StatsBase,PyCall,Interpolations,HDF5,Revise,Meshes,GLMakie,SplitApplyCombine,Combinatorics,Random,Statistics,ChainRulesCore, Zygote, KernelAbstractions,   FillArrays, LinearAlgebra,  Images, Optimisers, Plots



############################# Enzyme differentiation main apply weights to locs
function apply_weights_to_locs_kern_deff_add_a(sv_centers_mod, d_sv_centers_mod, control_points, d_control_points, control_points_out_in, d_control_points_out_in, weights, d_weights, cp_x::UInt32, cp_y::UInt32, cp_z::UInt32, num_blocks_z_pure, num_blocks_y_pure_w
)

    Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(apply_weights_to_locs_kern_add_a), Const,
     Duplicated(sv_centers_mod, d_sv_centers_mod), Duplicated(control_points, d_control_points)
     , Duplicated(control_points_out_in, d_control_points_out_in), Duplicated(weights, d_weights)
     , Const(cp_x), Const(cp_y), Const(cp_z), Const(num_blocks_z_pure), Const(num_blocks_y_pure_w)
    )
    return nothing
end


function call_apply_weights_to_locs_kern_add_a(sv_centers_mod, control_points, weights, threads, blocks, num_blocks_z_pure, num_blocks_y_pure_w,zero_der_beg)

    control_points_out = Float64.(copy(control_points))
    @cuda threads = threads blocks = blocks apply_weights_to_locs_kern_add_a(sv_centers_mod, control_points, control_points_out, weights, UInt32(size(control_points)[1]), UInt32(size(control_points)[2]), UInt32(size(control_points)[3]), num_blocks_z_pure, num_blocks_y_pure_w
    )
    return control_points_out
end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_apply_weights_to_locs_kern_add_a), sv_centers_mod, control_points, weights, threads, blocks, num_blocks_z_pure, num_blocks_y_pure_w,zero_der_beg)

    control_points_out = call_apply_weights_to_locs_kern_add_a(sv_centers_mod, control_points, weights, threads, blocks, num_blocks_z_pure, num_blocks_y_pure_w,zero_der_beg)
    function kernel1_pullback(d_control_points_out)

        current_time = Dates.now()

        d_weights = CUDA.zeros(Float64,size(weights)...)
        sizz = size(control_points)
        d_control_points = CUDA.zeros(Float64,sizz...)
        d_sv_centers_mod = CUDA.zeros(Float64,size(sv_centers_mod)...)
        sv_centers_mod=Float64.(sv_centers_mod)
        control_points=Float64.(control_points)

        #@device_code_warntype


        @cuda threads = threads blocks = blocks apply_weights_to_locs_kern_deff_add_a(sv_centers_mod, d_sv_centers_mod,
            control_points, d_control_points, control_points_out, CuArray(Float64.(Zygote.unthunk(d_control_points_out))), weights, d_weights, UInt32(sizz[1]), UInt32(sizz[2]), UInt32(sizz[3]), num_blocks_z_pure, num_blocks_y_pure_w
        )

        # dsss=sum(d_weights)
        # seconds_diff = Dates.value(Dates.now() - current_time)/ 1000
        # print("\n additional points kern backward d_weights sum $(dsss) ; $(seconds_diff) sec \n")
        # @info "additional points kern backward  " times=round(seconds_diff; digits = 2)

        # return NoTangent(),d_control_points_out,d_control_points_out_in,d_weights,NoTangent(),NoTangent(),NoTangent()
        if(zero_der_beg)
            return NoTangent(), d_sv_centers_mod, d_control_points, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),NoTangent()

        end    

        return NoTangent(), d_sv_centers_mod, d_control_points, d_weights, NoTangent(), NoTangent(), NoTangent(), NoTangent(),NoTangent()
    end

    return control_points_out, kernel1_pullback

end
    