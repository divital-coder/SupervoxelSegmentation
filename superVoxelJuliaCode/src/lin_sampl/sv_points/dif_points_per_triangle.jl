using LinearAlgebra
using Meshes
using GLMakie
using Statistics
using LinearAlgebra
using Random, Test
using LinearAlgebra, KernelAbstractions, CUDA
using Revise
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
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_per_triangle.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")


function get_random_point_in_tetrs_kern_deff(plan_tensor, d_plan_tensor, weights, d_weights, control_points_in, d_control_points_in, sv_centers_out, d_sv_centers_out, control_points_out, d_control_points_out, siz_c, max_index)

    p=Duplicated(plan_tensor, d_plan_tensor)
    w=Duplicated(weights, d_weights)
    c=Duplicated(control_points_in, d_control_points_in)
    s=Duplicated(sv_centers_out, d_sv_centers_out)
    co=Duplicated(control_points_out, d_control_points_out)
    
    Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(get_random_point_in_tetrs_kern),
        Enzyme.Const
        , p
        , w
        , c
        , s
        , co
        , Const(siz_c), Const(max_index)
    )

    return nothing
end


function apply_random_point_in_tetrs_kern(plan_tensor, weights, control_points_in, sv_centers_out, siz_c, max_index, blocks,zero_der_beg)


    control_points_out = Float64.(CuArray(copy(control_points_in)))
    @cuda threads = (len_get_random_point_in_tetrs_kern, 3) blocks = blocks get_random_point_in_tetrs_kern(plan_tensor, weights, control_points_in, sv_centers_out, control_points_out, siz_c, max_index)
    return control_points_out

end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(apply_random_point_in_tetrs_kern), plan_tensor, weights, control_points_in, sv_centers_out, siz_c, max_index, blocks,zero_der_beg)

    control_points_out = apply_random_point_in_tetrs_kern(plan_tensor, weights, control_points_in, sv_centers_out, siz_c, max_index, blocks,zero_der_beg)


    function kernel1_pullback(d_control_points_out)

        # current_time = Dates.now()


        d_control_points_out = CuArray(Float64.(Zygote.unthunk(d_control_points_out)))
        d_control_points_in = CUDA.zeros(Float64, size(control_points_in))
        d_plan_tensor = CUDA.zeros(Int16, size(plan_tensor))
        d_weights = CUDA.zeros(Float64, size(weights))
        d_sv_centers_out = CUDA.zeros(Float64, size(sv_centers_out))
        control_points_in=Float64.(control_points_in)
        sv_centers_out=Float64.(sv_centers_out)
        control_points_out=Float64.(control_points_out)
        # print("\n rrrrrrrrrrrrr $(typeof(control_points_in))\n")
        count_zeros(d_control_points_out, "d_control_points_out-apply_random_point_in_tetrs_kern")

        p=Duplicated(plan_tensor, d_plan_tensor)
        w=Duplicated(weights, d_weights)
        c=Duplicated(control_points_in, d_control_points_in)
        s=Duplicated(sv_centers_out, d_sv_centers_out)
        co=Duplicated(control_points_out, d_control_points_out)

        # print("\n sssssssssssss  siz_c $(siz_c) max_index $(max_index) blocks $(blocks)  len_get_random_point_in_tetrs_kern $(len_get_random_point_in_tetrs_kern)\n")

        @cuda threads = (len_get_random_point_in_tetrs_kern, 3) blocks = blocks get_random_point_in_tetrs_kern_deff(plan_tensor, d_plan_tensor, weights, d_weights, control_points_in, d_control_points_in, sv_centers_out, d_sv_centers_out,
            control_points_out, d_control_points_out, siz_c, max_index)

            # count_zeros(d_control_points_in, "d_control_points_in-d_control_points_in")

            # h5_path_b = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_gradients.h5"
            # fb = h5open(h5_path_b, "w")
    
            # write(fb, "d_control_points_out", Array(d_control_points_out))
            # write(fb, "d_control_points_in", Array(d_control_points_in))
            # write(fb, "d_plan_tensor", Array(d_plan_tensor))
            # write(fb, "d_weights", Array(d_weights))
            # write(fb, "d_sv_centers_out", Array(d_sv_centers_out))
            # write(fb, "plan_tensor", Array(plan_tensor))
            # write(fb, "weights", Array(weights))
            # write(fb, "control_points_in", Array(control_points_in))
            # write(fb, "sv_centers_out", Array(sv_centers_out))
            # write(fb, "control_points_out", Array(control_points_out))
    
            # close(fb)


        count_zeros(d_weights, "point per triangel d_weights")
        # print(size(d_weights))
        # count_zeros(d_weights[:,:,:,:,1], "d_weights-d_weights 1")
        # count_zeros(d_weights[:,:,:,:,2], "d_weights-d_weights 2")
        # count_zeros(d_weights[:,:,:,:,3], "d_weights-d_weights 3")

        # dsss=sum(d_weights)
        # seconds_diff = Dates.value(Dates.now() - current_time)/ 1000
        # print("\n point per triangle kern backward d_weights sum $(dsss) ; $(seconds_diff) sec \n")
        # @info "point per triangle backward  " times=round(seconds_diff; digits = 2)


        
        return NoTangent(), d_plan_tensor, d_weights, d_control_points_in, d_sv_centers_out, NoTangent(), NoTangent(), NoTangent(), NoTangent()
    end

    return control_points_out, kernel1_pullback

end

