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

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/additional_points_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/sv_centr_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/dif_points_per_triangle.jl")
# add EnzymeTestUtils,ChainRulesTestUtils,Revise,LLVMLoopInfo,Meshes,LinearAlgebra,GLMakie,Combinatorics,SplitApplyCombine,CUDA,Combinatorics,Random,Statistics,ChainRulesCore,ChainRulesCore, Zygote, CUDA, Enzyme, KernelAbstractions, Lux , FillArrays, LinearAlgebra,  Images, ImageFiltering,Optimisers,NNlib, Plots
# function get_point_on_a_line(vertex_0, vertex_1, weight)
#     diff_x = vertex_1[1] - vertex_0[1]
#     diff_y = vertex_1[2] - vertex_0[2]
#     diff_z = vertex_1[3] - vertex_0[3]
#     return [vertex_0[1] + (diff_x * weight), vertex_0[2] + (diff_y * weight), vertex_0[3] + (diff_z * weight)]
# end

"""
for linear we just get a point on a line between two adjacent sv centers in given axis
"""
macro save_point_on_line_for_lin(out_channel, x_add, y_add, z_add, weights_channel)
    return esc(quote
        for i in 1:3

            control_points_out[x, y, z, $out_channel, i, Int(Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure)))] = (sv_centers_mod[x+($x_add), y+($y_add), z+($z_add), i, Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))]
                                                                                                                   +
                                                                                                                   (sv_centers_mod[x+1, y+1, z+1, i, Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))] - (sv_centers_mod[x+($x_add), y+($y_add), z+($z_add), i, Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))])) * weights[x, y, z, ($weights_channel), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))])

        end
    end)
end



"""
we already have given the sv_centers_mod that are the centers of supervoxels with applied weights in previous step
we want to use a list of weights to move along a line between alowable maximum and minimum of points positions
for linear it is simply moving over a line between adjacent sv centers along each axis
for oblique it is selecting a position in a shape that is created by 8 sv centers

the function is designed to be broadcasted over the list of weights and the list of initialized 
control points 
    important we assume we have list of weights in a range from 0 to 1 (so for example after sigmoid) 
     control_points channel dimension is lin_x, lin_y, lin_z, oblique, 
    
    also  oblique_x, oblique_y, oblique_z will be calculated in separate kernel after oblique points are established
"""

function apply_weights_to_locs_kern(sv_centers_mod, control_points_out, weights, cp_x::UInt32, cp_y::UInt32, cp_z::UInt32, num_blocks_z_pure, num_blocks_y_pure)

    x = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))
    y = ((threadIdx().y + ((blockIdx().y - 1) * CUDA.blockDim_y())) % cp_y) + 1
    z = ((threadIdx().z + ((blockIdx().z - 1) * CUDA.blockDim_z())) % cp_z) + 1
    shared_arr = CuStaticSharedArray(Float64, (8, 8, 4, 4))

    if (x <= (cp_x) && y <= (cp_y) && z <= (cp_z) && x > 0 && y > 0 && z > 0)


        #lin_x
          control_points_out[x, y, z, 1, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int((ceil(CUDA.blockIdx().z / num_blocks_z_pure)))] = (sv_centers_mod[x+(0), y+(1), z+(1), Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))]
+
(sv_centers_mod[x+1, y+1, z+1, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))] - (sv_centers_mod[x+(0), y+(1), z+(1), Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))])) * weights[x, y, z, (4), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))])

                                                                                                               
                                                                                                               
        # #lin_y
          control_points_out[x, y, z, 2, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int((ceil(CUDA.blockIdx().z / num_blocks_z_pure)))] = (sv_centers_mod[x+(1), y+(0), z+(1), Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))]
+
(sv_centers_mod[x+1, y+1, z+1, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))] - (sv_centers_mod[x+(1), y+(0), z+(1), Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))])) * weights[x, y, z, (5), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))])

                                                                                                               
                                                                                                               
        #lin_z
          control_points_out[x, y, z, 3, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int((ceil(CUDA.blockIdx().z / num_blocks_z_pure)))] = (sv_centers_mod[x+(1), y+(1), z+(0), Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))]
+
(sv_centers_mod[x+1, y+1, z+1, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))] - (sv_centers_mod[x+(1), y+(1), z+(0), Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))])) * weights[x, y, z, (6), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))])

                                                                                                               
                                                                                                               
        #oblique main
        # save_point_for_oblique(1,x_add,y_add,z_add,x_add_b,y_add_b,z_add_b,7)
        #first we draw the line between 2 sv centers in x direction and get a point on it for each pair of sv centers that are neighbours in x
        #as we are moving in x we just need one weight for all of those points
            

        
shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 1] = (sv_centers_mod[x+(0), y+(0), z+(0)
, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(blockIdx().z  / num_blocks_z_pure))]
+
((sv_centers_mod[x+(1), y+(0), z+(0), Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure))
, Int(ceil(blockIdx().z  / num_blocks_z_pure))] - (sv_centers_mod[x+(0), y+(0), z+(0)
, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(blockIdx().z  / num_blocks_z_pure))])) 
* weights[x, y, z, (7), Int(ceil(blockIdx().z  / num_blocks_z_pure))]))

                                                                            
        
shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 2] = (sv_centers_mod[x+(0), y+(1), z+(0)
, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(blockIdx().z  / num_blocks_z_pure))]
+
((sv_centers_mod[x+(1), y+(1), z+(0), Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure))
, Int(ceil(blockIdx().z  / num_blocks_z_pure))] - (sv_centers_mod[x+(0), y+(1), z+(0)
, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(blockIdx().z  / num_blocks_z_pure))])) 
* weights[x, y, z, (7), Int(ceil(blockIdx().z  / num_blocks_z_pure))]))

                                                                            
        
shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 3] = (sv_centers_mod[x+(0), y+(0), z+(1)
, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(blockIdx().z  / num_blocks_z_pure))]
+
((sv_centers_mod[x+(1), y+(0), z+(1), Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure))
, Int(ceil(blockIdx().z  / num_blocks_z_pure))] - (sv_centers_mod[x+(0), y+(0), z+(1)
, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(blockIdx().z  / num_blocks_z_pure))])) 
* weights[x, y, z, (7), Int(ceil(blockIdx().z  / num_blocks_z_pure))]))

                                                                            
        
shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 4] = (sv_centers_mod[x+(0), y+(1), z+(1)
, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(blockIdx().z  / num_blocks_z_pure))]
+
((sv_centers_mod[x+(1), y+(1), z+(1), Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure))
, Int(ceil(blockIdx().z  / num_blocks_z_pure))] - (sv_centers_mod[x+(0), y+(1), z+(1)
, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(blockIdx().z  / num_blocks_z_pure))])) 
* weights[x, y, z, (7), Int(ceil(blockIdx().z  / num_blocks_z_pure))]))

                                                                            
        # #then we draw the line between 2 points calculated above centers in y direction and get a point on it for each pair of sv centers that are neighbours in y

            shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 1] = (shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 1]
    +
    (((shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 2] - (shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 1])) * weights[x, y, z, (8), Int(ceil(blockIdx().z  / num_blocks_z_pure))])))

            shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 3] = (shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 4]
    +
    (((shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 4] - (shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 4])) * weights[x, y, z, (8), Int(ceil(blockIdx().z  / num_blocks_z_pure))])))

            shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 1] = (shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 1]
    +
    (((shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 3] - (shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 1])) * weights[x, y, z, (9), Int(ceil(blockIdx().z  / num_blocks_z_pure))])))

        #saving output
        control_points_out[x, y, z, 4, Int(ceil(CUDA.blockIdx().y / num_blocks_y_pure)), Int(ceil(CUDA.blockIdx().z / num_blocks_z_pure))] = shared_arr[threadIdx().x, threadIdx().y, threadIdx().z, 1]
    end



    return nothing

end #apply_weights_to_locs


############################# Enzyme differentiation main apply weights to locs


function apply_weights_to_locs_kern_deff(sv_centers_mod, d_sv_centers_mod, control_points_out, d_control_points_out, weights, d_weights, cp_x::UInt32, cp_y::UInt32, cp_z::UInt32, num_blocks_z_pure, num_blocks_y_pure_w
)

    Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(apply_weights_to_locs_kern), Const, Duplicated(sv_centers_mod, d_sv_centers_mod), Duplicated(control_points_out, d_control_points_out), Duplicated(weights, d_weights), Const(cp_x), Const(cp_y), Const(cp_z), Const(num_blocks_z_pure), Const(num_blocks_y_pure_w)
    )
    return nothing
end


function call_apply_weights_to_locs_kern(sv_centers_mod, control_points_size, weights, threads, blocks, num_blocks_z_pure, num_blocks_y_pure_w,zero_der_beg)
    control_points_out = CUDA.zeros(control_points_size...)
    # print("\n sv_centers_mod $(size(sv_centers_mod))  control_points_size $(control_points_size) weights $(size(weights)) blocks $(blocks) threads $(threads) num_blocks_z_pure $(num_blocks_z_pure)\n")
    @cuda threads = threads blocks = blocks apply_weights_to_locs_kern(sv_centers_mod, control_points_out, weights, UInt32(control_points_size[1]), UInt32(control_points_size[2]), UInt32(control_points_size[3]), num_blocks_z_pure, num_blocks_y_pure_w
    )
    return control_points_out
end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_apply_weights_to_locs_kern), sv_centers_mod, control_points_size, weights, threads, blocks, num_blocks_z_pure, num_blocks_y_pure_w,zero_der_beg)

    control_points_out = call_apply_weights_to_locs_kern(sv_centers_mod, control_points_size, weights, threads, blocks, num_blocks_z_pure, num_blocks_y_pure_w,zero_der_beg)


    function kernel1_pullback(d_control_points_out)

        current_time = Dates.now()

        d_weights = CUDA.zeros(Float64,size(weights)...)
        # sizz = control_points_size
        # d_control_points = CUDA.zeros(sizz...)
        d_sv_centers_mod = CUDA.zeros(Float64,size(sv_centers_mod)...)
        control_points_out=Float64.(control_points_out)
        sv_centers_mod=Float64.(sv_centers_mod)

        #@device_code_warntype

        @cuda threads = threads blocks = blocks apply_weights_to_locs_kern_deff(
            sv_centers_mod, d_sv_centers_mod, control_points_out, CuArray(Float64.(Zygote.unthunk(d_control_points_out))), weights, d_weights, UInt32(control_points_size[1]), UInt32(control_points_size[2]), UInt32(control_points_size[3]), num_blocks_z_pure, num_blocks_y_pure_w)


        # dsss=sum(d_weights)
        # seconds_diff = Dates.value(Dates.now() - current_time)/ 1000
        # print("\n points weights basic kern backward d_weights sum $(dsss) ; $(seconds_diff) sec \n")
        # @info "points weights basic kern backward  " times=round(seconds_diff; digits = 2)

        if(zero_der_beg)
            return NoTangent(), d_sv_centers_mod, NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(), NoTangent(),NoTangent()
        end    
        return NoTangent(), d_sv_centers_mod, NoTangent(), d_weights, NoTangent(), NoTangent(), NoTangent(), NoTangent(),NoTangent()
    end

    return control_points_out, kernel1_pullback

end





############## lux definitions

"""
the main struct describing the apply weights kernel that is responsible to applying weights to 
control points that are used to define the borders of super voxels 
image_shape - shape of the image (or its patch) that we work on
"""
struct Points_weights_str <: Lux.Lux.AbstractLuxLayer
    radiuss::Tuple{Float32,Float32,Float32}
    batch_size::Int
    pad_voxels::Int
    image_shape::Tuple{Int,Int,Int}
    is_point_per_triangle::Bool
    zero_der_beg::Bool
    debug_with_source_arr::Bool
end

function Lux.initialparameters(rng::AbstractRNG, l::Points_weights_str)
    return ()
end


function Lux.initialstates(::AbstractRNG, l::Points_weights_str)::NamedTuple

    add_triangle_per_point = l.is_point_per_triangle


    max_index_for_per_triangle = -1
    x_blocks = -1
    blocks_per_triangle = -1
    siz_c = -1
    siz_plan = -1
    plan_tensors = CUDA.zeros(Int16, (1, 1))

    set_of_svs = initialize_centers_and_control_points(l.image_shape, l.radiuss, l.is_point_per_triangle)
    if (!l.is_point_per_triangle)
        sv_centers, control_points, tetrs, dims = set_of_svs
    else
        sv_centers, control_points, tetrs, dims, plan_tensors = set_of_svs
        siz_c = size(control_points)
        siz_plan = size(plan_tensors)
        x_blocks = Int(ceil((siz_c[1] * siz_c[2] * siz_c[3]) / len_get_random_point_in_tetrs_kern))
        max_index_for_per_triangle = siz_c[1] * siz_c[2] * siz_c[3]
        blocks_per_triangle = (x_blocks, siz_plan[1], l.batch_size)
    end

    batch_size = l.batch_size
    #repeat the control points and sv_centers for each element in batch
    control_points = repeat(control_points, inner=(1, 1, 1, 1, 1, l.batch_size))

    # sv_centers = repeat(sv_centers, inner=(1, 1, 1, 1, l.batch_size))



    control_points_size = size(control_points)
    control_points_size = (control_points_size[1], control_points_size[2], control_points_size[3], control_points_size[4], control_points_size[5], batch_size)

    # weights_begin = (UInt32(round((weights_shape[1] - control_points_shape[1]) / 2)), UInt32(round((weights_shape[2] - control_points_shape[2]) / 2)), UInt32(round((weights_shape[3] - control_points_shape[3]) / 2)))

    # weights_begin = (UInt32(1), UInt32(1), UInt32(1))
    weights_shape = (1, 1, 1)#legacy value
    threads_apply_sv, blocks_apply_sv, num_blocks_z_pure_sv, num_blocks_y_pure_sv = prepare_for_apply_weights_to_locs_kern(size(sv_centers), weights_shape, l.batch_size)

    threads_apply_w, blocks_apply_w, num_blocks_z_pure_w, num_blocks_y_pure_w = prepare_for_apply_weights_to_locs_kern(size(control_points), weights_shape, l.batch_size)

    sv_centers = Float64.(sv_centers)
    # control_points = Float32.(control_points)

    sv_centers_size = collect(size(sv_centers))[1:3]
    cp_size = collect(size(control_points))[1:3]

    return (debug_with_source_arr=l.debug_with_source_arr,zero_der_beg=l.zero_der_beg,plan_tensors=Int16.(plan_tensors), siz_plan=siz_plan, siz_c=siz_c, blocks_per_triangle=blocks_per_triangle, max_index_for_per_triangle=max_index_for_per_triangle, add_triangle_per_point=add_triangle_per_point, num_blocks_y_pure_w=num_blocks_y_pure_w, num_blocks_y_pure_sv=num_blocks_y_pure_sv, batch_size=l.batch_size, control_points_size=control_points_size, num_blocks_z_pure_w=num_blocks_z_pure_w, num_blocks_z_pure_sv=num_blocks_z_pure_sv, cp_size=cp_size, sv_centers_size=sv_centers_size, threads_apply_sv=threads_apply_sv, blocks_apply_sv=blocks_apply_sv, sv_centers=sv_centers, radiuss=l.radiuss, image_shape=l.image_shape, threads_apply_w=threads_apply_w, blocks_apply_w=blocks_apply_w)

end

function (l::Points_weights_str)(weights, ps, st::NamedTuple)

    if(st.debug_with_source_arr)
        source_arr,weights=weights
    end
    # print("\n wwwww min $(minimum(weights))  max $(maximum(weights)) mean $(mean(weights)) \n")
    # current_time = Dates.now()
    # weights=Float32.(weights)
    weights=Float64.(weights)
    zero_der_beg=st.zero_der_beg
    sv_centers_out = call_apply_weights_sv(CuArray(copy(st.sv_centers)), weights, st.radiuss, st.threads_apply_sv, st.blocks_apply_sv, st.num_blocks_z_pure_sv, st.batch_size, st.num_blocks_y_pure_sv,zero_der_beg)
    sv_centers_out=Float64.(sv_centers_out)
    # #now we can ingore weight from first entry
    # weights = weights[2:end, 2:end, 2:end, :, :]

    control_points_out = call_apply_weights_to_locs_kern(sv_centers_out, st.control_points_size, weights, st.threads_apply_w, st.blocks_apply_w, st.num_blocks_z_pure_w, st.num_blocks_y_pure_w,zero_der_beg)

    control_points_out = call_apply_weights_to_locs_kern_add_a(sv_centers_out, control_points_out, weights, st.threads_apply_w, st.blocks_apply_w, st.num_blocks_z_pure_w, st.num_blocks_y_pure_w,zero_der_beg)
    if (st.add_triangle_per_point)
        control_points_out = apply_random_point_in_tetrs_kern(st.plan_tensors, weights, control_points_out, sv_centers_out, st.siz_c, st.max_index_for_per_triangle, st.blocks_per_triangle,zero_der_beg)
    end



    # dsss=sum(control_points_out)
    # seconds_diff = Dates.value(Dates.now() - current_time)/ 1000
    # print("\n total control points forward control_points_out sum $(dsss) ; $(seconds_diff) sec \n")


    if(st.debug_with_source_arr)
        return (1.0,source_arr,(Float32.(sv_centers_out), Float32.(control_points_out))), st
    end

    return (Float32.(sv_centers_out), Float32.(control_points_out)), st

end




#(144, 2286, 9, 2, 5) must be consistent with array size (329232, 9, 2, 5)