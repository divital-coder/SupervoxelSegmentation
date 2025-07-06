using Revise, CUDA, HDF5
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


using Logging
using Interpolations
using KernelAbstractions, Dates
# using KernelGradients
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote
using Lux, Random, Optimisers, Zygote
using LinearAlgebra, MLUtils

using Revise



includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/get_lin_synth_dat.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/data_manag.jl")



"""
loss function need to first take all of the sv border points and associated variance from tetr_dat
    the higher the mean of those the better 
in tetr dat for each tetrahedron first entry is sv center and the rest are border points
    we need to take the border points and get variance which is the fourth in each point data 


Next we need to take the weighted variance of the points sampled from out_sampled_points separately for each supervoxel 
    as we have all of the tetrahedrons flattened we need to reshape the outsampled points array so we will be able to process
    each supervoxel separately and calculate separately variance of each
    so we out sample points are in the shape of (num tetrahedrons, num_sample points , 5) last dimension is info about the point
    first entry is interpolated value and next one its weight

To consider - using tulio or sth with einsum to get the variance of each supervoxel

finally mean of variance of supervoxels calculated from out sampled points should be as small as possible

"""
function get_border_loss(tetr_dat)
    
    arr = tetr_dat[:, 2:end, 4, :]
    sum(arr)
    return mean(arr)
end






function get_sv_variance_loss(out_sampled_points,is_point_per_triangle)
    sizz_out = size(out_sampled_points)#(65856, 9, 5)
    batch_size = size(out_sampled_points)[end]#(65856, 9, 5)
    # num_sv = sizz_out[1]#Int(round(sizz_out[1] / get_num_tetr_in_sv()))
    # print("\n ttttttttt  n_sv $(get_num_tetr_in_sv()) out_sampled_points $(size(out_sampled_points))  \n")

    # out_sampled_points_reshaped = reshape(out_sampled_points[:, :, 1:2, :], (get_num_tetr_in_sv(), Int(round(sizz_out[1] / get_num_tetr_in_sv())), sizz_out[2], 2, batch_size))
    # out_sampled_points_reshaped = permutedims(out_sampled_points_reshaped, [2, 1, 3, 4, 5])
    values_big = reshape(out_sampled_points[:, :, :, 1, :], :, get_num_tetr_in_sv(is_point_per_triangle) * 5, batch_size)
    weights_big = reshape(out_sampled_points[:, :, :, 2, :], :, get_num_tetr_in_sv(is_point_per_triangle) * 5, batch_size)
    
    big_weighted_values = values_big .* weights_big
    big_weighted_values_summed = sum(big_weighted_values, dims=2)
    
    big_weights_sum = sum(weights_big, dims=2)
    
    big_mean_weighted = big_weighted_values_summed ./ big_weights_sum
    
    variance_inside_power_big = values_big .- big_mean_weighted
    variance_inside_power_big = variance_inside_power_big .^ 2
    variance_inside_power_big = variance_inside_power_big .* weights_big
    variance_inside_power_big = sum(variance_inside_power_big, dims=2) ./ big_weights_sum
    res_no_kernel = sum(variance_inside_power_big)/length(variance_inside_power_big)
    return res_no_kernel
end



function loss_function_dummy(model, ps, st, data)
    y_pred, st = Lux.apply(model, data, ps, st)

    out_sampled_points, tetr_dat = y_pred
    # b_loss=get_sv_variance_loss(out_sampled_points)
    b_loss=sum(y_pred[1])

    # b_loss = (-(get_border_loss(tetr_dat)))
    # b_loss = ((get_sv_variance_loss(out_sampled_points)^2)/(get_border_loss(tetr_dat)))
    # b_loss = get_border_loss(y_pred[1]) 

    return b_loss, st, ()
end

# function loss_function_border_loss(model, ps, st, data,is_point_per_triangle)
#     print("border_loss_only")
#     y_pred, st = Lux.apply(model, data, ps, st)

#     out_sampled_points, tetr_dat = y_pred
#     b_loss=get_border_loss(tetr_dat)*(-1) 
#     return b_loss, st, ()
# end

# function loss_function_sv_variance_loss(model, ps, st, data)
#     y_pred, st = Lux.apply(model, data, ps, st)

#     out_sampled_points, tetr_dat = y_pred
#     b_loss=get_sv_variance_loss(out_sampled_points)
#     return b_loss, st, ()
# end

# function loss_function_var_border(model, ps, st, data)
#     y_pred, st = Lux.apply(model, data, ps, st)

#     out_sampled_points, tetr_dat = y_pred
#     b_loss=get_sv_variance_loss(out_sampled_points)+get_border_loss(tetr_dat)

#     return b_loss, st, ()
# end

# # function loss_function_varsq_div_border(model, ps, st, data)
# #     y_pred, st = Lux.apply(model, data, ps, st)

# #     out_sampled_points, tetr_dat = y_pred
# #     b_loss=(get_sv_variance_loss(out_sampled_points)^2)/get_border_loss(tetr_dat)
# #     return b_loss, st, ()
# # end

function loss_sinusoids(model, ps, st, data)
    y_pred, st = Lux.apply(model, data, ps, st)

    big_mean_weighted,weighted_sin_loss,tetr_dat,sin_p,texture_bank_p = y_pred
    b_loss=(mean(weighted_sin_loss)^2)/get_border_loss(tetr_dat)
    return b_loss, st, ()
end


function loss_function_inner(model, ps, st, data,loss_name)
    y_pred, st = Lux.apply(model, data, ps, st)

    out_sampled_points, tetr_dat = y_pred
    # b_loss=get_sv_variance_loss(out_sampled_points)


    if loss_name == "border_loss"
        b_loss=get_border_loss(tetr_dat)
    elseif loss_name == "sv_variance_loss"
        b_loss= get_sv_variance_loss(out_sampled_points)
    elseif loss_name == "var-border"
        b_loss= get_sv_variance_loss(out_sampled_points)-get_border_loss(tetr_dat)
    elseif loss_name == "varsq_div_border"
        b_loss= (get_sv_variance_loss(out_sampled_points)^2)/get_border_loss(tetr_dat)
    end

    # b_loss = (-(get_border_loss(tetr_dat)))
    # b_loss = ((get_sv_variance_loss(out_sampled_points)^2)/(get_border_loss(tetr_dat)))
    # b_loss = get_border_loss(y_pred[1]) 

    return b_loss, st, ()
end


"""
based on loss name return the loss function 
    ready to use by Lux
"""
function select_loss(loss_name)::Function
    return (model, ps, st, data) -> loss_function_inner(model, ps, st, data,loss_name)
end


function loss_simple(model, ps, st, data)
    y_pred, st = Lux.apply(model, data, ps, st)
    out_sampled_points, tetr_dat,b_loss = y_pred

    return b_loss, st, ()
end





# function loss_function_varsq_div_border(model, ps, st, data)
#     y_pred, st = Lux.apply(model, data, ps, st)
#     y_pred_true,y_pred_false=y_pred
    
#     out_sampled_points, tetr_dat = y_pred_true
#     b_loss=(get_sv_variance_loss(out_sampled_points,true)^2)/get_border_loss(tetr_dat)

#     out_sampled_points, tetr_dat = y_pred_false
#     b_loss=(get_sv_variance_loss(out_sampled_points,false)^2)/get_border_loss(tetr_dat)


#     return b_loss, st, ()
# end