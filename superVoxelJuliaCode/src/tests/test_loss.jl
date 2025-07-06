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
using LinearAlgebra

using Revise

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/Lux_model.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")


"""
our loss function has 2 components that we would test separately one is variance of the borders 
    it should increase when variance of the image increases and vice versa
second is variance of supervoxels in case of the images from synth data this variance should be small 
    and basically quite similar to the variance related to image with constant value like ones
    it should increase a lot in case of adding noise with high variance - test multiplying and adding noise

Hovewer first checkwether we calculate per sv variance correctly    
"""
# function test_loss_variance()

# out_sampled_points = rand(Float32, get_num_tetr_in_sv(), 9, 5,2)
# out_sampled_points = rand(Float32, get_num_tetr_in_sv(), 9, 5)
# out_sampled_points = repeat(out_sampled_points, inner=(1, 1, 1, 2))

# sizz_out = size(out_sampled_points) #(48, 9, 5, 2)

# out_sampled_points_reshaped = reshape(out_sampled_points[:, :, 1:2, :], (get_num_tetr_in_sv(), Int(round(sizz_out[1] / get_num_tetr_in_sv())), sizz_out[2], 2, sizz_out[4]))
# size(out_sampled_points_reshaped)
# out_sampled_points_reshaped = permutedims(out_sampled_points_reshaped, [2, 1, 3, 4, 5])
# size(out_sampled_points_reshaped) #(1, 48, 9, 2, 2)
# out_sampled_points = out_sampled_points[:, :, :, 1]

# values = reshape(out_sampled_points[1:get_num_tetr_in_sv(), :, 1], get_num_tetr_in_sv() * 9)
# weights = reshape(out_sampled_points[1:get_num_tetr_in_sv(), :, 2], get_num_tetr_in_sv() * 9)

# weighted_points = values .* weights
# mean_weighted = sum(weighted_points) / sum(weights)
# variance = sum(((values .- mean_weighted) .^ 2) .* weights) / sum(weights)

# res = call_get_per_sv_variance(CuArray(out_sampled_points_reshaped))
# res = Array(res)



# a = round(res[1], digits=4)
# b = round(variance[1], digits=4)

# @test isapprox(a, b)
# @test isapprox(res[1], res[2])

const is_point_per_triangle=true
get_num_tetr_in_sv()

out_sampled_points = rand(Float32, get_num_tetr_in_sv()*2, 9, 5,14)

sizz_out = size(out_sampled_points)#(65856, 9, 5)
batch_size = size(out_sampled_points)[4]#(65856, 9, 5)
num_sv = Int(round(sizz_out[1] / get_num_tetr_in_sv()))

out_sampled_points_reshaped = reshape(out_sampled_points[:, :, 1:2, :], (get_num_tetr_in_sv(), Int(round(sizz_out[1] / get_num_tetr_in_sv())), sizz_out[2], 2, batch_size))
out_sampled_points_reshaped = permutedims(out_sampled_points_reshaped, [2, 1, 3, 4, 5])
values_big = reshape(out_sampled_points_reshaped[:, :, :, 1, :], num_sv, get_num_tetr_in_sv() * 9, batch_size)
weights_big = reshape(out_sampled_points_reshaped[:, :, :, 2, :], num_sv, get_num_tetr_in_sv() * 9, batch_size)

big_weighted_values = values_big .* weights_big
big_weighted_values_summed = sum(big_weighted_values, dims=2)

big_weights_sum = sum(weights_big, dims=2)

big_mean_weighted = big_weighted_values_summed ./ big_weights_sum

variance_inside_power_big = values_big .- big_mean_weighted
variance_inside_power_big = variance_inside_power_big .^ 2
variance_inside_power_big = variance_inside_power_big .* weights_big
variance_inside_power_big = sum(variance_inside_power_big, dims=2) ./ big_weights_sum
res_no_kernel = mean(variance_inside_power_big)
res_no_kernel #0.08211113f0

#############3 non kernel loss calculation


# end

# test_loss_variance()




# function variance_loss(out_sampled_points)
#     sizz_out = size(out_sampled_points)#(65856, 9, 5)
#     num_sv=Int(round(sizz_out[1]/get_num_tetr_in_sv()))

#     out_sampled_points_reshaped = reshape(out_sampled_points[:, :, 1:2], (get_num_tetr_in_sv(), Int(round(sizz_out[1] / get_num_tetr_in_sv())), sizz_out[2], 2))
#     out_sampled_points_reshaped = permutedims(out_sampled_points_reshaped, [2, 1, 3, 4])
#     values_big=reshape(out_sampled_points_reshaped[:, :, :, 1],num_sv, get_num_tetr_in_sv()* 9)
#     weights_big=reshape(out_sampled_points_reshaped[:, :, :, 2],num_sv, get_num_tetr_in_sv()* 9)

#     big_weighted_values=values_big .* weights_big
#     big_weighted_values_summed=sum(big_weighted_values,dims=2)

#     big_weights_sum=sum(weights_big,dims=2)

#     big_mean_weighted=big_weighted_values_summed./big_weights_sum

#     variance_inside_power_big=values_big .- big_mean_weighted
#     variance_inside_power_big=variance_inside_power_big.^2
#     variance_inside_power_big=variance_inside_power_big.*weights_big
#     variance_inside_power_big=sum(variance_inside_power_big,dims=2)./big_weights_sum
#     return mean(variance_inside_power_big)
# end


# out_sampled_points = rand(Float32, 96, 9, 5,2)


# variance_loss(out_sampled_points)#0.08174637f0

# values = reshape(out_sampled_points[1:get_num_tetr_in_sv(), :, 1], get_num_tetr_in_sv() * 9)
# weights = reshape(out_sampled_points[1:get_num_tetr_in_sv(), :, 2], get_num_tetr_in_sv() * 9)
# weighted_points = values .* weights

# mean_weighted=sum(weighted_points)/sum(weights)

# aaa=big_weighted_values[1,:]

# weighted_points==aaa
# sum(weights)
# big_weights[1,:]==weights


# variance_inside_power_big

# variance_inside_power_small = sum(((values .- mean_weighted) .^ 2) .* weights) / sum(weights)
# isapprox(variance_inside_power_big[1,1], variance_inside_power_small)