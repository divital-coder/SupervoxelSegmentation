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
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/shape_similarity_loss.jl")


radiuss = (Float32(3.1), Float32(4.3), Float32(4.7))
num_weights_per_point = 8
spacing = (Float32(1.0), Float32(1.0), Float32(1.0))
batch_size = 2
a = 71
image_shape = (a, a, a, 2)

weights_shape = Int.(round.((a / 2, a / 2, a / 2, 24)))

sv_centers, control_points, tetrs, dims = example_set_of_svs=example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
size(sv_centers)
dims
#adding batch for testing
tetr_s=size(tetrs)
batch_size=1
tetrs_3d=reshape(tetrs, (tetr_s[1], tetr_s[2], tetr_s[3], batch_size))
tetrs_3d = reshape(tetrs_3d, (get_num_tetr_in_sv(), Int(round(tetr_s[1] / get_num_tetr_in_sv())), tetr_s[2], tetr_s[3], batch_size))
tetrs_3d=permutedims(tetrs_3d, [2, 1, 3, 4, 5])
tetr_s_b=size(tetrs_3d)
tetrs_3d=reshape(tetrs_3d, (dims[1]-2,dims[2]-2,dims[3]-2, tetr_s_b[2], tetr_s_b[3], tetr_s_b[4], tetr_s_b[5]))
tetrs_3d=tetrs_3d[:,:,:,:,1:4,1:3,:]
size(tetrs_3d)
#(x,y,z, tetr_index_in_sv, point_index_in_tetr, point_coord, batch_size)


call_get_sv_shape_similarity(CuArray(tetrs_3d))


# size(fb["tetr_dat"]) (82944, 5, 4, 1)