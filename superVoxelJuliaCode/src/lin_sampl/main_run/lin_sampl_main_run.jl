cd("/workspaces/superVoxelJuliaCode_lin_sampl")


# In the Julia REPL:
using Pkg
Pkg.activate(".")  # Activate the current directory as project


using MPI
using HDF5
# @assert HDF5.has_parallel()

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
using PyCall
# CUDA.allowscalar(true)
using Revise
using Pkg, Wavelets, LuxCUDA, JLD2
import Lux.Experimental.ADTypes  # Add this line

"""
to deal with some strange error
"""
function Zygote.accum(x::AbstractArray, ys::AbstractArray...)
    dev = gpu_device() 
    x = x |> dev
    ys = ys |> dev
    return Base.broadcast_preserving_zero_d(Zygote.accum, x, ys...)
end

# CUDA.allowscalar(true)
# Pkg.add(url="https://github.com/LuxDL/Lux.jl.git")

# Pkg.add(url="https://github.com/JuliaBinaryWrappers/Enzyme_jll.jl.git")
# Pkg.add(url="https://github.com/EnzymeAD/Enzyme.jl.git")

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/get_lin_synth_dat.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/data_manag.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/main_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/main_loop.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/Lux_model.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern _old.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/dif_get_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/hp_tuning.jl")
sitk = pyimport_conda("SimpleITK", "simpleitk")
np = pyimport_conda("numpy", "numpy")




# const gdev = gpu_device()
# const cdev = cpu_device()

# Distributed Training: NCCL for NVIDIA GPUs and MPI for anything else
# const distributed_backend = try
#     if gdev isa CUDADevice
#         DistributedUtils.initialize(NCCLBackend)
#         DistributedUtils.get_distributed_backend(NCCLBackend)
#     else
#         DistributedUtils.initialize(MPIBackend)
#         DistributedUtils.get_distributed_backend(MPIBackend)
#     end
# catch err
#     @error "Could not initialize distributed training. Error: $err"
#     nothing
# end

# DistributedUtils.initialize(NCCLBackend)
# distributed_backend = DistributedUtils.get_distributed_backend(NCCLBackend)

# local_rank=DistributedUtils.local_rank(distributed_backend)
# total_workers=DistributedUtils.total_workers(distributed_backend)
# print("\n local_rank $(local_rank) total_workers $(total_workers)MPI.has_cuda() $(MPI.has_cuda())\n")


# DistributedUtils.total_workers(distributed_backend)

# DistributedUtils.local_rank(distributed_backend)

# const local_rank = distributed_backend === nothing ? 0 :
#                    DistributedUtils.local_rank(distributed_backend)
# const total_workers = distributed_backend === nothing ? 1 :
#                       DistributedUtils.total_workers(distributed_backend)

const should_log = true
# const is_distributed = true
global const is_distributed = false
#for per triangle point calculations
global const len_get_random_point_in_tetrs_kern=Int(floor(256/3))
# global const is_point_per_triangle=false
#control additional point kernel
global const closeness_to_discr=1000000
global const num_additional_oblique_points_per_side=2
show_visualization=true
# needed for sinusoid loss 

# global const num_sinusoids_per_bank=4
# global const num_texture_banks=30

#load data
h5_path = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/brain_128_128_128.h5"
h5_path = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_viz/128_zero/synth_supervoxel_data.h5"


comm = MPI.COMM_WORLD
info = MPI.Info()
f = h5open(h5_path, "r")
# f = h5open(h5_path, "r" ,comm, info)


spacing = (Float32(1.0), Float32(1.0), Float32(1.0))
batch_size=4
max_keys= (-1) #batch_size*1000#how many images we will use in the training ; set it to -1 if on all images
# max_keys=batch_size*1000#how many images we will use in the training ; set it to -1 if on all images

main_tensor_board_path= "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/hp"
main_out_path="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/images3d"


hp_tuning(spacing,f,700,700,batch_size,main_tensor_board_path,main_out_path,max_keys,show_visualization)


#CUDA_VISIBLE_DEVICES=1 julia /workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/lin_sampl_main_run.jl
#CUDA_VISIBLE_DEVICES=0 julia /workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/lin_sampl_main_run.jl


# h5_path_b = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/locc.h5"
# fb = h5open(h5_path_b, "r")

# write(fb, "out_sampled_points", Array(out_sampled_points))
# write(fb, "tetr_dat", Array(tetr_dat))
# write(fb, "im", Array(get_sample_image_batched(f,batch_size)[:, :, :, 1, 1]))
# close(fb)


#tensorboard --logdir=//workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/hp --host localhost --port 8889
#mpiexec --allow-run-as-root -n 2 julia /workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/lin_sampl_main_run.jl

# /usr/local/share/julia/environments/v1.10 
#tensorboard --logdir=//workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/hp --host localhost --port 8889



# LEARNING_RATE_START="0.00013688731" \
# LEARNING_RATE_END="1.0517434e-6" \
# ADD_GRADIENT_ACCUM="true" \
# ADD_GRADIENT_NORM="true" \
# IS_WEIGHT_DECAY="true" \
# GRAD_ACCUM_VAL=4 \
# CLIP_NORM_VAL=10.0 \
# out_path="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/locc_b" \
# CUDA_VISIBLE_DEVICES=0 \
# restart=0 \
# julia /workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/lin_sampl_main_run.jl


# TODO
# https://docs.sciml.ai/Optimization/stable/optimization_packages/optimization/#Train-NN-with-Sophia

