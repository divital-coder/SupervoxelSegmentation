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
using JLD2
using TensorBoardLogger

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/data_manag.jl")

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

h5_path = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/patches.h5"
checkpointed_model = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/debug_inner/1/trained_model.jld2"
tb_path="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/hp_debug"

global const is_point_per_triangle=false


hp_dict=get_base_hp()
f = h5open(h5_path, "r")
batch_size=4
radiuss=hp_dict["radiuss"]
hp_dict["batch_size"]=batch_size

logger = TBLogger(tb_path)
# 1) Load data
image_batch = get_sample_image_batched(f, batch_size)
wWidth=700
wHeight=700
texW=size(image_batch,1)
texH=size(image_batch,2)
window=initializeWindow(wWidth, wHeight)
# 2) Load model checkpoint
opt = Optimisers.AdaDelta()
@load checkpointed_model ps st
st_opt=Optimisers.setup(opt, ps)

image_size = get_init_consts(hp_dict["batch_size"])  
spacing = (Float32(1.0), Float32(1.0), Float32(1.0))
max_wavelength,max_amplitude,min_value=get_consts_for_sin_loss(f, batch_size,hp_dict["radiuss"])

model, opt = get_model_consts(image_size,   spacing,hp_dict,max_wavelength,max_amplitude,min_value)

tstate= Lux.Training.TrainState(nothing, nothing, model, ps, st, opt, st_opt, 0)
dev= gpu_device() 
tstate = tstate |> dev
# 3) Run inference
y_pred = infer_model(tstate, model, CuArray(image_batch))
out_sampled_points, tetr_dat=y_pred
big_mean_weighted = get_sv_mean(out_sampled_points)
# 4) Display slice
epoch=-1

function process_and_render_full_image(im_name, axis,  tetr_dat, radiuss, f, imm_val,  epoch, windowWidth, windowHeight, texture_width, texture_height, window, sv_means,out_path)
    
    GLFW.Terminate()
    window=initializeWindow(windowWidth, windowHeight)

    imm_val_loc_a = Float32.(Array(imm_val)[:, :, :, 1, 1])
    tetr_dat = Array(tetr_dat)
    sv_means = sv_means .- minimum(sv_means)
    sv_means = sv_means ./ maximum(sv_means)


    im_lines_whole=[]
    im_gp_whole=[]
    im_base_whole=[]

    for plane_dist in 8:(size(imm_val_loc_a)[3]-8)
        
        imm_val_loc=copy(imm_val_loc_a)
        tetr_num = get_num_tetr_in_sv()
        image_size = size(imm_val_loc)
        triang_vec, sv_i, line_vertices, line_indices, imm = get_data_to_display(tetr_dat, tetr_num, axis, plane_dist, image_size, imm_val_loc)
        colorss = Float32.(sv_means[Int.(sv_i)])
        window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program, line_shader_program, textUreId = initialize_window_etc(windowWidth, windowHeight, texture_width, texture_height, window)
        GLFW.PollEvents()

        line_vao, line_vbo, line_ebo = initialize_lines(line_vertices, line_indices)
        GLFW.PollEvents()

        im_lines,im_base = render_and_save_separtated(windowWidth,windowHeight, texture_width, texture_height, imm_val_loc[:,:,plane_dist], line_vao, line_indices, line_shader_program, rectangle_vao, rectangle_shader_program, textUreId, window)
        GLFW.PollEvents()

        push!(im_lines_whole, sum(im_lines, dims=1)[:,:,:])
        push!(im_base_whole, sum(im_base, dims=1)[:,:,:])
    end



    for plane_dist in 8:(size(imm_val_loc_a)[3]-8)
        imm_val_loc=copy(imm_val_loc_a)
        tetr_num = get_num_tetr_in_sv()
        image_size = size(imm_val_loc)
        triang_vec, sv_i, line_vertices, line_indices, imm = get_data_to_display(tetr_dat, tetr_num, axis, plane_dist, image_size, imm_val_loc)
        colorss = Float32.(sv_means[Int.(sv_i)])
        window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program, line_shader_program, textUreId = initialize_window_etc(windowWidth, windowHeight, texture_width, texture_height, window)

        im_gp = render_grey_polygons_loc(triang_vec, colorss, window, windowWidth, windowHeight, im_name, step)
        GLFW.PollEvents()
        push!(im_gp_whole, sum(im_gp, dims=1)[:,:,:])
    end


    im_lines_whole_im=vcat(im_lines_whole...)
    im_base_whole_im=vcat(im_base_whole...)
    im_gp_whole_im=vcat(im_gp_whole...)
    p_out_im_base="$(out_path)/im_base_whole_im.nii.gz"
    p_out_im_lines="$(out_path)/im_base_lines_im.nii.gz"
    p_out_im_gp="$(out_path)/im_base_gp_im.nii.gz"
    
    im_lines_whole_im=UInt8.(round.(im_lines_whole_im))
    sitk.WriteImage(sitk.GetImageFromArray(im_base_whole_im), p_out_im_base)
    sitk.WriteImage(sitk.GetImageFromArray(im_lines_whole_im), p_out_im_lines)
    sitk.WriteImage(sitk.GetImageFromArray(im_gp_whole_im), p_out_im_gp)





end



out_p="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/debug_b"
process_and_render_full_image("slice_example", 3, Array(tetr_dat), radiuss, f, image_batch,  epoch, wWidth, wHeight, texW, texH, window, Array(big_mean_weighted),out_p)

# size(im_lines_whole[1])

# im_lines_whole_im=vcat(im_lines_whole...)
# im_base_whole_im=vcat(im_base_whole...)
# im_gp_whole_im=vcat(im_gp_whole...)
# p_out_im_base="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/debug_b/im_base_whole_im.nii.gz"
# p_out_im_lines="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/debug_b/im_base_lines_im.nii.gz"
# p_out_im_gp="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/debug_b/im_base_gp_im.nii.gz"

# im_lines_whole_im=UInt8.(round.(im_lines_whole_im))
# sitk.WriteImage(sitk.GetImageFromArray(im_base_whole_im), p_out_im_base)
# sitk.WriteImage(sitk.GetImageFromArray(im_lines_whole_im), p_out_im_lines)
# sitk.WriteImage(sitk.GetImageFromArray(im_gp_whole_im), p_out_im_gp)