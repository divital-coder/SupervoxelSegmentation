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
using Test,Logging,Interpolations
using KernelAbstractions, Dates
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote
using Lux, Random, Optimisers, Zygote
using LinearAlgebra, MLUtils
using Revise
using Pkg,JLD2
using TensorBoardLogger, Logging, Random
using ParameterSchedulers,GLFW
using MLDataDevices,DelaunayTriangulation
using Combinatorics

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/prepare_polygons.jl")

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/render_for_tensor_board.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/render_grey_poligons.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/main_loop.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/OpenGLUtils.jl")


debug_source="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/dd.h5"
image_size=(128,128,128)
const is_point_per_triangle=false
texture_width = image_size[1]
texture_height = image_size[2]
windowWidth, windowHeight=700,700
window=initializeWindow(windowWidth, windowHeight)
image_consts=window
radiuss=Float32(3.0), Float32(3.0), Float32(3.0)
imm_val=rand(Float32,image_size...).*100
im_name="debug"
axis=3
plane_dist=80


t_path="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/debugg"
if isdir(t_path)
    rm(t_path; force=true, recursive=true)
end
mkdir(t_path)
lg=TBLogger(t_path, min_level=Logging.Info)
h5file = h5open(debug_source, "r")
tetr_dat = read(h5file, "tetr_dat")
big_mean_weighted= read(h5file, "big_mean_weighted")
f=h5file
epoch=1.0
sv_means=big_mean_weighted[:,1,1]
num_tetr=get_num_tetr_in_sv()
imm_val_loc=Float32.(Array(imm_val)[:,:,:,1,1])




triang_vec,sv_i,line_vertices,line_indices,imm=get_data_to_display(tetr_dat, num_tetr, axis, plane_dist,image_size,imm_val_loc)
colorss=Float32.(big_mean_weighted[Int.(sv_i)])
window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program, line_shader_program, textUreId=initialize_window_etc(windowWidth, windowHeight,texture_width, texture_height,window)

line_vao, line_vbo, line_ebo = initialize_lines(line_vertices, line_indices)
im = render_and_save(lg, "$(im_name)_lines", epoch, windowWidth, windowHeight, texture_width, texture_height, imm_val_loc, line_vao, line_indices, line_shader_program, rectangle_vao, rectangle_shader_program, textUreId, window)

render_grey_polygons(triang_vec,colorss,window,windowWidth,windowHeight,lg,im_name,step)


close(h5file)

###
#for texture viz

###


locc_hdf5="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/locc.h5"
h5file = h5open(locc_hdf5, "r")
keys(h5file)

tetr_dat = read(h5file, "tetr_dat")
out_sampled_points = read(h5file, "out_sampled_points")
im = read(h5file, "im")
imm_val_loc=Float32.(Array(im)[:,:,:,1,1])


sizz_out = size(out_sampled_points)#(65856, 9, 5)
batch_size = size(out_sampled_points)[end]#(65856, 9, 5)
num_sv = sizz_out[1]#Int(round(sizz_out[1] / get_num_tetr_in_sv()))
# print("\n ttttttttt  n_sv $(get_num_tetr_in_sv()) out_sampled_points $(size(out_sampled_points))  \n")

out_sampled_points = reshape(out_sampled_points[:, :, 1:2, :], (get_num_tetr_in_sv(), Int(round(sizz_out[1] / get_num_tetr_in_sv())), sizz_out[2], 2, batch_size))
out_sampled_points = permutedims(out_sampled_points, [2, 1, 3, 4, 5])
values_big = reshape(out_sampled_points[:, :, :, 1, :], :, get_num_tetr_in_sv() * 9, batch_size)
weights_big = reshape(out_sampled_points[:, :, :, 2, :], :, get_num_tetr_in_sv() * 9, batch_size)

big_weighted_values = values_big .* weights_big
big_weighted_values_summed = sum(big_weighted_values, dims=2)

big_weights_sum = sum(weights_big, dims=2)

big_mean_weighted = vec(big_weighted_values_summed ./ big_weights_sum)





triang_vec,sv_i,line_vertices,line_indices,imm=get_data_to_display(tetr_dat, num_tetr, axis, plane_dist,image_size,imm_val_loc)
colorss=Float32.(big_mean_weighted[Int.(sv_i)])
window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program, line_shader_program, textUreId=initialize_window_etc(windowWidth, windowHeight,texture_width, texture_height,window)

line_vao, line_vbo, line_ebo = initialize_lines(line_vertices, line_indices)
im = render_and_save(lg, "$(im_name)_lines", epoch, windowWidth, windowHeight, texture_width, texture_height, imm_val_loc, line_vao, line_indices, line_shader_program, rectangle_vao, rectangle_shader_program, textUreId, window)

render_grey_polygons(triang_vec,colorss,window,windowWidth,windowHeight,lg,im_name,step)



### prepare data for texture
sizz_out = size(out_sampled_points)
batch_size = sizz_out[end]
n_tetr = get_num_tetr_in_sv()
num_sv = Int(round(sizz_out[1] / n_tetr))
num_texture_banks=32 
num_sinusoids_per_bank=4
texture_bank_p = rand(Float32, num_texture_banks, num_sinusoids_per_bank, 5)

sin_p = rand(Float32, sizz_out[1], num_texture_banks + 6) .* 2
sin_p_a = sin_p[:, 1:5]
sin_p_b = softmax(sin_p[:, 6:end], dims=2)
sin_p = cat(sin_p_a, sin_p_b, dims=2)


