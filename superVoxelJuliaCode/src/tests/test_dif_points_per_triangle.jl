using LinearAlgebra
using Meshes
using GLMakie
using Statistics
using LinearAlgebra
using Random,Test
using LinearAlgebra,KernelAbstractions,CUDA
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
using HDF5,LuxCUDA
using Plots,JLD2
using Wavelets, ParameterSchedulers, NNlib, LuxCUDA, JLD2
using Meshes

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_per_triangle.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/dif_points_per_triangle.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/main_loop.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")


function count_zeros_r(arr)
    num_zeros = count(x -> x == 0.0, arr)
    num_entries = length(arr)
    percentt = (num_zeros / num_entries) * 100
    return percentt
end

function infer_model(tstate_glob, model, imagee)
    y_pred, st = Lux.apply(model, CuArray(imagee), tstate_glob.parameters, tstate_glob.states)
    out_sampled_points, tetr_dat = y_pred
    return y_pred
end


radiuss = (Float32(3.5), Float32(3.5), Float32(3.5))
batch_size=3
pad_voxels=true
image_shape=(128, 128, 128)

global const len_get_random_point_in_tetrs_kern=Int(floor(256/3))
global const is_point_per_triangle=true
add_triangle_per_point=is_point_per_triangle


conv5_a = (in, out) -> Lux.Conv((7, 7, 7), in => out, NNlib.gelu, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
conv5_s = (in, out) -> Lux.Conv((7, 7, 7), in => out, NNlib.sigmoid, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)

ch=168
radiuss= (Float32(3.5), Float32(3.5), Float32(3.5))
spacing=(Float32(1.0), Float32(1.0), Float32(1.0))
model=Lux.Chain(conv5_a(ch,ch),conv5_a(ch,ch),conv5_a(ch,ch),conv5_a(ch,ch) ,conv5_s(ch,ch),Points_weights_str(radiuss,batch_size,pad_voxels,image_shape,add_triangle_per_point))
# weights=CuArray(rand(Float32,20,20,20,168,3))
# weights=CuArray(rand(Float32,22,22,22,ch,batch_size))
weights=CuArray(ones(Float32,22,22,22,ch,batch_size))
minimum(weights)
maximum(weights)

dev= gpu_device() 
rng = Random.default_rng()
vjp = Lux.Experimental.ADTypes.AutoZygote()
ps, st = Lux.setup(rng, model) |> dev

opt=Optimisers.Lion(0.000001)

tstate = Lux.Training.TrainState(model, ps, st, opt)

function loss_function_dummy(model, ps, st, data)
    y_pred, st = Lux.apply(model, data, ps, st)
    #y_pred[1] (17, 17, 17, 31, 3, 3)
    #y_pred[2]  (18, 18, 18, 3, 3)
    sv_centers_out=y_pred[1][2:end,2:end,2:end,:,:,:]
    if(is_point_per_triangle)
    sv_centers_expanded = repeat(sv_centers_out, 1, 1, 1, 1, 1,31)
    else
        sv_centers_expanded = repeat(sv_centers_out, 1, 1, 1, 1, 1,7)
    end
    sv_centers_expanded=permutedims(sv_centers_expanded, [1, 2, 3, 6, 4, 5]) #(17, 17, 17, 31, 3, 3)
    b_loss= sv_centers_expanded-y_pred[2]
    b_loss=b_loss.^2
    return mean(b_loss), st, ()
end


gs, loss, stats, tstate = Training.single_train_step!(vjp, loss_function_dummy,weights, tstate)
fin_loss=0.0
for i in 1:400
    gs, loss, stats, tstate = Training.single_train_step!(vjp, loss_function_dummy,weights, tstate)
    fin_loss=loss
end
fin_loss


function fill_tetrahedron_data(tetr_dat, sv_centers, control_points, index)
    center = map(axis -> sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), axis], [1, 2, 3])
    corners = map(corner_num ->
            map(axis -> control_points[Int(tetr_dat[index, corner_num, 1]), Int(tetr_dat[index, corner_num, 2]), Int(tetr_dat[index, corner_num, 3]), Int(tetr_dat[index, corner_num, 4]), axis], [1, 2, 3]), [2, 3, 4])
    corners = [center, corners...]
    return corners
end

function get_tetrahedrons_from_corners(corners)
    points = map(el -> (el[1], el[2], el[3]), corners)
    return Meshes.Tetrahedron(points...)
end
function get_base_triangles_from_corners(corners)
    points = map(el -> (el[1], el[2], el[3]), corners)
    return points[2:end]
end


function get_tetr_for_vis(curr_batch, n, tetr_dat_out, sv_centers_out, control_points_out, tetrs)
    size(sv_centers_out)
    size(sv_centers_out)
    size(control_points_out)
    # tetrs=tetrs[:,:,:,curr_batch]
    sv_centers_out = sv_centers_out[:, :, :, :, curr_batch]
    control_points_out = control_points_out[:, :, :, :, :, curr_batch]
    first_sv_tetrs_a = map(index -> fill_tetrahedron_data(tetrs, Array(sv_centers_out), Array(control_points_out), index), ((get_num_tetr_in_sv()*n)+1):get_num_tetr_in_sv()*(n+2))
    first_sv_tetrs = map(get_tetrahedrons_from_corners, first_sv_tetrs_a)
    first_sv_tetrs_base_triangles = map(get_base_triangles_from_corners, first_sv_tetrs_a)

    # curr=first_sv_tetrs_base_triangles[24:30]
    curr = first_sv_tetrs_base_triangles[1:get_num_tetr_in_sv()]
    return curr
end


sv_centers_out,control_points_out=infer_model(tstate, model, weights)


set_of_svs = initialize_centers_and_control_points(image_shape, radiuss, is_point_per_triangle)
if(is_point_per_triangle)
    sv_centers, control_points, tetrs, dims, plan_tensors = set_of_svs
else
    sv_centers, control_points, tetrs, dims = set_of_svs
end
curr = get_tetr_for_vis(2, 1, tetrs, sv_centers_out, control_points_out, tetrs)

trs=[]
for i in 1:length(curr)
    try
        p1=Meshes.Point(curr[i][1]...)
        p2=Meshes.Point(curr[i][2]...)
        p3=Meshes.Point(curr[i][3]...)
        viz(Meshes.Triangle(p1,p2,p3))
        push!(trs,Meshes.Triangle(p1,p2,p3))
    catch
        println("Error $i")
    end
end

viz(trs, color=1:length(trs))#,alpha=0.5

# viz(Meshes.Triangle(Meshes.Point(1.0, 2.0, 3.0),Meshes.Point(1.0, 4.0, 3.0),Meshes.Point(3.0, 4.0, 3.0)))

a=1

#MethodError: no method matching ntuple(::Meshes.var"#738#739"{…}, ::Unitful.Quantity{…})

# source_arr = CuArray(rand(Float32, image_shape[1],image_shape[2], image_shape[3], 2, batch_size))
# threads_tetr_set, blocks_tetr_set = prepare_for_set_tetr_dat(image_shape, size(tetrs), batch_size)

# tetr_dat_out = call_set_tetr_dat_kern(CuArray(tetrs), CuArray(source_arr), control_points_out, sv_centers_out, threads_tetr_set, blocks_tetr_set, spacing, batch_size)
# CUDA.synchronize()
# num_sv_per_tetr=48*3
# obb=[]
# ind_tetr=7

# ind_tetr=7
# tetr_inf=Array(tetr_dat_out)[(ind_tetr-1)*num_sv_per_tetr+1:ind_tetr*num_sv_per_tetr,:,:,1]
# for i in 1:num_sv_per_tetr
#     push!(obb,Meshes.PolyArea(Meshes.Point(tetr_inf[i,2,1:3]...),Meshes.Point(tetr_inf[i,3,1:3]...),Meshes.Point(tetr_inf[i,4,1:3]...)))
# end 


# cc=collect(1:length(obb))
# viz(obb,color=cc,pointsize=10.2)#  ,alpha=[0.7,0.7,0.7,0.0]  , color=1:length(first_sv_tetrs)



# sv_centers_expanded = repeat(sv_centers_out, 1, 1, 1, 1, 1,31)
# sv_centers_expanded=permutedims(sv_centers_expanded, [1, 2, 3, 6, 4, 5])
# sv_centers_expanded
# sv_centers_out

# sv_centers_expanded[:,:,:,8,:,:]==sv_centers_out















######
# check from real model - saved during training in hdf5 from chain rule
####

check_path="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/debugg"

# Cast train_state to CPU
cpu_train_state = cpu_device()(train_state)
ps=cpu_train_state.parameters
st=cpu_train_state.states
@save "$(check_path)/trained_model.jld2" ps st

# @load "$(check_path)/trained_model.jld2" ps st

# Save the train state to the specified path

h5_path_b = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_gradients.h5"
fb = h5open(h5_path_b, "r")

d_control_points_out = CuArray(read(fb, "d_control_points_out"))
d_control_points_in = CuArray(read(fb, "d_control_points_in"))
d_plan_tensor = CuArray(read(fb, "d_plan_tensor"))
d_weights = CuArray(read(fb, "d_weights"))
d_sv_centers_out = CuArray(read(fb, "d_sv_centers_out"))
plan_tensor = CuArray(read(fb, "plan_tensor"))
weights = CuArray(read(fb, "weights"))
control_points_in = CuArray(read(fb, "control_points_in"))
sv_centers_out = CuArray(read(fb, "sv_centers_out"))
control_points_out = CuArray(read(fb, "control_points_out"))
siz_c=(20, 20, 20, 31, 3)
max_index=8000
blocks=(95, 24, 3)
const len_get_random_point_in_tetrs_kern=85

close(fb)

# percent_zeros = [count_zeros_r(d_weights[:,:,:,i,:]) for i in 1:168]
# percent_zeros = [count_zeros_r(d_weights[:,:,:,:,i]) for i in 1:3]
percent_zeros = [count_zeros_r(d_weights[:,:,:,:,i]) for i in 1:3]

ppp=zeros(6)
for by in 1:24
    for wi in 1:6
        ppp[wi]+=count_zeros_r(d_weights[3:end-3,3:end-3,3:end-3,((by-1)*6)+24+wi,:])
    end
    # weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  5 ,blockIdx().z]
end

ppp=ppp./24

bar(ppp, xlabel="Index", ylabel="Percentage of Zeros", title="Percentage of Zeros in d_weights")

percent_zeros = [count_zeros_r(d_weights[3:end-3,3:end-3,3:end-3,i,:]) for i in 25:168]
bar(percent_zeros, xlabel="Index", ylabel="Percentage of Zeros", title="Percentage of Zeros in d_weights")

percent_zeros = [count_zeros_r(d_weights[i,3:end-5,3:end-3,25:end,:]) for i in 3:17]
bar(percent_zeros, xlabel="Index", ylabel="Percentage of Zeros", title="Percentage of Zeros in d_weights")


count_zeros_r(d_weights[3:end-3,3:end-3,3:end-3,25:end,:])
count_zeros_r(d_control_points_in)

size(d_weights)
# Sum the last 3 dimensions of the 5-dimensional tensor
summed_matrix = sum(d_weights, dims=(3,4, 5))

# Convert the resulting matrix to a 2D array for heatmap
summed_matrix_2d = dropdims(summed_matrix, dims=(3,4, 5))

# for i in 1:17 ; j in 1:17
#     # summed_matrix_2d[i,j]=sum(summed_matrix[i,j,:,:])
# end
summed_matrix_2d=Array(summed_matrix_2d)[3:17,3:17]
summed_matrix_2d=summed_matrix_2d.>0
# Display the resulting matrix as a heatmap
Plots.heatmap(summed_matrix_2d, xlabel="Dimension 1", ylabel="Dimension 2", title="Heatmap of Summed Tensor")

# function count_zeros_r(arr)
#     num_zeros = count(x -> x == 0.0, arr)
#     num_entries = length(arr)
#     percentt = (num_zeros / num_entries) * 100
#     return percentt
# end

# count_zeros_r(d_weights[:,:,:,:,:])


# size(d_weights)



# cpu_train_state = cpu_device()(train_state)
# ps=cpu_train_state.parameters
# st=cpu_train_state.states
# @save path_check_p ps st




# percent_zeros = [count_zeros_r(d_weights[:,:,:,i,:]) for i in 1:168]
# percent_zeros = [count_zeros_r(d_weights[:,:,:,:,i]) for i in 1:3]

# using Plots
# bar(percent_zeros, xlabel="Index", ylabel="Percentage of Zeros", title="Percentage of Zeros in d_weights")

# apt update
# apt install -y mysql-server
#systemctl start mysql.service



# # Use a base image with a minimal Linux distribution
# FROM ubuntu:latest

# # Update the package repository and install necessary packages
# RUN apt-get update && apt-get install -y \
#     wget \
#     lsb-release \
#     gnupg

# # Add MySQL APT repository
# RUN wget https://dev.mysql.com/get/mysql-apt-config_0.8.17-1_all.deb && \
#     dpkg -i mysql-apt-config_0.8.17-1_all.deb && \
#     apt-get update

# # Install MySQL server
# RUN apt-get install -y mysql-server

# # Expose MySQL port
# EXPOSE 3306

# # Start MySQL service
# CMD ["mysqld"]
