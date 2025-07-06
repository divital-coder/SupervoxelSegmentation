using Test, Revise
using PyCall
using Revise
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
# 

using Logging
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")

# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_a.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_b.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern _old.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/sv_centr_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/additional_points_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")


includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights_kerns/get_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/dif_get_weights.jl")
sitk = pyimport_conda("SimpleITK", "simpleitk")
np = pyimport_conda("numpy", "numpy")
#  get_dir_indicies((1, 0, 0),(3.1,3.1,3.1))


# radiuss = (Float32(8.1), Float32(8.3), Float32(8.2))
# directions_indicies, voxel_counts = get_directions_info(radiuss)
# voxel_counts

"""
Create test that will check if the function get_directions_info works correctly by invoking it and then on the basis of the indicies of the tensor that it returns
  populate empty array and set diffrent integer value for each direction, so all points related to given direction has the same integer associated, than using simple itk save this
  array as a 3d image so one could display it in 3d and check if the directions are correct
"""
function save_get_dirs_for_vis()
  tensor = get_directions_info((4, 4, 4))

  dummy = zeros(128, 128, 128)
  dummy_b = ones(128, 128, 128)
  sizz = size(tensor)
  for i in 1:sizz[1]
    for j in 1:sizz[2]

      x, y, z = tensor[i, j, :]
      if x > -1000
        dummy[Int(x + 64), Int(y + 64), Int(z + 64)] = i
      end
    end
  end

  sitk.WriteImage(sitk.GetImageFromArray(UInt16.(dummy)), "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/inferred.nii.gz")
  sitk.WriteImage(sitk.GetImageFromArray(dummy_b), "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/white.nii.gz")
  return tensor
end

tt = save_get_dirs_for_vis()
size(tt)

# map(el->length(el),tt)


function get_test_data()

  # function test_get_weights_from_directions()
  radiuss = (Float32(3.1), Float32(3.3), Float32(3.2))

  # radiuss = (Float32(3.1), Float32(4.3), Float32(4.7))
  spacing = (Float32(1.0), Float32(1.0), Float32(1.0))
  batch_size = 2
  #maximum shared memory we allow to use in the kernel
  max_shared_get_weights = 200
  #for parameters constants
  num_params_exec = 5
  num_channels = 2
  num_directions = 14
  control_points_weights_num = 24
  num_shared_repr = 1
  num_per_shared_mem_spot = 3
  #how many indicies we we will do iteratively - the more the slower it compile but less memory and generally better run
  num_indicies_per_block = 7



  a = 71
  image_shape = (a, a, a, num_channels)


  example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
  sv_centers, control_points, tetrs, dims = example_set_of_svs

  #get image
  image_shape = (a, a, a, 2)
  source_arr = ones(image_shape...)
  source_arr = repeat(source_arr, inner=(1, 1, 1, 1, batch_size))

  #time to flatten sv centers
  flat_sv_centers = reshape(sv_centers, (size(sv_centers, 1) * size(sv_centers, 2) * size(sv_centers, 3), size(sv_centers, 4)))

  directions_indicies, voxel_counts = get_directions_info(radiuss)
  #initialize parameters


  ds = size(directions_indicies)
  padded_dim = Int(ceil(ds[2] / num_indicies_per_block) * num_indicies_per_block)
  padded_ds = zeros(Int64, ds[1], padded_dim, ds[3])
  padded_ds[:, 1:ds[2], :] = directions_indicies
  directions_indicies = padded_ds
  directions_indicies = Int32.(directions_indicies)
  max_len_dir = size(directions_indicies, 2)



  #we need to accumulate information from diffrent sets of parameters (num_params_exec sets of parameters )
  #and all of the points in each direction while we are doing num_indicies_per_block at once so we will partly 
  #solve this issue by using shared memory and storing the results in shared memory and then summing them up - reduction
  # but we have limits of size of shared memory per block so we may need to add additional dimension to accomodate for it

  #how many threads need to execute he same direction of diffrent parameter sets of the same sv
  times_exec_per_dir_per_param = ceil(max_len_dir / num_indicies_per_block)




  param_matrix_a = ones(Float32, num_params_exec, num_channels, num_directions, padded_dim + 1, num_shared_repr, num_per_shared_mem_spot)

  #!!!! it is important that last index will be always zero as it is used for padding
  for i in eachindex(voxel_counts)
    param_matrix_a[:, :, i, (voxel_counts[i]+1):end, :, :] .= 0.0
  end

  conv_kernels = ones(Float32, 3, 3, 3, num_channels, num_params_exec, num_directions)
  out_summarised = zeros(Float32, batch_size, num_channels, size(flat_sv_centers, 1), num_shared_repr * num_params_exec, num_directions)



  #calculating required padding
  unique_directions = sort(unique(directions_indicies))
  min_direction = unique_directions[2]
  max_direction = unique_directions[end]
  min_sv_centers = minimum(flat_sv_centers)
  max_sv_centers = maximum(flat_sv_centers)

  beg_axis_pad = (Int(ceil(abs(min_sv_centers + min_direction))) + 1) + 2
  end_axis_pad = (Int(ceil(abs(maximum(collect(image_shape)) - (max_sv_centers + max_direction)))) + 1) + 2
  pad_size = beg_axis_pad + end_axis_pad
  curr_image = ones(image_shape[1] + pad_size, image_shape[2] + pad_size, image_shape[3] + pad_size, image_shape[4], batch_size)
  curr_image[beg_axis_pad:end-(end_axis_pad+1), beg_axis_pad:end-(end_axis_pad+1), beg_axis_pad:end-(end_axis_pad+1), :, :] = source_arr

  #we iterate over all of the parameters and reiterations of the same parameter sets on y dimension of the threads
  #but we may divide it into diffrent spots in memory if there is too much to squeeze in single thread block 
  #if we have such situation place_for_multi_param_needed is bigger than 1
  threads_y = Int(ceil(times_exec_per_dir_per_param))
  threads_x = Int(ceil(max_shared_get_weights / threads_y))
  max_index = size(flat_sv_centers, 1)
  threads = (threads_x, threads_y)

  #calculating the size for params_reduction
  for_red = 2
  for ii in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for_red += Int(ceil(threads_y / ((ii) * 2)))
  end




  true_blocks_x = Int(ceil(max_index / threads[1]))
  #blocks x get index of sv center but take into account also times_exec_per_dir
  #blocks y - indicate which direction we are analyzing currently
  #blocks z - indicate which batch and channel we are on
  # blocks = (true_blocks_x, (num_directions*num_params_exec)+12, batch_size * num_channels)
  blocks = (true_blocks_x, (num_directions * num_params_exec), batch_size * num_channels)
  out_summarised = CuArray(out_summarised)

  place_for_multi_param_needed = -100 #legacy code
  times_exec_per_dir = -100 #legacy code
  return times_exec_per_dir_per_param, place_for_multi_param_needed, times_exec_per_dir, a, max_len_dir, num_indicies_per_block, true_blocks_x, num_shared_repr, num_directions, num_params_exec, voxel_counts, flat_sv_centers, curr_image, directions_indicies, param_matrix_a, conv_kernels, num_channels, control_points_weights_num, out_summarised, a, max_index, beg_axis_pad, threads, blocks
end


"""
time for some tests to check if the function works correctly
1) we set all weights to 1 and all image to 1, so we should basically for each sv and each direction  that is not on edges 
to be the same value as voxel_counts in respective spots in out_summarised
"""
function test1()
  times_exec_per_dir_per_param, place_for_multi_param_needed, times_exec_per_dir, a, max_len_dir, num_indicies_per_block, true_blocks_x, num_shared_repr, num_directions, num_params_exec, voxel_counts, flat_sv_centers, curr_image, directions_indicies, param_matrix_a, conv_kernels, num_channels, control_points_weights_num, out_summarised, a, max_index, beg_axis_pad, threads, blocks = get_test_data()

  #!!!! it is important that last index will be always zero as it is used for padding
  for i in eachindex(voxel_counts)
    param_matrix_a[:, :, i, (voxel_counts[i]+1):end, :, :] .= 0.0
  end
  @cuda threads = threads blocks = blocks get_weights_from_directions(
    CuArray(flat_sv_centers),
    CuArray(curr_image),
    CuArray(directions_indicies),
    CuArray(param_matrix_a),
    CuArray(conv_kernels), num_channels,
    out_summarised, max_index, beg_axis_pad, num_indicies_per_block, num_directions, CuArray(voxel_counts))

  # curr_image=curr_image[beg_axis_pad:end-(end_axis_pad+1),beg_axis_pad:end-(end_axis_pad+1),beg_axis_pad:end-(end_axis_pad+1),:,:]
  CUDA.synchronize()


  out_summarised = Array(out_summarised)

  num_sv = 300
  flat_sv_centers[num_sv, :]

  #  out_summarised = zeros(Float32, batch_size, num_channels, size(flat_sv_centers, 1), num_shared_repr*num_params_exec, num_directions)

  curr = out_summarised[1, 1, num_sv, :, :]

  # repeated_voxel_counts = [repeat([x], 3) for x in voxel_counts]
  # repeated_voxel_counts = (vcat(repeated_voxel_counts...) .* (num_params_exec * 28))
  @test isapprox(mean(permutedims(curr, (2, 1))), 28)

end

function test2()
  times_exec_per_dir_per_param, place_for_multi_param_needed, times_exec_per_dir, a, max_len_dir, num_indicies_per_block, true_blocks_x, num_shared_repr, num_directions, num_params_exec, voxel_counts, flat_sv_centers, curr_image, directions_indicies, param_matrix_a, conv_kernels, num_channels, control_points_weights_num, out_summarised, a, max_index, beg_axis_pad, threads, blocks = get_test_data()

  #### checking with single 2 in convolutions
  out_summarised = CuArray(out_summarised)
  conv_kernels
  conv_kernels[2, 2, 2, :, :, :] .= 2
  #  conv_kernels=ones(Float32, 3,3,3,num_channels,num_params_exec,num_directions).*2

  #!!!! it is important that last index will be always zero as it is used for padding
  for i in eachindex(voxel_counts)
    param_matrix_a[:, :, i, (voxel_counts[i]+1):end, :, :] .= 0.0
  end
  @cuda threads = threads blocks = blocks get_weights_from_directions(
    CuArray(flat_sv_centers),
    CuArray(curr_image),
    CuArray(directions_indicies),
    CuArray(param_matrix_a),
    CuArray(conv_kernels), num_channels,
    out_summarised, max_index, beg_axis_pad, num_indicies_per_block, num_directions, CuArray(voxel_counts))

  out_summarised = Array(out_summarised)
  num_sv = 300
  flat_sv_centers[num_sv, :]
  curr = out_summarised[1, 1, num_sv, :, :]

  # repeated_voxel_counts = [repeat([x], 3) for x in voxel_counts]
  # repeated_voxel_counts = (vcat(repeated_voxel_counts...) .* (num_params_exec * 28))
  @test isapprox(mean(permutedims(curr, (2, 1))), 29)

end

### changed addition to 2
function test3()
  times_exec_per_dir_per_param, place_for_multi_param_needed, times_exec_per_dir, a, max_len_dir, num_indicies_per_block, true_blocks_x, num_shared_repr, num_directions, num_params_exec, voxel_counts, flat_sv_centers, curr_image, directions_indicies, param_matrix_a, conv_kernels, num_channels, control_points_weights_num, out_summarised, a, max_index, beg_axis_pad, threads, blocks = get_test_data()

  param_matrix_a
  param_matrix_a[:, :, :, :, :, 2] .= 7
  out_summarised = CuArray(out_summarised)
  conv_kernels

  conv_kernels = ones(Float32, 3, 3, 3, num_channels, num_params_exec, num_directions)
  #!!!! it is important that last index will be always zero as it is used for padding
  for i in eachindex(voxel_counts)
    param_matrix_a[:, :, i, (voxel_counts[i]+1):end, :, :] .= 0.0
  end
  @cuda threads = threads blocks = blocks get_weights_from_directions(
    CuArray(flat_sv_centers),
    CuArray(curr_image),
    CuArray(directions_indicies),
    CuArray(param_matrix_a),
    CuArray(conv_kernels), num_channels,
    out_summarised, max_index, beg_axis_pad, num_indicies_per_block, num_directions, CuArray(voxel_counts))

  out_summarised = Array(out_summarised)
  num_sv = 300
  flat_sv_centers[num_sv, :]
  curr = out_summarised[1, 1, num_sv, :, :]


  @test isapprox(mean(permutedims(curr, (2, 1))), 34)

end

function test4()

  times_exec_per_dir_per_param, place_for_multi_param_needed, times_exec_per_dir, a, max_len_dir, num_indicies_per_block, true_blocks_x, num_shared_repr, num_directions, num_params_exec, voxel_counts, flat_sv_centers, curr_image, directions_indicies, param_matrix_a, conv_kernels, num_channels, control_points_weights_num, out_summarised, a, max_index, beg_axis_pad, threads, blocks = get_test_data()

  #!!!! it is important that last index will be always zero as it is used for padding
  for i in eachindex(voxel_counts)
    param_matrix_a[:, :, i, (voxel_counts[i]+1):end, :, :] .= 0.0
  end
  #second direction second index first parameter third parameter set
  param_matrix_a[3, :, 2, 2, :, 1] .= 2

  @cuda threads = threads blocks = blocks get_weights_from_directions(
    CuArray(flat_sv_centers),
    CuArray(curr_image),
    CuArray(directions_indicies),
    CuArray(param_matrix_a),
    CuArray(conv_kernels), num_channels,
    out_summarised, max_index, beg_axis_pad, num_indicies_per_block, num_directions, CuArray(voxel_counts))

  # curr_image=curr_image[beg_axis_pad:end-(end_axis_pad+1),beg_axis_pad:end-(end_axis_pad+1),beg_axis_pad:end-(end_axis_pad+1),:,:]
  CUDA.synchronize()


  out_summarised = Array(out_summarised)

  num_sv = 300
  flat_sv_centers[num_sv, :]

  #  out_summarised = zeros(Float32, batch_size, num_channels, size(flat_sv_centers, 1), num_shared_repr*num_params_exec, num_directions)

  curr = out_summarised[1, 1, num_sv, :, :]

  @test curr[3, 2] == ((voxel_counts[2]) * 28) + 27


end


function test5()

  times_exec_per_dir_per_param, place_for_multi_param_needed, times_exec_per_dir, a, max_len_dir, num_indicies_per_block, true_blocks_x, num_shared_repr, num_directions, num_params_exec, voxel_counts, flat_sv_centers, curr_image, directions_indicies, param_matrix_a, conv_kernels, num_channels, control_points_weights_num, out_summarised, a, max_index, beg_axis_pad, threads, blocks = get_test_data()

  #!!!! it is important that last index will be always zero as it is used for padding
  for i in eachindex(voxel_counts)
    param_matrix_a[:, :, i, (voxel_counts[i]+1):end, :, :] .= 0.0
  end
  #second direction second index first parameter third parameter set
  size(conv_kernels)
  size(param_matrix_a)

  #first channel second parameter set third direction
  conv_kernels = Array(conv_kernels)
  conv_kernels[2, 2, 2, 1, 2, 3] = 3
  conv_kernels = CuArray(conv_kernels)

  @cuda threads = threads blocks = blocks get_weights_from_directions(
    CuArray(flat_sv_centers),
    CuArray(curr_image),
    CuArray(directions_indicies),
    CuArray(param_matrix_a),
    CuArray(conv_kernels), num_channels,
    out_summarised, max_index, beg_axis_pad, num_indicies_per_block, num_directions, CuArray(voxel_counts))

  # curr_image=curr_image[beg_axis_pad:end-(end_axis_pad+1),beg_axis_pad:end-(end_axis_pad+1),beg_axis_pad:end-(end_axis_pad+1),:,:]
  CUDA.synchronize()


  out_summarised = Array(out_summarised)

  num_sv = 300
  flat_sv_centers[num_sv, :]

  #  out_summarised = zeros(Float32, batch_size, num_channels, size(flat_sv_centers, 1), num_shared_repr*num_params_exec, num_directions)

  curr = out_summarised[1, 1, num_sv, :, :]

  @test curr[2, 3] == (30 * voxel_counts[3])


end



times_exec_per_dir_per_param, place_for_multi_param_needed, times_exec_per_dir, a, max_len_dir, num_indicies_per_block, true_blocks_x, num_shared_repr, num_directions, num_params_exec, voxel_counts, flat_sv_centers, curr_image, directions_indicies, param_matrix_a, conv_kernels, num_channels, control_points_weights_num, out_summarised, a, max_index, beg_axis_pad, threads, blocks = get_test_data()

print("\n ** directions_indicies $(size(directions_indicies)) flat_sv_centers $(size(flat_sv_centers))  param_matrix_a $(size(param_matrix_a))  conv_kernels $(size(conv_kernels))  out_summarised $(size(out_summarised)) ** \n")

threads

size(param_matrix_a)
threads[2] * 7


const global x_dim_sh_block = threads[1]
const global y_dim_sh_block = threads[2]




test1()

test2()

test3()




# test4()

# threads = (x_dim_sh_block, 14)
# true_blocks_x = Int(ceil(max_index / threads[1]))
# blocks_per_dir = Int(ceil(max_len_dir / num_indicies_per_block))
# blocks = (true_blocks_x * blocks_per_dir, num_params_exec, batch_size * num_channels)


# blockIdxv = blocks[1]
# true_blocks_x
# blockIdxv = 311

# ((Int(floor((blockIdxv / true_blocks_x))) * num_indicies_per_block) + 1)
# (Int(ceil((blockIdxv / true_blocks_x))) * num_indicies_per_block)

max_len_dir = size(directions_indicies, 2)


out_summarised = CuArray(out_summarised)
conv_kernels


#!!!! it is important that last index will be always zero as it is used for padding
for i in eachindex(voxel_counts)
  param_matrix_a[:, :, i, voxel_counts[i]:end, :, :] .= 0.0
end
@cuda threads = threads blocks = blocks get_weights_from_directions(
  CuArray(flat_sv_centers),
  CuArray(curr_image),
  CuArray(directions_indicies),
  CuArray(param_matrix_a),
  CuArray(conv_kernels), num_channels,
  out_summarised, max_index, beg_axis_pad, num_indicies_per_block, num_directions, CuArray(voxel_counts))

CUDA.synchronize()



directions_indicies_size = size(directions_indicies)

d_flat_sv_centers = CuArray(zeros(Float32, size(flat_sv_centers)...))
d_source_arr = CuArray(zeros(Float32, size(curr_image)...))
d_directions_indicies = CuArray(zeros(Int32, size(directions_indicies)...))
d_param_matrix_a = CuArray(zeros(Float32, size(param_matrix_a)...))
d_conv_kernels = CuArray(zeros(Float32, size(conv_kernels)...))
d_out_summarised = CuArray(ones(Float32, size(out_summarised)...))
d_voxel_counts = CuArray(zeros(Float32, size(voxel_counts)...))

flat_sv_centers = CuArray(flat_sv_centers)
source_arr = CuArray(Float32.(rand(Float32, size(curr_image))))
# source_arr = CuArray(Float32.(curr_image))
directions_indicies = CuArray(directions_indicies)
param_matrix_a = CuArray(rand(Float32, size(param_matrix_a)))
conv_kernels = CuArray(rand(Float32, size(conv_kernels)))
out_summarised = CuArray(out_summarised)
voxel_counts = CuArray(Float32.(voxel_counts))

@cuda threads = threads blocks = blocks get_weights_from_directions_deff(flat_sv_centers, d_flat_sv_centers,
  source_arr, d_source_arr,
  directions_indicies, d_directions_indicies,
  param_matrix_a, d_param_matrix_a,
  conv_kernels, d_conv_kernels, num_channels,
  out_summarised, d_out_summarised, max_index, beg_axis_pad, num_indicies_per_block, num_directions, voxel_counts, d_voxel_counts)

CUDA.synchronize()

mean(d_source_arr)
mean(d_param_matrix_a)


batch_sizee = 2
call_get_weights_from_directions(batch_sizee, num_channels, flat_sv_centers, num_params_exec, num_directions, threads, blocks, source_arr, CuArray(directions_indicies), param_matrix_a, conv_kernels, max_index, beg_axis_pad, num_indicies_per_block, CuArray(voxel_counts))

batch_sizee # 2
num_channels #2
flat_sv_centers # (1210, 3)
size(flat_sv_centers)#(1210, 3)
threads# (14,15)
num_params_exec # 5
size(source_arr) #(88, 88, 88, 2, 2)
size(directions_indicies)#(14, 105, 3)
size(param_matrix_a)#(5, 2, 14, 106, 1, 3)
size(conv_kernels)#(3, 3, 3, 2, 5, 14)
max_index#1210
beg_axis_pad#8
num_indicies_per_block#7
voxel_counts #  30.0 30.0 ... 103.0 103.0


CUDA.synchronize()

res, pullbackk = ChainRulesCore.rrule(call_get_weights_from_directions, batch_sizee, num_channels, flat_sv_centers, num_params_exec, num_directions, threads, blocks, source_arr, directions_indicies, param_matrix_a, conv_kernels, max_index, beg_axis_pad, num_indicies_per_block, voxel_counts)

pullbackk(out_summarised)


#############
using CUDA
function get_modified_part(param_matrix_a, i, voxel_counts)
  dims = size(param_matrix_a)
  CUDA.@allowscalar curr_vox_len = voxel_counts[i]
  part = param_matrix_a[:, :, i, 1:curr_vox_len, :, :]
  zeros_part = CUDA.zeros(eltype(param_matrix_a), dims[1], dims[2], dims[4] - curr_vox_len, dims[5], dims[6])
  res = cat(part, zeros_part, dims=3)
  res_size = size(res)
  res = reshape(res, (res_size[1], res_size[2], 1, res_size[3], res_size[4], res_size[5]))
  return res
end



# Define the shapes and padding sizes
image_shape = (64, 64, 64)  # Example shape
pad_size = 2
num_channels = 3
batch_size = 4
beg_axis_pad = 1
end_axis_pad = 1

# Create the source array
source_arr = CuArray(rand(Float32, image_shape[1], image_shape[2], image_shape[3], num_channels, batch_size))

# Create the initial padded array with zeros
curr_image = CuArray(zeros(Float32, image_shape[1] + pad_size, image_shape[2] + pad_size, image_shape[3] + pad_size, num_channels, batch_size))

# Pad the source array
padded_image = cat(
  CUDA.zeros(Float32, beg_axis_pad, size(source_arr, 2), size(source_arr, 3), num_channels, batch_size),
  source_arr,
  CUDA.zeros(Float32, end_axis_pad, size(source_arr, 2), size(source_arr, 3), num_channels, batch_size);
  dims=1
)

padded_image = cat(
  CUDA.zeros(Float32, size(padded_image, 1), beg_axis_pad, size(padded_image, 3), num_channels, batch_size),
  padded_image,
  CUDA.zeros(Float32, size(padded_image, 1), end_axis_pad, size(padded_image, 3), num_channels, batch_size);
  dims=2
)

padded_image = cat(
  CUDA.zeros(Float32, size(padded_image, 1), size(padded_image, 2), beg_axis_pad, num_channels, batch_size),
  padded_image,
  CUDA.zeros(Float32, size(padded_image, 1), size(padded_image, 2), end_axis_pad, num_channels, batch_size);
  dims=3
)

# Assign the padded image to curr_image
curr_image = padded_image

source_arr == curr_image[2:end-1, 2:end-1, 2:end-1, :, :]
# Verify the shape
println(size(curr_image))

aa = CuArray(ones(Float32, 5, 5, 5, 2, 3))
bb = Float32.(CuArray(falses(5, 5, 5, 2, 3)))