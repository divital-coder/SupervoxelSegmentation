using Pkg
using ChainRulesCore, Zygote, CUDA, Enzyme
# using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote, LLVMLoopInfo
using LinearAlgebra
using Revise
using Base.Threads

# Newest idea is to simplify it futher get a separate convolution weights per point in sv area 
# have a fixed area like 5x5x5 or 6x6x6 and work on every second point in the area at once
# each point would have diffrent convolution weights so we will have for now sth akin to strided convolution 
# next we will apply non linearity save to shared memory 
# next we will do a parametrarized multiplication and addition between all entries in shared memory 
# and in the end parametrirized reduction, we will do reduction couple times and save it in couple entries in global memory
# we need to execute it multiple times with diffrent parameters as we need to have in the end suffiient data for 24 parameters
#   for get points from weights 

# krowa we can instead of just adding the info from each point in each get_directions_info
# reduce the information binarly so as we are now in the and adding all up by adding every second thread
# we can parametrarize this process by not only adding but also taking the previous value and next value 
# and in a prametrarized way combining them so important idea is that given the convolutions  
# had found some edge etc we need to keep the information about relative value of the points  
# best would be to combine in by parametrirized multiplication and addition each pair of the values 
# and we will attempt it as we have loaded each point in the shared memory and we can compare it with all other
# on x dimension of thread block and then combine them futher still in parametrarized way via binary reduction

"""
goal is to get information about the image that is stored in diffrent channels (we should have image and its wavelet transform /transforms at first channels)
we need also to keep information about directions so how does values are changing in a given direction - classic convolutions and max pooling may loose this info
So 1) in x we will work on a lines of length 2r+1 in some direction - diffrent directions will be present in diffrent y coords of the thread block
2) for each point we will do the 3by 3 or 5 by 5 convolution - parameters of this will be stored in a parameter array in global memory  (it will lead to a lot of recomputations but will
  reduce memory consumption )
  a) apply convolution by 3 nested loop we just multiply and add
  b) apply non linearity like relu
  c) multiply by parameter matrix for given kernel to shorten the representation of the line
  d) apply non linearity 
  e) add atomically to global memory represenation 
3)then we have all of the directions and we need to apply nonlinearities and mlp mixers and then add to global memory via atomic add 
  (atomic as diffent blocks will do diffrent paameters; but may work on the same entry) ; probably will be good also to get separate parametrization of each of the channles we will use for each 
  final weight for getting the point location
  a) we have some matrix that is representation of a line on one dimension and on other dimension number of directions
  b) now we multiply by some matrix in order to reduce size of it 
  c) apply non linearity
  d) add atomicaly to global memory represantation on global memory of given 
4) we need mlp mixing between channels for single sv and between diffrent svs to exchange information between channels and in some neighberhood; first mlp mixing is quite straightforward second
  can be probably done in very similar way as in step 2 so look in all direction make convolutions.... 

"""

"""
@workspace You are Julia language expert

first one need to get all the directions of the lines we want to work on those lines will be of diffrent lengths and will be in diffrent directions and the radius may be diffrent
  for each axis , also oblique radius will be longer then non oblique radius morover to cover more space we will take more points into consideration the more distant we are from the center
    so we will basically have 2 cones that will be attached by their apex to the center of the supervoxel and their base will be at the end of the supervoxel and a bit futher.
    Hovewer we need to remember that we need additional parametrization of this side directions of the cone (so all that is in cone but not in line) when we will project them to the line
    hence we will need also to calculate the required size of parametrization for this 
All required information will be calculated before running the kernel and will be stored in global memory as a tuple 

We want to generate the 3 dimensional tensor where the first dimension would indicate which direction we are currently working on and a second dimension would indicate the current iteration of a thread in the third dimension there would be indicated in x,y,z coordinates relative to the current sv_center point. So for example we are iterating over direction 1 we look into first entry in first dimension then in loop we iterate over second dimension to look through all relevant voxels for this direction we know which voxel to choose on the basis of the index differences in third dimension for example first voxel in direction can be (-1,-1,-1), then (-2,-2,-2) etc. 
We want to have length of the direction be related to seize of supervoxel in given direction - the size is controlled by the tuple "radiuss" this tuple has radius in consecutively x,y,z direction. so the length of direction need to be calculated on the basis of this , the length for axis oriented directions will be just radius plus 1 in case of the oblique directions one need to calculate it from pythagorean theorem. 
Moreover the direction is a cone not a line so every second voxel in the direction we add the indices in both directions perpendicular to the main direction. So for example at first index in direction we look just at one voxel in a plane perpendicular to direction, at third voxels we look at 5 voxels at plane perpendicular to direction  as we have 1 voxel in each direction perpendiclar to the main direction plus center voxel in the direction line. at 5th voxel we should have 12 xoxels in perpendicular plane to diraction as we have voxel from all sides of the centre so 9 plus voxels with distant 2 in each axis perpendicular to direction. 
As the super voxel is not isovolumetric directions may not be of the same length - hence we need to initialize 3 dimensional tensor with second dimension big enough to store all indicies for the longest cone and pad the indicies of shorter directions with -1000 ; additionally return how many voxels are analyzed per direction. 
The second dimension of the tensor are diffrent directions , there should be 6 axis oriented directions so up,down,left,right,anterior,posterior and 8 oblique ones that would be oriented on a corners of the cube ; where sv center so index (0,0,0) is in the center of the cube and the length of the cube edge in each axis is related to 2*radius+2. Hence the second dimension of the result should have length of 14.

      """


"""
looking for all indicies that are in the distance of max_dist from the curr_index
"""


function euclidean_distance(p1::Tuple{Int, Int, Int}, p2::Tuple{Int, Int, Int})
  return sqrt(sum((p1 .- p2) .^ 2))
end

# voxel_radius=(5,5,5)


function parse_to_array(vector_of_vectors)
  # Initialize an empty array with the desired shape (14, 95, 3)
  parsed_array = Array{Int, 3}(undef, length(vector_of_vectors), length(vector_of_vectors[1]), 3)
  
  # Populate the array
  for i in 1:length(vector_of_vectors)
      for j in 1: length(vector_of_vectors[1])
          parsed_array[i, j, :] = collect(vector_of_vectors[i][j])
      end
  end
  
  return parsed_array
end

function get_directions_info(voxel_radius)

# Example usage
directions = [
  (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),  # Axis-aligned directions
  (1, 1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1), (1, 1, -1), (-1, -1, 1), (1, -1, -1), (-1, 1, 1)  # Oblique directions
]
points=map(t-> (t[1]*voxel_radius[1], t[2]*voxel_radius[2],t[3]*voxel_radius[3]),directions)

cartesian_coords = [(x, y, z) for x in -voxel_radius[1]:voxel_radius[1], y in -voxel_radius[2]:voxel_radius[2], z in -voxel_radius[3]:voxel_radius[3]]
cartesian_coords = vcat(cartesian_coords...)

# result_matrix = assign_cartesian_coordinates(cartesian_coords, points)
# println(result_matrix)

  num_points = length(points)
  num_coords = length(cartesian_coords)
  
  # Step 1: Calculate distances and initial assignment
  assignments = map(i->[],range(1,num_points))
  points[1]
  for coord in cartesian_coords
  # oord =cartesian_coords[1]
      distances = [euclidean_distance(coord, point) for point in points]
      closest_point = argmin(distances)
      push!(assignments[closest_point], coord)
  end
  
  # Step 2: Analyze distribution
  counts = [length(assignments[i]) for i in 1:num_points]
  target_count = div(num_coords, num_points)
  
  # Step 3: Reassign for equal distribution
  while(maximum([length(assignments[i]) for i in 1:num_points])>(minimum([length(assignments[i]) for i in 1:num_points])+1) )
      for j in 1:num_points
              # Find a neighbor to reassign
              coord=assignments[j][rand(1:length(assignments[j]))]
                  neighbors = [
                      (coord[1]+1, coord[2], coord[3]), (coord[1]-1, coord[2], coord[3]),
                      (coord[1], coord[2]+1, coord[3]), (coord[1], coord[2]-1, coord[3]),
                      (coord[1], coord[2], coord[3]+1), (coord[1], coord[2], coord[3]-1)
                  ]
                  neighbor=neighbors[rand(1:6)]
                    for n_i in 1:num_points
                      if neighbor in assignments[n_i]
                        if(length(assignments[n_i])>length(assignments[j]))
                            # deleteat!(assignments[n_i], findfirst(x -> x == coord, assignments[n_i]))
                            assignments[n_i]=filter(x -> x != neighbor, assignments[n_i])
                            push!(assignments[j], neighbor)
                            break
                        end  
                      end
                    end 
          end
        end
      
      return parse_to_array(assignments)
      end

      # voxel_radius=(5,5,5)
      # assignments=assign_cartesian_coordinates(voxel_radius)











"""

The main kernel for learning the weight for each control point in the super voxel each thread will work on single direction and single supervoxel 
  Directions will be stored in a tensor that we got from get_directions_info function where the first dimension is the direction and the second dimension 
    is the index of the point in the direction and in third dimension we have change in x,y and z coordinates that indicate the point in the direction relative to index of the supervoxel center
  The kernel will do the following steps:
  1) each thread will work on the direction depending on its y index and then iterate o0f the lenght of the second dimension of supplied tensor
  in each iteration it will add the x,y,z coordinates to the index and apply 3x3x3 convolution of this point with the kernel that is stored in global memory and which location 
  is indicated by the block index y; the result of the convolution will be stored in the shared memory then immidiately apply multiplication by the parameter entry that is stored in global memory
  and that you can find on the basis of the index of the thread in the second dimension of the tensor and the y index of the block; 
  then apply nonlinearity and store the result in the shared memory - you will apply 3 diffrent parameters to every single point after convolution and store it in 3 diffrent spots in shared memory
  as each thread will have assigned 4 entries in shared memory
  2) then iterate over all points and add information to those 3 entries of the supervoxels that are in shared memory for given thread at each step; remember to iterate over indicies only if
  they are not -1000 as this is the padding for shorter directions
  3) apply multiplication by parameter parameter addition and nonlinearity while adding entries from 2 directions that are next to each other in y direction of a thread block
    then do the same evry fourth direction and then every 8th direction and so on until you reach the end of the thread block and you will reduce it to 1 entry in shared memory ; perform such operation 3 times 
    with diffrent parameters 
  4) repeat step 3 for each from 24 weights that are required by apply weights to control points functions, each time you will have diffrent parameters for each weight
    save the resulting value by adding it atomically to the global memory of the supervoxel that one will find by x,y,z coordinates of the supervoxel center and the index of the weight 
    of control points that you are currently working on

Work step by step implementing each step as a function - implement all of the code. Do not overthink and act logically.    
"""



"""
before we need initialize direction_indicies using get_directions_info function
We also assume flattened sv centers indicies to make indexing easier 
we will iterate over directions using y dimension so y dimension of the block need to be 14
block y dimension will be used to switch between diffrent instatntiations of the kernels weights and 
parametrization matricies - as in that way we will basically increase number of parameters 
without significant increase in memory consumption
source_arr- is the image we are working on it may be that we have more than one channel and more than
one image in the batch so we need to take both into the consideration using block_id_z

param_matrix_a will be 6 dimensional

  1) blockid.y - so how many sets of parameters we want to use  
  2) channels - number of channels for input image
  3) direction - number of directions we are working on (default 14)
  4) point_index - number of points in the biggest direction  
  5) number of shared memory spots we are using to summarise direction (default 3)
  6) number of parameters we are using per point and per shared memory spot
  
param_matrix_b will be similar to param_matrix_a but will be used for the aggregation and dimension 4 will be control point weight index specific

out_summarised
  1) batch index
  2) channel index
  3) supervoxel index in flattened array
  4) control point weight index

"""

function get_weights_from_directions(
        flat_sv_centers,
        source_arr,
        directions_indicies,
        param_matrix_a,
        conv_kernels, channels_num,
        out_summarised, max_index::Int64,
        beg_axis_pad,  num_indicies_per_block
        ,num_directions
        ,voxel_counts)
        
        if (threadIdx().x + (blockIdx().x * CUDA.blockDim_x())) > max_index
                return nothing
                 
        end

        #load kernels to shared memory
        shared_arr_kern = CuStaticSharedArray(Float32, (3, 3, 3))
        shared_arr = CuStaticSharedArray(Float32, (x_dim_sh_block, y_dim_sh_block, 1))

        #on x dimension we are iterating over sv centers
        #threads on y are iterating on the same sv center the same parameter set and channel - divided just becouse of enzyme compilation issues
        #blocks on y encode direction by (( blockIdx().y%num_directions )+1) and which parameter set by (Int(ceil(blockIdx().y/num_directions)))
        # blocks z encode batch and channel


        #variable storing info about current work done
        #setting shared memory to zero
        shared_arr[threadIdx().x, threadIdx().y, 1] = 0.0

        if (threadIdx().x == 1 &  threadIdx().y == 1 )
 
          shared_arr_kern[1, 1, 1] = (conv_kernels[1, 1, 1, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[1, 1, 2] = (conv_kernels[1, 1, 2, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[1, 1, 3] = (conv_kernels[1, 1, 3, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[1, 2, 1] = (conv_kernels[1, 2, 1, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[1, 2, 2] = (conv_kernels[1, 2, 2, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[1, 2, 3] = (conv_kernels[1, 2, 3, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[1, 3, 1] = (conv_kernels[1, 3, 1, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[1, 3, 2] = (conv_kernels[1, 3, 2, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[1, 3, 3] = (conv_kernels[1, 3, 3, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[2, 1, 1] = (conv_kernels[2, 1, 1, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[2, 1, 2] = (conv_kernels[2, 1, 2, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[2, 1, 3] = (conv_kernels[2, 1, 3, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[2, 2, 1] = (conv_kernels[2, 2, 1, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[2, 2, 2] = (conv_kernels[2, 2, 2, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[2, 2, 3] = (conv_kernels[2, 2, 3, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[2, 3, 1] = (conv_kernels[2, 3, 1, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[2, 3, 2] = (conv_kernels[2, 3, 2, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[2, 3, 3] = (conv_kernels[2, 3, 3, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[3, 1, 1] = (conv_kernels[3, 1, 1, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[3, 1, 2] = (conv_kernels[3, 1, 2, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[3, 1, 3] = (conv_kernels[3, 1, 3, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[3, 2, 1] = (conv_kernels[3, 2, 1, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[3, 2, 2] = (conv_kernels[3, 2, 2, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[3, 2, 3] = (conv_kernels[3, 2, 3, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[3, 3, 1] = (conv_kernels[3, 3, 1, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[3, 3, 2] = (conv_kernels[3, 3, 2, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
         
          shared_arr_kern[3, 3, 3] = (conv_kernels[3, 3, 3, Int(CUDA.blockIdx().z % channels_num)+1,(Int(ceil(blockIdx().y/num_directions)))
          ,(( blockIdx().y%num_directions )+1) ])
        end
        sync_threads()
        
          shared_arr[threadIdx().x,threadIdx().y, 1]=(((max(0,(((source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,1+2])
        )*(param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1,1])+
        param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1,2] ))
        * (param_matrix_a[(Int(ceil(blockIdx().y/num_directions))),Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+1),1,3])) + (max(0,(((source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,1+2])
        )*(param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1,1])+
        param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1,2] ))
        * (param_matrix_a[(Int(ceil(blockIdx().y/num_directions))),Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+2),1,3])) + (max(0,(((source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,1+2])
        )*(param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1,1])+
        param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1,2] ))
        * (param_matrix_a[(Int(ceil(blockIdx().y/num_directions))),Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+3),1,3])) + (max(0,(((source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,1+2])
        )*(param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1,1])+
        param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1,2] ))
        * (param_matrix_a[(Int(ceil(blockIdx().y/num_directions))),Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+4),1,3])) + (max(0,(((source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,1+2])
        )*(param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1,1])+
        param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1,2] ))
        * (param_matrix_a[(Int(ceil(blockIdx().y/num_directions))),Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+5),1,3])) + (max(0,(((source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,1+2])
        )*(param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1,1])+
        param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1,2] ))
        * (param_matrix_a[(Int(ceil(blockIdx().y/num_directions))),Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+6),1,3])) + (max(0,(((source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[-1+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[0+2,1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(-1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,-1+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(0)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,0+2,1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(-1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,-1+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(0)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,0+2])
         + (source_arr[ Int(floor((directions_indicies[(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),1])+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),2]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),2]+(1)+beg_axis_pad))
        ,Int(floor(directions_indicies[(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),3]+flat_sv_centers[(threadIdx().x + (blockIdx().x * CUDA.blockDim_x())),3]+(1)+beg_axis_pad))
        ,Int(CUDA.blockIdx().z %channels_num)+1,Int(ceil(CUDA.blockIdx().z / channels_num))  ] *shared_arr_kern[1+2,1+2,1+2])
        )*(param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1,1])+
        param_matrix_a[(Int(ceil(blockIdx().y/num_directions)))
        ,Int(CUDA.blockIdx().z %channels_num)+1,(( blockIdx().y%num_directions )+1)
        ,(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1,2] ))
        * (param_matrix_a[(Int(ceil(blockIdx().y/num_directions))),Int(CUDA.blockIdx().z %channels_num)+1
        ,(( blockIdx().y%num_directions )+1),(( ((threadIdx().y)-1)*num_indicies_per_block)+7),1,3])))/7)
        sync_threads()
        
          ####1
            if ((threadIdx().y-1)%(2*1)==0)
              if ((threadIdx().y+(1))<=CUDA.blockDim_y())
        
                shared_arr[threadIdx().x,threadIdx().y,1]=((shared_arr[threadIdx().x,threadIdx().y,1]+shared_arr[threadIdx().x,threadIdx().y+(1),1]))
        
        
              end
            end
          
          sync_threads()
          
          ####2
            if ((threadIdx().y-1)%(2*2)==0)
              if ((threadIdx().y+(2))<=CUDA.blockDim_y())
        
                shared_arr[threadIdx().x,threadIdx().y,1]=((shared_arr[threadIdx().x,threadIdx().y,1]+shared_arr[threadIdx().x,threadIdx().y+(2),1]))
        
        
              end
            end
          
          sync_threads()
          
          ####4
            if ((threadIdx().y-1)%(2*4)==0)
              if ((threadIdx().y+(4))<=CUDA.blockDim_y())
        
                shared_arr[threadIdx().x,threadIdx().y,1]=((shared_arr[threadIdx().x,threadIdx().y,1]+shared_arr[threadIdx().x,threadIdx().y+(4),1]))
        
        
              end
            end
          
          sync_threads()
          
          ####8
            if ((threadIdx().y-1)%(2*8)==0)
              if ((threadIdx().y+(8))<=CUDA.blockDim_y())
        
                shared_arr[threadIdx().x,threadIdx().y,1]=((shared_arr[threadIdx().x,threadIdx().y,1]+shared_arr[threadIdx().x,threadIdx().y+(8),1]))
        
        
              end
            end
          
          sync_threads()
          
          ####16
            if ((threadIdx().y-1)%(2*16)==0)
              if ((threadIdx().y+(16))<=CUDA.blockDim_y())
        
                shared_arr[threadIdx().x,threadIdx().y,1]=((shared_arr[threadIdx().x,threadIdx().y,1]+shared_arr[threadIdx().x,threadIdx().y+(16),1]))
        
        
              end
            end
          
          sync_threads()
          
          ####32
            if ((threadIdx().y-1)%(2*32)==0)
              if ((threadIdx().y+(32))<=CUDA.blockDim_y())
        
                shared_arr[threadIdx().x,threadIdx().y,1]=((shared_arr[threadIdx().x,threadIdx().y,1]+shared_arr[threadIdx().x,threadIdx().y+(32),1]))
        
        
              end
            end
          
          sync_threads()
          
          ####64
            if ((threadIdx().y-1)%(2*64)==0)
              if ((threadIdx().y+(64))<=CUDA.blockDim_y())
        
                shared_arr[threadIdx().x,threadIdx().y,1]=((shared_arr[threadIdx().x,threadIdx().y,1]+shared_arr[threadIdx().x,threadIdx().y+(64),1]))
        
        
              end
            end
          
          sync_threads()
          
          ####128
            if ((threadIdx().y-1)%(2*128)==0)
              if ((threadIdx().y+(128))<=CUDA.blockDim_y())
        
                shared_arr[threadIdx().x,threadIdx().y,1]=((shared_arr[threadIdx().x,threadIdx().y,1]+shared_arr[threadIdx().x,threadIdx().y+(128),1]))
        
        
              end
            end
          
          sync_threads()
          
          ####256
            if ((threadIdx().y-1)%(2*256)==0)
              if ((threadIdx().y+(256))<=CUDA.blockDim_y())
        
                shared_arr[threadIdx().x,threadIdx().y,1]=((shared_arr[threadIdx().x,threadIdx().y,1]+shared_arr[threadIdx().x,threadIdx().y+(256),1]))
        
        
              end
            end
          
          sync_threads()
          
          ####512
            if ((threadIdx().y-1)%(2*512)==0)
              if ((threadIdx().y+(512))<=CUDA.blockDim_y())
        
                shared_arr[threadIdx().x,threadIdx().y,1]=((shared_arr[threadIdx().x,threadIdx().y,1]+shared_arr[threadIdx().x,threadIdx().y+(512),1]))
        
        
              end
            end
          
          sync_threads()
          
           if (threadIdx().y==1)
        
            # for i in 2:CUDA.blockDim_y()
            #         shared_arr[threadIdx().x,1, 1]+=shared_arr[threadIdx().x,i, 1]
            #         shared_arr[threadIdx().x,1, 2]+=shared_arr[threadIdx().x,i, 2]
            #         shared_arr[threadIdx().x,1, 3]+=shared_arr[threadIdx().x,i, 3]
            # end
        
        
        out_summarised[Int(ceil(CUDA.blockIdx().z / channels_num)) 
        ,Int(CUDA.blockIdx().z %channels_num)+1
        ,(threadIdx().x + (blockIdx().x * CUDA.blockDim_x()))
        , (1+(Int(ceil(blockIdx().y/num_directions)-1)))
        , (( blockIdx().y%num_directions )+1) ]+=(shared_arr[threadIdx().x,1, 1]/ (voxel_counts[(( blockIdx().y%num_directions )+1)]/7))
            
            end
                
        
        return nothing

end

############## differentiable kernel

function get_weights_from_directions_deff(
        flat_sv_centers, d_flat_sv_centers,
        source_arr, d_source_arr,
        directions_indicies, d_directions_indicies,
        param_matrix_a, d_param_matrix_a,
        conv_kernels, d_conv_kernels,
        channels_num,
        out_summarised, d_out_summarised, 
        max_index,
        beg_axis_pad,  num_indicies_per_block
        ,num_directions,voxel_counts,d_voxel_counts
)
        Enzyme.autodiff_deferred(
                Enzyme.Reverse, Enzyme.Const(get_weights_from_directions), Enzyme.Const,
                Enzyme.Duplicated(flat_sv_centers, d_flat_sv_centers),
                Enzyme.Duplicated(source_arr, d_source_arr),
                Enzyme.Duplicated(directions_indicies, d_directions_indicies),
                Enzyme.Duplicated(param_matrix_a, d_param_matrix_a),
                Enzyme.Duplicated(conv_kernels, d_conv_kernels),
                Enzyme.Const(channels_num),
                Enzyme.Duplicated(out_summarised, d_out_summarised),
                Enzyme.Const(max_index),
                Enzyme.Const(beg_axis_pad),
                Enzyme.Const(num_indicies_per_block),
                Enzyme.Const(num_directions),
                Enzyme.Duplicated(voxel_counts,d_voxel_counts)
                )
        return nothing
end







"""
after we have direction information per control point weight we need to mix information in stages 
1) between diffent directions for the same control point weight
2) between diffrent control point weights
3) between diffrent channels
4) between diffrent supervoxels


1 and 2) we will apply dense to the last dimension it will use basically last two dimensions as this is matrix multiply we will then mix information
   from within the same supervoxel ! consider reshaping to make it smoother
3) we can do parametrised add so broadcast multiply and add the parameters to sv representation, then add apply nonlinearity and multiply by the third parameter
4) we can do similarly like in 3 but we will do diffrent amount of apdding to the copies of the  reshaped out_summarised     
  a) we need to reshape out_summarised to regain x,y,z dimensions sth like  sv_centers_resh_back = reshape(flat_sv_centers, size(sv_centers, 1),size(sv_centers, 2),size(sv_centers, 3),size(sv_centers, 4))
  b) we need to pad the reshaped out_summarised we will pad it multiple times in a diffrent way so we will e able to apply parametrarised add between diffrent neighbouring supervoxels

As atomic add is not working well would be good to try primary aggregation on shared memory and then add it to global memory
        for it we need to have shared memory that is dynamically set - maybe can do it throgh global constants?

Remember to add normalization layers after each step


Later we can decrease memory consumption of loss layers by 
1) fuse the set tetr data by dithing variance calculation wit set point kernel
2) on set point kernel do not save all sampled points for variance calculation but apply some parameters and add for per voxel vector representation
        then this vector representation will be processed to get parameters for for example 3 sinusoids
        
we can make it work also by aggregating some info in the shared memory after the data of each point is calculated


3) then we execute point info kern one more time but now each point and weight we use to accumulate the diffrence between the point and the sinusoids
        value at this point and the mean diffrence between the point and the sinusoids value at this point from all the points in tetrahedron will be saved
        then loss is the mean of those values
        
point 2 and 3 will make it not necessary for saving out points and tetr data out - significantly reducing memory consumption        

"""

