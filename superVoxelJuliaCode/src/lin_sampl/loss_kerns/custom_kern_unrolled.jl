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

import CUDA
using KernelAbstractions



"""
each tetrahedron will have a set of sample points that are on the line between sv center and triangle center
and additional one between corner of the triangle and the last main sample point 
function is created to find ith sample point when max is is the number of main sample points (num_base_samp_points) +3 (for additional ones)
  now we have in tetrs all of the triangles that create the outer skin of sv volume 
  we need to no 
  1. get a center of each triangle
  2. draw a line between the center of the triangle and the center of the sv lets call it AB
  3. divide AB into n sections (the bigger n the more sample points)
  4. division between sections will be our main sample points morover we will get point in a middle between
      last main zsample points and a verticies of main triangle that we got as input
  5. we weight each point by getting the distance to the edges of the tetrahedron in which we are operating
      the bigger the distance the bigger the importance of this sample point. 
      a)in order to get this distance for main sample points we need to define lines between triangle verticies and the center of the sv
      and then we need to project the sample point onto this line pluc the distance to previous and or next sample point
      b)in case for additional ones - those that branch out from last main sample point we will approximate the spread by just getting the distance between
      the last main sample point and the the vartex of the trangle that we used for it and using it as a diameter of the sphere which volume we will use for weighting those points
  
  
  Implementation details:
  probably the bst is to get all of the sample point per tetrahedron to be calculated in sequence over single thread
  and then parallelize info on tetrahedrons 
  In case of using float32 and reserving space for shadow memory for enzyme we probably can keep 2-3 floats32 in shared memory per thread
  for x,y,z indicies of the supervoxels probably uint8 will be sufficient - morover we can unroll the loop
  we will also try to use generic names of the variables to keep track of the register memory usage 
  
  tetr_dat - array of 5 points indicies that first is sv center next 3 are verticies of the triangle that creates the base of the tetrahedron plus this triangle center we also have added point to indicate location in control_points so 5x4 array
  num_base_samp_points - how many main sample points we want to have between each triangle center and sv center in each tetrahedron
  shared_arr - array of length 3 where we have allocated space for temporarly storing the sample point that we are calculating
  out_sampled_points - array for storing the value of the sampled point that we got from interpolation and its weight that depends on the proximity to the edges of the tetrahedron and to other points
      hence the shape of the array is (num_base_samp_points+3)x5 where in second dimension first is the interpolated value of the point and second is its weight the last 3 entries are x,y,z coordinates of sampled point
  source_arr - array with image on which we are working    
  control_points - array of where we have the coordinates of the control points of the tetrahedron

  num_additional_samp_points - how many additional sample points we want to have between the last main sample point and the verticies of the triangle that we used for it
"""
# @kernel function set_tetr_dat_kern_unrolled(@Const(tetr_dat),tetr_dat_out,@Const(source_arr),@Const(control_points),@Const(sv_centers))
function set_tetr_dat_kern_unrolled(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers, max_index, spacing,dims)
   index = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))

   # @cuprintf("index %d\n", index)

   if index > max_index
      return nothing
   end
   # source_arr = CUDA.Const(source_arr)
   # control_points = CUDA.Const(control_points)
   # sv_centers = CUDA.Const(sv_centers)
   shared_arr = CuStaticSharedArray(Float32, (256, 4))
   # shared_arr = @localmem Float32 (@groupsize()[1], 4) 
   # index = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) 
   # index = @index(Global)
   shared_arr[threadIdx().x, 1] = sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), 1,blockIdx().y]
   shared_arr[threadIdx().x, 2] = sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), 2,blockIdx().y]
   shared_arr[threadIdx().x, 3] = sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), 3,blockIdx().y]
   tetr_dat_out[index, 1, 1,blockIdx().y] = shared_arr[threadIdx().x, 1]
   tetr_dat_out[index, 1, 2,blockIdx().y] = shared_arr[threadIdx().x, 2]
   tetr_dat_out[index, 1, 3,blockIdx().y] = shared_arr[threadIdx().x, 3]
   tetr_dat_out[index, 1, 4,blockIdx().y] =  (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) )))
 
 
   shared_arr[threadIdx().x, 1] = control_points[Int(tetr_dat[index, 2, 1])
 , Int(tetr_dat[index, 2, 2])
 , Int(tetr_dat[index, 2, 3])
 , Int(tetr_dat[index, 2, 4]), 1]
    shared_arr[threadIdx().x, 2] = control_points[Int(tetr_dat[index, 2, 1])
 , Int(tetr_dat[index, 2, 2])
 , Int(tetr_dat[index, 2, 3])
 , Int(tetr_dat[index, 2, 4]), 2]
    shared_arr[threadIdx().x, 3] = control_points[Int(tetr_dat[index, 2, 1])
 , Int(tetr_dat[index, 2, 2])
 , Int(tetr_dat[index, 2, 3])
 , Int(tetr_dat[index, 2, 4]), 3]
    tetr_dat_out[index, 2, 1,blockIdx().y] = shared_arr[threadIdx().x, 1]
   tetr_dat_out[index, 2, 2,blockIdx().y] = shared_arr[threadIdx().x, 2]
   tetr_dat_out[index, 2, 3,blockIdx().y] = shared_arr[threadIdx().x, 3]
 shared_arr[threadIdx().x, 4] =  (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) )))
 
 
 tetr_dat_out[index, 2, 4,blockIdx().y] =  (( (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) )))
 
    shared_arr[threadIdx().x, 1] = control_points[Int(tetr_dat[index, 3, 1])
 , Int(tetr_dat[index, 3, 2])
 , Int(tetr_dat[index, 3, 3])
 , Int(tetr_dat[index, 3, 4]), 1]
    shared_arr[threadIdx().x, 2] = control_points[Int(tetr_dat[index, 3, 1])
 , Int(tetr_dat[index, 3, 2])
 , Int(tetr_dat[index, 3, 3])
 , Int(tetr_dat[index, 3, 4]), 2]
    shared_arr[threadIdx().x, 3] = control_points[Int(tetr_dat[index, 3, 1])
 , Int(tetr_dat[index, 3, 2])
 , Int(tetr_dat[index, 3, 3])
 , Int(tetr_dat[index, 3, 4]), 3]
    tetr_dat_out[index, 3, 1,blockIdx().y] = shared_arr[threadIdx().x, 1]
   tetr_dat_out[index, 3, 2,blockIdx().y] = shared_arr[threadIdx().x, 2]
   tetr_dat_out[index, 3, 3,blockIdx().y] = shared_arr[threadIdx().x, 3]
 shared_arr[threadIdx().x, 4] =  (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) )))
 
 
 tetr_dat_out[index, 3, 4,blockIdx().y] =  (( (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) )))
 
    shared_arr[threadIdx().x, 1] = control_points[Int(tetr_dat[index, 4, 1])
 , Int(tetr_dat[index, 4, 2])
 , Int(tetr_dat[index, 4, 3])
 , Int(tetr_dat[index, 4, 4]), 1]
    shared_arr[threadIdx().x, 2] = control_points[Int(tetr_dat[index, 4, 1])
 , Int(tetr_dat[index, 4, 2])
 , Int(tetr_dat[index, 4, 3])
 , Int(tetr_dat[index, 4, 4]), 2]
    shared_arr[threadIdx().x, 3] = control_points[Int(tetr_dat[index, 4, 1])
 , Int(tetr_dat[index, 4, 2])
 , Int(tetr_dat[index, 4, 3])
 , Int(tetr_dat[index, 4, 4]), 3]
    tetr_dat_out[index, 4, 1,blockIdx().y] = shared_arr[threadIdx().x, 1]
   tetr_dat_out[index, 4, 2,blockIdx().y] = shared_arr[threadIdx().x, 2]
   tetr_dat_out[index, 4, 3,blockIdx().y] = shared_arr[threadIdx().x, 3]
 shared_arr[threadIdx().x, 4] =  (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) )))
 
 
 tetr_dat_out[index, 4, 4,blockIdx().y] =  (( (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) )))
 
    shared_arr[threadIdx().x, 1] += tetr_dat_out[index, 2, 1,blockIdx().y]
   shared_arr[threadIdx().x, 2] += tetr_dat_out[index, 2, 2,blockIdx().y]
   shared_arr[threadIdx().x, 3] += tetr_dat_out[index, 2, 3,blockIdx().y]
   shared_arr[threadIdx().x, 1] += tetr_dat_out[index, 3, 1,blockIdx().y]
   shared_arr[threadIdx().x, 2] += tetr_dat_out[index, 3, 2,blockIdx().y]
   shared_arr[threadIdx().x, 3] += tetr_dat_out[index, 3, 3,blockIdx().y]
       shared_arr[threadIdx().x, 1] = shared_arr[threadIdx().x, 1] / 3
 tetr_dat_out[index, 5, 1,blockIdx().y] = shared_arr[threadIdx().x, 1]
               shared_arr[threadIdx().x, 2] = shared_arr[threadIdx().x, 2] / 3
 tetr_dat_out[index, 5, 2,blockIdx().y] = shared_arr[threadIdx().x, 2]
               shared_arr[threadIdx().x, 3] = shared_arr[threadIdx().x, 3] / 3
 tetr_dat_out[index, 5, 3,blockIdx().y] = shared_arr[threadIdx().x, 3]
         shared_arr[threadIdx().x, 4] =  (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) )))
 
 
 tetr_dat_out[index, 5, 4,blockIdx().y] =  (( (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))   +  (((source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
 ,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
 ,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
 ,1,blockIdx().y] -shared_arr[threadIdx().x,4])^2)*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
 +(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
 + (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
 )) )))
 
  
 
  


   return nothing
end











function point_info_kern_unrolled(tetr_dat, out_sampled_points, source_arr, num_base_samp_points, num_additional_samp_points, max_index, spacing,dims)
   shared_arr = CuStaticSharedArray(Float32, (256, 4))
   index = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))
   if index > max_index
      return nothing
   end
#we get the diffrence between the sv center and the triangle center
shared_arr[threadIdx().x,1]= ((tetr_dat[index,5,1,blockIdx().y]-tetr_dat[index,1,1,blockIdx().y])/(2+1))
shared_arr[threadIdx().x,2]=((tetr_dat[index,5,2,blockIdx().y]-tetr_dat[index,1,2,blockIdx().y])/(2+1))
shared_arr[threadIdx().x,3]=((tetr_dat[index,5,3,blockIdx().y]-tetr_dat[index,1,3,blockIdx().y])/(2+1))



##calculate weight of the point
#first distance from next and previous point on the line between sv center and triangle center
shared_arr[threadIdx().x,4]=sqrt((shared_arr[threadIdx().x,1])^2 +(shared_arr[threadIdx().x,2])^2+(shared_arr[threadIdx().x,3])^2)*2 #distance between main sample points (two times for distance to previous and next)
#now we get the distance to the lines that get from sv center to the triangle corners - for simplicity
# we can assume that sv center location is 0.0,0.0,0.0 as we need only diffrences 
#now we get the location of sample point

shared_arr[threadIdx().x,1]= tetr_dat[index,1,1,blockIdx().y]+(shared_arr[threadIdx().x,1]*1)
shared_arr[threadIdx().x,2]= tetr_dat[index,1,2,blockIdx().y]+(shared_arr[threadIdx().x,2]*1)
shared_arr[threadIdx().x,3]= tetr_dat[index,1,3,blockIdx().y]+(shared_arr[threadIdx().x,3]*1)


shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,1,2,blockIdx().y]
-tetr_dat[index,1+1,2,blockIdx().y])
*(tetr_dat[index,1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,1,3,blockIdx().y]-tetr_dat[index,1+1,3,blockIdx().y])
*(tetr_dat[index,1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
    (((tetr_dat[index,1,3,blockIdx().y]
    -tetr_dat[index,1+1,3,blockIdx().y])
    *(tetr_dat[index,1,1,blockIdx().y]-shared_arr[threadIdx().x,1])
     - (tetr_dat[index,1,1,blockIdx().y]-tetr_dat[index,1+1,1,blockIdx().y])
     *(tetr_dat[index,1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
    ((tetr_dat[index,1,1,blockIdx().y]-tetr_dat[index,1+1,1,blockIdx().y])
    *(tetr_dat[index,1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
    - (tetr_dat[index,1,2,blockIdx().y]-tetr_dat[index,1+1,2,blockIdx().y])
    *(tetr_dat[index,1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
    / (sqrt((tetr_dat[index,1+1,2,blockIdx().y]
    -tetr_dat[index,1,2,blockIdx().y])^2
    +(tetr_dat[index,1+1,3,blockIdx().y]
    -tetr_dat[index,1,3,blockIdx().y])^2
    +(tetr_dat[index,1+1,1,blockIdx().y]
    -tetr_dat[index,1,1,blockIdx().y])^2)+0.00001)  )
    
    out_sampled_points[index,1,1,blockIdx().y]= (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) )))


    #saving sample points coordinates mainly for debugging and visualizations
    out_sampled_points[index,1,3,blockIdx().y]=shared_arr[threadIdx().x,1]
    out_sampled_points[index,1,4,blockIdx().y]=shared_arr[threadIdx().x,2]
    out_sampled_points[index,1,5,blockIdx().y]=shared_arr[threadIdx().x,3]

  
shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,1,2,blockIdx().y]
-tetr_dat[index,2+1,2,blockIdx().y])
*(tetr_dat[index,1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,1,3,blockIdx().y]-tetr_dat[index,2+1,3,blockIdx().y])
*(tetr_dat[index,1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
    (((tetr_dat[index,1,3,blockIdx().y]
    -tetr_dat[index,2+1,3,blockIdx().y])
    *(tetr_dat[index,1,1,blockIdx().y]-shared_arr[threadIdx().x,1])
     - (tetr_dat[index,1,1,blockIdx().y]-tetr_dat[index,2+1,1,blockIdx().y])
     *(tetr_dat[index,1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
    ((tetr_dat[index,1,1,blockIdx().y]-tetr_dat[index,2+1,1,blockIdx().y])
    *(tetr_dat[index,1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
    - (tetr_dat[index,1,2,blockIdx().y]-tetr_dat[index,2+1,2,blockIdx().y])
    *(tetr_dat[index,1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
    / (sqrt((tetr_dat[index,2+1,2,blockIdx().y]
    -tetr_dat[index,1,2,blockIdx().y])^2
    +(tetr_dat[index,2+1,3,blockIdx().y]
    -tetr_dat[index,1,3,blockIdx().y])^2
    +(tetr_dat[index,2+1,1,blockIdx().y]
    -tetr_dat[index,1,1,blockIdx().y])^2)+0.00001)  )
    
    out_sampled_points[index,1,1,blockIdx().y]= (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) )))


    #saving sample points coordinates mainly for debugging and visualizations
    out_sampled_points[index,1,3,blockIdx().y]=shared_arr[threadIdx().x,1]
    out_sampled_points[index,1,4,blockIdx().y]=shared_arr[threadIdx().x,2]
    out_sampled_points[index,1,5,blockIdx().y]=shared_arr[threadIdx().x,3]

  
shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,1,2,blockIdx().y]
-tetr_dat[index,3+1,2,blockIdx().y])
*(tetr_dat[index,1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,1,3,blockIdx().y]-tetr_dat[index,3+1,3,blockIdx().y])
*(tetr_dat[index,1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
    (((tetr_dat[index,1,3,blockIdx().y]
    -tetr_dat[index,3+1,3,blockIdx().y])
    *(tetr_dat[index,1,1,blockIdx().y]-shared_arr[threadIdx().x,1])
     - (tetr_dat[index,1,1,blockIdx().y]-tetr_dat[index,3+1,1,blockIdx().y])
     *(tetr_dat[index,1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
    ((tetr_dat[index,1,1,blockIdx().y]-tetr_dat[index,3+1,1,blockIdx().y])
    *(tetr_dat[index,1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
    - (tetr_dat[index,1,2,blockIdx().y]-tetr_dat[index,3+1,2,blockIdx().y])
    *(tetr_dat[index,1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
    / (sqrt((tetr_dat[index,3+1,2,blockIdx().y]
    -tetr_dat[index,1,2,blockIdx().y])^2
    +(tetr_dat[index,3+1,3,blockIdx().y]
    -tetr_dat[index,1,3,blockIdx().y])^2
    +(tetr_dat[index,3+1,1,blockIdx().y]
    -tetr_dat[index,1,1,blockIdx().y])^2)+0.00001)  )
    
    out_sampled_points[index,1,1,blockIdx().y]= (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) )))


    #saving sample points coordinates mainly for debugging and visualizations
    out_sampled_points[index,1,3,blockIdx().y]=shared_arr[threadIdx().x,1]
    out_sampled_points[index,1,4,blockIdx().y]=shared_arr[threadIdx().x,2]
    out_sampled_points[index,1,5,blockIdx().y]=shared_arr[threadIdx().x,3]

   out_sampled_points[index,1,2,blockIdx().y]= (((shared_arr[threadIdx().x,4])/5)^3)
out_sampled_points[index,1,1,blockIdx().y]= (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) )))


#saving sample points coordinates mainly for debugging and visualizations
out_sampled_points[index,1,3,blockIdx().y]=shared_arr[threadIdx().x,1]
out_sampled_points[index,1,4,blockIdx().y]=shared_arr[threadIdx().x,2]
out_sampled_points[index,1,5,blockIdx().y]=shared_arr[threadIdx().x,3]


#we get the diffrence between the sv center and the triangle center
shared_arr[threadIdx().x,1]= ((tetr_dat[index,5,1,blockIdx().y]-tetr_dat[index,1,1,blockIdx().y])/(2+1))
shared_arr[threadIdx().x,2]=((tetr_dat[index,5,2,blockIdx().y]-tetr_dat[index,1,2,blockIdx().y])/(2+1))
shared_arr[threadIdx().x,3]=((tetr_dat[index,5,3,blockIdx().y]-tetr_dat[index,1,3,blockIdx().y])/(2+1))



##calculate weight of the point
#first distance from next and previous point on the line between sv center and triangle center
shared_arr[threadIdx().x,4]=sqrt((shared_arr[threadIdx().x,1])^2 +(shared_arr[threadIdx().x,2])^2+(shared_arr[threadIdx().x,3])^2)*2 #distance between main sample points (two times for distance to previous and next)
#now we get the distance to the lines that get from sv center to the triangle corners - for simplicity
# we can assume that sv center location is 0.0,0.0,0.0 as we need only diffrences 
#now we get the location of sample point

shared_arr[threadIdx().x,1]= tetr_dat[index,1,1,blockIdx().y]+(shared_arr[threadIdx().x,1]*2)
shared_arr[threadIdx().x,2]= tetr_dat[index,1,2,blockIdx().y]+(shared_arr[threadIdx().x,2]*2)
shared_arr[threadIdx().x,3]= tetr_dat[index,1,3,blockIdx().y]+(shared_arr[threadIdx().x,3]*2)


shared_arr[threadIdx().x,4]+=sqrt( ((tetr_dat[index,1+1,1,blockIdx().y]-shared_arr[threadIdx().x,1])
*(1/(num_additional_samp_points+1)))^2
+((tetr_dat[index,1+1,2,blockIdx().y]-shared_arr[threadIdx().x,2])
*(1/(num_additional_samp_points+1)))^2
+((tetr_dat[index,1+1,3,blockIdx().y]-shared_arr[threadIdx().x,3])
*(1/(num_additional_samp_points+1)))^2)

 
shared_arr[threadIdx().x,4]+=sqrt( ((tetr_dat[index,2+1,1,blockIdx().y]-shared_arr[threadIdx().x,1])
*(1/(num_additional_samp_points+1)))^2
+((tetr_dat[index,2+1,2,blockIdx().y]-shared_arr[threadIdx().x,2])
*(1/(num_additional_samp_points+1)))^2
+((tetr_dat[index,2+1,3,blockIdx().y]-shared_arr[threadIdx().x,3])
*(1/(num_additional_samp_points+1)))^2)

 
shared_arr[threadIdx().x,4]+=sqrt( ((tetr_dat[index,3+1,1,blockIdx().y]-shared_arr[threadIdx().x,1])
*(1/(num_additional_samp_points+1)))^2
+((tetr_dat[index,3+1,2,blockIdx().y]-shared_arr[threadIdx().x,2])
*(1/(num_additional_samp_points+1)))^2
+((tetr_dat[index,3+1,3,blockIdx().y]-shared_arr[threadIdx().x,3])
*(1/(num_additional_samp_points+1)))^2)

  out_sampled_points[index,2,2,blockIdx().y]= (((shared_arr[threadIdx().x,4])/5)^3)
out_sampled_points[index,2,1,blockIdx().y]= (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) )))


#saving sample points coordinates mainly for debugging and visualizations
out_sampled_points[index,2,3,blockIdx().y]=shared_arr[threadIdx().x,1]
out_sampled_points[index,2,4,blockIdx().y]=shared_arr[threadIdx().x,2]
out_sampled_points[index,2,5,blockIdx().y]=shared_arr[threadIdx().x,3]



shared_arr[threadIdx().x,1]=((tetr_dat[index,1+1,1,blockIdx().y]-out_sampled_points[index,num_base_samp_points,3,blockIdx().y])/(1+1))
shared_arr[threadIdx().x,2]=((tetr_dat[index,1+1,2,blockIdx().y]-out_sampled_points[index,num_base_samp_points,4,blockIdx().y])/(1+1))
shared_arr[threadIdx().x,3]=((tetr_dat[index,1+1,3,blockIdx().y]-out_sampled_points[index,num_base_samp_points,5,blockIdx().y])/(1+1))

#### calculate weight of the point 
#first we get distance to the previous and next point on the line between the last main sample point and the triangle corner
shared_arr[threadIdx().x,4]=sqrt( shared_arr[threadIdx().x,1]^2+shared_arr[threadIdx().x,2]^2+shared_arr[threadIdx().x,3]^2)*2
#now we get the location of sample point
shared_arr[threadIdx().x,1]= out_sampled_points[index,num_base_samp_points,3,blockIdx().y]+(shared_arr[threadIdx().x,1]*(1))
shared_arr[threadIdx().x,2]= out_sampled_points[index,num_base_samp_points,4,blockIdx().y]+(shared_arr[threadIdx().x,2]*(1))
shared_arr[threadIdx().x,3]= out_sampled_points[index,num_base_samp_points,5,blockIdx().y]+(shared_arr[threadIdx().x,3]*(1))



shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,1+1,2,blockIdx().y]
-tetr_dat[index,1,2,blockIdx().y])
*(tetr_dat[index,1+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,1+1,3,blockIdx().y]
-tetr_dat[index,1,3,blockIdx().y])
*(tetr_dat[index,1+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
(((tetr_dat[index,1+1,3,blockIdx().y]
-tetr_dat[index,1,3,blockIdx().y])
*(tetr_dat[index,1+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]) 
- (tetr_dat[index,1+1,1,blockIdx().y]
-tetr_dat[index,1,1,blockIdx().y])
*(tetr_dat[index,1+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
((tetr_dat[index,1+1,1,blockIdx().y]
-tetr_dat[index,1,1,blockIdx().y])
*(tetr_dat[index,1+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
- (tetr_dat[index,1+1,2,blockIdx().y]
-tetr_dat[index,1,2,blockIdx().y])
*(tetr_dat[index,1+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
/ (sqrt((tetr_dat[index,1,2,blockIdx().y]
-tetr_dat[index,1+1,2,blockIdx().y])^2
+(tetr_dat[index,1,3,blockIdx().y]
-tetr_dat[index,1+1,3,blockIdx().y])^2
+(tetr_dat[index,1,1,blockIdx().y]
-tetr_dat[index,1+1,1,blockIdx().y])^2) +0.00000001) )




shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,1+1,2,blockIdx().y]
-tetr_dat[index,3,2,blockIdx().y])
*(tetr_dat[index,1+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,1+1,3,blockIdx().y]
-tetr_dat[index,3,3,blockIdx().y])
*(tetr_dat[index,1+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
(((tetr_dat[index,1+1,3,blockIdx().y]
-tetr_dat[index,3,3,blockIdx().y])
*(tetr_dat[index,1+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]) 
- (tetr_dat[index,1+1,1,blockIdx().y]
-tetr_dat[index,3,1,blockIdx().y])
*(tetr_dat[index,1+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
((tetr_dat[index,1+1,1,blockIdx().y]
-tetr_dat[index,3,1,blockIdx().y])
*(tetr_dat[index,1+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
- (tetr_dat[index,1+1,2,blockIdx().y]
-tetr_dat[index,3,2,blockIdx().y])
*(tetr_dat[index,1+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
/ (sqrt((tetr_dat[index,3,2,blockIdx().y]
-tetr_dat[index,1+1,2,blockIdx().y])^2
+(tetr_dat[index,3,3,blockIdx().y]
-tetr_dat[index,1+1,3,blockIdx().y])^2
+(tetr_dat[index,3,1,blockIdx().y]
-tetr_dat[index,1+1,1,blockIdx().y])^2) +0.00000001) )




shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,1+1,2,blockIdx().y]
-tetr_dat[index,4,2,blockIdx().y])
*(tetr_dat[index,1+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,1+1,3,blockIdx().y]
-tetr_dat[index,4,3,blockIdx().y])
*(tetr_dat[index,1+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
(((tetr_dat[index,1+1,3,blockIdx().y]
-tetr_dat[index,4,3,blockIdx().y])
*(tetr_dat[index,1+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]) 
- (tetr_dat[index,1+1,1,blockIdx().y]
-tetr_dat[index,4,1,blockIdx().y])
*(tetr_dat[index,1+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
((tetr_dat[index,1+1,1,blockIdx().y]
-tetr_dat[index,4,1,blockIdx().y])
*(tetr_dat[index,1+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
- (tetr_dat[index,1+1,2,blockIdx().y]
-tetr_dat[index,4,2,blockIdx().y])
*(tetr_dat[index,1+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
/ (sqrt((tetr_dat[index,4,2,blockIdx().y]
-tetr_dat[index,1+1,2,blockIdx().y])^2
+(tetr_dat[index,4,3,blockIdx().y]
-tetr_dat[index,1+1,3,blockIdx().y])^2
+(tetr_dat[index,4,1,blockIdx().y]
-tetr_dat[index,1+1,1,blockIdx().y])^2) +0.00000001) )



out_sampled_points[index,(num_base_samp_points+1)+(1-1)*3,2,blockIdx().y]=(((shared_arr[threadIdx().x,4])/5)^3)      
#performing interpolation result is in var2 and it get data from shared_arr
#saving the result of interpolated value to the out_sampled_points
out_sampled_points[index,(num_base_samp_points+1)+(1-1)*3,1,blockIdx().y]= (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) )))


# #saving sample points coordinates
out_sampled_points[index,(num_base_samp_points+1)+(1-1)*3,3,blockIdx().y]=shared_arr[threadIdx().x,1]
out_sampled_points[index,(num_base_samp_points+1)+(1-1)*3,4,blockIdx().y]=shared_arr[threadIdx().x,2]
out_sampled_points[index,(num_base_samp_points+1)+(1-1)*3,5,blockIdx().y]=shared_arr[threadIdx().x,3]


shared_arr[threadIdx().x,1]=((tetr_dat[index,2+1,1,blockIdx().y]-out_sampled_points[index,num_base_samp_points,3,blockIdx().y])/(1+1))
shared_arr[threadIdx().x,2]=((tetr_dat[index,2+1,2,blockIdx().y]-out_sampled_points[index,num_base_samp_points,4,blockIdx().y])/(1+1))
shared_arr[threadIdx().x,3]=((tetr_dat[index,2+1,3,blockIdx().y]-out_sampled_points[index,num_base_samp_points,5,blockIdx().y])/(1+1))

#### calculate weight of the point 
#first we get distance to the previous and next point on the line between the last main sample point and the triangle corner
shared_arr[threadIdx().x,4]=sqrt( shared_arr[threadIdx().x,1]^2+shared_arr[threadIdx().x,2]^2+shared_arr[threadIdx().x,3]^2)*2
#now we get the location of sample point
shared_arr[threadIdx().x,1]= out_sampled_points[index,num_base_samp_points,3,blockIdx().y]+(shared_arr[threadIdx().x,1]*(1))
shared_arr[threadIdx().x,2]= out_sampled_points[index,num_base_samp_points,4,blockIdx().y]+(shared_arr[threadIdx().x,2]*(1))
shared_arr[threadIdx().x,3]= out_sampled_points[index,num_base_samp_points,5,blockIdx().y]+(shared_arr[threadIdx().x,3]*(1))



shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,2+1,2,blockIdx().y]
-tetr_dat[index,1,2,blockIdx().y])
*(tetr_dat[index,2+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,2+1,3,blockIdx().y]
-tetr_dat[index,1,3,blockIdx().y])
*(tetr_dat[index,2+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
(((tetr_dat[index,2+1,3,blockIdx().y]
-tetr_dat[index,1,3,blockIdx().y])
*(tetr_dat[index,2+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]) 
- (tetr_dat[index,2+1,1,blockIdx().y]
-tetr_dat[index,1,1,blockIdx().y])
*(tetr_dat[index,2+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
((tetr_dat[index,2+1,1,blockIdx().y]
-tetr_dat[index,1,1,blockIdx().y])
*(tetr_dat[index,2+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
- (tetr_dat[index,2+1,2,blockIdx().y]
-tetr_dat[index,1,2,blockIdx().y])
*(tetr_dat[index,2+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
/ (sqrt((tetr_dat[index,1,2,blockIdx().y]
-tetr_dat[index,2+1,2,blockIdx().y])^2
+(tetr_dat[index,1,3,blockIdx().y]
-tetr_dat[index,2+1,3,blockIdx().y])^2
+(tetr_dat[index,1,1,blockIdx().y]
-tetr_dat[index,2+1,1,blockIdx().y])^2) +0.00000001) )




shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,2+1,2,blockIdx().y]
-tetr_dat[index,2,2,blockIdx().y])
*(tetr_dat[index,2+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,2+1,3,blockIdx().y]
-tetr_dat[index,2,3,blockIdx().y])
*(tetr_dat[index,2+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
(((tetr_dat[index,2+1,3,blockIdx().y]
-tetr_dat[index,2,3,blockIdx().y])
*(tetr_dat[index,2+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]) 
- (tetr_dat[index,2+1,1,blockIdx().y]
-tetr_dat[index,2,1,blockIdx().y])
*(tetr_dat[index,2+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
((tetr_dat[index,2+1,1,blockIdx().y]
-tetr_dat[index,2,1,blockIdx().y])
*(tetr_dat[index,2+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
- (tetr_dat[index,2+1,2,blockIdx().y]
-tetr_dat[index,2,2,blockIdx().y])
*(tetr_dat[index,2+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
/ (sqrt((tetr_dat[index,2,2,blockIdx().y]
-tetr_dat[index,2+1,2,blockIdx().y])^2
+(tetr_dat[index,2,3,blockIdx().y]
-tetr_dat[index,2+1,3,blockIdx().y])^2
+(tetr_dat[index,2,1,blockIdx().y]
-tetr_dat[index,2+1,1,blockIdx().y])^2) +0.00000001) )




shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,2+1,2,blockIdx().y]
-tetr_dat[index,4,2,blockIdx().y])
*(tetr_dat[index,2+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,2+1,3,blockIdx().y]
-tetr_dat[index,4,3,blockIdx().y])
*(tetr_dat[index,2+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
(((tetr_dat[index,2+1,3,blockIdx().y]
-tetr_dat[index,4,3,blockIdx().y])
*(tetr_dat[index,2+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]) 
- (tetr_dat[index,2+1,1,blockIdx().y]
-tetr_dat[index,4,1,blockIdx().y])
*(tetr_dat[index,2+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
((tetr_dat[index,2+1,1,blockIdx().y]
-tetr_dat[index,4,1,blockIdx().y])
*(tetr_dat[index,2+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
- (tetr_dat[index,2+1,2,blockIdx().y]
-tetr_dat[index,4,2,blockIdx().y])
*(tetr_dat[index,2+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
/ (sqrt((tetr_dat[index,4,2,blockIdx().y]
-tetr_dat[index,2+1,2,blockIdx().y])^2
+(tetr_dat[index,4,3,blockIdx().y]
-tetr_dat[index,2+1,3,blockIdx().y])^2
+(tetr_dat[index,4,1,blockIdx().y]
-tetr_dat[index,2+1,1,blockIdx().y])^2) +0.00000001) )



out_sampled_points[index,(num_base_samp_points+2)+(1-1)*3,2,blockIdx().y]=(((shared_arr[threadIdx().x,4])/5)^3)      
#performing interpolation result is in var2 and it get data from shared_arr
#saving the result of interpolated value to the out_sampled_points
out_sampled_points[index,(num_base_samp_points+2)+(1-1)*3,1,blockIdx().y]= (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) )))


# #saving sample points coordinates
out_sampled_points[index,(num_base_samp_points+2)+(1-1)*3,3,blockIdx().y]=shared_arr[threadIdx().x,1]
out_sampled_points[index,(num_base_samp_points+2)+(1-1)*3,4,blockIdx().y]=shared_arr[threadIdx().x,2]
out_sampled_points[index,(num_base_samp_points+2)+(1-1)*3,5,blockIdx().y]=shared_arr[threadIdx().x,3]


shared_arr[threadIdx().x,1]=((tetr_dat[index,3+1,1,blockIdx().y]-out_sampled_points[index,num_base_samp_points,3,blockIdx().y])/(1+1))
shared_arr[threadIdx().x,2]=((tetr_dat[index,3+1,2,blockIdx().y]-out_sampled_points[index,num_base_samp_points,4,blockIdx().y])/(1+1))
shared_arr[threadIdx().x,3]=((tetr_dat[index,3+1,3,blockIdx().y]-out_sampled_points[index,num_base_samp_points,5,blockIdx().y])/(1+1))

#### calculate weight of the point 
#first we get distance to the previous and next point on the line between the last main sample point and the triangle corner
shared_arr[threadIdx().x,4]=sqrt( shared_arr[threadIdx().x,1]^2+shared_arr[threadIdx().x,2]^2+shared_arr[threadIdx().x,3]^2)*2
#now we get the location of sample point
shared_arr[threadIdx().x,1]= out_sampled_points[index,num_base_samp_points,3,blockIdx().y]+(shared_arr[threadIdx().x,1]*(1))
shared_arr[threadIdx().x,2]= out_sampled_points[index,num_base_samp_points,4,blockIdx().y]+(shared_arr[threadIdx().x,2]*(1))
shared_arr[threadIdx().x,3]= out_sampled_points[index,num_base_samp_points,5,blockIdx().y]+(shared_arr[threadIdx().x,3]*(1))



shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,3+1,2,blockIdx().y]
-tetr_dat[index,1,2,blockIdx().y])
*(tetr_dat[index,3+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,3+1,3,blockIdx().y]
-tetr_dat[index,1,3,blockIdx().y])
*(tetr_dat[index,3+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
(((tetr_dat[index,3+1,3,blockIdx().y]
-tetr_dat[index,1,3,blockIdx().y])
*(tetr_dat[index,3+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]) 
- (tetr_dat[index,3+1,1,blockIdx().y]
-tetr_dat[index,1,1,blockIdx().y])
*(tetr_dat[index,3+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
((tetr_dat[index,3+1,1,blockIdx().y]
-tetr_dat[index,1,1,blockIdx().y])
*(tetr_dat[index,3+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
- (tetr_dat[index,3+1,2,blockIdx().y]
-tetr_dat[index,1,2,blockIdx().y])
*(tetr_dat[index,3+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
/ (sqrt((tetr_dat[index,1,2,blockIdx().y]
-tetr_dat[index,3+1,2,blockIdx().y])^2
+(tetr_dat[index,1,3,blockIdx().y]
-tetr_dat[index,3+1,3,blockIdx().y])^2
+(tetr_dat[index,1,1,blockIdx().y]
-tetr_dat[index,3+1,1,blockIdx().y])^2) +0.00000001) )




shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,3+1,2,blockIdx().y]
-tetr_dat[index,2,2,blockIdx().y])
*(tetr_dat[index,3+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,3+1,3,blockIdx().y]
-tetr_dat[index,2,3,blockIdx().y])
*(tetr_dat[index,3+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
(((tetr_dat[index,3+1,3,blockIdx().y]
-tetr_dat[index,2,3,blockIdx().y])
*(tetr_dat[index,3+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]) 
- (tetr_dat[index,3+1,1,blockIdx().y]
-tetr_dat[index,2,1,blockIdx().y])
*(tetr_dat[index,3+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
((tetr_dat[index,3+1,1,blockIdx().y]
-tetr_dat[index,2,1,blockIdx().y])
*(tetr_dat[index,3+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
- (tetr_dat[index,3+1,2,blockIdx().y]
-tetr_dat[index,2,2,blockIdx().y])
*(tetr_dat[index,3+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
/ (sqrt((tetr_dat[index,2,2,blockIdx().y]
-tetr_dat[index,3+1,2,blockIdx().y])^2
+(tetr_dat[index,2,3,blockIdx().y]
-tetr_dat[index,3+1,3,blockIdx().y])^2
+(tetr_dat[index,2,1,blockIdx().y]
-tetr_dat[index,3+1,1,blockIdx().y])^2) +0.00000001) )




shared_arr[threadIdx().x,4]+=(sqrt(((((tetr_dat[index,3+1,2,blockIdx().y]
-tetr_dat[index,3,2,blockIdx().y])
*(tetr_dat[index,3+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]) 
- (tetr_dat[index,3+1,3,blockIdx().y]
-tetr_dat[index,3,3,blockIdx().y])
*(tetr_dat[index,3+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]))^2)+
(((tetr_dat[index,3+1,3,blockIdx().y]
-tetr_dat[index,3,3,blockIdx().y])
*(tetr_dat[index,3+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]) 
- (tetr_dat[index,3+1,1,blockIdx().y]
-tetr_dat[index,3,1,blockIdx().y])
*(tetr_dat[index,3+1,3,blockIdx().y]-shared_arr[threadIdx().x,3]))^2)+
((tetr_dat[index,3+1,1,blockIdx().y]
-tetr_dat[index,3,1,blockIdx().y])
*(tetr_dat[index,3+1,2,blockIdx().y]-shared_arr[threadIdx().x,2]) 
- (tetr_dat[index,3+1,2,blockIdx().y]
-tetr_dat[index,3,2,blockIdx().y])
*(tetr_dat[index,3+1,1,blockIdx().y]-shared_arr[threadIdx().x,1]))^2)) 
/ (sqrt((tetr_dat[index,3,2,blockIdx().y]
-tetr_dat[index,3+1,2,blockIdx().y])^2
+(tetr_dat[index,3,3,blockIdx().y]
-tetr_dat[index,3+1,3,blockIdx().y])^2
+(tetr_dat[index,3,1,blockIdx().y]
-tetr_dat[index,3+1,1,blockIdx().y])^2) +0.00000001) )



out_sampled_points[index,(num_base_samp_points+3)+(1-1)*3,2,blockIdx().y]=(((shared_arr[threadIdx().x,4])/5)^3)      
#performing interpolation result is in var2 and it get data from shared_arr
#saving the result of interpolated value to the out_sampled_points
out_sampled_points[index,(num_base_samp_points+3)+(1-1)*3,1,blockIdx().y]= (( (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))   +  (source_arr[  min(dims[1],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))))
,  min(dims[2],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))))
,  min(dims[3],max(1,Int(round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))))
,1,blockIdx().y]*( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ))  )/(( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round(0.5 - 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) ) + ( (1/( ((((shared_arr[threadIdx().x,1]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,1]))*spacing[1])^2)
+(((shared_arr[threadIdx().x,2]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,2]))*spacing[2])^2)
+ (((shared_arr[threadIdx().x,3]-round((-0.5) + 1e-10+ shared_arr[threadIdx().x,3]))*spacing[3])^2) )+0.000001
)) )))


# #saving sample points coordinates
out_sampled_points[index,(num_base_samp_points+3)+(1-1)*3,3,blockIdx().y]=shared_arr[threadIdx().x,1]
out_sampled_points[index,(num_base_samp_points+3)+(1-1)*3,4,blockIdx().y]=shared_arr[threadIdx().x,2]
out_sampled_points[index,(num_base_samp_points+3)+(1-1)*3,5,blockIdx().y]=shared_arr[threadIdx().x,3]



   return nothing

end




















# @kernel function point_info_kern_unrolled(@Const(tetr_dat),out_sampled_points  ,@Const(source_arr),num_base_samp_points,num_additional_samp_points)


