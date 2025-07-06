# using Pkg
# using ChainRulesCore, Zygote, CUDA, Enzyme
# # using CUDAKernels
# using KernelAbstractions
# # using KernelGradients
# using Zygote, Lux
# using Lux, Random
# import Optimisers, Plots, Random, Statistics, Zygote

# using LinearAlgebra

# using Revise




# """
# calculate distance between set location and neighbouring voxel coordinates
# """
# macro get_dist(a, b, c)
#   return esc(:(
#     (1 / ((((shared_arr[threadIdx().x, 1] - round(shared_arr[threadIdx().x, 1] + ($a)))^2 + (shared_arr[threadIdx().x, 2] - round(shared_arr[threadIdx().x, 2] + ($b)))^2 + (shared_arr[threadIdx().x, 3] - round(shared_arr[threadIdx().x, 3] + ($c)))^2)^2) + 0.00000001))#we add a small number to avoid division by 0
#   ))
# end


# """
# get a diffrence between coordinates in given axis of sv center and triangle center
# """
# macro get_diff_on_line_sv_tetr(coord_i, tetr_dat_coord, point_num)
#   return esc(quote
#     ((tetr_dat[$tetr_dat_coord, $coord_i] - tetr_dat[1, $coord_i]) * ($point_num / (num_base_samp_points + 1)))
#   end)
# end



# """
# unrolled version of trilinear interpolation
# """
# function unrolled_trilin_interpol(shared_arr, source_arr)
#   return (((
#     source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
#     source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
#   )
#            *
#            (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
#            (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#             +
#             source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#            *
#            (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
#           *
#           (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
#           +
#           ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#             +
#             source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#            *
#            (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
#            (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#             +
#             source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#            *
#            (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
#           *
#           (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
# end

# """
# unrolled version of calculation of local variance based on trilinear interpolation
# """
# function unrolled_trilin_variance(shared_arr, source_arr)
#   #first saving meaking mean 
#   shared_arr[threadIdx().x, 4] = unrolled_trilin_interpol(shared_arr, source_arr)
#   return (((
#     ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
#     *
#     (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
#     ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
#     *
#     (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
#   )
#            *
#            (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
#            (((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
#             *
#             (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#             +
#             ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
#             *
#             (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#            *
#            (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
#           *
#           (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
#           +
#           ((((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
#             *
#             (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#             +
#             ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
#             *
#             (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#            *
#            (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
#            (((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
#             *
#             (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#             +
#             ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
#             *
#             (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
#            *
#            (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
#           *
#           (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
# end


# """
# given point p1 and line l where line l is defined by two points p2 and p3
# we calculate the distance between point p1 and line l
# """
# function dist_of_p_to_line(p1, p2, p3)
#   return sqrt(((((p2[2] - p3[2]) * (p2[3] - p1[3]) - (p2[3] - p3[3]) * (p2[2] - p1[2]))^2) +
#                (((p2[3] - p3[3]) * (p2[1] - p1[1]) - (p2[1] - p3[1]) * (p2[3] - p1[3]))^2) +
#                ((p2[1] - p3[1]) * (p2[2] - p1[2]) - (p2[2] - p3[2]) * (p2[1] - p1[1]))^2)) / sqrt((p3[2] - p2[2])^2 + (p3[3] - p2[3])^2 + (p3[1] - p2[1])^2)
# end


# """
# each tetrahedron will have a set of sample points that are on the line between sv center and triangle center
# and additional one between corner of the triangle and the last main sample point 
# function is created to find ith sample point when max is is the number of main sample points (num_base_samp_points) +3 (for additional ones)
#   now we have in tetrs all of the triangles that create the outer skin of sv volume 
#   we need to no 
#   1. get a center of each triangle
#   2. draw a line between the center of the triangle and the center of the sv lets call it AB
#   3. divide AB into n sections (the bigger n the more sample points)
#   4. division between sections will be our main sample points morover we will get point in a middle between
#       last main zsample points and a verticies of main triangle that we got as input
#   5. we weight each point by getting the distance to the edges of the tetrahedron in which we are operating
#       the bigger the distance the bigger the importance of this sample point. 
#       a)in order to get this distance for main sample points we need to define lines between triangle verticies and the center of the sv
#       and then we need to project the sample point onto this line pluc the distance to previous and or next sample point
#       b)in case for additional ones - those that branch out from last main sample point we will approximate the spread by just getting the distance between
#       the last main sample point and the the vartex of the trangle that we used for it and using it as a diameter of the sphere which volume we will use for weighting those points
  
  
#   Implementation details:
#   probably the bst is to get all of the sample point per tetrahedron to be calculated in sequence over single thread
#   and then parallelize info on tetrahedrons 
#   In case of using float32 and reserving space for shadow memory for enzyme we probably can keep 2-3 floats32 in shared memory per thread
#   for x,y,z indicies of the supervoxels probably uint8 will be sufficient - morover we can unroll the loop
#   we will also try to use generic names of the variables to keep track of the register memory usage 
  
#   tetr_dat - array of 5 points indicies that first is sv center next 3 are verticies of the triangle that creates the base of the tetrahedron plus this triangle center we also have added point to indicate location in control_points so 5x4 array
#   num_base_samp_points - how many main sample points we want to have between each triangle center and sv center in each tetrahedron
#   shared_arr - array of length 3 where we have allocated space for temporarly storing the sample point that we are calculating
#   out_sampled_points - array for storing the value of the sampled point that we got from interpolation and its weight that depends on the proximity to the edges of the tetrahedron and to other points
#       hence the shape of the array is (num_base_samp_points+3)x5 where in second dimension first is the interpolated value of the point and second is its weight the last 3 entries are x,y,z coordinates of sampled point
#   source_arr - array with image on which we are working    
#   control_points - array of where we have the coordinates of the control points of the tetrahedron

#   num_additional_samp_points - how many additional sample points we want to have between the last main sample point and the verticies of the triangle that we used for it
# """
# function set_tetr_dat_kern_forward(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers, max_index)

#   source_arr = CUDA.Const(source_arr)
#   control_points = CUDA.Const(control_points)
#   sv_centers = CUDA.Const(sv_centers)


#   # shared_arr = CuStaticSharedArray(Float32, (CUDA.blockDim_x(),3))
#   shared_arr = CuStaticSharedArray(Float32, (256, 4))
#   index = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))

#   if (index > max_index)
#     return nothing
#   end

#   #setting sv centers data
#   shared_arr[threadIdx().x, 1] = sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), 1]
#   shared_arr[threadIdx().x, 2] = sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), 2]
#   shared_arr[threadIdx().x, 3] = sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), 3]


#   tetr_dat_out[index, 1, 1] = shared_arr[threadIdx().x, 1] #populate tetr dat wth coordinates of sv center
#   tetr_dat_out[index, 1, 2] = shared_arr[threadIdx().x, 2] #populate tetr dat wth coordinates of sv center
#   tetr_dat_out[index, 1, 3] = shared_arr[threadIdx().x, 3] #populate tetr dat wth coordinates of sv center

#   #performing interpolation result is in var2 and it get data from shared_arr

#   #saving the result of sv center value
#   ###interpolate
#   tetr_dat_out[index, 1, 4] = unrolled_trilin_interpol(shared_arr, source_arr)


#   ### end interpolate


#   #get the coordinates of the triangle corners and save them to tetr_dat_out
#   @loopinfo unroll for triangle_corner_num in UInt8(2):UInt8(4)

#     shared_arr[threadIdx().x, 1] = control_points[Int(tetr_dat[index, triangle_corner_num, 1]), Int(tetr_dat[index, triangle_corner_num, 2]), Int(tetr_dat[index, triangle_corner_num, 3]), Int(tetr_dat[index, triangle_corner_num, 4]), 1]
#     shared_arr[threadIdx().x, 2] = control_points[Int(tetr_dat[index, triangle_corner_num, 1]), Int(tetr_dat[index, triangle_corner_num, 2]), Int(tetr_dat[index, triangle_corner_num, 3]), Int(tetr_dat[index, triangle_corner_num, 4]), 2]
#     shared_arr[threadIdx().x, 3] = control_points[Int(tetr_dat[index, triangle_corner_num, 1]), Int(tetr_dat[index, triangle_corner_num, 2]), Int(tetr_dat[index, triangle_corner_num, 3]), Int(tetr_dat[index, triangle_corner_num, 4]), 3]

#     tetr_dat_out[index, triangle_corner_num, 1] = shared_arr[threadIdx().x, 1] #populate tetr dat wth coordinates of triangle corners
#     tetr_dat_out[index, triangle_corner_num, 2] = shared_arr[threadIdx().x, 2] #populate tetr dat wth coordinates of triangle corners
#     tetr_dat_out[index, triangle_corner_num, 3] = shared_arr[threadIdx().x, 3] #populate tetr dat wth coordinates of triangle corners
#     #performing interpolation result is in var2 and it get data from shared_arr
#     #saving the result of control point value
#     tetr_dat_out[index, triangle_corner_num, 4] = unrolled_trilin_variance(shared_arr, source_arr)
#   end#for triangle_corner_num

#   #get the center of the triangle
#   @loopinfo unroll for triangle_corner_num in UInt8(2):UInt8(3)#we do not not need to loop over last triangle as it is already in the shared memory
#     #we add up x,y,z info of all triangles and in the end we divide by 3 to get the center
#     shared_arr[threadIdx().x, 1] += tetr_dat_out[index, triangle_corner_num, 1]
#     shared_arr[threadIdx().x, 2] += tetr_dat_out[index, triangle_corner_num, 2]
#     shared_arr[threadIdx().x, 3] += tetr_dat_out[index, triangle_corner_num, 3]
#   end#for triangle_corner_num
#   for axis in UInt8(1):UInt8(3)
#     shared_arr[threadIdx().x, axis] = shared_arr[threadIdx().x, axis] / 3
#     tetr_dat_out[index, 5, axis] = shared_arr[threadIdx().x, axis]
#   end#for axis
#   #saving the result of centroid value
#   tetr_dat_out[index, 5, 4] = unrolled_trilin_variance(shared_arr, source_arr)

#   return nothing
# end



# function point_info_kern_forward(tetr_dat, out_sampled_points, source_arr, num_base_samp_points, num_additional_samp_points, max_index)
#   shared_arr = CuStaticSharedArray(Float32, (256, 4))
#   index = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))

#   if (index > max_index)
#     return nothing
#   end

#   # we iterate over rest of points in main sample points
#   @loopinfo unroll for point_num in UInt8(1):UInt8(num_base_samp_points)

#     #we get the diffrence between the sv center and the triangle center
#     shared_arr[threadIdx().x, 1] = ((tetr_dat[index, 5, 1] - tetr_dat[index, 1, 1]) / (num_base_samp_points + 1))
#     shared_arr[threadIdx().x, 2] = ((tetr_dat[index, 5, 2] - tetr_dat[index, 1, 2]) / (num_base_samp_points + 1))
#     shared_arr[threadIdx().x, 3] = ((tetr_dat[index, 5, 3] - tetr_dat[index, 1, 3]) / (num_base_samp_points + 1))



#     ##calculate weight of the point
#     #first distance from next and previous point on the line between sv center and triangle center
#     shared_arr[threadIdx().x, 4] = sqrt((shared_arr[threadIdx().x, 1])^2 + (shared_arr[threadIdx().x, 2])^2 + (shared_arr[threadIdx().x, 3])^2) * 2 #distance between main sample points (two times for distance to previous and next)
#     #now we get the distance to the lines that get from sv center to the triangle corners - for simplicity
#     # we can assume that sv center location is 0.0,0.0,0.0 as we need only diffrences 
#     #now we get the location of sample point

#     shared_arr[threadIdx().x, 1] = tetr_dat[index, 1, 1] + (shared_arr[threadIdx().x, 1] * point_num)
#     shared_arr[threadIdx().x, 2] = tetr_dat[index, 1, 2] + (shared_arr[threadIdx().x, 2] * point_num)
#     shared_arr[threadIdx().x, 3] = tetr_dat[index, 1, 3] + (shared_arr[threadIdx().x, 3] * point_num)
#     for triangle_corner_num in UInt8(1):UInt8(3)
#       #if it is last point from base sample points we need to measure its distance to the first additional sample points not to the lines between sv center and triangle corners
#       if (point_num == num_base_samp_points)
#         shared_arr[threadIdx().x, 4] += sqrt(((tetr_dat[index, triangle_corner_num+1, 1] - shared_arr[threadIdx().x, 1])
#                                               *
#                                               (1 / (num_additional_samp_points + 1)))^2
#                                              + ((tetr_dat[index, triangle_corner_num+1, 2] - shared_arr[threadIdx().x, 2])
#                                                 *
#                                                 (1 / (num_additional_samp_points + 1)))^2
#                                              + ((tetr_dat[index, triangle_corner_num+1, 3] - shared_arr[threadIdx().x, 3])
#                                                 *
#                                                 (1 / (num_additional_samp_points + 1)))^2)
#       else
#         #distance to the line between sv center and the one of triangle corners (triangle that is a base of tetrahedron)
#         #in all cases line start as sv center and end as triangle corner and a point is current point in shared memory

#         shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1, 2] - tetr_dat[index, triangle_corner_num+1, 2]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1, 3] - tetr_dat[index, triangle_corner_num+1, 3]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]))^2) +
#                                               (((tetr_dat[index, 1, 3] - tetr_dat[index, triangle_corner_num+1, 3]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1, 1] - tetr_dat[index, triangle_corner_num+1, 1]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]))^2) +
#                                               ((tetr_dat[index, 1, 1] - tetr_dat[index, triangle_corner_num+1, 1]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1, 2] - tetr_dat[index, triangle_corner_num+1, 2]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, triangle_corner_num+1, 2] - tetr_dat[index, 1, 2])^2 + (tetr_dat[index, triangle_corner_num+1, 3] - tetr_dat[index, 1, 3])^2 + (tetr_dat[index, triangle_corner_num+1, 1] - tetr_dat[index, 1, 1])^2)
#       end#if else point_num==num_base_samp_points  
#     end#for triangle_corner_num   


#     #now as we had looked into distance to other points in 5 directions we divide by 5 and save it to the out_sampled_points
#     out_sampled_points[index, point_num, 2] = (((shared_arr[threadIdx().x, 4]) / 5)^3)

#     ##time to get value by interpolation and save it to the out_sampled_points

#     #performing interpolation result is in var2 and it get data from shared_arr
#     #saving the result of interpolated value to the out_sampled_points
#     out_sampled_points[index, point_num, 1] = unrolled_trilin_interpol(shared_arr, source_arr)
#     #saving sample points coordinates mainly for debugging and visualizations
#     out_sampled_points[index, point_num, 3] = shared_arr[threadIdx().x, 1]
#     out_sampled_points[index, point_num, 4] = shared_arr[threadIdx().x, 2]
#     out_sampled_points[index, point_num, 5] = shared_arr[threadIdx().x, 3]




#   end#for num_base_samp_points

#   ##### now we need to calculate the additional sample points that are branching out from the last main sample point
#   @loopinfo unroll for n_add_samp in UInt8(1):UInt8(num_additional_samp_points)
#     @loopinfo unroll for triangle_corner_num in UInt8(1):UInt8(3)

#       #now we need to get diffrence between the last main sample point and the triangle corner
#       shared_arr[threadIdx().x, 1] = ((tetr_dat[index, triangle_corner_num+1, 1] - out_sampled_points[index, num_base_samp_points, 3]) / (num_additional_samp_points + 1))
#       shared_arr[threadIdx().x, 2] = ((tetr_dat[index, triangle_corner_num+1, 2] - out_sampled_points[index, num_base_samp_points, 4]) / (num_additional_samp_points + 1))
#       shared_arr[threadIdx().x, 3] = ((tetr_dat[index, triangle_corner_num+1, 3] - out_sampled_points[index, num_base_samp_points, 5]) / (num_additional_samp_points + 1))

#       #### calculate weight of the point 
#       #first we get distance to the previous and next point on the line between the last main sample point and the triangle corner
#       shared_arr[threadIdx().x, 4] = sqrt(shared_arr[threadIdx().x, 1]^2 + shared_arr[threadIdx().x, 2]^2 + shared_arr[threadIdx().x, 3]^2) * 2
#       #now we get the location of sample point
#       shared_arr[threadIdx().x, 1] = out_sampled_points[index, num_base_samp_points, 3] + (shared_arr[threadIdx().x, 1] * (n_add_samp))
#       shared_arr[threadIdx().x, 2] = out_sampled_points[index, num_base_samp_points, 4] + (shared_arr[threadIdx().x, 2] * (n_add_samp))
#       shared_arr[threadIdx().x, 3] = out_sampled_points[index, num_base_samp_points, 5] + (shared_arr[threadIdx().x, 3] * (n_add_samp))
#       ### now we get the distance of a point to lines joining the corner toward which current line is going and all other verticies of tetrahedron
#       for tetr_dat_index in UInt8(1):UInt8(4)
#         if (tetr_dat_index != (triangle_corner_num + 1)) #this is the corner we are going to in the line for additional sampling points
#           # shared_arr[threadIdx().x,4]+=(tetr_dat[index,triangle_corner_num+1,2]-tetr_dat[index,tetr_dat_index,2]- shared_arr[threadIdx().x,2])
#           shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, triangle_corner_num+1, 2] - tetr_dat[index, tetr_dat_index, 2]) * (tetr_dat[index, triangle_corner_num+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, triangle_corner_num+1, 3] - tetr_dat[index, tetr_dat_index, 3]) * (tetr_dat[index, triangle_corner_num+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
#                                                 (((tetr_dat[index, triangle_corner_num+1, 3] - tetr_dat[index, tetr_dat_index, 3]) * (tetr_dat[index, triangle_corner_num+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, triangle_corner_num+1, 1] - tetr_dat[index, tetr_dat_index, 1]) * (tetr_dat[index, triangle_corner_num+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
#                                                 ((tetr_dat[index, triangle_corner_num+1, 1] - tetr_dat[index, tetr_dat_index, 1]) * (tetr_dat[index, triangle_corner_num+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, triangle_corner_num+1, 2] - tetr_dat[index, tetr_dat_index, 2]) * (tetr_dat[index, triangle_corner_num+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, tetr_dat_index, 2] - tetr_dat[index, triangle_corner_num+1, 2])^2 + (tetr_dat[index, tetr_dat_index, 3] - tetr_dat[index, triangle_corner_num+1, 3])^2 + (tetr_dat[index, tetr_dat_index, 1] - tetr_dat[index, triangle_corner_num+1, 1])^2)
#         end #if
#       end#for

#       out_sampled_points[index, (num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3, 2] = (((shared_arr[threadIdx().x, 4]) / 5)^3)





#       #performing interpolation result is in var2 and it get data from shared_arr
#       #saving the result of interpolated value to the out_sampled_points
#       out_sampled_points[index, (num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3, 1] = unrolled_trilin_interpol(shared_arr, source_arr)
#       # #saving sample points coordinates
#       out_sampled_points[index, (num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3, 3] = shared_arr[threadIdx().x, 1]
#       out_sampled_points[index, (num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3, 4] = shared_arr[threadIdx().x, 2]
#       out_sampled_points[index, (num_base_samp_points+triangle_corner_num)+(n_add_samp-1)*3, 5] = shared_arr[threadIdx().x, 3]







#     end #for triangle_corner_num
#   end #for n_add_samp


#   return nothing
# end

