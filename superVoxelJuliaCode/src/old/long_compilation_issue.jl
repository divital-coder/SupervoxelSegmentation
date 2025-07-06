using ChainRulesCore, Zygote, CUDA, Enzyme, Test, SplitApplyCombine, Dates, KernelAbstractions,LLVMLoopInfo





function set_tetr_dat_kern_unrolled(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers, max_index)

  index = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))
  if index > max_index
    return nothing
  end

  source_arr = CUDA.Const(source_arr)
  control_points = CUDA.Const(control_points)
  sv_centers = CUDA.Const(sv_centers)
  shared_arr = CuStaticSharedArray(Float32, (256, 4))
  # shared_arr = @localmem Float32 (@groupsize()[1], 4) 

  # index = @index(Global)

  #setting sv centers data
  shared_arr[threadIdx().x, 1] = sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), 1]
  shared_arr[threadIdx().x, 2] = sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), 2]
  shared_arr[threadIdx().x, 3] = sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), 3]


  tetr_dat_out[index, 1, 1] = shared_arr[threadIdx().x, 1] #populate tetr dat wth coordinates of sv center
  tetr_dat_out[index, 1, 2] = shared_arr[threadIdx().x, 2] #populate tetr dat wth coordinates of sv center
  tetr_dat_out[index, 1, 3] = shared_arr[threadIdx().x, 3] #populate tetr dat wth coordinates of sv center

  #performing interpolation result is in var2 and it get data from shared_arr

  #saving the result of sv center value
  ###interpolate
  tetr_dat_out[index, 1, 4] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                *
                                (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                               *
                               (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                               +
                               ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                               *
                               (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))


  ### end interpolate


  #get the coordinates of the triangle corners and save them to tetr_dat_out
  # @loopinfo unroll for triangle_corner_num in UInt8(2):UInt8(4)

  shared_arr[threadIdx().x, 1] = control_points[Int(tetr_dat[index, 2, 1]), Int(tetr_dat[index, 2, 2]), Int(tetr_dat[index, 2, 3]), Int(tetr_dat[index, 2, 4]), 1]
  shared_arr[threadIdx().x, 2] = control_points[Int(tetr_dat[index, 2, 1]), Int(tetr_dat[index, 2, 2]), Int(tetr_dat[index, 2, 3]), Int(tetr_dat[index, 2, 4]), 2]
  shared_arr[threadIdx().x, 3] = control_points[Int(tetr_dat[index, 2, 1]), Int(tetr_dat[index, 2, 2]), Int(tetr_dat[index, 2, 3]), Int(tetr_dat[index, 2, 4]), 3]

  tetr_dat_out[index, 2, 1] = shared_arr[threadIdx().x, 1] #populate tetr dat wth coordinates of triangle corners
  tetr_dat_out[index, 2, 2] = shared_arr[threadIdx().x, 2] #populate tetr dat wth coordinates of triangle corners
  tetr_dat_out[index, 2, 3] = shared_arr[threadIdx().x, 3] #populate tetr dat wth coordinates of triangle corners
  #performing interpolation result is in var2 and it get data from shared_arr
  #saving the result of control point value
  shared_arr[threadIdx().x, 4] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                   *
                                   (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                   (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                  *
                                  (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                  +
                                  ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                   (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                  *
                                  (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
  tetr_dat_out[index, 2, 4] = (((
    ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
    *
    (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
    *
    (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                *
                                (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                (((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                               *
                               (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                               +
                               ((((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                (((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                               *
                               (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))





  shared_arr[threadIdx().x, 1] = control_points[Int(tetr_dat[index, 3, 1]), Int(tetr_dat[index, 3, 2]), Int(tetr_dat[index, 3, 3]), Int(tetr_dat[index, 3, 4]), 1]
  shared_arr[threadIdx().x, 2] = control_points[Int(tetr_dat[index, 3, 1]), Int(tetr_dat[index, 3, 2]), Int(tetr_dat[index, 3, 3]), Int(tetr_dat[index, 3, 4]), 2]
  shared_arr[threadIdx().x, 3] = control_points[Int(tetr_dat[index, 3, 1]), Int(tetr_dat[index, 3, 2]), Int(tetr_dat[index, 3, 3]), Int(tetr_dat[index, 3, 4]), 3]

  tetr_dat_out[index, 3, 1] = shared_arr[threadIdx().x, 1] #populate tetr dat wth coordinates of triangle corners
  tetr_dat_out[index, 3, 2] = shared_arr[threadIdx().x, 2] #populate tetr dat wth coordinates of triangle corners
  tetr_dat_out[index, 3, 3] = shared_arr[threadIdx().x, 3] #populate tetr dat wth coordinates of triangle corners
  #performing interpolation result is in var2 and it get data from shared_arr
  #saving the result of control point value
  shared_arr[threadIdx().x, 4] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                   *
                                   (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                   (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                  *
                                  (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                  +
                                  ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                   (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                  *
                                  (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
  tetr_dat_out[index, 3, 4] = (((
    ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
    *
    (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
    *
    (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                *
                                (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                (((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                               *
                               (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                               +
                               ((((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                (((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                               *
                               (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))





  shared_arr[threadIdx().x, 1] = control_points[Int(tetr_dat[index, 4, 1]), Int(tetr_dat[index, 4, 2]), Int(tetr_dat[index, 4, 3]), Int(tetr_dat[index, 4, 4]), 1]
  shared_arr[threadIdx().x, 2] = control_points[Int(tetr_dat[index, 4, 1]), Int(tetr_dat[index, 4, 2]), Int(tetr_dat[index, 4, 3]), Int(tetr_dat[index, 4, 4]), 2]
  shared_arr[threadIdx().x, 3] = control_points[Int(tetr_dat[index, 4, 1]), Int(tetr_dat[index, 4, 2]), Int(tetr_dat[index, 4, 3]), Int(tetr_dat[index, 4, 4]), 3]

  tetr_dat_out[index, 4, 1] = shared_arr[threadIdx().x, 1] #populate tetr dat wth coordinates of triangle corners
  tetr_dat_out[index, 4, 2] = shared_arr[threadIdx().x, 2] #populate tetr dat wth coordinates of triangle corners
  tetr_dat_out[index, 4, 3] = shared_arr[threadIdx().x, 3] #populate tetr dat wth coordinates of triangle corners
  #performing interpolation result is in var2 and it get data from shared_arr
  #saving the result of control point value
  shared_arr[threadIdx().x, 4] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                   *
                                   (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                   (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                  *
                                  (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                  +
                                  ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                   (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                  *
                                  (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
  tetr_dat_out[index, 4, 4] = (((
    ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
    *
    (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
    *
    (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                *
                                (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                (((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                               *
                               (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                               +
                               ((((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                (((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                               *
                               (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))







  #get the center of the triangle
  # @loopinfo unroll for triangle_corner_num in UInt8(2):UInt8(3)#we do not not need to loop over last triangle as it is already in the shared memory
  #we add up x,y,z info of all triangles and in the end we divide by 3 to get the center
  shared_arr[threadIdx().x, 1] += tetr_dat_out[index, 2, 1]
  shared_arr[threadIdx().x, 2] += tetr_dat_out[index, 2, 2]
  shared_arr[threadIdx().x, 3] += tetr_dat_out[index, 2, 3]

  shared_arr[threadIdx().x, 1] += tetr_dat_out[index, 3, 1]
  shared_arr[threadIdx().x, 2] += tetr_dat_out[index, 3, 2]
  shared_arr[threadIdx().x, 3] += tetr_dat_out[index, 3, 3]

  # for axis in UInt8(1):UInt8(3)
  shared_arr[threadIdx().x, 1] = shared_arr[threadIdx().x, 1] / 3
  tetr_dat_out[index, 5, 1] = shared_arr[threadIdx().x, 1]

  shared_arr[threadIdx().x, 2] = shared_arr[threadIdx().x, 2] / 3
  tetr_dat_out[index, 5, 2] = shared_arr[threadIdx().x, 2]

  shared_arr[threadIdx().x, 3] = shared_arr[threadIdx().x, 3] / 3
  tetr_dat_out[index, 5, 3] = shared_arr[threadIdx().x, 3]
  # end#for axis
  #saving the result of centroid value
  shared_arr[threadIdx().x, 4] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                   *
                                   (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                   (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                  *
                                  (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                  +
                                  ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                   (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                    +
                                    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                   *
                                   (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                  *
                                  (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
  tetr_dat_out[index, 5, 4] = (((
    ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
    *
    (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
    *
    (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                *
                                (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                (((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                               *
                               (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                               +
                               ((((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                (((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                 +
                                 ((source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] - shared_arr[threadIdx().x, 4])^2)
                                 *
                                 (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                *
                                (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                               *
                               (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
  return nothing
end


function point_info_kern_unrolled(tetr_dat, out_sampled_points, source_arr, num_base_samp_points, num_additional_samp_points, max_index)
  shared_arr = CuStaticSharedArray(Float32, (256, 4))
  index = (threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))
  if index > max_index
    return nothing
  end


  # shared_arr = @localmem Float32 (@groupsize()[1], 4) 
  # index = @index(Global)

  #we get the diffrence between the sv center and the triangle center
  shared_arr[threadIdx().x, 1] = ((tetr_dat[index, 5, 1] - tetr_dat[index, 1, 1]) / (3 + 1))
  shared_arr[threadIdx().x, 2] = ((tetr_dat[index, 5, 2] - tetr_dat[index, 1, 2]) / (3 + 1))
  shared_arr[threadIdx().x, 3] = ((tetr_dat[index, 5, 3] - tetr_dat[index, 1, 3]) / (3 + 1))



  ##calculate weight of the point
  #first distance from next and previous point on the line between sv center and triangle center
  shared_arr[threadIdx().x, 4] = sqrt((shared_arr[threadIdx().x, 1])^2 + (shared_arr[threadIdx().x, 2])^2 + (shared_arr[threadIdx().x, 3])^2) * 2 #distance between main sample points (two times for distance to previous and next)
  #now we get the distance to the lines that get from sv center to the triangle corners - for simplicity
  # we can assume that sv center location is 0.0,0.0,0.0 as we need only diffrences 
  #now we get the location of sample point

  shared_arr[threadIdx().x, 1] = tetr_dat[index, 1, 1] + (shared_arr[threadIdx().x, 1] * 1)
  shared_arr[threadIdx().x, 2] = tetr_dat[index, 1, 2] + (shared_arr[threadIdx().x, 2] * 1)
  shared_arr[threadIdx().x, 3] = tetr_dat[index, 1, 3] + (shared_arr[threadIdx().x, 3] * 1)


  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1, 2] - tetr_dat[index, 1+1, 2]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1, 3] - tetr_dat[index, 1+1, 3]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1, 3] - tetr_dat[index, 1+1, 3]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1, 1] - tetr_dat[index, 1+1, 1]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1, 1] - tetr_dat[index, 1+1, 1]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1, 2] - tetr_dat[index, 1+1, 2]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 1+1, 2] - tetr_dat[index, 1, 2])^2 + (tetr_dat[index, 1+1, 3] - tetr_dat[index, 1, 3])^2 + (tetr_dat[index, 1+1, 1] - tetr_dat[index, 1, 1])^2)
  out_sampled_points[index, 1, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                     +
                                     ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  #saving sample points coordinates mainly for debugging and visualizations
  out_sampled_points[index, 1, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, 1, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, 1, 5] = shared_arr[threadIdx().x, 3]


  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1, 2] - tetr_dat[index, 2+1, 2]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1, 3] - tetr_dat[index, 2+1, 3]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1, 3] - tetr_dat[index, 2+1, 3]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1, 1] - tetr_dat[index, 2+1, 1]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1, 1] - tetr_dat[index, 2+1, 1]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1, 2] - tetr_dat[index, 2+1, 2]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 2+1, 2] - tetr_dat[index, 1, 2])^2 + (tetr_dat[index, 2+1, 3] - tetr_dat[index, 1, 3])^2 + (tetr_dat[index, 2+1, 1] - tetr_dat[index, 1, 1])^2)
  out_sampled_points[index, 1, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                     +
                                     ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  #saving sample points coordinates mainly for debugging and visualizations
  out_sampled_points[index, 1, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, 1, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, 1, 5] = shared_arr[threadIdx().x, 3]


  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1, 2] - tetr_dat[index, 3+1, 2]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1, 3] - tetr_dat[index, 3+1, 3]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1, 3] - tetr_dat[index, 3+1, 3]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1, 1] - tetr_dat[index, 3+1, 1]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1, 1] - tetr_dat[index, 3+1, 1]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1, 2] - tetr_dat[index, 3+1, 2]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 3+1, 2] - tetr_dat[index, 1, 2])^2 + (tetr_dat[index, 3+1, 3] - tetr_dat[index, 1, 3])^2 + (tetr_dat[index, 3+1, 1] - tetr_dat[index, 1, 1])^2)
  out_sampled_points[index, 1, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                     +
                                     ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  #saving sample points coordinates mainly for debugging and visualizations
  out_sampled_points[index, 1, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, 1, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, 1, 5] = shared_arr[threadIdx().x, 3]

  out_sampled_points[index, 1, 2] = (((shared_arr[threadIdx().x, 4]) / 5)^3)
  out_sampled_points[index, 1, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                     +
                                     ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  #saving sample points coordinates mainly for debugging and visualizations
  out_sampled_points[index, 1, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, 1, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, 1, 5] = shared_arr[threadIdx().x, 3]


  #we get the diffrence between the sv center and the triangle center
  shared_arr[threadIdx().x, 1] = ((tetr_dat[index, 5, 1] - tetr_dat[index, 1, 1]) / (3 + 1))
  shared_arr[threadIdx().x, 2] = ((tetr_dat[index, 5, 2] - tetr_dat[index, 1, 2]) / (3 + 1))
  shared_arr[threadIdx().x, 3] = ((tetr_dat[index, 5, 3] - tetr_dat[index, 1, 3]) / (3 + 1))



  ##calculate weight of the point
  #first distance from next and previous point on the line between sv center and triangle center
  shared_arr[threadIdx().x, 4] = sqrt((shared_arr[threadIdx().x, 1])^2 + (shared_arr[threadIdx().x, 2])^2 + (shared_arr[threadIdx().x, 3])^2) * 2 #distance between main sample points (two times for distance to previous and next)
  #now we get the distance to the lines that get from sv center to the triangle corners - for simplicity
  # we can assume that sv center location is 0.0,0.0,0.0 as we need only diffrences 
  #now we get the location of sample point

  shared_arr[threadIdx().x, 1] = tetr_dat[index, 1, 1] + (shared_arr[threadIdx().x, 1] * 2)
  shared_arr[threadIdx().x, 2] = tetr_dat[index, 1, 2] + (shared_arr[threadIdx().x, 2] * 2)
  shared_arr[threadIdx().x, 3] = tetr_dat[index, 1, 3] + (shared_arr[threadIdx().x, 3] * 2)


  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1, 2] - tetr_dat[index, 1+1, 2]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1, 3] - tetr_dat[index, 1+1, 3]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1, 3] - tetr_dat[index, 1+1, 3]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1, 1] - tetr_dat[index, 1+1, 1]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1, 1] - tetr_dat[index, 1+1, 1]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1, 2] - tetr_dat[index, 1+1, 2]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 1+1, 2] - tetr_dat[index, 1, 2])^2 + (tetr_dat[index, 1+1, 3] - tetr_dat[index, 1, 3])^2 + (tetr_dat[index, 1+1, 1] - tetr_dat[index, 1, 1])^2)
  out_sampled_points[index, 2, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                     +
                                     ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  #saving sample points coordinates mainly for debugging and visualizations
  out_sampled_points[index, 2, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, 2, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, 2, 5] = shared_arr[threadIdx().x, 3]


  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1, 2] - tetr_dat[index, 2+1, 2]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1, 3] - tetr_dat[index, 2+1, 3]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1, 3] - tetr_dat[index, 2+1, 3]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1, 1] - tetr_dat[index, 2+1, 1]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1, 1] - tetr_dat[index, 2+1, 1]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1, 2] - tetr_dat[index, 2+1, 2]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 2+1, 2] - tetr_dat[index, 1, 2])^2 + (tetr_dat[index, 2+1, 3] - tetr_dat[index, 1, 3])^2 + (tetr_dat[index, 2+1, 1] - tetr_dat[index, 1, 1])^2)
  out_sampled_points[index, 2, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                     +
                                     ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  #saving sample points coordinates mainly for debugging and visualizations
  out_sampled_points[index, 2, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, 2, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, 2, 5] = shared_arr[threadIdx().x, 3]


  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1, 2] - tetr_dat[index, 3+1, 2]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1, 3] - tetr_dat[index, 3+1, 3]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1, 3] - tetr_dat[index, 3+1, 3]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1, 1] - tetr_dat[index, 3+1, 1]) * (tetr_dat[index, 1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1, 1] - tetr_dat[index, 3+1, 1]) * (tetr_dat[index, 1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1, 2] - tetr_dat[index, 3+1, 2]) * (tetr_dat[index, 1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 3+1, 2] - tetr_dat[index, 1, 2])^2 + (tetr_dat[index, 3+1, 3] - tetr_dat[index, 1, 3])^2 + (tetr_dat[index, 3+1, 1] - tetr_dat[index, 1, 1])^2)
  out_sampled_points[index, 2, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                     +
                                     ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  #saving sample points coordinates mainly for debugging and visualizations
  out_sampled_points[index, 2, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, 2, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, 2, 5] = shared_arr[threadIdx().x, 3]

  out_sampled_points[index, 2, 2] = (((shared_arr[threadIdx().x, 4]) / 5)^3)
  out_sampled_points[index, 2, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                     +
                                     ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  #saving sample points coordinates mainly for debugging and visualizations
  out_sampled_points[index, 2, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, 2, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, 2, 5] = shared_arr[threadIdx().x, 3]


  #we get the diffrence between the sv center and the triangle center
  shared_arr[threadIdx().x, 1] = ((tetr_dat[index, 5, 1] - tetr_dat[index, 1, 1]) / (3 + 1))
  shared_arr[threadIdx().x, 2] = ((tetr_dat[index, 5, 2] - tetr_dat[index, 1, 2]) / (3 + 1))
  shared_arr[threadIdx().x, 3] = ((tetr_dat[index, 5, 3] - tetr_dat[index, 1, 3]) / (3 + 1))



  ##calculate weight of the point
  #first distance from next and previous point on the line between sv center and triangle center
  shared_arr[threadIdx().x, 4] = sqrt((shared_arr[threadIdx().x, 1])^2 + (shared_arr[threadIdx().x, 2])^2 + (shared_arr[threadIdx().x, 3])^2) * 2 #distance between main sample points (two times for distance to previous and next)
  #now we get the distance to the lines that get from sv center to the triangle corners - for simplicity
  # we can assume that sv center location is 0.0,0.0,0.0 as we need only diffrences 
  #now we get the location of sample point

  shared_arr[threadIdx().x, 1] = tetr_dat[index, 1, 1] + (shared_arr[threadIdx().x, 1] * 3)
  shared_arr[threadIdx().x, 2] = tetr_dat[index, 1, 2] + (shared_arr[threadIdx().x, 2] * 3)
  shared_arr[threadIdx().x, 3] = tetr_dat[index, 1, 3] + (shared_arr[threadIdx().x, 3] * 3)


  shared_arr[threadIdx().x, 4] += sqrt(((tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1])
                                        *
                                        (1 / (num_additional_samp_points + 1)))^2
                                       + ((tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2])
                                          *
                                          (1 / (num_additional_samp_points + 1)))^2
                                       + ((tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3])
                                          *
                                          (1 / (num_additional_samp_points + 1)))^2)


  shared_arr[threadIdx().x, 4] += sqrt(((tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1])
                                        *
                                        (1 / (num_additional_samp_points + 1)))^2
                                       + ((tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2])
                                          *
                                          (1 / (num_additional_samp_points + 1)))^2
                                       + ((tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3])
                                          *
                                          (1 / (num_additional_samp_points + 1)))^2)


  shared_arr[threadIdx().x, 4] += sqrt(((tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1])
                                        *
                                        (1 / (num_additional_samp_points + 1)))^2
                                       + ((tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2])
                                          *
                                          (1 / (num_additional_samp_points + 1)))^2
                                       + ((tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3])
                                          *
                                          (1 / (num_additional_samp_points + 1)))^2)

  out_sampled_points[index, 3, 2] = (((shared_arr[threadIdx().x, 4]) / 5)^3)
  out_sampled_points[index, 3, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                     +
                                     ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                      (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                       +
                                       source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                      *
                                      (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                     *
                                     (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  #saving sample points coordinates mainly for debugging and visualizations
  out_sampled_points[index, 3, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, 3, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, 3, 5] = shared_arr[threadIdx().x, 3]



  shared_arr[threadIdx().x, 1] = ((tetr_dat[index, 1+1, 1] - out_sampled_points[index, num_base_samp_points, 3]) / (2 + 1))
  shared_arr[threadIdx().x, 2] = ((tetr_dat[index, 1+1, 2] - out_sampled_points[index, num_base_samp_points, 4]) / (2 + 1))
  shared_arr[threadIdx().x, 3] = ((tetr_dat[index, 1+1, 3] - out_sampled_points[index, num_base_samp_points, 5]) / (2 + 1))

  #### calculate weight of the point 
  #first we get distance to the previous and next point on the line between the last main sample point and the triangle corner
  shared_arr[threadIdx().x, 4] = sqrt(shared_arr[threadIdx().x, 1]^2 + shared_arr[threadIdx().x, 2]^2 + shared_arr[threadIdx().x, 3]^2) * 2
  #now we get the location of sample point
  shared_arr[threadIdx().x, 1] = out_sampled_points[index, num_base_samp_points, 3] + (shared_arr[threadIdx().x, 1] * (1))
  shared_arr[threadIdx().x, 2] = out_sampled_points[index, num_base_samp_points, 4] + (shared_arr[threadIdx().x, 2] * (1))
  shared_arr[threadIdx().x, 3] = out_sampled_points[index, num_base_samp_points, 5] + (shared_arr[threadIdx().x, 3] * (1))



  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 1, 2] - tetr_dat[index, 1+1, 2])^2 + (tetr_dat[index, 1, 3] - tetr_dat[index, 1+1, 3])^2 + (tetr_dat[index, 1, 1] - tetr_dat[index, 1+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1+1, 2] - tetr_dat[index, 3, 2]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1+1, 3] - tetr_dat[index, 3, 3]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1+1, 3] - tetr_dat[index, 3, 3]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1+1, 1] - tetr_dat[index, 3, 1]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1+1, 1] - tetr_dat[index, 3, 1]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1+1, 2] - tetr_dat[index, 3, 2]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 3, 2] - tetr_dat[index, 1+1, 2])^2 + (tetr_dat[index, 3, 3] - tetr_dat[index, 1+1, 3])^2 + (tetr_dat[index, 3, 1] - tetr_dat[index, 1+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1+1, 2] - tetr_dat[index, 4, 2]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1+1, 3] - tetr_dat[index, 4, 3]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1+1, 3] - tetr_dat[index, 4, 3]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1+1, 1] - tetr_dat[index, 4, 1]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1+1, 1] - tetr_dat[index, 4, 1]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1+1, 2] - tetr_dat[index, 4, 2]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 4, 2] - tetr_dat[index, 1+1, 2])^2 + (tetr_dat[index, 4, 3] - tetr_dat[index, 1+1, 3])^2 + (tetr_dat[index, 4, 1] - tetr_dat[index, 1+1, 1])^2)



  out_sampled_points[index, (num_base_samp_points+1)+(1-1)*3, 2] = (((shared_arr[threadIdx().x, 4]) / 5)^3)
  #performing interpolation result is in var2 and it get data from shared_arr
  #saving the result of interpolated value to the out_sampled_points
  out_sampled_points[index, (num_base_samp_points+1)+(1-1)*3, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                                                    +
                                                                    ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  # #saving sample points coordinates
  out_sampled_points[index, (num_base_samp_points+1)+(1-1)*3, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, (num_base_samp_points+1)+(1-1)*3, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, (num_base_samp_points+1)+(1-1)*3, 5] = shared_arr[threadIdx().x, 3]


  shared_arr[threadIdx().x, 1] = ((tetr_dat[index, 2+1, 1] - out_sampled_points[index, num_base_samp_points, 3]) / (2 + 1))
  shared_arr[threadIdx().x, 2] = ((tetr_dat[index, 2+1, 2] - out_sampled_points[index, num_base_samp_points, 4]) / (2 + 1))
  shared_arr[threadIdx().x, 3] = ((tetr_dat[index, 2+1, 3] - out_sampled_points[index, num_base_samp_points, 5]) / (2 + 1))

  #### calculate weight of the point 
  #first we get distance to the previous and next point on the line between the last main sample point and the triangle corner
  shared_arr[threadIdx().x, 4] = sqrt(shared_arr[threadIdx().x, 1]^2 + shared_arr[threadIdx().x, 2]^2 + shared_arr[threadIdx().x, 3]^2) * 2
  #now we get the location of sample point
  shared_arr[threadIdx().x, 1] = out_sampled_points[index, num_base_samp_points, 3] + (shared_arr[threadIdx().x, 1] * (1))
  shared_arr[threadIdx().x, 2] = out_sampled_points[index, num_base_samp_points, 4] + (shared_arr[threadIdx().x, 2] * (1))
  shared_arr[threadIdx().x, 3] = out_sampled_points[index, num_base_samp_points, 5] + (shared_arr[threadIdx().x, 3] * (1))



  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 2+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 2+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 2+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 2+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 2+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 2+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 1, 2] - tetr_dat[index, 2+1, 2])^2 + (tetr_dat[index, 1, 3] - tetr_dat[index, 2+1, 3])^2 + (tetr_dat[index, 1, 1] - tetr_dat[index, 2+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 2+1, 2] - tetr_dat[index, 2, 2]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 2+1, 3] - tetr_dat[index, 2, 3]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 2+1, 3] - tetr_dat[index, 2, 3]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 2+1, 1] - tetr_dat[index, 2, 1]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 2+1, 1] - tetr_dat[index, 2, 1]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 2+1, 2] - tetr_dat[index, 2, 2]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 2, 2] - tetr_dat[index, 2+1, 2])^2 + (tetr_dat[index, 2, 3] - tetr_dat[index, 2+1, 3])^2 + (tetr_dat[index, 2, 1] - tetr_dat[index, 2+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 2+1, 2] - tetr_dat[index, 4, 2]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 2+1, 3] - tetr_dat[index, 4, 3]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 2+1, 3] - tetr_dat[index, 4, 3]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 2+1, 1] - tetr_dat[index, 4, 1]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 2+1, 1] - tetr_dat[index, 4, 1]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 2+1, 2] - tetr_dat[index, 4, 2]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 4, 2] - tetr_dat[index, 2+1, 2])^2 + (tetr_dat[index, 4, 3] - tetr_dat[index, 2+1, 3])^2 + (tetr_dat[index, 4, 1] - tetr_dat[index, 2+1, 1])^2)



  out_sampled_points[index, (num_base_samp_points+2)+(1-1)*3, 2] = (((shared_arr[threadIdx().x, 4]) / 5)^3)
  #performing interpolation result is in var2 and it get data from shared_arr
  #saving the result of interpolated value to the out_sampled_points
  out_sampled_points[index, (num_base_samp_points+2)+(1-1)*3, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                                                    +
                                                                    ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  # #saving sample points coordinates
  out_sampled_points[index, (num_base_samp_points+2)+(1-1)*3, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, (num_base_samp_points+2)+(1-1)*3, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, (num_base_samp_points+2)+(1-1)*3, 5] = shared_arr[threadIdx().x, 3]


  shared_arr[threadIdx().x, 1] = ((tetr_dat[index, 3+1, 1] - out_sampled_points[index, num_base_samp_points, 3]) / (2 + 1))
  shared_arr[threadIdx().x, 2] = ((tetr_dat[index, 3+1, 2] - out_sampled_points[index, num_base_samp_points, 4]) / (2 + 1))
  shared_arr[threadIdx().x, 3] = ((tetr_dat[index, 3+1, 3] - out_sampled_points[index, num_base_samp_points, 5]) / (2 + 1))

  #### calculate weight of the point 
  #first we get distance to the previous and next point on the line between the last main sample point and the triangle corner
  shared_arr[threadIdx().x, 4] = sqrt(shared_arr[threadIdx().x, 1]^2 + shared_arr[threadIdx().x, 2]^2 + shared_arr[threadIdx().x, 3]^2) * 2
  #now we get the location of sample point
  shared_arr[threadIdx().x, 1] = out_sampled_points[index, num_base_samp_points, 3] + (shared_arr[threadIdx().x, 1] * (1))
  shared_arr[threadIdx().x, 2] = out_sampled_points[index, num_base_samp_points, 4] + (shared_arr[threadIdx().x, 2] * (1))
  shared_arr[threadIdx().x, 3] = out_sampled_points[index, num_base_samp_points, 5] + (shared_arr[threadIdx().x, 3] * (1))



  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 3+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 3+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 3+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 3+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 3+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 3+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 1, 2] - tetr_dat[index, 3+1, 2])^2 + (tetr_dat[index, 1, 3] - tetr_dat[index, 3+1, 3])^2 + (tetr_dat[index, 1, 1] - tetr_dat[index, 3+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 3+1, 2] - tetr_dat[index, 2, 2]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 3+1, 3] - tetr_dat[index, 2, 3]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 3+1, 3] - tetr_dat[index, 2, 3]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 3+1, 1] - tetr_dat[index, 2, 1]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 3+1, 1] - tetr_dat[index, 2, 1]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 3+1, 2] - tetr_dat[index, 2, 2]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 2, 2] - tetr_dat[index, 3+1, 2])^2 + (tetr_dat[index, 2, 3] - tetr_dat[index, 3+1, 3])^2 + (tetr_dat[index, 2, 1] - tetr_dat[index, 3+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 3+1, 2] - tetr_dat[index, 3, 2]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 3+1, 3] - tetr_dat[index, 3, 3]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 3+1, 3] - tetr_dat[index, 3, 3]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 3+1, 1] - tetr_dat[index, 3, 1]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 3+1, 1] - tetr_dat[index, 3, 1]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 3+1, 2] - tetr_dat[index, 3, 2]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 3, 2] - tetr_dat[index, 3+1, 2])^2 + (tetr_dat[index, 3, 3] - tetr_dat[index, 3+1, 3])^2 + (tetr_dat[index, 3, 1] - tetr_dat[index, 3+1, 1])^2)



  out_sampled_points[index, (num_base_samp_points+3)+(1-1)*3, 2] = (((shared_arr[threadIdx().x, 4]) / 5)^3)
  #performing interpolation result is in var2 and it get data from shared_arr
  #saving the result of interpolated value to the out_sampled_points
  out_sampled_points[index, (num_base_samp_points+3)+(1-1)*3, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                                                    +
                                                                    ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  # #saving sample points coordinates
  out_sampled_points[index, (num_base_samp_points+3)+(1-1)*3, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, (num_base_samp_points+3)+(1-1)*3, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, (num_base_samp_points+3)+(1-1)*3, 5] = shared_arr[threadIdx().x, 3]


  shared_arr[threadIdx().x, 1] = ((tetr_dat[index, 1+1, 1] - out_sampled_points[index, num_base_samp_points, 3]) / (2 + 1))
  shared_arr[threadIdx().x, 2] = ((tetr_dat[index, 1+1, 2] - out_sampled_points[index, num_base_samp_points, 4]) / (2 + 1))
  shared_arr[threadIdx().x, 3] = ((tetr_dat[index, 1+1, 3] - out_sampled_points[index, num_base_samp_points, 5]) / (2 + 1))

  #### calculate weight of the point 
  #first we get distance to the previous and next point on the line between the last main sample point and the triangle corner
  shared_arr[threadIdx().x, 4] = sqrt(shared_arr[threadIdx().x, 1]^2 + shared_arr[threadIdx().x, 2]^2 + shared_arr[threadIdx().x, 3]^2) * 2
  #now we get the location of sample point
  shared_arr[threadIdx().x, 1] = out_sampled_points[index, num_base_samp_points, 3] + (shared_arr[threadIdx().x, 1] * (2))
  shared_arr[threadIdx().x, 2] = out_sampled_points[index, num_base_samp_points, 4] + (shared_arr[threadIdx().x, 2] * (2))
  shared_arr[threadIdx().x, 3] = out_sampled_points[index, num_base_samp_points, 5] + (shared_arr[threadIdx().x, 3] * (2))



  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 1, 2] - tetr_dat[index, 1+1, 2])^2 + (tetr_dat[index, 1, 3] - tetr_dat[index, 1+1, 3])^2 + (tetr_dat[index, 1, 1] - tetr_dat[index, 1+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1+1, 2] - tetr_dat[index, 3, 2]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1+1, 3] - tetr_dat[index, 3, 3]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1+1, 3] - tetr_dat[index, 3, 3]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1+1, 1] - tetr_dat[index, 3, 1]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1+1, 1] - tetr_dat[index, 3, 1]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1+1, 2] - tetr_dat[index, 3, 2]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 3, 2] - tetr_dat[index, 1+1, 2])^2 + (tetr_dat[index, 3, 3] - tetr_dat[index, 1+1, 3])^2 + (tetr_dat[index, 3, 1] - tetr_dat[index, 1+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 1+1, 2] - tetr_dat[index, 4, 2]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 1+1, 3] - tetr_dat[index, 4, 3]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 1+1, 3] - tetr_dat[index, 4, 3]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 1+1, 1] - tetr_dat[index, 4, 1]) * (tetr_dat[index, 1+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 1+1, 1] - tetr_dat[index, 4, 1]) * (tetr_dat[index, 1+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 1+1, 2] - tetr_dat[index, 4, 2]) * (tetr_dat[index, 1+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 4, 2] - tetr_dat[index, 1+1, 2])^2 + (tetr_dat[index, 4, 3] - tetr_dat[index, 1+1, 3])^2 + (tetr_dat[index, 4, 1] - tetr_dat[index, 1+1, 1])^2)



  out_sampled_points[index, (num_base_samp_points+1)+(2-1)*3, 2] = (((shared_arr[threadIdx().x, 4]) / 5)^3)
  #performing interpolation result is in var2 and it get data from shared_arr
  #saving the result of interpolated value to the out_sampled_points
  out_sampled_points[index, (num_base_samp_points+1)+(2-1)*3, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                                                    +
                                                                    ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  # #saving sample points coordinates
  out_sampled_points[index, (num_base_samp_points+1)+(2-1)*3, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, (num_base_samp_points+1)+(2-1)*3, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, (num_base_samp_points+1)+(2-1)*3, 5] = shared_arr[threadIdx().x, 3]


  shared_arr[threadIdx().x, 1] = ((tetr_dat[index, 2+1, 1] - out_sampled_points[index, num_base_samp_points, 3]) / (2 + 1))
  shared_arr[threadIdx().x, 2] = ((tetr_dat[index, 2+1, 2] - out_sampled_points[index, num_base_samp_points, 4]) / (2 + 1))
  shared_arr[threadIdx().x, 3] = ((tetr_dat[index, 2+1, 3] - out_sampled_points[index, num_base_samp_points, 5]) / (2 + 1))

  #### calculate weight of the point 
  #first we get distance to the previous and next point on the line between the last main sample point and the triangle corner
  shared_arr[threadIdx().x, 4] = sqrt(shared_arr[threadIdx().x, 1]^2 + shared_arr[threadIdx().x, 2]^2 + shared_arr[threadIdx().x, 3]^2) * 2
  #now we get the location of sample point
  shared_arr[threadIdx().x, 1] = out_sampled_points[index, num_base_samp_points, 3] + (shared_arr[threadIdx().x, 1] * (2))
  shared_arr[threadIdx().x, 2] = out_sampled_points[index, num_base_samp_points, 4] + (shared_arr[threadIdx().x, 2] * (2))
  shared_arr[threadIdx().x, 3] = out_sampled_points[index, num_base_samp_points, 5] + (shared_arr[threadIdx().x, 3] * (2))



  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 2+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 2+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 2+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 2+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 2+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 2+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 1, 2] - tetr_dat[index, 2+1, 2])^2 + (tetr_dat[index, 1, 3] - tetr_dat[index, 2+1, 3])^2 + (tetr_dat[index, 1, 1] - tetr_dat[index, 2+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 2+1, 2] - tetr_dat[index, 2, 2]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 2+1, 3] - tetr_dat[index, 2, 3]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 2+1, 3] - tetr_dat[index, 2, 3]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 2+1, 1] - tetr_dat[index, 2, 1]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 2+1, 1] - tetr_dat[index, 2, 1]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 2+1, 2] - tetr_dat[index, 2, 2]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 2, 2] - tetr_dat[index, 2+1, 2])^2 + (tetr_dat[index, 2, 3] - tetr_dat[index, 2+1, 3])^2 + (tetr_dat[index, 2, 1] - tetr_dat[index, 2+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 2+1, 2] - tetr_dat[index, 4, 2]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 2+1, 3] - tetr_dat[index, 4, 3]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 2+1, 3] - tetr_dat[index, 4, 3]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 2+1, 1] - tetr_dat[index, 4, 1]) * (tetr_dat[index, 2+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 2+1, 1] - tetr_dat[index, 4, 1]) * (tetr_dat[index, 2+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 2+1, 2] - tetr_dat[index, 4, 2]) * (tetr_dat[index, 2+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 4, 2] - tetr_dat[index, 2+1, 2])^2 + (tetr_dat[index, 4, 3] - tetr_dat[index, 2+1, 3])^2 + (tetr_dat[index, 4, 1] - tetr_dat[index, 2+1, 1])^2)



  out_sampled_points[index, (num_base_samp_points+2)+(2-1)*3, 2] = (((shared_arr[threadIdx().x, 4]) / 5)^3)
  #performing interpolation result is in var2 and it get data from shared_arr
  #saving the result of interpolated value to the out_sampled_points
  out_sampled_points[index, (num_base_samp_points+2)+(2-1)*3, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                                                    +
                                                                    ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  # #saving sample points coordinates
  out_sampled_points[index, (num_base_samp_points+2)+(2-1)*3, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, (num_base_samp_points+2)+(2-1)*3, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, (num_base_samp_points+2)+(2-1)*3, 5] = shared_arr[threadIdx().x, 3]


  shared_arr[threadIdx().x, 1] = ((tetr_dat[index, 3+1, 1] - out_sampled_points[index, num_base_samp_points, 3]) / (2 + 1))
  shared_arr[threadIdx().x, 2] = ((tetr_dat[index, 3+1, 2] - out_sampled_points[index, num_base_samp_points, 4]) / (2 + 1))
  shared_arr[threadIdx().x, 3] = ((tetr_dat[index, 3+1, 3] - out_sampled_points[index, num_base_samp_points, 5]) / (2 + 1))

  #### calculate weight of the point 
  #first we get distance to the previous and next point on the line between the last main sample point and the triangle corner
  shared_arr[threadIdx().x, 4] = sqrt(shared_arr[threadIdx().x, 1]^2 + shared_arr[threadIdx().x, 2]^2 + shared_arr[threadIdx().x, 3]^2) * 2
  #now we get the location of sample point
  shared_arr[threadIdx().x, 1] = out_sampled_points[index, num_base_samp_points, 3] + (shared_arr[threadIdx().x, 1] * (2))
  shared_arr[threadIdx().x, 2] = out_sampled_points[index, num_base_samp_points, 4] + (shared_arr[threadIdx().x, 2] * (2))
  shared_arr[threadIdx().x, 3] = out_sampled_points[index, num_base_samp_points, 5] + (shared_arr[threadIdx().x, 3] * (2))



  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 3+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 3+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 3+1, 3] - tetr_dat[index, 1, 3]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 3+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 3+1, 1] - tetr_dat[index, 1, 1]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 3+1, 2] - tetr_dat[index, 1, 2]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 1, 2] - tetr_dat[index, 3+1, 2])^2 + (tetr_dat[index, 1, 3] - tetr_dat[index, 3+1, 3])^2 + (tetr_dat[index, 1, 1] - tetr_dat[index, 3+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 3+1, 2] - tetr_dat[index, 2, 2]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 3+1, 3] - tetr_dat[index, 2, 3]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 3+1, 3] - tetr_dat[index, 2, 3]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 3+1, 1] - tetr_dat[index, 2, 1]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 3+1, 1] - tetr_dat[index, 2, 1]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 3+1, 2] - tetr_dat[index, 2, 2]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 2, 2] - tetr_dat[index, 3+1, 2])^2 + (tetr_dat[index, 2, 3] - tetr_dat[index, 3+1, 3])^2 + (tetr_dat[index, 2, 1] - tetr_dat[index, 3+1, 1])^2)




  shared_arr[threadIdx().x, 4] += sqrt(((((tetr_dat[index, 3+1, 2] - tetr_dat[index, 3, 2]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]) - (tetr_dat[index, 3+1, 3] - tetr_dat[index, 3, 3]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]))^2) +
                                        (((tetr_dat[index, 3+1, 3] - tetr_dat[index, 3, 3]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]) - (tetr_dat[index, 3+1, 1] - tetr_dat[index, 3, 1]) * (tetr_dat[index, 3+1, 3] - shared_arr[threadIdx().x, 3]))^2) +
                                        ((tetr_dat[index, 3+1, 1] - tetr_dat[index, 3, 1]) * (tetr_dat[index, 3+1, 2] - shared_arr[threadIdx().x, 2]) - (tetr_dat[index, 3+1, 2] - tetr_dat[index, 3, 2]) * (tetr_dat[index, 3+1, 1] - shared_arr[threadIdx().x, 1]))^2)) / sqrt((tetr_dat[index, 3, 2] - tetr_dat[index, 3+1, 2])^2 + (tetr_dat[index, 3, 3] - tetr_dat[index, 3+1, 3])^2 + (tetr_dat[index, 3, 1] - tetr_dat[index, 3+1, 1])^2)



  out_sampled_points[index, (num_base_samp_points+3)+(2-1)*3, 2] = (((shared_arr[threadIdx().x, 4]) / 5)^3)
  #performing interpolation result is in var2 and it get data from shared_arr
  #saving the result of interpolated value to the out_sampled_points
  out_sampled_points[index, (num_base_samp_points+3)+(2-1)*3, 1] = (((
    source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))) +
    source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1])))
  )
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(floor(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (1 - (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))
                                                                    +
                                                                    ((source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(floor(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (1 - (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2])))) +
                                                                     (source_arr[Int(floor(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (1 - (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                      +
                                                                      source_arr[Int(ceil(shared_arr[threadIdx().x, 1])), Int(ceil(shared_arr[threadIdx().x, 2])), Int(ceil(shared_arr[threadIdx().x, 3]))] * (shared_arr[threadIdx().x, 1] - Int(floor(shared_arr[threadIdx().x, 1]))))
                                                                     *
                                                                     (shared_arr[threadIdx().x, 2] - Int(floor(shared_arr[threadIdx().x, 2]))))
                                                                    *
                                                                    (shared_arr[threadIdx().x, 3] - Int(floor(shared_arr[threadIdx().x, 3]))))

  (shared_arr, source_arr)
  # #saving sample points coordinates
  out_sampled_points[index, (num_base_samp_points+3)+(2-1)*3, 3] = shared_arr[threadIdx().x, 1]
  out_sampled_points[index, (num_base_samp_points+3)+(2-1)*3, 4] = shared_arr[threadIdx().x, 2]
  out_sampled_points[index, (num_base_samp_points+3)+(2-1)*3, 5] = shared_arr[threadIdx().x, 3]



  return nothing

end



















"""
get 4 dimensional array of cartesian indicies of a 3 dimensional array
thats size is passed as an argument dims
"""
function get_base_indicies_arr(dims)
  indices = CartesianIndices(dims)
  # indices=collect.(Tuple.(collect(indices)))
  indices = Tuple.(collect(indices))
  indices = collect(Iterators.flatten(indices))
  indices = reshape(indices, (3, dims[1], dims[2], dims[3]))
  indices = permutedims(indices, (2, 3, 4, 1))
  return indices
end#get_base_indicies_arr

function get_corrected_dim(ax, radius, image_shape)
  diam = radius * 2
  return Int(ceil((image_shape[ax] - 5) / diam)) - 2
end

function get_dif(ax, image_shape, dims, diam)
  return max(floor((image_shape[ax] - ((dims[ax] + 1) .* diam)) / 2), 2.0)
end

"""
initialize sv centers coordinates 
"""
function get_sv_centers(radius, image_shape)
  diam = radius * 2
  dims = (get_corrected_dim(1, radius, image_shape), get_corrected_dim(2, radius, image_shape), get_corrected_dim(3, radius, image_shape))
  diffs = (get_dif(1, image_shape, dims, diam), get_dif(2, image_shape, dims, diam), get_dif(3, image_shape, dims, diam))
  # diffs= (1.0,1.0,1.0)
  res = get_base_indicies_arr(dims) * diam
  res[:, :, :, 1] = res[:, :, :, 1] .+ diffs[1]
  res[:, :, :, 2] = res[:, :, :, 2] .+ diffs[2]
  res[:, :, :, 3] = res[:, :, :, 3] .+ diffs[3]
  return res, dims, diffs

end



"""
given the size of the x,y,z dimension of control weights (what in basic architecture get as output of convolutions)
we get linear control points - so points that are on the lines between each of the sv_centers - hence their modifications will require just one weight
we will create linear points by moving by radius in each axis
"""
function get_linear_control_points(dims, axis, diam, radius, diffs)
  #increasing dimension as we need to have them both up and down the axis
  # dim_new=collect(Iterators.flatten(dims))#.+1
  # dim_new[axis]=dim_new[axis]+1
  dim_new = collect(Iterators.flatten(dims)) .+ 1
  indicies = get_base_indicies_arr(Tuple(dim_new)) .- 1
  indicies = indicies .* diam
  indicies = indicies .+ diam
  indicies_ax = indicies[:, :, :, axis] .- radius
  indicies[:, :, :, axis] = indicies_ax
  res = indicies
  res[:, :, :, 1] = res[:, :, :, 1] .+ diffs[1]
  res[:, :, :, 2] = res[:, :, :, 2] .+ diffs[2]
  res[:, :, :, 3] = res[:, :, :, 3] .+ diffs[3]

  return res
end#get_linear_control_points


"""
will get oblique control points - so points that are on the corners of the cube that is enclosing a volume of
non modified supervoxel area 
"""
function get_oblique_control_points(dims, diam, radius, diffs)
  #increasing dimension as we need to have them both up and down the axis
  dim_new = collect(Iterators.flatten(dims)) .+ 1
  indicies = get_base_indicies_arr(Tuple(dim_new)) .- 1
  indicies = indicies .* diam
  res = indicies .+ radius
  res[:, :, :, 1] = res[:, :, :, 1] .+ diffs[1]
  res[:, :, :, 2] = res[:, :, :, 2] .+ diffs[2]
  res[:, :, :, 3] = res[:, :, :, 3] .+ diffs[3]

  return res


end#get_oblique_control_points


"""
flips the value of the index of the tuple at the position ind needed for get_linear_between function
"""
function flip_num(base_ind, tupl, ind)
  arr = collect(tupl)
  # arr=append!(arr,[4])
  if (arr[ind] == base_ind[ind])
    arr[ind] = base_ind[ind] + 1
  else
    arr[ind] = base_ind[ind]
  end
  return arr
end

"""
we can identify the line between two corners that go obliquely through the wall of the cube
it connects points that has 2 coordinates diffrent and one the same 
we can also find a point in the middle so it will be in lin_x if this common index is 1 and in lin_y if it is 2 and lin_z if 3
next if we have 1 it is pre and if 2 post
    control_points first dimension is lin_x, lin_y, lin_z, oblique
"""
function get_linear_between(base_ind, ind_1, ind_2)
  if (ind_1[1] == ind_2[1])
    return [ind_1[1], base_ind[2], base_ind[3], 1]
  end
  if (ind_1[2] == ind_2[2])
    return [base_ind[1], ind_1[2], base_ind[3], 2]
  end

  return [base_ind[1], base_ind[2], ind_1[3], 3]
end



"""
from control points and base index of the supervoxel we get the tetrahedron that is created by the center of the supervoxel
and the control points we are creating here all tetrahedrons that cover one of the corners of the cube
"""
function get_tetr_triangles_in_corner(base_ind, corner)
  corner = Float32.(append!(collect(corner), [4]))

  sv_center = Float32.([base_ind[1], base_ind[2], base_ind[3], -1.0])
  p_a = Float32.(flip_num(base_ind, corner, 1))
  p_b = Float32.(flip_num(base_ind, corner, 2))
  p_c = Float32.(flip_num(base_ind, corner, 3))


  p_ab = Float32.(get_linear_between(base_ind, p_a, p_b))
  p_ac = Float32.(get_linear_between(base_ind, p_a, p_c))
  p_bc = Float32.(get_linear_between(base_ind, p_b, p_c))

  dummy = Float32.([-1.0, -1.0, -1.0, -1.0])

  res = [[sv_center;; corner;; p_a;; p_ab;; dummy], [sv_center;; corner;; p_ab;; p_b;; dummy], [sv_center;; corner;; p_b;; p_bc;; dummy], [sv_center;; corner;; p_bc;; p_c;; dummy], [sv_center;; corner;; p_a;; p_ac;; dummy], [sv_center;; corner;; p_ac;; p_c;; dummy]
  ]

  res = map(el -> permutedims(el, (2, 1)), res)
  res = map(el -> reshape(el, (1, size(el)...)), res)
  res = vcat(res...)
  return res

end



"""
given indicies of current supervoxel and control points we get all triangles that are covering the surface of the supervoxel
"""
function get_all_surface_triangles_of_sv(base_ind)
  return [
    get_tetr_triangles_in_corner(base_ind, (base_ind[1], base_ind[2], base_ind[3])), get_tetr_triangles_in_corner(base_ind, (base_ind[1] + 1, base_ind[2] + 1, base_ind[3])), get_tetr_triangles_in_corner(base_ind, (base_ind[1], base_ind[2] + 1, base_ind[3] + 1)), get_tetr_triangles_in_corner(base_ind, (base_ind[1] + 1, base_ind[2], base_ind[3] + 1))
  ]
end #get_tetrahedrons_of_sv

"""
get a flattened array of all surface triangles of all supervoxels
in first dimension every 24 elements are a single supervoxel
second dimension is size 5 and is in orde sv_center, point a,point b,point c,centroid 
    where centroid is a placeholder for centroid of the triangle a,b,c
in last dimension we have x,y,z coordinates of the point
currently we have just indicies to the appropriate arrays -> it need to be populated after weights get applied        
"""
function get_flattened_triangle_data(dims)
  indices = CartesianIndices(dims)
  # indices=collect.(Tuple.(collect(indices)))
  indices = Tuple.(collect(indices))
  indices = collect(Iterators.flatten(indices))
  indices = reshape(indices, (3, dims[1] * dims[2] * dims[3]))
  indices = permutedims(indices, (2, 1))

  indices = splitdims(indices, 1)

  all_surf_triangles = map(el -> get_all_surface_triangles_of_sv(el), indices)
  #concatenate all on first dimension
  all_surf_triangles = map(el -> vcat(el...), all_surf_triangles)
  all_surf_triangles = vcat(all_surf_triangles...)


  return all_surf_triangles
end


"""
given the size of the x,y,z dimension of control weights (what in basic architecture get as output of convolutions)
and the radius of supervoxels will return the grid of points that will be used as centers of supervoxels 
and the intilia positions of the control points
"""
function initialize_centers_and_control_points(image_shape, radius)
  diam = radius * 2
  sv_centers, dims, diffs = get_sv_centers(radius, image_shape)
  lin_x = get_linear_control_points(dims, 1, diam, radius, diffs)
  lin_y = get_linear_control_points(dims, 2, diam, radius, diffs)
  lin_z = get_linear_control_points(dims, 3, diam, radius, diffs)
  oblique = get_oblique_control_points(dims, diam, radius, diffs)

  flattened_triangles = get_flattened_triangle_data(dims)

  return sv_centers, combinedims([lin_x, lin_y, lin_z, oblique], 4), flattened_triangles, dims
end#initialize_centeris_and_control_points    

function get_current_time()
  return Dates.now()
end

function prepare_for_kern(tetr_dat_shape)
  threads = 256

  needed_blocks = ceil(Int, tetr_dat_shape[1] / threads)
  to_pad = (threads * needed_blocks) - tetr_dat_shape[1]

  return threads, needed_blocks, to_pad
end


radiuss = Float32(4.0)
diam = radiuss * 2
num_weights_per_point = 6
a = 36
image_shape = (a, a, a)

example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
sv_centers, control_points, tetrs, dims = example_set_of_svs
source_arr = rand(Float32, image_shape)
num_base_samp_points, num_additional_samp_points = 3, 2


#get the number of threads and blocks needed for the kernel
threads_point_info, blocks_point_info, pad_point_info = prepare_for_kern(size(tetrs))
max_index = size(tetrs)[1]
num_base_samp_points, num_additional_samp_points = 3, 2
#put on GPU
tetr_dat = CuArray(Float32.(tetrs))
sv_centers = CuArray(Float32.(sv_centers))
control_points = CuArray(Float32.(control_points))
source_arr = CuArray(Float32.(source_arr))
tetr_dat_out = CUDA.zeros(size(tetrs)...)

out_sampled_points = CUDA.zeros((size(tetr_dat)[1], num_base_samp_points + (3 * num_additional_samp_points), 5))
#initialize shadow memory
d_tetr_dat_out = CUDA.ones(size(tetr_dat)...)
d_out_sampled_points = CUDA.ones(size(out_sampled_points)...)

d_tetr_dat = CUDA.zeros(size(tetr_dat)...)
d_source_arr = CUDA.zeros(size(source_arr)...)
d_control_points = CUDA.zeros(size(control_points)...)
d_sv_centers = CUDA.zeros(size(sv_centers)...)


### execute kernel no autodiff
@cuda threads = threads_point_info blocks = blocks_point_info set_tetr_dat_kern_unrolled(tetr_dat, tetr_dat_out, source_arr, control_points, sv_centers, max_index)
@cuda threads = threads_point_info blocks = blocks_point_info point_info_kern_unrolled(tetr_dat_out, out_sampled_points, source_arr, num_base_samp_points, num_additional_samp_points, max_index)
tetr_dat = tetr_dat_out
### execute kernel with autodiff

function set_tetr_dat_kern_deff(tetr_dat, d_tetr_dat, tetr_dat_out, d_tetr_dat_out, source_arr, d_source_arr, control_points, d_control_points, sv_centers, d_sv_centers, max_index)
  Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(set_tetr_dat_kern_unrolled), Const, Duplicated(tetr_dat, d_tetr_dat), Duplicated(tetr_dat_out, d_tetr_dat_out), Duplicated(source_arr, d_source_arr), Duplicated(control_points, d_control_points), Duplicated(sv_centers, d_sv_centers), Const(max_index))
  return nothing
end

function set_point_info_kern_deff(tetr_dat, d_tetr_dat, out_sampled_points, d_out_sampled_points, source_arr, d_source_arr, num_base_samp_points, num_additional_samp_points, max_index)
  Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(point_info_kern_unrolled), Const, Duplicated(tetr_dat, d_tetr_dat), Duplicated(out_sampled_points, d_out_sampled_points), Duplicated(source_arr, d_source_arr), Const(num_base_samp_points), Const(num_additional_samp_points), Const(max_index))
  return nothing
end




current_time = get_current_time()
@cuda threads = threads_point_info blocks = blocks_point_info set_tetr_dat_kern_deff(tetr_dat,d_tetr_dat, tetr_dat_out, d_tetr_dat_out, source_arr,d_source_arr, control_points,d_control_points, sv_centers,d_sv_centers,max_index)
CUDA.synchronize()

println("Time taken smaller kernel (minutes): ", Dates.value(get_current_time() - current_time) / 60000.0) #15.297383333333334

current_time = get_current_time()
@cuda threads = threads_point_info blocks = blocks_point_info set_point_info_kern_deff(tetr_dat, d_tetr_dat, out_sampled_points, d_out_sampled_points, source_arr, d_source_arr, num_base_samp_points, num_additional_samp_points, max_index)

println("Time taken bigger kernel (minutes): ", Dates.value(get_current_time() - current_time) / 60000.0)






