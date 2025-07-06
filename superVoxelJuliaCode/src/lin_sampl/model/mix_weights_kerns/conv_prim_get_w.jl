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

function kernel_analysis_conv(source_arr, flat_sv_centers, conved, beg_axis_pad, conv_kernels
    ,num_blocks_y_true,half_num_params)
    
      # channel = UInt32(round((blockIdx().y % num_blocks_y_true)+1))
      # param_set_idx = (UInt32(ceil((blockIdx().y / num_blocks_y_true)))+half_num_params*(((threadIdx().x) - 1) ÷ 5))

    shared_mem = CuStaticSharedArray(UInt32, (3,10,5,5))
    shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=flat_sv_centers[blockIdx().x, 1] + beg_axis_pad + (((mod(threadIdx().x - 1, 5) + 1)-3)*2)
    shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]=flat_sv_centers[blockIdx().x, 2] + beg_axis_pad + ((threadIdx().y-3)*2)
    shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]=flat_sv_centers[blockIdx().x, 3] + beg_axis_pad + ((threadIdx().z-3)*2)
    
     conved[blockIdx().x
        ,(mod(threadIdx().x - 1, 5) + 1)
        , threadIdx().y
        , threadIdx().z
        ,(UInt32(ceil((blockIdx().y / num_blocks_y_true)))+half_num_params*(((threadIdx().x) - 1) ÷ 5))
        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] =( (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[1,1,1,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[1,1,2,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[1,1,3,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[1,2,1,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[1,2,2,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[1,2,3,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[1,3,1,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[1,3,2,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[1,3,3,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[2,1,1,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[2,1,2,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[2,1,3,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[2,2,1,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[2,2,2,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[2,2,3,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[2,3,1,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[2,3,2,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[2,3,3,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[3,1,1,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[3,1,2,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[3,1,3,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[3,2,1,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[3,2,2,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[3,2,3,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[3,3,1,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[3,3,2,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        + (source_arr[
                        (shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                        , (shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                        , (shared_mem[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
                    * conv_kernels[3,3,3,threadIdx().x,threadIdx().y,threadIdx().z
                    , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
                    ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
        )

return nothing

end
