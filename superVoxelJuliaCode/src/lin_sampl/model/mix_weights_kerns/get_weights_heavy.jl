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


```
first get all points around the sv center in this case for 5 voxels in each direction in 3 axes , then it 
    uses 3x3x convolution to reduce number of points we are intrested in - so in practice after this convolution we can analize every second voxel
   then we are using lrelu to mix information from all loaded convolved points 
   last we are doing reductions - hevewer reductions are parametrirized and puprosfully performed multiple times in paralles so we will get 
   5x5 result from 5x5x5 reduction
   we are performing all operations separately for each parameter set, each channel and each batch
   in order to make it a bit faster we will do 2 parameter sets at once in one kernel
```

function kernel_analysis(source_arr,flat_sv_centers,conv_kernels, param_mixing
        , param_reducing,param_reducing_b, param_reducing_c,result
        ,num_blocks_y_true,dims,half_num_params,beg_axis_pad)
        
      # Calculate thread and block indices
      # x = threadIdx().x
      # y = threadIdx().y
      # z = threadIdx().z
      # bx = blockIdx().x
      # by = blockIdx().y
      # bz = blockIdx().z
      # Calculate channel and parameter set index
      
      # channel = UInt32(round((blockIdx().y % num_blocks_y_true)+1))
      # param_set_idx = (UInt32(ceil((blockIdx().y / num_blocks_y_true)))+half_num_params*(((threadIdx().x) - 1) ÷ 5))

        shared_mem = CuStaticSharedArray(Float32, (2,10,5,5))
        shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=0.0
         shared_mem_conv = CuStaticSharedArray(UInt16, (3,10,5,5))
shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z]=flat_sv_centers[blockIdx().x, 1] + beg_axis_pad + (((mod(threadIdx().x - 1, 5) + 1)-3)*2)
shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]=flat_sv_centers[blockIdx().x, 2] + beg_axis_pad + ((threadIdx().y-3)*2)
shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]=flat_sv_centers[blockIdx().x, 3] + beg_axis_pad + ((threadIdx().z-3)*2)
shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]=( (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[1,1,1,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[1,1,2,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[1,1,3,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[1,2,1,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[1,2,2,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[1,2,3,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[1,3,1,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[1,3,2,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(-1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[1,3,3,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[2,1,1,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[2,1,2,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[2,1,3,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[2,2,1,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[2,2,2,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[2,2,3,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[2,3,1,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[2,3,2,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(0)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[2,3,3,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[3,1,1,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[3,1,2,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[3,1,3,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[3,2,1,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[3,2,2,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[3,2,3,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(-1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[3,3,1,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(0))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[3,3,2,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
+ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +(1)) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+(1))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[3,3,3,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
)

 
shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=0.0
 
shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]+(((
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[1,1+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,1+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,1+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[3,1+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,1+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,1+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    ))
 
shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]+(((
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[1,2+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,2+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,2+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[3,2+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,2+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,2+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    ))
 
shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]+(((
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[1,3+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,3+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,3+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[3,3+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,3+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,3+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    ))
 
shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]+(((
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[1,4+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,4+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,4+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[3,4+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,4+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,4+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    ))
 
shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]+(((
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,1] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),1,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,2] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),1,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,3] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),1,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,4] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),1,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),1,5] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),1,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,1] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),2,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,2] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),2,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,3] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),2,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,4] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),2,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),2,5] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),2,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,1] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),3,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,2] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),3,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,3] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),3,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,4] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),3,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),3,5] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),3,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,1] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),4,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,2] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),4,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,3] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),4,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,4] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),4,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),4,5] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),4,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,1] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),5,1,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,2] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),5,2,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,3] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),5,3,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,4] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),5,4,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    +(
    # Sigmoid activation
    (1.0 / (1.0 + exp(-(
        shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[1,5+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] + 
        param_mixing[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_mixing[5,5+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Rational function activation (bounded)
    (shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) /
    (1.0 + abs(shared_mem[2,5+((threadIdx().x - 1) ÷ 5 * 5),5,5] * param_mixing[3,5+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
     param_mixing[4,5+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) * 
    param_mixing[6,5+((threadIdx().x - 1) ÷ 5 * 5),5,5,threadIdx().x,threadIdx().y,threadIdx().z,UInt32(round((blockIdx().y % num_blocks_y_true)+1)),UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
)
    ))
 
shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]/250
sync_threads()



# Synchronize threads
    sync_threads()
    #adding skip connection
    #shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]+shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]

    # shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]
    sync_threads()
        # Step 4: reduction on x, y, and z axes 
 if ((mod(threadIdx().x - 1, 5) + 1)==1)           
        
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing[1, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[2, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing[3, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing[4, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[5, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing[6, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x+1, threadIdx().y, threadIdx().z] * param_reducing[7, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x+1, threadIdx().y, threadIdx().z] * param_reducing[7, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing[8, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing[9, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x+1, threadIdx().y, threadIdx().z] * param_reducing[10, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[11, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing[12, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
              
        
    end
    if ((mod(threadIdx().x - 1, 5) + 1)==4)           
        
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing[1, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[2, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing[3, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing[4, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[5, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing[6, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x+1, threadIdx().y, threadIdx().z] * param_reducing[7, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x+1, threadIdx().y, threadIdx().z] * param_reducing[7, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing[8, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing[9, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x+1, threadIdx().y, threadIdx().z] * param_reducing[10, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[11, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing[12, 1, 1, 1, threadIdx().x+1, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
             
        
    end


        

        if ((mod(threadIdx().x - 1, 5) + 1)==1 )           
            #x3
            
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing[1, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[2, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing[3, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing[4, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[5, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing[6, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x+2, threadIdx().y, threadIdx().z] * param_reducing[7, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x+2, threadIdx().y, threadIdx().z] * param_reducing[7, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing[8, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing[9, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x+2, threadIdx().y, threadIdx().z] * param_reducing[10, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[11, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing[12, 1, 1, 1, threadIdx().x+2, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
     
            #x4
            
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing[1, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[2, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing[3, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing[4, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[5, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing[6, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x+3, threadIdx().y, threadIdx().z] * param_reducing[7, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x+3, threadIdx().y, threadIdx().z] * param_reducing[7, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing[8, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing[9, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x+3, threadIdx().y, threadIdx().z] * param_reducing[10, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing[11, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing[12, 1, 1, 1, threadIdx().x+3, threadIdx().y, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
            
        end    
        sync_threads()



        #before we will get into y reduction let's populate all shared memory in x with accumulated values
        shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]/8
        
        sync_threads()   
        #y reduction
        if (threadIdx().y==1)           
    
            
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_b[1, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[2, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing_b[3, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_b[4, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[5, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing_b[6, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x, 2, threadIdx().z] * param_reducing_b[7, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x, 2, threadIdx().z] * param_reducing_b[7, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing_b[8, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing_b[9, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x, 2, threadIdx().z] * param_reducing_b[10, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[11, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing_b[12, 1, 1, 1, threadIdx().x, 1, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
    
        end
        if (threadIdx().y==4)           
            
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_b[1, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[2, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing_b[3, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_b[4, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[5, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing_b[6, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x, 5, threadIdx().z] * param_reducing_b[7, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x, 5, threadIdx().z] * param_reducing_b[7, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing_b[8, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing_b[9, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x, 5, threadIdx().z] * param_reducing_b[10, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[11, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing_b[12, 1, 1, 1, threadIdx().x, 2, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
    
        end
        sync_threads()
        #we have in y1 info from y1 and y2 and in y4 infor from y4 and y5
        #so we just need to get info from y3 and y4 to y1
        if (threadIdx().y==1 )           
            #x3
            
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_b[1, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[2, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing_b[3, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_b[4, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[5, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing_b[6, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x, 3, threadIdx().z] * param_reducing_b[7, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x, 3, threadIdx().z] * param_reducing_b[7, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing_b[8, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing_b[9, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x, 3, threadIdx().z] * param_reducing_b[10, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[11, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing_b[12, 1, 1, 1, threadIdx().x, 3, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
    
            #x4
            
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_b[1, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[2, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing_b[3, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_b[4, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[5, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing_b[6, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x, 4, threadIdx().z] * param_reducing_b[7, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x, 4, threadIdx().z] * param_reducing_b[7, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing_b[8, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing_b[9, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x, 4, threadIdx().z] * param_reducing_b[10, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_b[11, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing_b[12, 1, 1, 1, threadIdx().x, 4, threadIdx().z, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
    
        end   
        sync_threads()      
        #we had accumulated all info on y dimension on y1 so we want to get it to all y
        shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,1,threadIdx().z]/8
        sync_threads()      

        # z reducing
        if (threadIdx().z==1 )          
            
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_c[1, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[2, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing_c[3, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_c[4, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[5, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing_c[6, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x, threadIdx().y, 2] * param_reducing_c[7, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x, threadIdx().y, 2] * param_reducing_c[7, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing_c[8, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing_c[9, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x, threadIdx().y, 2] * param_reducing_c[10, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[11, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing_c[12, 1, 1, 1, threadIdx().x, threadIdx().y, 1, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
    
        end
        if (threadIdx().z==4 )          
            
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_c[1, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[2, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing_c[3, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_c[4, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[5, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing_c[6, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x, threadIdx().y, 5] * param_reducing_c[7, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x, threadIdx().y, 5] * param_reducing_c[7, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing_c[8, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing_c[9, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x, threadIdx().y, 5] * param_reducing_c[10, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[11, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing_c[12, 1, 1, 1, threadIdx().x, threadIdx().y, 2, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
    
        end
        sync_threads()      
    
        if (threadIdx().z==1 )          
            
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_c[1, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[2, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing_c[3, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_c[4, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[5, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing_c[6, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x, threadIdx().y, 3] * param_reducing_c[7, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x, threadIdx().y, 3] * param_reducing_c[7, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing_c[8, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing_c[9, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x, threadIdx().y, 3] * param_reducing_c[10, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[11, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing_c[12, 1, 1, 1, threadIdx().x, threadIdx().y, 3, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
    
            #x4
            
shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z ] = (
    # First non-linear transformation using sigmoid
    (shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] +(1.0 / (1.0 + exp(-(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_c[1, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[2, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )))) * param_reducing_c[3, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Second non-linear transformation using tanh
    tanh(
        shared_mem[1, threadIdx().x, threadIdx().y, threadIdx().z] * param_reducing_c[4, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[5, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    ) * param_reducing_c[6, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Third non-linear transformation using rational function
    ((shared_mem[1, threadIdx().x, threadIdx().y, 4] * param_reducing_c[7, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) / 
    (1.0 + abs(shared_mem[1, threadIdx().x, threadIdx().y, 4] * param_reducing_c[7, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]) + 
    param_reducing_c[8, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))])) * 
    param_reducing_c[9, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
    
    # Fourth non-linear transformation using softplus (smooth ReLU)
    log(1.0 + exp(
        shared_mem[1, threadIdx().x, threadIdx().y, 4] * param_reducing_c[10, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))] +
        param_reducing_c[11, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
    )) * param_reducing_c[12, 1, 1, 1, threadIdx().x, threadIdx().y, 4, UInt32(round((blockIdx().y % num_blocks_y_true)+1)), UInt32(ceil((blockIdx().y / num_blocks_y_true)))]
))
    
        end    
        sync_threads()      
        # we had accumulated values in all axes so we can save it to global memory
        # we had accumulated first x values and later we accumulated y and z block dim X times 
        # so we avoided summarising information from threadblock into single number and now we have for example 5x5 values summarizing whole area 
        # results are in all x and threads but z thread = 1 
    
    
        # Step 5: Organize result
        if (threadIdx().z==1 )       

        result[mod(blockIdx().x - 1, dims[3]) + 1
            ,div(mod(blockIdx().x  - 1, dims[2] * dims[3]), dims[3]) + 1
            ,div(blockIdx().x  - 1, dims[2] * dims[3]) + 1 
          ,(mod(threadIdx().x - 1, 5) + 1)
          , threadIdx().y,(UInt32(ceil((blockIdx().y / num_blocks_y_true)))+half_num_params*(((threadIdx().x) - 1) ÷ 5))
        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] = (shared_mem[1,threadIdx().x, threadIdx().y,1]
        +shared_mem[2,threadIdx().x,threadIdx().y,1] #adding here as skip connection directly from initial convolutions
        +shared_mem[2,threadIdx().x,threadIdx().y,2] 
        +shared_mem[2,threadIdx().x,threadIdx().y,3] 
        +shared_mem[2,threadIdx().x,threadIdx().y,4] 
        +shared_mem[2,threadIdx().x,threadIdx().y,5]         
        )

        end
        return nothing
    end


