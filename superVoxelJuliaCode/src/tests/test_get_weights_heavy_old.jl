using CUDA
using Test,Revise,Lux
using Pkg, Wavelets,LuxCUDA,JLD2


includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights_kerns/get_weights_heavy.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/dif_get_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights_kerns/conv_prim_get_w.jl")

"""
In example case the supervoxel center is set at 7,7,7 with beg_axis_pad=1 so we iterate over area centered at 8,8,8 in source_arr
for distance 5 in all x,y,z axis up and down the axis.

1) check the conv_kernels weather they work as should . setting any entry in first 3 dimensions to 2 should have exactly the same influence on the output. 
2) setting the value in source array in coordinate 4,4,4 and to 2 and in conv_kernels in [1,1,1,:,:,:,1,1] to 2 should lead to bigger increase in result then setting the value in source array in coordinate 4,4,4 and to 2 and to 2 in conv_kernels in [2,1,1,:,:,:,:,:] or [2,2,1,:,:,:,:,:] or [2,2,2,:,:,:,:,:] or [1,1,1,3,3,3,:,:] or [1,1,1,:,:,:,2,1] or [1,1,1,:,:,:,1,2]
3) test interaction in shared memory, as each entry in the area around sv center has diffrent weights so first set conv_kernels in [1,1,1,1,1,1,1,1] to 2 then in param_mixing[1:4,1,1,1,1,1,1,:,:] to 2 and it should lead to bigger increase then when setting  param_mixing[1:4,2,1,1,1,1,1,:,:] to 2 or param_mixing[1:4,1,2,1,1,1,1,:,:] to 2 or param_mixing[1:4,1,1,2,1,1,1,:,:] to 2 or param_mixing[1:4,1,1,1,2,1,1,:,:] to 2 or param_mixing[1:4,1,1,1,1,2,1,:,:] to 2 or param_mixing[1:4,1,1,1,1,1,2,:,:] to 2 or param_mixing[1:4,1,1,1,1,1,1,2,:] to 2 or param_mixing[1:4,1,1,1,1,1,1,:,2] to 2
4) test reduction changing any of the field in param_reducing or param_reducing_b in first 4 dimensions should lead to change in the second and third dimension but the change in last 2 dimensions should lead to change only in the same entries in last 2 dimensions of res
5) for param_reducing_c change in dimension 2 should be reflected in the same entries of dimension 2 of a result; change in dimension 3 should be reflected in changes of dimension 3 in the result

"""
# get block dim on x increased from 5 to 10
# get assertion that parma length is divisible by 2
# increase param mixing x dim to 10
# but decrease the param mixing dimension two times

function launch_kernel(source_arr, flat_sv_centers,  beg_axis_pad
    , conv_kernels, param_mixing, param_reducing,param_reducing_b
    , param_reducing_c,threads,blocks, num_blocks_y_true,param_set_length
    , num_channels, batch_size,l_relu_weight,return_conved=false)
    
    # Define grid and block dimensions
    # Allocate result array
    #1,1,1 becouse it is shape of dummy sv centers
    
    result = CUDA.zeros(Float32, 1,1,1  ,5,threads[2], param_set_length, num_channels, batch_size)
    
    conved = CUDA.zeros(Float32, 1  ,5,threads[2],threads[3], param_set_length, num_channels, batch_size)
    
    source_arr=Float32.(CuArray(source_arr))
    conv_kernels=Float32.(CuArray(conv_kernels))
    param_mixing=Float32.(CuArray(param_mixing))
    param_reducing=Float32.(CuArray(param_reducing))
    param_reducing_b=Float32.(CuArray(param_reducing_b))
    param_reducing_c=Float32.(CuArray(param_reducing_c))
    dims=(1,1,1)
    is_to_mix_weights=true

    half_num_params=Int(param_set_length/4)


    # as threads are diffrent in kernels blocks also need to be adjusted
    # Launch the kernel
    # kernel_analysis(source_arr, flat_sv_centers,  beg_axis_pad, conv_kernels, param_mixing, param_reducing,param_reducing_b, result,num_blocks_y_true,l_relu_weight)
    
    @cuda threads=(10,5,5) blocks=(blocks[1],num_channels * Int(param_set_length/2),blocks[3]) kernel_analysis_conv(source_arr, flat_sv_centers, conved, beg_axis_pad, conv_kernels
    ,num_blocks_y_true,Int(param_set_length/2))
    CUDA.synchronize()
    CUDA.synchronize(conved)
    print("cccccc $(sum(conved))")
    cccc=sum(conved)
    @cuda threads=threads blocks=blocks kernel_analysis(conved, flat_sv_centers,  beg_axis_pad, conv_kernels, param_mixing, param_reducing,param_reducing_b,param_reducing_c, result,num_blocks_y_true,l_relu_weight,dims,is_to_mix_weights,half_num_params)
    CUDA.synchronize()
    CUDA.synchronize(result)
    print("cccccc $(sum(result))")
    cccc=sum(conved)
    CUDA.synchronize()

    if(return_conved)
        return Array(result),Array(conved)
    end

    return Array(result)
end


function setup_inputs()
    num_channels=2
    batch_size=2
    source_arr = ones(Float32, 12, 12, 12, num_channels, batch_size)  # Example 5D tensor
    flat_sv_centers = CUDA.ones(UInt32, 1, 3).+5  # Example SV centers
    param_set_length = 8
    half_num_params=Int(param_set_length/4)
    beg_axis_pad = UInt32(1)

    threads = (20, 5, 5)    
    
    conv_kernels = ones(Float32, 3, 3, 3, 10,threads[2],threads[3],num_channels, half_num_params*2)  # Example convolution kernels
    
    param_mixing = ones(Float32, 5,threads[1],threads[2],threads[3], threads[1],threads[2],threads[3], num_channels, half_num_params) 
    param_reducing = ones(Float32, 5,1,1,1, threads[1],threads[2],threads[3], num_channels, half_num_params)  
    param_reducing_b = ones(Float32, 5,1,1,1, threads[1],threads[2]-1,threads[3], num_channels, half_num_params)  
    param_reducing_c = ones(Float32, 5,1,1,1, threads[1],threads[2],threads[3]-1, num_channels, half_num_params) 

    # param_mixing[5,:,:,:,:,:,:,:,:].= 0.5
    # param_reducing[5,:,:,:,:,:,:,:,:].= 0.5
    # param_reducing_b[5,:,:,:,:,:,:,:,:].= 0.5
    # param_reducing_c[5,:,:,:,:,:,:,:,:].= 0.5

    blocks = (size(flat_sv_centers, 1), num_channels * half_num_params, batch_size)
    num_blocks_y_true = num_channels
    return source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size
end    


function get_convv(i, j, k, conv_kernels, source_arr, flat_sv_centers, beg_axis_pad, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight, res_prim)
    source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()

    conv_kernels[i, j, k, 1, 1, 1, 1, 1] = 2000
    res = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
    CUDA.synchronize()
    # @test res[1,1,1, 1, 1, 1, 1, 1] > res_prim[1,1,1, 1, 1, 1, 1, 1]
    # conv_kernels[i, j, k, 1, 1, 1, 1, 1] = 1  # Reset to original value
    return res[1,1,1, 1, 1, 1, 1, 1]
end 

# Test Convolutional params
function test_convolutional_kernels()

    l_relu_weight=0.00001
    source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
    res_prim,conved_prim = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight,true)
    CUDA.synchronize()
    aa341a=sum(conved_prim)
    CUDA.synchronize()

    # Task 1: Check conv_kernels influence
    for i in 1:3, j in 1:3, k in 1:3
        source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
        conv_kernels[i, j, k, 1, 1, 1, 1, 1] = 3000
        ooo=sum(conv_kernels)
        CUDA.synchronize()

        raa,conved = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight,true)
        CUDA.synchronize()


        @test sum(conved) > aa341a
        @test sum(conved[:,:,:,:,:,2,1]) != sum(conved[:,:,:,:,:,1,1])
    end



    # l_relu_weight=0.00001
    # source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
    # res_prim = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
    # CUDA.synchronize()

    # Task 1: Check conv_kernels influence
    for i in 1:3, j in 1:3, k in 1:3
        source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
        conv_kernels[i, j, k, 1, 1, 1, 1, 1] = 3000
        ooo=sum(conv_kernels)
        res = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
        aaaa=sum(res)
        CUDA.synchronize()
        aaaa=sum(res)

        @test sum(res) > sum(res_prim)
    end

    ll=[get_convv(i, j, k, conv_kernels, source_arr, flat_sv_centers, beg_axis_pad, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight, res_prim) for i in 1:3, j in 1:3, k in 1:3]


    for nl in ll
        @test isapprox(nl,ll[1],rtol=0.1)
    end

    # Task 2: Influence of specific coordinates should strenthen the result
    source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
    source_arr[3,3, 3, 1, 1] = 2000
    conv_kernels[1, 1, 1, 1, 1, 1, 1, 1] = 2000
    res = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
    CUDA.synchronize()

    @test res[1,1,1, 1, 1, 1, 1, 1] > res_prim[1,1,1, 1, 1, 1, 1, 1]
    res_bigger=res
    # Additional checks for other coordinates

    for i in 1:3, j in 1:3, k in 1:3
        source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
        conv_kernels[i, j, k, 1, 1, 1, 1, 1] = 2000
        res = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
        CUDA.synchronize()
        @test res[1,1,1, 1, 1, 1, 2, 1] != res_bigger[1,1,1, 1, 1, 1, 1, 1]
        @test res[1,1,1, 1, 1, 1, 1, 2] != res_bigger[1,1,1, 1, 1, 1, 1, 1]
        # conv_kernels.= 1  # Reset to original value
    end


    # for i in 2:3, j in 1:3, k in 1:3,c in 1:2,b in 1:2
    #     source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
    #     source_arr[3,3, 3, 1, 1] = 2000
    #     conv_kernels[1, 1, 1, i, j, k, 1, b]= 2000
    #     res = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
    #     CUDA.synchronize()
    #     print("\n res $(res[1,1,1, 1, 1, 1, 1, 1]) res_bigger $(res_bigger[1,1,1, 1, 1, 1, 1, 1])\n")
    #     @test res[1,1,1, 1, 1, 1, 1, 1] < res_bigger[1,1,1, 1, 1, 1, 1, 1]
    #     # @test sum(res)< sum(res_bigger)

    #     conv_kernels[i, j, k, 1, 1, 1, 1, 1] = 1  # Reset to original value
    # end



end


# Test Shared Memory Interaction
function test_shared_memory_interaction()
    l_relu_weight=0.00001
    source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()

    # Set conv_kernels and param_mixing as specified
    conv_kernels[1, 1, 1, 1, 1, 1, 1, 1] = 2
    param_mixing[1:3, 1, :, :, :, :,:, :, :].= 2

    # Launch kernel and synchronize
    res = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
    CUDA.synchronize()

    # Check results
    resbigger =res[1,1,1, 1, 1, 1, 1, 1]  

    # for i in 2:7
    for i in 2:4
        # Reset and test other param_mixing entries
        source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
        conv_kernels[1, 1, 1, 1, 1, 1, 1, 1] = 2
        param_mixing[1:3, i, :, :, :, :, :, :, :].= 2
        res = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
        CUDA.synchronize()
        print(" \n aaa $(res[1,1,1, 1, 1, 1, 1, 1] )  sum $(sum((res ))) \n ")
        @test res[1,1,1, 1, 1, 1, 1, 1] < resbigger
    end

end


# Define the test function
function test_parameter_reduction()
    # Initial setup
    l_relu_weight=0.00001
    source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
    res_prim = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
    CUDA.synchronize()
    # Modify specific entries in the parameters
    source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
    param_reducing[1:3,:, :, :, :, :, :, :, :].= 40.0
    res = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
    CUDA.synchronize()
    @test sum(res)>sum(res_prim)
    source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
    param_reducing_b[1:3,:, :, :, :, :, :, :, :].= 40.0
    res = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
    CUDA.synchronize()
    @test sum(res)>sum(res_prim)
    source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
    param_reducing_c[1:3,:, :, :, :,:, :, :, :].= 40.0
    res = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
    CUDA.synchronize()
    @test sum(res)>sum(res_prim)
    # Launch the kernel
    res = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
    CUDA.synchronize()

end


l_relu_weight=0.00001
source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
res_prim = launch_kernel(source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size, l_relu_weight)
CUDA.synchronize()


#tests
test_convolutional_kernels()
# # test_shared_memory_interaction()
test_parameter_reduction()

# TDO count zeros in derivative calculations

function get_weights_from_directions_deff(
    source_arr, d_source_arr
    , flat_sv_centers, d_flat_sv_centers
    ,beg_axis_pad
    ,conv_kernels, d_conv_kernels
    ,param_mixing,d_param_mixing
    ,param_reducing,d_param_reducing
    ,param_reducing_b,d_param_reducing_b
    ,param_reducing_c,d_param_reducing_c
    ,result,d_result
    ,num_blocks_y_true
    ,l_relu_weight
    ,dims
    ,half_num_params
)


    Enzyme.autodiff_deferred(
            Enzyme.Reverse
            ,Enzyme.Const(kernel_analysis), Enzyme.Const
            ,Enzyme.Duplicated(source_arr, d_source_arr)
            ,Enzyme.Duplicated(flat_sv_centers, d_flat_sv_centers),
            Enzyme.Const(beg_axis_pad),
            Enzyme.Duplicated(conv_kernels, d_conv_kernels),
            Enzyme.Duplicated(param_mixing, d_param_mixing),
            Enzyme.Duplicated(param_reducing, d_param_reducing),
            Enzyme.Duplicated(param_reducing_b, d_param_reducing_b),
            Enzyme.Duplicated(param_reducing_c, d_param_reducing_c),
            Enzyme.Duplicated(result, d_result),
            Enzyme.Const(num_blocks_y_true),
            Enzyme.Const(l_relu_weight),
            Enzyme.Const(dims),
            Enzyme.Const(half_num_params)
            )
    return nothing
end

function launch_kernel_deff(source_arr, flat_sv_centers,  beg_axis_pad, conv_kernels, param_mixing, param_reducing,param_reducing_b, param_reducing_c,threads,blocks, num_blocks_y_true,param_set_length, num_channels, batch_size,l_relu_weight)
    
    result = CUDA.zeros(Float32, 1,1,1  ,5,threads[2], param_set_length, num_channels, batch_size)
    conved = CUDA.zeros(Float32, 1  ,5,threads[2],threads[3], param_set_length, num_channels, batch_size)



    # result = CUDA.zeros(Float32, 1,1,1,threads[1],threads[2], param_set_length, num_channels, batch_size)
    source_arr=CUDA.rand(Float32, size(source_arr)...)
    conv_kernels=CUDA.rand(Float32, size(conv_kernels)...)
    param_mixing=CUDA.rand(Float32, size(param_mixing)...)
    param_reducing=CUDA.rand(Float32, size(param_reducing)...)
    param_reducing_b=CUDA.rand(Float32, size(param_reducing_b)...)
    param_reducing_c=CUDA.rand(Float32, size(param_reducing_c)...)
    flat_sv_centers=CuArray(UInt32.(round.(flat_sv_centers)))
    dims=(1,1,1)
    half_num_params=Int(param_set_length/4)
    # Launch the kernel
    # kernel_analysis(source_arr, flat_sv_centers,  beg_axis_pad, conv_kernels, param_mixing, param_reducing,param_reducing_b, result,num_blocks_y_true,l_relu_weight)
    # @cuda threads=threads blocks=blocks kernel_analysis(source_arr, flat_sv_centers,  beg_axis_pad, conv_kernels, param_mixing, param_reducing,param_reducing_b,param_reducing_c, result,num_blocks_y_true,l_relu_weight,dims,Int(param_set_length/2))
    @cuda threads=(10,5,5) blocks=(blocks[1],num_channels * Int(param_set_length/2),blocks[3]) kernel_analysis_conv(source_arr, flat_sv_centers, conved, beg_axis_pad, conv_kernels
    ,num_blocks_y_true,Int(param_set_length/2))

    CUDA.synchronize()

    d_result =           CUDA.ones(Float32, size(result)...)
    d_conved =       CUDA.zeros(Float32, size(conved)...)
    d_conv_kernels =     CUDA.zeros(Float32, size(conv_kernels)...)
    d_param_mixing =     CUDA.zeros(Float32, size(param_mixing)...)
    d_param_reducing =   CUDA.zeros(Float32, size(param_reducing)...)
    d_param_reducing_b = CUDA.zeros(Float32, size(param_reducing_b)...)
    d_param_reducing_c = CUDA.zeros(Float32, size(param_reducing_c)...)
    d_flat_sv_centers = CUDA.zeros(UInt32, size(flat_sv_centers)...)

    # Launch the kernel
    @cuda threads=threads blocks=blocks get_weights_from_directions_deff(
        conved, d_conved
            , flat_sv_centers, d_flat_sv_centers
            ,beg_axis_pad
            ,conv_kernels, d_conv_kernels
            ,param_mixing,d_param_mixing
            ,param_reducing,d_param_reducing
            ,param_reducing_b,d_param_reducing_b
            ,param_reducing_c,d_param_reducing_c
            ,result,d_result
            ,num_blocks_y_true
            ,l_relu_weight
            ,dims
            ,half_num_params
        )
    CUDA.synchronize()    
    # return Array(result)
    return d_result,d_source_arr,d_conv_kernels,d_param_mixing,d_param_reducing,d_param_reducing_b,d_param_reducing_c
end






function difff()

    l_relu_weight=0.00001
    source_arr, flat_sv_centers, beg_axis_pad, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c, threads, blocks, num_blocks_y_true, param_set_length, num_channels, batch_size=setup_inputs()
    d_result,d_source_arr,d_conv_kernels,d_param_mixing,d_param_reducing,d_param_reducing_b,d_param_reducing_c=launch_kernel_deff(source_arr, flat_sv_centers,  beg_axis_pad, conv_kernels, param_mixing, param_reducing,param_reducing_b, param_reducing_c,threads,blocks, num_blocks_y_true,param_set_length, num_channels, batch_size,l_relu_weight)
    CUDA.synchronize()

    return d_result,d_source_arr,d_conv_kernels,d_param_mixing,d_param_reducing,d_param_reducing_b,d_param_reducing_c

end    

difff()


function forward_pass_example()
    rng = Random.default_rng()
    radiuss = (1.5f0, 1.8f0, 1.2f0)
    spacing = (1.0f0, 1.0f0, 1.0f0)
    batch_size = 2
    num_channels = 3
    num_params_exec = 10
    image_shape = (32, 32, 32, num_channels, batch_size)
    final_sv_repr = 64


    example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
    sv_centers, control_points, tetrs, dims = example_set_of_svs

    curr_cent=sv_centers[3,4,5,:]

    model = GetWeightsFromDirections_str(radiuss, spacing, batch_size, num_channels, num_params_exec, (image_shape[1],image_shape[2],image_shape[3]), final_sv_repr)
    opt = Optimisers.AdamW(0.001)
    
    ps, st = Lux.setup(rng, model)
    tstate_glob = Lux.Training.TrainState(model, ps, st, opt)
    dev = gpu_device()
    ps, st = Lux.setup(rng, model)|> dev
    tstate_glob = Lux.Training.TrainState(model,ps, st, opt)
    # Example input data
    input_data = CuArray(ones(Float32, image_shape...))
    # input_data[Int(curr_cent[1]), Int(curr_cent[2]),Int(curr_cent[3]), :, :].= 10000000.0
    tstate_glob.states.flat_sv_centers
    y_pred, st = Lux.apply(model, input_data, tstate_glob.parameters, tstate_glob.states)
    CUDA.synchronize()

    return y_pred, st,curr_cent
end






function test_sum_y_pred()
    y_pred, _, _ = forward_pass_example()
    @test !isnan(sum(y_pred))
    @test !isinf(sum(y_pred))
end

# Run the additional test

# test_convolutional_kernels() 
# # test_shared_memory_interaction()
# test_parameter_reduction() 
test_sum_y_pred()
d_result,d_source_arr,d_conv_kernels,d_param_mixing,d_param_reducing,d_param_reducing_b,d_param_reducing_c=difff()

function count_zeros(arr, name::String)
    num_zeros = count(x -> x == 0.0, arr)
    num_entries = length(arr)
    percentt=(num_zeros/num_entries)*100
    println("percent of zeros in $name: $percentt %")
end
count_zeros(d_source_arr,"d_source_arr")
# count_zeros(d_flat_sv_centers,"d_flat_sv_centers")
count_zeros(d_conv_kernels,"d_conv_kernels")
count_zeros(d_param_mixing,"d_param_mixing")
count_zeros(d_param_reducing,"d_param_reducing")
count_zeros(d_param_reducing_b,"d_param_reducing_b")
count_zeros(d_param_reducing_c,"d_param_reducing_c")

size(d_conv_kernels)

count_zeros(d_conv_kernels[:,:,:,1,1,1,2,7],"d_conv_kernels")
count_zeros(d_conv_kernels[:,:,:,:,:,:,1:2,1:7],"d_conv_kernels")


bin=Float32.(d_source_arr[:,:,:,1,1].==0)
bin
using PyCall
sitk = pyimport_conda("SimpleITK", "simpleitk")
np = pyimport_conda("numpy", "numpy")



# Save the image to the specified path
path = "superVoxelJuliaCode/data/debug_directions/inferred.nii.gz"
sitk.WriteImage(sitk.GetImageFromArray(Array(bin)), path)














rng = Random.default_rng()
radiuss = (1.5f0, 1.8f0, 1.2f0)
spacing = (1.0f0, 1.0f0, 1.0f0)
batch_size = 2
num_channels = 3
num_params_exec = 10
image_shape = (32, 32, 32, num_channels, batch_size)
final_sv_repr = 64


example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
sv_centers, control_points, tetrs, dims = example_set_of_svs


# Function to convert linear index to Cartesian index and retrieve the element
function get_element_from_linear_index(arr, linear_index)
    # Get the dimensions of the array
    dims = (size(arr,1),size(arr,2),size(arr,3))
    
    # Calculate the Cartesian indices
    
    k = mod(linear_index - 1, dims[1]) + 1    
    j = div(mod(linear_index - 1, dims[2] * dims[1]), dims[1]) + 1
    i = div(linear_index - 1, dims[2] * dims[1]) + 1

    print("\n  k j i $(k) $(j) $(i) \n ")
    # Retrieve the element at the Cartesian index
    return arr[k,j,i,:]
end

dims = (size(sv_centers,1),size(sv_centers,2),size(sv_centers,3))


    
# Flatten the tensor
flattened_tensor = reshape(sv_centers, (size(sv_centers, 1) * size(sv_centers, 2) * size(sv_centers, 3), size(sv_centers, 4)))

# linear_index=2310
# get_element_from_linear_index(tensor, linear_index)
# sv_centers[15,13,11,:]


# # Test the function
# function test_function()
#     # Generate a 3D tensor
#     tensor = sv_centers
    
#     # Flatten the tensor
#     # flattened_tensor = vec(tensor)
    
#     # Test for all linear indices
#     for linear_index in 1:size(flattened_tensor,1)
#         element_from_flattened = flattened_tensor[linear_index,:]
#         element_from_function = get_element_from_linear_index(tensor, linear_index)
#         print(" $element_from_flattened $element_from_function \n")
#         @assert element_from_flattened == element_from_function "Test failed at index $linear_index"
#     end
    
#     println("All tests passed!")
# end

# # # Run the test function
# test_function()


# for x in 1:20
#     b=((x%5)+1) ,Int(ceil(x/5))
#     print(" \n $(b)")
# end
