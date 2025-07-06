using CUDA, Zygote, Enzyme
using Test

function test_call_get_weights_from_directions()
    # Initialize small test inputs
    batch_size = 1
    num_channels = 2
    param_set_length = 4
    param_set_length_reduced = param_set_length ÷ 2
    
    # Create small sample dimensions
    dims = (3, 3, 3)
    
    # Create small arrays for testing
    source_arr = CUDA.randn(Float32, 5, 5, 5, num_channels, batch_size)
    flat_sv_centers = CUDA.ones(UInt32, 5, 4)
    
    # Initialize parameters with small random values
    threads_mix_weights_kern = (2, 2, 2)
    blocks = (5, num_channels * param_set_length_reduced, batch_size)
    
    # Create small convolution kernels and parameters
    conv_kernels = CUDA.randn(Float32, 3, 3, 3, threads_mix_weights_kern[1], threads_mix_weights_kern[2], threads_mix_weights_kern[3], num_channels, param_set_length_reduced)
    param_mixing = CUDA.randn(Float32, 6, threads_mix_weights_kern[1], threads_mix_weights_kern[2], threads_mix_weights_kern[3], threads_mix_weights_kern[1], threads_mix_weights_kern[2], threads_mix_weights_kern[3], num_channels, param_set_length_reduced)
    param_reducing = CUDA.randn(Float32, 12, 1, 1, 1, threads_mix_weights_kern[1], threads_mix_weights_kern[2], threads_mix_weights_kern[3], num_channels, param_set_length_reduced)
    param_reducing_b = CUDA.randn(Float32, 12, 1, 1, 1, threads_mix_weights_kern[1], threads_mix_weights_kern[2]-1, threads_mix_weights_kern[3], num_channels, param_set_length_reduced)
    param_reducing_c = CUDA.randn(Float32, 12, 1, 1, 1, threads_mix_weights_kern[1], threads_mix_weights_kern[2], threads_mix_weights_kern[3]-1, num_channels, param_set_length_reduced)
    
    # Other required parameters
    num_blocks_y_true = num_channels
    beg_axis_pad = 2
    
    # Define a simple loss function
    function loss_fn(source_arr, flat_sv_centers, conv_kernels, param_mixing, param_reducing, param_reducing_b, param_reducing_c)
        result = call_get_weights_from_directions(
            source_arr, flat_sv_centers, conv_kernels, 
            param_mixing, param_reducing, param_reducing_b, param_reducing_c,
            num_blocks_y_true, dims, param_set_length_reduced, beg_axis_pad, 
            threads_mix_weights_kern, blocks, param_set_length
        )
        return sum(result)
    end
    
    # Forward pass
    println("Running forward pass...")
    result = call_get_weights_from_directions(
        source_arr, flat_sv_centers, conv_kernels, 
        param_mixing, param_reducing, param_reducing_b, param_reducing_c,
        num_blocks_y_true, dims, param_set_length_reduced, beg_axis_pad, 
        threads_mix_weights_kern, blocks, param_set_length
    )
    
    # Test that the result has expected dimensions
    @test size(result, 1) == dims[1]
    @test size(result, 2) == dims[2]
    @test size(result, 3) == dims[3]
    
    # Test backpropagation
    println("Testing backpropagation...")
    try
        # Skip param_mixing for simplicity as it's the largest tensor
        # and focus on smaller parameters for quicker testing
        gs = gradient(loss_fn, source_arr, flat_sv_centers, conv_kernels, 
                      param_mixing, param_reducing, param_reducing_b, param_reducing_c)
        
        # Check that gradients exist and are not all zeros
        @test !isnothing(gs[3])  # Check conv_kernels gradient
        @test sum(abs.(Array(gs[3]))) > 0  # Should have non-zero values
        
        println("Backpropagation successful!")
    catch e
        println("Error during backpropagation: ", e)
        rethrow(e)
    end
end


function debug_kernel_analysis()
    # Create minimal inputs just for testing kernel compilation
    threads_mix_weights_kern = (2, 2, 2)
    blocks = (2, 2, 1)
    dims = (3, 3, 3)
    num_blocks_y_true = 2
    beg_axis_pad = 2
    half_num_params = 2
    param_set_length = 4
    batch_size = 1
    
    # Minimal arrays
    source_arr = CUDA.zeros(Float32, 5, 5, 5, 2, batch_size)
    flat_sv_centers = CUDA.ones(UInt32, 2, 4)
    conv_kernels = CUDA.zeros(Float32, 3, 3, 3, 2, 2, 2, 2, 2)
    param_mixing = CUDA.zeros(Float32, 6, 2, 2, 2, 2, 2, 2, 2, 2)
    param_reducing = CUDA.zeros(Float32, 12, 1, 1, 1, 2, 2, 2, 2, 2)
    param_reducing_b = CUDA.zeros(Float32, 12, 1, 1, 1, 2, 1, 2, 2, 2)
    param_reducing_c = CUDA.zeros(Float32, 12, 1, 1, 1, 2, 2, 1, 2, 2)
    
    # Create result tensor with minimal size
    result = CUDA.zeros(Float32, dims[1], dims[2], dims[3], 5, threads_mix_weights_kern[2], param_set_length, num_blocks_y_true, batch_size)
    
    # Try launching just the kernel to test compilation
    println("Testing kernel compilation...")
    try
        @cuda threads=threads_mix_weights_kern blocks=blocks kernel_analysis(
            source_arr, flat_sv_centers, conv_kernels, 
            param_mixing, param_reducing, param_reducing_b, param_reducing_c,
            result, num_blocks_y_true, dims, half_num_params, beg_axis_pad
        )
        println("Kernel compiled successfully!")
        return true
    catch e
        println("Kernel compilation error: ", e)
        return false
    end
end




# Run the test
test_call_get_weights_from_directions()


debug_kernel_analysis()
