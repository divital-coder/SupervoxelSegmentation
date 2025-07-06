using cuTENSOR,TensorOperations
using Test, Revise,Lux
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
using Test,LuxCUDA
using CUDA
using Test

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights.jl")

final_sv_repr=64
global const final_sv_repr_for_shared=final_sv_repr
const global y_shared_for_sv_info=4 #currently just set manually can experiment with making it bigger

radiuss = (Float32(3.1), Float32(3.3), Float32(3.2))
num_channels_after_first_conv=2
# radiuss = (Float32(3.1), Float32(4.3), Float32(4.7))
spacing = (Float32(1.0), Float32(1.0), Float32(1.0))
batch_size = 2
a = 71
image_shape = (a, a, a, num_channels_after_first_conv)



dummy_image=CuArray(rand(Float32,image_shape[1],image_shape[2],image_shape[3],image_shape[4],batch_size))
sv_centers,dims,diffs= get_sv_centers(radiuss,image_shape)

cp_x=size(sv_centers, 1)
cp_y=size(sv_centers, 2)
cp_z=size(sv_centers, 3)

mixing_radius=3

threads=(final_sv_repr_for_shared,y_shared_for_sv_info)
blocks=(cp_x,Int(ceil(cp_y/y_shared_for_sv_info)),cp_z*batch_size)

function test_mix()

    y_pred=CUDA.ones((final_sv_repr, size(sv_centers, 1),size(sv_centers, 2), size(sv_centers, 3),batch_size))
    mix_params=CUDA.ones((mixing_radius*2)+1,(mixing_radius*2)+1,(mixing_radius*2)+1,final_sv_repr,3)
    output = CUDA.zeros(final_sv_repr, size(sv_centers, 1),size(sv_centers, 2), size(sv_centers, 3),batch_size)

    # Initial run
    y_pred.=1.0f0
    @cuda threads = threads blocks = blocks  mix_sv_info(y_pred,cp_x,cp_y,cp_z, mix_params,output,mixing_radius)
    CUDA.synchronize()
    initial_output = Array(output)[1, 3, 3, 3, 1]

    y_pred.=1.0f0
    CUDA.@allowscalar y_pred[1, 3, 3, 3, 1] = 5
    @cuda threads=threads blocks=blocks mix_sv_info(y_pred, cp_x, cp_y, cp_z, mix_params, output, mixing_radius)
    CUDA.synchronize()
    later_output = Array(output)[1, 3, 3, 3, 1]

    @test sum(Array(output).==0)==0

    # Mutate y_pred
    for dx in -2:2
        for dy in -2:2
            for dz in -2:2
                if dx != 0 || dy != 0 || dz != 0
                    CUDA.@allowscalar y_pred[1, 3 + dx, 3 + dy, 3 + dz, 1] = 2.0f0

                    # Run after mutating y_pred
                    output .= 0.0f0
                    @cuda threads=threads blocks=blocks mix_sv_info(y_pred, cp_x, cp_y, cp_z, mix_params, output, mixing_radius)
                    CUDA.synchronize()
                    CUDA.@allowscalar mutated_y_pred_output = output[1, 3, 3, 3, 1]

                    # Check if output increased
                    @test mutated_y_pred_output > initial_output

                    # Reset y_pred
                    y_pred .= 1.0f0



                end
            end
        end
    end



    # Mutate mix_params
    for dx in -2:2
        for dy in -2:2
            for dz in -2:2
                if dx != 0 || dy != 0 || dz != 0
                    CUDA.@allowscalar mix_params[5 + dx, 5 + dy, 5 + dz, :, :] .= 2.0f0
                    # Run after mutating mix_params
                    output .= 0.0f0
                    @cuda threads=threads blocks=blocks mix_sv_info(y_pred, cp_x, cp_y, cp_z, mix_params, output, mixing_radius)
                    CUDA.synchronize()
                    CUDA.@allowscalar mutated_mix_params_output = output[1, 3, 3, 3, 1]

                    # Check if output increased
                    @test mutated_mix_params_output > initial_output


                end
            end
        end
    end

end
# end

# Run the test
# test_mix_sv_info()
test_mix()






