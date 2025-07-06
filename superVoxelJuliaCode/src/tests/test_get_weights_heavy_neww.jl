using CUDA
using Test,Revise,Lux
using Pkg, Wavelets,LuxCUDA,JLD2
using Dates
import Dates
using Plots
using Combinatorics,NNlib


includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights_kerns/get_weights_heavy.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/dif_get_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights_kerns/conv_prim_get_w.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights.jl")



includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern_unrolled.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern _old.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_point.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/dif_get_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/mix_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sinusoid_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/Lux_model.jl")


global const is_point_per_triangle = true
global const len_get_random_point_in_tetrs_kern=Int(floor(256/3))

function forward_pass_example()
    rng = Random.default_rng()
    radiuss = (1.5f0, 1.8f0, 1.2f0)
    spacing = (1.0f0, 1.0f0, 1.0f0)
    batch_size = 2
    num_channels = 3
    num_params_exec = 8
    image_shape = (32, 32, 32, num_channels, batch_size)
    final_sv_repr = 64  
    mixing_param_sets=3
    num_weights_for_set_point_weights=48+48*3
    is_to_mix_weights=true
    example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
    sv_centers, control_points, tetrs, dims = example_set_of_svs

    curr_cent=sv_centers[3,4,5,:]

    # model = GetWeightsFromDirections_str(radiuss, spacing, batch_size, num_channels, num_params_exec
    # , (image_shape[1],image_shape[2],image_shape[3]), final_sv_repr,true)
    model = Lux.Chain(get_mix_weights_full_layers(image_shape, radiuss, spacing, batch_size, num_channels
    , num_params_exec, mixing_param_sets
    ,num_weights_for_set_point_weights,is_to_mix_weights)...) 

    opt = Optimisers.AdamW(0.001)
    
    ps, st = Lux.setup(rng, model)
    tstate_glob = Lux.Training.TrainState(model, ps, st, opt)
    dev = gpu_device()
    ps, st = Lux.setup(rng, model)|> dev
    tstate_glob = Lux.Training.TrainState(model,ps, st, opt)
    # Example input data
    input_data = CuArray(ones(Float32, image_shape...))
    # input_data[Int(curr_cent[1]), Int(curr_cent[2]),Int(curr_cent[3]), :, :].= 10000000.0

    y_pred, st = Lux.apply(model, input_data, tstate_glob.parameters, tstate_glob.states)
    CUDA.synchronize()

    return y_pred, st,curr_cent
end

function loss_ff(model, ps, st, data)
    y_pred, st = Lux.apply(model, data, ps, st)
    print("\n ooooo $(typeof(y_pred))  \n")
    return sum(y_pred[1])+sum(y_pred[2]), st, ()
    # return sum(y_pred), st, ()
end



function back_pass_example()
    rng = Random.default_rng()
    radiuss = (1.5f0, 1.8f0, 1.2f0)
    spacing = (1.0f0, 1.0f0, 1.0f0)
    batch_size = 2
    num_channels = 3
    num_params_exec = 8
    image_shape = (64, 64, 64, num_channels, batch_size)
    mixing_param_sets=3
    
    num_weights_for_set_point_weights=24+24*6
    is_to_mix_weights=true
    pad_voxels=true
    example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
    sv_centers, control_points, tetrs, dims = example_set_of_svs

    curr_cent=sv_centers[3,4,5,:]

    # model = GetWeightsFromDirections_str(radiuss, spacing, batch_size, num_channels, num_params_exec
    # , (image_shape[1],image_shape[2],image_shape[3]), final_sv_repr,true)

    # model = Lux.Chain(get_mix_weights_full_layers(image_shape, radiuss, spacing, batch_size, num_channels
    # , num_params_exec, mixing_param_sets
    # ,num_weights_for_set_point_weights,is_to_mix_weights)...) 


    # model = Lux.Chain(get_mix_weights_full_layers(image_shape, radiuss, spacing, batch_size, num_channels
    # , num_params_exec, mixing_param_sets
    # ,num_weights_for_set_point_weights,is_to_mix_weights)...,
    # Points_weights_str(radiuss, batch_size, pad_voxels, (image_shape[1], image_shape[2], image_shape[3]),is_point_per_triangle)) 
    get_weigts_layers = get_weigts_layers = Lux.Chain(
        #get and mix weights in directions
        get_mix_weights_full_layers(image_shape, radiuss, spacing
        , batch_size, image_shape[4], num_params_exec
        , mixing_param_sets
        ,num_weights_for_set_point_weights,is_to_mix_weights)...,
        #get weights to move control points
        Points_weights_str(radiuss, batch_size, pad_voxels, (image_shape[1], image_shape[2], image_shape[3]),is_point_per_triangle)
    )

    before_point_kerns = SkipConnection(get_weigts_layers, connection_before_set_tetr_dat_kern)
    model = Lux.Chain(before_point_kerns, Set_tetr_dat_str(radiuss,spacing,batch_size, 0, (image_shape[1], image_shape[2], image_shape[3])))
    model=get_weigts_layers

    opt = Optimisers.Adam(0.001)
    
    # ps, st = Lux.setup(rng, model)
    # tstate_glob = Lux.Training.TrainState(model, ps, st, opt)
    dev = gpu_device()
    ps, st = Lux.setup(rng, model)|> dev
    tstate = Lux.Training.TrainState(model,ps, st, opt)
    # Example input data
    input_data = CuArray(rand(Float32, image_shape...))
    # input_data[Int(curr_cent[1]), Int(curr_cent[2]),Int(curr_cent[3]), :, :].= 10000000.0
    vjp = Lux.Experimental.ADTypes.AutoZygote()
    
    # _, loss, _, tstate = Lux.Training.single_train_step!(vjp, loss_ff, input_data, tstate)
    gs, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_ff, input_data, tstate)
    tstate = Training.apply_gradients!(tstate, gs) 

    for i in 1:3
        gs, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_ff, input_data, tstate)
        tstate = Training.apply_gradients!(tstate, gs) 
        print("** $loss **")
    end
    
    CUDA.synchronize()
    # y_pred, st = Lux.apply(model, input_data, ps, st)
    # return gs,loss
    return loss
end



function test_sum_y_pred()
    y_pred, _, _ = forward_pass_example()
    @test !isnan(sum(y_pred))
    @test !isinf(sum(y_pred))
    @test (sum(y_pred))!=0.0
end


test_sum_y_pred()


y_pred, _, _ = forward_pass_example()
y_pred
minimum(y_pred)
maximum(y_pred)

# gs,s=back_pass_example()
s=back_pass_example()

# for i in 1:10
#     gs,s=back_pass_example()
#     print("** $s **")
# end

# gs,s=back_pass_example()
# #@NamedTuple{conv_kernels::CuArray{Float32, 8, CUDA.DeviceMemory}, param_mixing::CuArray{Float32, 9, CUDA.DeviceMemory}, param_reducing::CuArray{Float32, 9, CUDA.DeviceMemory}, param_reducing_b::CuArray{Float32, 9, CUDA.DeviceMemory}, param_reducing_c::CuArray{Float32, 9, CUDA.DeviceMemory}}
# ck=gs[:conv_kernels]
# size(ck)#(3, 3, 3, 10, 5, 5, 3, 4)

# for i in 1:10
#     print("$i ; $(count_zeros(ck[:, :, :, i, :, :, :, :]," "))  \n")
# end 

# pm=gs[:param_mixing]
# sizz=size(pm)# (6, 10, 5, 5, 10, 5, 5, 3, 4)
# for n in 1:9
#     print("***** nnn **** $(n) \n")
#     for i in 1:sizz[n]
#         pm_slice = selectdim(pm, n, i)
#         print("$(count_zeros(pm_slice,"$i "))  \n")
#         # print("$(count_zeros(pm[:, :, :, :, :, :, :, :, :],"$i "))  \n")
#     end 
# end


# pm=Array(pm)
# count_zeros(pm,"pm")


# pm_size = size(pm)  # (6, 10, 5, 5, 10, 5, 5, 3, 4)
# dim_indices = collect(1:length(pm_size))  # [1,2,3,4,5,6,7,8,9]

# # Generate all combinations of two dimensions
# dim_pairs = collect(combinations(dim_indices, 2))

# for dims_to_keep in dim_pairs
#     # Sum over the other dimensions
#     dims_to_sum_over = setdiff(dim_indices, dims_to_keep)
#     pm_reduced = sum(pm; dims=tuple(dims_to_sum_over...))
    
#     # Remove singleton dimensions
#     pm_reduced = dropdims(pm_reduced; dims=tuple(dims_to_sum_over...))
#     pm_reduced[pm_reduced .== 0.0] .= 1000.0
#     # Plot heatmap
#     plt=Plots.heatmap(
#         pm_reduced,
#         xlabel = "Dimension $(dims_to_keep[1])",
#         ylabel = "Dimension $(dims_to_keep[2])",
#         title = "Heatmap of Dimensions $(dims_to_keep[1]) and $(dims_to_keep[2])"
#     )
#     # sleep(2)
#     display(plt)
#     filename = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/heatmap_dims_$(dims_to_keep[1])_$(dims_to_keep[2]).png"
#     savefig(plt, filename)

# end



# # pr=gs[:param_reducing]
# # size(pr)# (12, 1, 1, 1, 10, 5, 5, 3, 4)
# # for i in 1:4
# #     print("$(count_zeros(pr[:, :, :, :, :, :, :, :, i],"$i "))  \n")
# # end 




# # for x in 1:10
# #     print("$x ; $((mod(x - 1, 5)+1)) \n ")
# # end    


