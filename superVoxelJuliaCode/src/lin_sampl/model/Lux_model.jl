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
using Wavelets, ParameterSchedulers, NNlib, LuxCUDA, JLD2
using TensorBoardLogger, Logging, Random
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

#TODO set definitions of the lux original layers to be rematerialized https://github.com/FluxML/Zygote.jl/issues/884

"""
overwritten in order to add gradient checkpointing and/or time per layer
"""

function myapply(layer, x, ps, st)
    # print("***  $(layer) ***") # Point_info_kern_str  GetWeightsFromDirections_str Set_tetr_dat_str TensorOpLayer_str LayerNorm

    if((layer isa Point_info_kern_str) || (layer isa Set_tetr_dat_str)  || (layer isa Conv) || (layer isa Points_weights_str)|| (layer isa Lux.BatchNorm)  ) 

        return Zygote.checkpointed(Lux.apply, layer, x, ps, st)
    end
    return Lux.apply(layer, x, ps, st)
end

@generated function Lux.applychain(
    layers::NamedTuple{fields}, x, ps, st::NamedTuple{fields}) where {fields}
N = length(fields)
x_symbols = vcat([:x], [gensym() for _ in 1:N])
st_symbols = [gensym() for _ in 1:N]
calls = [:(($(x_symbols[i + 1]), $(st_symbols[i])) = @inline myapply(
             layers.$(fields[i]), $(x_symbols[i]), ps.$(fields[i]), st.$(fields[i]))   )
         for i in 1:N]
push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))
push!(calls, :(return $(x_symbols[N + 1]), st))
return Expr(:block, calls...)
end







#get convolutions
conv1 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.sigmoid, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
conv2 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.sigmoid, stride=2, pad=Lux.SamePad(), init_weight=glorot_uniform)
conv_simple = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.sigmoid, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
conv2_b = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.gelu, stride=2, pad=Lux.SamePad(), init_weight=glorot_uniform)
conv5_b = (in, out) -> Lux.Conv((5, 5, 5), in => out, NNlib.gelu, stride=2, pad=Lux.SamePad(), init_weight=glorot_uniform)
conv5_a = (in, out) -> Lux.Conv((5, 5, 5), in => out, NNlib.gelu, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
conv3_a = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.gelu, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
conv3_depth = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.gelu, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform, groups=in)
convsigm2 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.sigmoid, stride=2, pad=Lux.SamePad())




conv5_c = (in, out) -> Lux.Conv((5, 5, 5), in => out, NNlib.gelu, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
conv5_d = (in, out) -> Lux.Conv((5, 5, 5), in => out, NNlib.gelu, stride=2, pad=Lux.SamePad(), init_weight=glorot_uniform)

function connection_before_set_tetr_dat_kern(x, y)
    return (x, y)
end #connection_before_set_tetr_dat_kern
function connection_parallel(x, y)

    return (x, y)
end #connection_parallel

function connection_before_mix_params(x, y)
    return cat(x, y, dims=4)
end #connection_before_set_tetr_dat_kern


# try Checkpointing.jl


function get_conv(image_shape)
    num_channelss = 5
    num_channels_bigger = num_channelss * 2 * 2 * 2
    # num_channels_bigger=num_channelss*4*4*4
    im_shape_reduced = Int.(round.(image_shape[1:3] ./ 2))
    num_c = 60
    # im_shape_reduced=Int.(round.(image_shape[1:3]./4))
    conv_part = Lux.Chain(
        conv5_a(2, num_c), Lux.LayerNorm((Int(round(image_shape[1])), Int(round(image_shape[2])), Int(round(image_shape[3])), num_c)), conv5_a(num_c, num_c), Lux.LayerNorm((Int(round(image_shape[1])), Int(round(image_shape[2])), Int(round(image_shape[3])), num_c)), conv5_b(num_c, num_c), Lux.LayerNorm((Int(round(image_shape[1] / 2)), Int(round(image_shape[2] / 2)), Int(round(image_shape[3] / 2)), num_c)), conv5_a(num_c, num_c), Lux.LayerNorm((Int(round(image_shape[1] / 2)), Int(round(image_shape[2] / 2)), Int(round(image_shape[3] / 2)), num_c)), conv5_b(num_c, num_c * 2), Lux.LayerNorm((Int(round(image_shape[1] / 4)), Int(round(image_shape[2] / 4)), Int(round(image_shape[3] / 4)), num_c * 2)), conv5_a(num_c * 2, num_c * 2)
        # ,Lux.LayerNorm((Int(round(image_shape[1])),Int(round(image_shape[2])),Int(round(image_shape[3])),image_shape[4]))    
        # , ReshapeLayer((im_shape_reduced[1], im_shape_reduced[2], im_shape_reduced[3], num_channels_bigger)), conv3_a(num_channels_bigger, num_channels_bigger)

        # ,Lux.LayerNorm((im_shape_reduced[1],im_shape_reduced[2],im_shape_reduced[3],image_shape[4]))
        # ,conv3_depth(num_channels_bigger, num_channels_bigger)#depth wise convolutions
        # ,Lux.LayerNorm((im_shape_reduced[1],im_shape_reduced[2],im_shape_reduced[3],image_shape[4]))
        # ,conv5_depth(num_channels_bigger, num_channels_bigger)#depth wise convolutions
        # ,Lux.LayerNorm((im_shape_reduced[1],im_shape_reduced[2],im_shape_reduced[3],image_shape[4]))
        # ,conv_simple(num_channels_bigger, 24))
        , conv_simple(num_c * 2, 24))
    return conv_part
end

# function debug_mix_weights(image_shape, radiuss, batch_size, spacing, num_params_exec, final_sv_repr, first_dense_out, second_dense_out, third_dense_out, primary_sv_repr, mixing_radius, num_prim_convs, num_weights_for_set_point_weights, num_channels_after_first_conv, pad_voxels)
#     num_channels = num_channels_after_first_conv
#     get_w_from_dir = GetWeightsFromDirections_str(radiuss, spacing, batch_size, num_channels, num_params_exec, (image_shape[1], image_shape[2], image_shape[3]), first_dense_out, second_dense_out, third_dense_out, primary_sv_repr, final_sv_repr
#     )
#     mix_sv_inf = mix_sv_info_str(radiuss, image_shape, mixing_radius, final_sv_repr, batch_size, num_weights_for_set_point_weights)


#     get_weigts_layers = get_weigts_layers = Lux.Chain(
#         #get some convolutions for the bagining to increase receptive field of mix weights
#         # SkipConnection(conv5_c(image_shape[4],62), connection_before_mix_params) ,
#         #get and mix weights in directions
#         conv5_d(image_shape[4], 64),
#         conv5_d(64, 64),
#         # ReshapeLayer((2,image_shape[1], image_shape[2], image_shape[3], 64)),
#         mix_sv_inf,
#         #get weights to move control points
#         Points_weights_str(radiuss, batch_size, pad_voxels, (image_shape[1], image_shape[2], image_shape[3]),zero_der_beg)
#     )
#     return get_weigts_layers
# end


"""
return model and optimizer
"""
function get_model_consts(image_shape, spacing,hp_dict,max_wavelength,max_amplitude,min_value)

    print("\n iiiiiiiiii mage_shape",image_shape)

    radiuss=hp_dict["radiuss"]
    batch_size=hp_dict["batch_size"]
    num_params_exec=hp_dict["num_params_exec"]
    mixing_param_sets=hp_dict["mixing_param_sets"]
    num_weights_for_set_point_weights=hp_dict["num_weights_for_set_point_weights"]
    learning_rate_start=hp_dict["learning_rate_start"]
    learning_rate_end=hp_dict["learning_rate_end"]
    is_to_mix_weights=hp_dict["is_to_mix_weights"]
    is_sinusoid_loss=hp_dict["is_sinusoid_loss"]

    num_texture_banks=hp_dict["num_texture_banks"]
    num_sinusoids_per_bank=hp_dict["num_sinusoids_per_bank"]

   
    num_base_samp_points=hp_dict["num_base_samp_points"]
    num_additional_samp_points=hp_dict["num_additional_samp_points"]

    add_gradient_norm=hp_dict["add_gradient_norm"]
    add_gradient_accum=hp_dict["add_gradient_accum"]
    is_WeightDecay=hp_dict["is_WeightDecay"]
    grad_accum_val=hp_dict["grad_accum_val"]
    clip_norm_val=hp_dict["clip_norm_val"]
    opt_str=hp_dict["optimiser"]
    is_point_per_triangle=hp_dict["is_point_per_triangle"]
    zero_der_beg=hp_dict["zero_der_beg"]

    pad_voxels = 0 #legacy value

    get_weigts_layers = get_weigts_layers = Lux.Chain(
        #get and mix weights in directions
        get_mix_weights_full_layers(image_shape, radiuss, spacing
        , batch_size, image_shape[4], num_params_exec
        , mixing_param_sets
        ,num_weights_for_set_point_weights,is_to_mix_weights,is_point_per_triangle)...,
        #get weights to move control points
        Points_weights_str(radiuss, batch_size, pad_voxels, (image_shape[1], image_shape[2], image_shape[3]),is_point_per_triangle,zero_der_beg,false)
    )
    # conv_part = get_conv(image_shape)

    # before_point_kerns = SkipConnection(debug_mix_weights(image_shape, radiuss, batch_size, spacing
    # ,num_params_exec,final_sv_repr,first_dense_out,second_dense_out
    # ,third_dense_out,primary_sv_repr,mixing_radius
    # ,num_prim_convs,num_weights_for_set_point_weights,num_channels_after_first_conv,pad_voxels ), connection_before_set_tetr_dat_kern)
    before_point_kerns = SkipConnection(get_weigts_layers, connection_before_set_tetr_dat_kern)

    #calculate all needed for loss caclulations
    model = Lux.Chain(before_point_kerns
    , Lux.Chain(Set_tetr_dat_str(radiuss, spacing, batch_size, 0, (image_shape[1], image_shape[2], image_shape[3]),false,false)
    , Point_info_kern_str(radiuss, spacing, batch_size, (image_shape[1], image_shape[2], image_shape[3]), num_base_samp_points, num_additional_samp_points,false,false))
    
    #     ,Lux.Chain(Set_tetr_dat_str(radiuss, spacing, batch_size, 0, (image_shape[1], image_shape[2], image_shape[3]),true,false)
    #    , Point_info_kern_str(radiuss, spacing, batch_size, (image_shape[1], image_shape[2], image_shape[3]), num_base_samp_points, num_additional_samp_points,true,false))
     
    
    
    
    )
    if(is_sinusoid_loss)

        set_of_svs = initialize_centers_and_control_points(image_shape, radiuss,is_point_per_triangle)
        if(!is_point_per_triangle)
            sv_centers, control_points, tetrs, dims = set_of_svs
        else
            sv_centers,control_points,tetrs,dims,plan_tensors=set_of_svs
        end    

        model = Lux.Chain(model,
        get_sinusoid_loss_layers(num_base_samp_points
        ,num_additional_samp_points
        ,size(tetrs)[1]
        ,is_point_per_triangle
        ,Float32(max_wavelength)
        ,Float32(max_amplitude)
        ,Float32(min_value),num_texture_banks,num_sinusoids_per_bank
        ,hp_dict["nn2"]
        ,hp_dict["p2"]
        ,hp_dict["nn3"]
        ,hp_dict["svn2"]
        ,hp_dict["svn3"]
        ,hp_dict["p3"]                              
        ,hp_dict["svn4"]
        ,hp_dict["p4"]
                
        )... )    
    end

    # model = Lux.Chain(before_point_kerns, Set_tetr_dat_str(radiuss,spacing,batch_size, 0, (image_shape[1], image_shape[2], image_shape[3])))
    # ,learning_rate_start,learning_rate_end
    # opt = Optimisers.AdamW(learning_rate_start)
    # opt = Optimisers.AdaDelta()
    # opt = Optimisers.AMSGrad()


    #configuring optimizer from hyperparameters
    opt = Optimisers.Lion(learning_rate_start)
    if(opt_str=="AdaDelta")
        opt = Optimisers.AdaDelta()
    end    
    if(opt_str=="NAdam")
        opt = Optimisers.NAdam()
    end    

  
    # Initialize the optimizer chain with the base optimizer
    opt_chain = []

    # Conditionally add WeightDecay
    if is_WeightDecay
        push!(opt_chain, WeightDecay())
    end
    push!(opt_chain, opt)

    # Add other optimizers based on conditions
    if add_gradient_accum && add_gradient_norm
        push!(opt_chain, AccumGrad(grad_accum_val))
        push!(opt_chain, ClipNorm(clip_norm_val; throw = false))
    elseif add_gradient_accum
        push!(opt_chain, AccumGrad(grad_accum_val))
    elseif add_gradient_norm
        push!(opt_chain, ClipNorm(clip_norm_val; throw = false))

    end

    # Create the OptimiserChain
    opt = OptimiserChain(opt_chain...)


    return model, opt
end #get_model_consts 




# 13186.39 
# #get convolutions
# conv1 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.sigmoid, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
# conv2 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.sigmoid, stride=2, pad=Lux.SamePad(), init_weight=glorot_uniform)
# conv_simple = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.sigmoid, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
# conv2_b = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.gelu, stride=2, pad=Lux.SamePad(), init_weight=glorot_uniform)
# conv5_b = (in, out) -> Lux.Conv((5, 5, 5), in => out, NNlib.gelu, stride=2, pad=Lux.SamePad(), init_weight=glorot_uniform)
# conv5_a = (in, out) -> Lux.Conv((5, 5, 5), in => out, NNlib.gelu, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
# conv3_a = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.gelu, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
# conv3_depth = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.gelu, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform, groups=in)
# convsigm2 = (in, out) -> Lux.Conv((3, 3, 3), in => out, NNlib.sigmoid, stride=2, pad=Lux.SamePad())

# num_channelss = 5
# num_channels_bigger = num_channelss * 2 * 2 * 2
# # num_channels_bigger=num_channelss*4*4*4
# im_shape_reduced = Int.(round.(image_shape[1:3] ./ 2))
# # im_shape_reduced=Int.(round.(image_shape[1:3]./4))
# conv_part = Lux.Chain(
#     conv5_a(2, 40), Lux.LayerNorm((Int(round(image_shape[1])), Int(round(image_shape[2])), Int(round(image_shape[3])), 40)), conv5_a(40, num_channelss)
#     # ,Lux.LayerNorm((Int(round(image_shape[1])),Int(round(image_shape[2])),Int(round(image_shape[3])),image_shape[4]))    
#     , ReshapeLayer((im_shape_reduced[1], im_shape_reduced[2], im_shape_reduced[3], num_channels_bigger)), conv3_a(num_channels_bigger, num_channels_bigger)

#     # ,Lux.LayerNorm((im_shape_reduced[1],im_shape_reduced[2],im_shape_reduced[3],image_shape[4]))
#     # ,conv3_depth(num_channels_bigger, num_channels_bigger)#depth wise convolutions
#     # ,Lux.LayerNorm((im_shape_reduced[1],im_shape_reduced[2],im_shape_reduced[3],image_shape[4]))
#     # ,conv5_depth(num_channels_bigger, num_channels_bigger)#depth wise convolutions
#     # ,Lux.LayerNorm((im_shape_reduced[1],im_shape_reduced[2],im_shape_reduced[3],image_shape[4]))
#     # ,conv_simple(num_channels_bigger, 24))
#     , conv_simple(num_channels_bigger, 24))


