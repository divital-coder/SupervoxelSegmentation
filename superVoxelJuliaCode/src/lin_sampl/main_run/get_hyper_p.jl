
function get_base_hp()
    lr_cycle_len=40
    num_epochs = lr_cycle_len*300
    return Dict(
        "batch_size" => 1,
        "num_base_samp_points" => 2,
        "num_additional_samp_points" => 1,
        "lr_cycle_len"=>lr_cycle_len,
        "radiuss" => (Float32(3.5), Float32(3.5), Float32(3.5)),
        # "radiuss" => (Float32(3.5), Float32(3.5), Float32(3.5)),
        "learning_rate_start" => 0.00008, #start for learning rate
        "learning_rate_end" => 0.00000008, #end for learning rate
        "num_params_exec" => 48, # controls the number of parameter used in get weights from directions kernel
        "mixing_param_sets" => 10, # final super voxel representation from mixing weights before mixing information of neighbouring sv and reducing it to num_weights_for_set_point_weights
        "is_to_mix_weights" => true, #if true will perform additional mixing of weights inside kernel (IMPORTANT leads to long compilation)
        "loss_function"=>[ ("varsq_div_border",1) ] ,
        # "loss_function"=>[ ("border_loss",1) ] ,
        "num_epochs"=>num_epochs,
        "num_weights_for_set_point_weights" => 24, #do not change this
        "is_sinusoid_loss"=>false,#weather to use sinusoid loss or not
        #setting for sinusoid loss
        "num_texture_banks"=>32,
        "num_sinusoids_per_bank"=>4
        ,"nn2"=>10
        ,"p2"=>10
        ,"nn3"=>10
        ,"svn2"=>10
        ,"svn3"=>10
        ,"p3"=>10                                
        ,"svn4"=>12
        ,"p4"=>12
        ,"add_gradient_norm"=>false
        ,"add_gradient_accum"=>false
        ,"patience"=>1000
        ,"is_WeightDecay"=>true
        ,"grad_accum_val"=>10
        ,"clip_norm_val"=>100
        ,"optimiser"=>"LiOn"#AdaDelta
        ,"is_point_per_triangle"=>false
        ,"zero_der_beg"=>false
        )
    
end