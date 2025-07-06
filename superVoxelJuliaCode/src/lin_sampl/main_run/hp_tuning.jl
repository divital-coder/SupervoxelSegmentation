using Revise, CUDA, HDF5
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
using Test,Logging,Interpolations
using KernelAbstractions, Dates
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote
using Lux, Random, Optimisers, Zygote
using LinearAlgebra, MLUtils
using Revise
using Pkg,JLD2,GLFW
using TensorBoardLogger, Logging, Random
using ParameterSchedulers
using LuxCUDA,Setfield
import MPI # Enables distributed training in Lux. NCCL is needed for CUDA GPUs
using DataFrames, CSV

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/get_lin_synth_dat.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/data_manag.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/main_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/main_loop.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/Lux_model.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern _old.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/dif_get_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/get_hyper_p.jl")

sitk = pyimport_conda("SimpleITK", "simpleitk")
np = pyimport_conda("numpy", "numpy")


#constants for distributed training


# """
# modification to execute always on the same threads so one can decide on the basis of this on what gpu to run
# """
# function HyperTuning.evaluate_objective_threads(f::Function, scenario::Scenario)
    
#     verbose = scenario.verbose
#     trials = sample(scenario)

#     tasks = Vector{Task}(undef, length(trials))
#     print("\n  hhhhhhhhhhhhhhhhhhhhhhhhh $(length(tasks))\n")

#     for i in eachindex(trials)
#         if iseven(i)
#             # Schedule on thread 2
#             tasks[i] = Threads.@spawnat 2 begin
#                 evaluate_trial!(f, trials[i], verbose)
#             end
#         else
#             # Schedule on thread 3
#             tasks[i] = Threads.@spawnat 3 begin
#                 evaluate_trial!(f, trials[i], verbose)
#             end
#         end
#     end

#     # Wait for all tasks to complete
#     for t in tasks
#         fetch(t)
#     end

# end


"""
return list of hyperparameter dictionaries to try
"""
function get_hps(hp_dict)
    num_epochs=hp_dict["num_epochs"]
    lr_cycle_len=hp_dict["lr_cycle_len"]

    lossa0=deepcopy(hp_dict)
    lossa0["add_gradient_accum"]=true
    lossa0["add_gradient_norm"]=true
    # lossa0["loss_function"]=[ ("border_loss",1) ]
    lossa0["num_epochs"]=2

    lossa0_b=deepcopy(lossa0)
    lossa0_b["is_sinusoid_loss"]=false

    lossa1=deepcopy(hp_dict)
    lossa1["learning_rate_start"]=0.0001
    lossa1["learning_rate_end"]=0.00000001
 
    lossa2=deepcopy(hp_dict)
    lossa2["learning_rate_start"]=0.00001
    lossa2["learning_rate_end"]=0.000000001
 
    lossa3=deepcopy(hp_dict)
    lossa3["learning_rate_start"]=0.001
    lossa3["learning_rate_end"]=0.0000001


    lossa4=deepcopy(hp_dict)
    lossa4["add_gradient_accum"]=false

    lossa5=deepcopy(hp_dict)
    lossa5["add_gradient_norm"]=true

    # if get(ENV, "CUDA_VISIBLE_DEVICES", "") == "1"

    #     return [lossa2,lossa3,lossa4,lossa5 ]#,lossa1,lossa2,lossa3,lossa4,lossa5
    # end
    # return [lossa2,lossa3,lossa4,lossa5 ,lossa1,lossa0,lossa0_b]
    return [lossa0]

end



function get_consts_for_sin_loss(f, batch_size,radiuss)
    imm=get_sample_image_batched(f, batch_size)
    maxx=maximum(imm)
    minn=minimum(imm)
    max_wavelength=minimum(collect(radiuss))/2
    max_amplitude= abs(maxx-minn)
    min_value=0.0#abs(minn)
    return max_wavelength,max_amplitude,min_value
end    


function modify_hp_dict(hp_dict)
    # @suggest learning_rate_start in trial
    # @suggest learning_rate_end in trial
    # @suggest add_gradient_accum in trial
    # @suggest add_gradient_norm in trial
    # @suggest is_WeightDecay in trial
    # @suggest grad_accum_val in trial
    # @suggest clip_norm_val in trial

    learning_rate_start = parse(Float32, get(ENV, "LEARNING_RATE_START", "0.00001"))
    learning_rate_end = parse(Float32, get(ENV, "LEARNING_RATE_END", "0.00000001"))
    add_gradient_accum = parse(Bool, get(ENV, "ADD_GRADIENT_ACCUM", "true"))
    add_gradient_norm = parse(Bool, get(ENV, "ADD_GRADIENT_NORM", "true"))
    is_WeightDecay = parse(Bool, get(ENV, "IS_WEIGHT_DECAY", "true"))
    grad_accum_val = parse(Int, get(ENV, "GRAD_ACCUM_VAL", "4"))
    clip_norm_val = parse(Float32, get(ENV, "CLIP_NORM_VAL", "1.0"))

    hp_dict_loc=deepcopy(hp_dict)

    hp_dict_loc["learning_rate_start"]=learning_rate_start[1]+learning_rate_end[1]
    hp_dict_loc["learning_rate_end"]=learning_rate_end[1]
    hp_dict_loc["add_gradient_accum"]=add_gradient_accum
    hp_dict_loc["add_gradient_norm"]=add_gradient_norm
    hp_dict_loc["is_WeightDecay"]=is_WeightDecay
    hp_dict_loc["grad_accum_val"]=grad_accum_val
    hp_dict_loc["clip_norm_val"]=clip_norm_val
    hp_dict_loc["optimiser"]="LiOn"
    hp_dict_loc["CUDA_VISIBLE_DEVICES"]=parse(Int, get(ENV, "CUDA_VISIBLE_DEVICES", "1"))

    print("hhhhhhhhhhhhh hp chosen $(hp_dict_loc) \n")


    return hp_dict_loc
end    


function run_to_precompile(
    f, batch_size, hp_dict, image_size, spacing, curr_tensor_board_path, 
    plane_dist, epoch, windowWidth, windowHeight, texture_width, texture_height, window, 
    rng, vjp, dev, render_channel, is_point_per_triangle,show_visualization
)
    # Existing code remains the same
    max_wavelength, max_amplitude, min_value = get_consts_for_sin_loss(f, batch_size, hp_dict["radiuss"])
    
    model, opt = get_model_consts(image_size, spacing, hp_dict, max_wavelength, max_amplitude, min_value)
    lg = TBLogger(curr_tensor_board_path, min_level=Logging.Info)
    radiuss = hp_dict["radiuss"]
    
    # Move parameters to GPU
    ps, st = Lux.setup(rng, model) |> dev
    
    # Move optimizer state to GPU explicitly
    st_opt = if dev isa CUDADevice
        Optimisers.setup(opt, ps) |> dev
    else
        Optimisers.setup(opt, ps)
    end
    
    # Create train state with GPU-located components
    tstate = Lux.Training.TrainState(nothing, nothing, model, ps, st, opt, st_opt, 0)                      
    
    # Forward pass remains the same
    forward_pass_single(rng, model, opt, f, batch_size, hp_dict, plane_dist, radiuss, lg, epoch
    , windowWidth, windowHeight, texture_width, texture_height, window, tstate, render_channel, dev, is_point_per_triangle,show_visualization)

    # For the backward pass, make sure data is on GPU correctly
    loss_f = loss_simple
    is_sinusoid_loss = hp_dict["is_sinusoid_loss"]
    if(is_sinusoid_loss)
        loss_f = loss_sinusoids
    end    

    # Ensure the data is on GPU with the correct type
    imagee = get_sample_image_batched(f, batch_size) |> dev
    imagee = Float32.(imagee)
    imagee= imagee |> dev
    # Run the training step
    _, loss, _, tstate = Lux.Training.single_train_step!(vjp, loss_f, imagee, tstate)
end

# mutable struct Atomic{T}; @atomic x::T; end

function load_ps(path_check_p)
    @load path_check_p ps st
    return ps
end


# function get_restart_tstate(rng, model, path_check_p, opt)
#     psb, st = Lux.setup(rng, model) 
#     ps=load_ps(path_check_p)
#     st_opt = Optimisers.setup(opt, ps)
#     tstate = Lux.Training.TrainState(nothing, nothing, model, ps, st, opt, st_opt, 0)
#     return tstate
# end

function get_restart_tstate(rng, model, path_check_p, opt, dev)
    psb, st = Lux.setup(rng, model) 
    ps = load_ps(path_check_p) |> dev
    st = st |> dev
    st_opt = Optimisers.setup(opt, ps) |> dev
    tstate = Lux.Training.TrainState(nothing, nothing, model, ps, st, opt, st_opt, 0)
    return tstate
end


function hp_tuning(spacing,f,windowWidth, windowHeight,batch_size
    ,main_tensor_board_path,main_out_path,max_keys,show_visualization)

    dev= gpu_device() 
    dev_num=1

    # # Initialize a channel at global scope for asynchronous rendering (and logging rendered images to TensorBoard)
    render_channel = Channel{Tuple}(30000)

    # # Start a task that consumes from the channel and calls process_and_render
    @async begin
        for data in render_channel
            process_and_render(data...)
        end
    end

    #get seed
    Random.seed!(12)

    #for case we are restarting
    cud_dev=parse(Int, get(ENV, "CUDA_VISIBLE_DEVICES", '1'))
    out_path = get(ENV, "out_path", "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/debug_inner/$(cud_dev)")
    
    # to_restart = parse(Int, get(ENV, "restart", "0")) TODO unhash
    to_restart=false
    path_check_p="$(out_path)/trained_model.jld2"


    rng = Random.default_rng()
    # Random.seed!(12)
    #clear tensorboard folder
    # if isdir(main_tensor_board_path)
    #     rm(main_tensor_board_path; force=true, recursive=true)
    # end
    # if isdir(main_out_path)
    #     rm(main_out_path; force=true, recursive=true)
    # end

    #get keys and constantsfor training
    vjp = Lux.Experimental.ADTypes.AutoZygote()
    # keyss = keys(f)
    # # given list of strings "keyss" divide it into sublists of equal length "batch_size"
    # # if length of "keyss" is not divisible by "batch_size" then remove  last sublist which would  have length less than "batch_size"
    # batched_keys=divide_keys_into_batches(keyss,batch_size)[1:7]
    
    train_data_loader=get_train_data_loader(f,batch_size,[],[],max_keys,rng,dev)

    #initialize hyperparameters
    hp_dict=get_base_hp()
    hp_dict["batch_size"]=batch_size
    is_point_per_triangle=hp_dict["is_point_per_triangle"]

    #initialize image size and display constants
    image_size = get_init_consts(hp_dict["batch_size"])   
    texture_width = image_size[1]
    texture_height = image_size[2]
    # Close all windows and terminate GLFW and set the image consts for logging
    window=[]
    image_consts=[]
    if(show_visualization)
        GLFW.Terminate()
        window=initializeWindow(windowWidth, windowHeight)
        image_consts=window
    end

    

    #place to modify hyperparameters
    # hp_dicts_list=get_hps(hp_dict)
    exp=1

    # for_gpu_counter_1 = Atomic(1)
    # for_gpu_counter_2 = Atomic(0)



    #iterate over diffrent hyperparameters
    # for hp_dict in hp_dicts_list

        #initialize model
    #precompilation 
    plane_dist=30.0 #arbitrary
    epoch=0#arbitrary
    curr_tensor_board_path="$(main_tensor_board_path)/00_$(exp)"

    #try to precompile on both gpu's
    run_to_precompile(
        f, batch_size, hp_dict, image_size, spacing, curr_tensor_board_path, 
        plane_dist, epoch, windowWidth, windowHeight, texture_width, texture_height, window, 
        rng, vjp,gpu_device(),render_channel,is_point_per_triangle,show_visualization
    )

    # run_to_precompile(
    #     f, batch_size, hp_dict, image_size, spacing, curr_tensor_board_path, 
    #     plane_dist, epoch, windowWidth, windowHeight, texture_width, texture_height, window, 
    #     rng, vjp,gpu_device(1)
    # )

        # function objective(trial)
            exp+=1
            #setting diffrent gpu's to be used on diffrent runs
            # dev= gpu_device() 
            # dev_num=1
            # if((@atomic for_gpu_counter_1.x)>(@atomic for_gpu_counter_2.x))
            #     @atomic for_gpu_counter_2.x += 1
            #     dev= gpu_device(2)
            #     dev_num=2

            # else
            #     @atomic for_gpu_counter_1.x += 1
            #     dev= gpu_device(1)    
            #     dev_num=1
            # end
            # print("\n ddddddddddddddddddddddd dev used $(dev_num)  \n")

            curr_tensor_board_path="$(main_tensor_board_path)/exp_$(exp)"
            
            hp_dict=modify_hp_dict(hp_dict)


            max_wavelength,max_amplitude,min_value=get_consts_for_sin_loss(f, batch_size,hp_dict["radiuss"])

            model, opt = get_model_consts(image_size,   spacing,hp_dict,max_wavelength,max_amplitude,min_value)
            # Distributed training: wrap the optimizer in a distributed optimizer
            opt = is_distributed ?
            DistributedUtils.DistributedOptimizer(distributed_backend, opt) :
            opt


            #main loop for training
            

            if(to_restart==1)
                print("\n restartingggg \n")
                #we get new state we use only loaded parameters
                tstate= get_restart_tstate(rng, model, path_check_p, opt,dev)
            else
                ps, st = Lux.setup(rng, model) |> dev

                if(is_distributed)
                    ps = DistributedUtils.synchronize!!(distributed_backend, ps)
                    st = DistributedUtils.synchronize!!(distributed_backend, st)
                end
                
                # tstate = Lux.Training.TrainState(model, ps, st, opt)
                
                st_opt = if dev isa ReactantDevice
                    ps_cpu = ps |> cpu_device()
                    Optimisers.setup(opt, ps_cpu) |> dev
                else
                    Optimisers.setup(opt, ps)
                end
                tstate= Lux.Training.TrainState(nothing, nothing, model, ps, st, opt, st_opt, 0)
            end
  
            
            tstate = tstate |> dev
            


            lg=TBLogger(curr_tensor_board_path, min_level=Logging.Info)
            #single forward pass just to check weather it compile

            radiuss=hp_dict["radiuss"]
            # forward_pass_single(rng, model, opt, f, batch_size, hp_dict, plane_dist, radiuss, lg, epoch, windowWidth, windowHeight, texture_width, texture_height, window,tstate,render_channel,dev,is_point_per_triangle)
            
            


            if is_distributed
                @set! tstate.optimizer_state = DistributedUtils.synchronize!!(
                    distributed_backend, tstate.optimizer_state)
            end

            # JLD2.@load "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/checkpoints/trained_model.jld2" tstate
            tstate_trained,metricc = main_loop(hp_dict["num_epochs"], rng, model, opt, tstate
            , f, batch_size, image_size,image_consts,hp_dict,train_data_loader,curr_tensor_board_path,windowWidth
            , windowHeight,texture_width, texture_height,lg,render_channel,dev ,path_check_p,to_restart,out_path,show_visualization)
            




            #save_image for 3d visualization
            is_sinusoid_loss=hp_dict["is_sinusoid_loss"]
            if(is_sinusoid_loss)
                big_mean_weighted,weighted_sin_loss,tetr_dat,sin_p,texture_bank_p= infer_model(tstate_trained, model, get_sample_image_batched(f, batch_size))
            else
                out_sampled_points, tetr_dat = infer_model(tstate_trained, model, get_sample_image_batched(f, batch_size))
            end


            # out_path="$(main_out_path)"
            # if !isdir(out_path)
            #     mkdir(out_path)
            # end

            # out_path="$(main_out_path)/exp_$(exp)"
            # if !isdir(out_path)
            #     mkdir(out_path)
            # end


            #reset the atomics controlling which gpu is used
            # if(dev_num==1)
            #     @atomic for_gpu_counter_1.x -= 1
            # end
            # if(dev_num==2)
            #     @atomic for_gpu_counter_2.x -= 1
            # end
            out_csv= "$(out_path)/res.csv"

            # Create a DataFrame with a single column "res"
            df = DataFrame(res = [metricc])

            # Save the DataFrame to a CSV file
            CSV.write(out_csv, df)


            # sitk.WriteImage(sitk.GetImageFromArray(out_image_to_vis), "$(out_path)/inferred.nii.gz")
            # sitk.WriteImage(sitk.GetImageFromArray(get_sample_image_batched(f, batch_size)[:, :, :, 1, 1]), "$(out_path)/res_im_orig.nii.gz")

            #save example data
            # h5_path_b ="$(out_path)/locc.h5"
            # fb = h5open(h5_path_b, "w")
            # if(is_sinusoid_loss)
            #     write(fb, "big_mean_weighted", Array(big_mean_weighted))
            #     write(fb, "tetr_dat", Array(tetr_dat))
            #     write(fb, "sin_p", Array(sin_p))
            #     write(fb, "texture_bank_p", Array(texture_bank_p))
            #     write(fb, "im", Array(get_sample_image_batched(f,batch_size)[:, :, :, 1, 1]))
            # else
            #     write(fb, "out_sampled_points", Array(out_sampled_points))
            #     write(fb, "tetr_dat", Array(tetr_dat))
            #     write(fb, "im", Array(get_sample_image_batched(f,batch_size)[:, :, :, 1, 1]))
            # end
            # close(fb)

    #         return metricc
    # end


    # scenario = Scenario(### hyperparameters
    #                 # learning rates
                    # learning_rate_start= [0.00001,0.0001,0.0002,0.0007,0.001],#0.0000001,0.000001
                    # learning_rate_end = [0.0000000001,0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001,0.0002,0.0007,0.001],
                    # add_gradient_accum = [true, false],
                    # add_gradient_norm = [true, false],
                    # is_WeightDecay = [true, false],
                    # grad_accum_val = [2,6,10],
                    # clip_norm_val = [1.5,10.0,100.0],
    #                 # pruner= MedianPruner(start_after = 5#=trials=#, prune_after = 10#=epochs=#),
    #                 verbose = true, # show the log
    #                 max_trials = 1200, # maximum number of hyperparameters computed
    #                 batch_size=2,
    #                 )

    # display(scenario)

    # # minimize accuracy error
    # HyperTuning.optimize(objective, scenario)

    # # end
        





end
