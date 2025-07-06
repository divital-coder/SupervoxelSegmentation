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
using Test, Logging, Interpolations
using KernelAbstractions, Dates
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote
using Lux, Random, Optimisers, Zygote
using LinearAlgebra, MLUtils
using Revise
using Pkg, JLD2
using TensorBoardLogger, Logging, Random
using ParameterSchedulers, GLFW
using MLDataDevices, DelaunayTriangulation
using DataFrames, CSV


includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/get_lin_synth_dat.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/data_manag.jl")

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/prepare_polygons.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/render_for_tensor_board.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/render_grey_poligons.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/OpenGLUtils.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/prepare_polygons.jl")







"""
apply state to model
"""
function infer_model(tstate_glob, model, imagee)
    y_pred, st = Lux.apply(model, imagee, tstate_glob.parameters, tstate_glob.states)
    out_sampled_points, tetr_dat,b_loss = y_pred
    return y_pred
end

function get_case(f, keyss, i)
    imagee = f[keyss[i]]["image"][:, :, :]
    imagee = reshape(imagee, (size(imagee)[1], size(imagee)[2], size(imagee)[3], 1, 1))
    imagee = Float32.(imagee)
    wt = wavelet(WT.haar)
    wtt = dwt(imagee[:, :, :, 1, 1], wt)
    wtt = reshape(wtt, size(wtt)..., 1, 1)
    wtt = Float32.(wtt)
    imagee = cat(imagee, wtt, dims=4)
end




"""
visualize 
"""
function process_and_render(im_name, axis, plane_dist, tetr_dat, radiuss, f, imm_val, lg, epoch, windowWidth, windowHeight, texture_width, texture_height, window, sv_means,is_point_per_triangle)
   
    GLFW.Terminate()
    window=initializeWindow(windowWidth, windowHeight)

   
    imm_val_loc = Float32.(Array(imm_val)[:, :, :, 1, 1])
    tetr_dat = Array(tetr_dat)
    sv_means = sv_means .- minimum(sv_means)
    sv_means = sv_means ./ maximum(sv_means)

    tetr_num = get_num_tetr_in_sv(is_point_per_triangle)
    image_size = size(imm_val_loc)
    triang_vec, sv_i, line_vertices, line_indices, imm = get_data_to_display(tetr_dat, tetr_num, axis, plane_dist, image_size, imm_val_loc)
    colorss = Float32.(sv_means[Int.(sv_i)])
    window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program, line_shader_program, textUreId = initialize_window_etc(windowWidth, windowHeight, texture_width, texture_height, window)

    line_vao, line_vbo, line_ebo = initialize_lines(line_vertices, line_indices)
    im_lines = render_and_save(lg, "$(im_name)_lines", epoch, windowWidth, windowHeight, texture_width, texture_height, imm_val_loc[:,:,plane_dist], line_vao, line_indices, line_shader_program, rectangle_vao, rectangle_shader_program, textUreId, window)
    sleep(0.5)
    im_gp = render_grey_polygons(triang_vec, colorss, window, windowWidth, windowHeight, lg, im_name, step)

    log_image(lg, "$(im_name)_lines", im_lines, CWH)
    log_image(lg, "$(im_name)_gp", im_gp, CWH)

    # all_res,dat, line_vertices, line_indices=main_get_poligon_data(tetr_dat, axis, plane_dist, radiuss,get_num_tetr_in_sv(),imm_val_loc)    
    # # print("\n ddddddd dat $(size(dat)) min $(minimum(dat))  dat $(maximum(dat))  \n")

    # window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program, line_shader_program, textUreId=initialize_edge_viz.initialize_window_etc(windowWidth, windowHeight,texture_width, texture_height,window)

    # line_vao, line_vbo, line_ebo = initialize_edge_viz.initialize_lines(line_vertices, line_indices)
    # im = render_and_save(lg, "$(im_name)_lines", epoch, windowWidth, windowHeight, texture_width, texture_height, dat, line_vao, line_indices, line_shader_program, rectangle_vao, rectangle_shader_program, textUreId, window)

    # all_res=main_get_poligon_data(tetr_dat, axis, plane_dist, radiuss,get_num_tetr_in_sv())



    # RenderGreyPoligons.render_grey_poligons(all_res, sv_means,lg,window,windowWidth, windowHeight,im_name,epoch) 

    return imm_val_loc
end





"""
visualization saved to nifti file for more precise analisis
"""
function process_and_render_full_image(im_name, axis,  tetr_dat, radiuss, f, imm_val,  epoch, windowWidth, windowHeight, texture_width, texture_height, window, sv_means,out_path,is_point_per_triangle)
    
    GLFW.Terminate()
    window=initializeWindow(windowWidth, windowHeight)

    imm_val_loc_a = Float32.(Array(imm_val)[:, :, :, 1, 1])
    tetr_dat = Array(tetr_dat)
    sv_means = sv_means .- minimum(sv_means)
    sv_means = sv_means ./ maximum(sv_means)


    im_lines_whole=[]
    im_gp_whole=[]
    im_base_whole=[]

    for plane_dist in 8:(size(imm_val_loc_a)[3]-8)
        try
            imm_val_loc=copy(imm_val_loc_a)
            tetr_num = get_num_tetr_in_sv(is_point_per_triangle)
            image_size = size(imm_val_loc)
            triang_vec, sv_i, line_vertices, line_indices, imm = get_data_to_display(tetr_dat, tetr_num, axis, plane_dist, image_size, imm_val_loc)
            colorss = Float32.(sv_means[Int.(sv_i)])
            window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program, line_shader_program, textUreId = initialize_window_etc(windowWidth, windowHeight, texture_width, texture_height, window)
            GLFW.PollEvents()

            line_vao, line_vbo, line_ebo = initialize_lines(line_vertices, line_indices)
            GLFW.PollEvents()

            im_lines,im_base = render_and_save_separtated(windowWidth,windowHeight, texture_width, texture_height, imm_val_loc[:,:,plane_dist], line_vao, line_indices, line_shader_program, rectangle_vao, rectangle_shader_program, textUreId, window)
            GLFW.PollEvents()

            push!(im_lines_whole, sum(im_lines, dims=1)[:,:,:])
            push!(im_base_whole, sum(im_base, dims=1)[:,:,:])
        catch e
            println("Error in processing plane_dist $(plane_dist): $e")
            continue
        end    
    end



    for plane_dist in 8:(size(imm_val_loc_a)[3]-8)
        try
            imm_val_loc=copy(imm_val_loc_a)
            tetr_num = get_num_tetr_in_sv(is_point_per_triangle)
            image_size = size(imm_val_loc)
            triang_vec, sv_i, line_vertices, line_indices, imm = get_data_to_display(tetr_dat, tetr_num, axis, plane_dist, image_size, imm_val_loc)
            colorss = Float32.(sv_means[Int.(sv_i)])
            window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program, line_shader_program, textUreId = initialize_window_etc(windowWidth, windowHeight, texture_width, texture_height, window)

            im_gp = render_grey_polygons_loc(triang_vec, colorss, window, windowWidth, windowHeight, im_name, step)
            GLFW.PollEvents()
            push!(im_gp_whole, sum(im_gp, dims=1)[:,:,:])
        catch e
            println("Error in processing plane_dist $(plane_dist): $e")
            continue
        end
    end


    im_lines_whole_im=vcat(im_lines_whole...)
    im_base_whole_im=vcat(im_base_whole...)
    im_gp_whole_im=vcat(im_gp_whole...)
    p_out_im_base="$(out_path)/im_base_whole_im.nii.gz"
    p_out_im_lines="$(out_path)/im_base_lines_im.nii.gz"
    p_out_im_gp="$(out_path)/im_base_gp_im.nii.gz"
    
    im_lines_whole_im=UInt8.(round.(im_lines_whole_im))
    sitk.WriteImage(sitk.GetImageFromArray(im_base_whole_im), p_out_im_base)
    sitk.WriteImage(sitk.GetImageFromArray(im_lines_whole_im), p_out_im_lines)
    sitk.WriteImage(sitk.GetImageFromArray(im_gp_whole_im), p_out_im_gp)





end


"""
    get_applicable_loss(hp_dict::Dict, epoch::Int)

Given a dictionary `hp_dict` with an entry named "loss_function" that is a list of tuples of length 2, where the first entry is the loss name and the second is the beginning epoch when this loss applies, this function returns the loss name if the given `epoch` is greater than or equal to the beginning epoch. If multiple losses meet this criterion, the function selects the one with the highest beginning epoch.

# Arguments
- `hp_dict::Dict`: A dictionary containing the hyperparameters, including the "loss_function" entry.
- `epoch::Int`: The current epoch as an integer.

# Returns
- `String`: The name of the applicable loss function.
- `Nothing`: If no loss function applies to the given epoch.

# Example
```julia
hp_dict = Dict("loss_function" => [("var-border", 1), ("sv_variance_loss", 200)])
epoch = 150
loss_name = get_applicable_loss(hp_dict, epoch)
println(loss_name)  # Output should be "var-border"
"""
function get_applicable_loss(hp_dict, epoch)
    loss_tuples = hp_dict["loss_function"]
    applicable_losses = filter(x -> epoch >= x[2], loss_tuples)

    if isempty(applicable_losses)
        return nothing
    end
    # Find the loss with the highest beginning epoch
    selected_loss = argmax(x -> x[2], applicable_losses)[1]
    return selected_loss
end



function get_sv_mean(out_sampled_points,is_point_per_triangle)
    sizz_out = size(out_sampled_points)#(65856, 9, 5)
    batch_size = size(out_sampled_points)[end]#(65856, 9, 5)
    num_sv = Int(round(sizz_out[1] / get_num_tetr_in_sv(is_point_per_triangle)))

    # out_sampled_points = reshape(out_sampled_points[:, :, 1:2, :], (get_num_tetr_in_sv(), Int(round(sizz_out[1] / get_num_tetr_in_sv())), sizz_out[2], 2, batch_size))
    # out_sampled_points = permutedims(out_sampled_points, [2, 1, 3, 4, 5])
    values_big = reshape(out_sampled_points[:, :, :, 1, :], :, get_num_tetr_in_sv(is_point_per_triangle) * 5, batch_size)
    weights_big = reshape(out_sampled_points[:, :, :, 2, :], :, get_num_tetr_in_sv(is_point_per_triangle) * 5, batch_size)

    big_weighted_values = values_big .* weights_big
    big_weighted_values_summed = sum(big_weighted_values, dims=2)

    big_weights_sum = sum(weights_big, dims=2)

    big_mean_weighted = big_weighted_values_summed ./ big_weights_sum
    return big_mean_weighted

end







"""
single forward pass just to check weather it compiles 
some sanity check
"""
function forward_pass_single(rng, model, opt, f, batch_size, hp_dict, plane_dist, radiuss, lg, epoch, windowWidth, windowHeight, texture_width, texture_height, window, tstate, render_channel, dev,is_point_per_triangle,show_visualization)


    imagee = CuArray(get_sample_image_batched(f, batch_size))
    imagee=cu(Float32.(Array(imagee)))



    ps, st = Lux.setup(rng, model) |> dev
    tstate_glob = Lux.Training.TrainState(model, ps, st, opt)
    tstate_glob = tstate_glob |> dev
    # tstate_glob = cu(tstate_glob)
    is_sinusoid_loss = hp_dict["is_sinusoid_loss"]



    #first for compilation
    if (is_sinusoid_loss)
        loss_sinusoids(model, tstate_glob.parameters, tstate_glob.states, imagee)
    else

        loss_simple(model, tstate_glob.parameters, tstate_glob.states, imagee)
    end

    #measure time of forward pass
    current_time = Dates.now()
    if (is_sinusoid_loss)
        loss_sinusoids(model, tstate_glob.parameters, tstate_glob.states, imagee)
    else

        loss_simple(model, tstate_glob.parameters, tstate_glob.states, imagee)
    end

    current_time_diff = Dates.now() - current_time
    seconds_diff = Dates.value(current_time_diff) / 1000  # Convert milliseconds to seconds
    print("\n forward pass time single case with loss $(seconds_diff) sec   \n")

    ##### test visualizations
    if(show_visualization)
        imm_val = Float32.(Array(imagee))


        if (is_sinusoid_loss)
            big_mean_weighted, weighted_sin_loss, tetr_dat, sin_p, texture_bank_p = infer_model(tstate, model, Float32.(CuArray(imm_val)))
        else
            out_sampled_points, tetr_dat,b_loss = infer_model(tstate, model, Float32.(CuArray(imm_val)))
            big_mean_weighted = get_sv_mean(out_sampled_points,is_point_per_triangle)
        end

        im_name = "ax_1"
        axis = 3
        big_mean_weighted = Array(big_mean_weighted)
        with_logger(lg) do
            put!(render_channel, (im_name, axis, plane_dist, tetr_dat, radiuss, f, imm_val, lg, epoch, windowWidth, windowHeight, texture_width, texture_height, window, big_mean_weighted[:, 1, 1],is_point_per_triangle))
            # process_and_render(im_name, axis, plane_dist, tetr_dat, radiuss, f, imm_val, lg, epoch, windowWidth, windowHeight, texture_width, texture_height, window, big_mean_weighted[:, 1, 1])
            
            # path_d="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions/dd.h5"

            # h5open(path_d, "w") do h5file
            #     write(h5file,"big_mean_weighted",Array(big_mean_weighted))
            #     write(h5file,"tetr_dat", Array(tetr_dat))
            # end

        end
    end


end











"""
main loop for training
"""
function main_loop(num_epochs, rng, model, opt, tstate, f, batch_size, image_size, image_consts, hp_dict, train_dataloader, curr_tensor_board_path, windowWidth, windowHeight, texture_width, texture_height
    , lg, render_channel,dev,path_check_p,to_restart,out_path,show_visualization)


    window = image_consts

    tstate = tstate |> dev
    vjp = Lux.Experimental.ADTypes.AutoZygote()





    hp_dict_str = Dict{String,String}()
    for (key, value) in hp_dict
        value_str = string(value)
        hp_dict_str[key] = value_str
    end
    write_hparams!(lg, hp_dict_str, ["loss"])

    learning_rate_start = hp_dict["learning_rate_start"]
    learning_rate_end = hp_dict["learning_rate_end"]
    is_sinusoid_loss = hp_dict["is_sinusoid_loss"]
    opt_str=hp_dict["optimiser"]
    scheduler = ParameterSchedulers.CosAnneal(learning_rate_start, learning_rate_end, hp_dict["lr_cycle_len"])

    #if the model was restarted we start from the saved epoch otherwise from 1
    epoch_start=1

    # out_path = get(ENV, "out_path", "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions")
    out_path_epoch=joinpath(out_path, "epoch.csv")
    if (isfile(out_path_epoch))
        df = CSV.read(out_path_epoch, DataFrame)
        epoch_start = df[1, "e"]
    end

    losses_all = [1000000.0]
    patience_count = 0
    with_logger(lg) do

        for epoch in epoch_start:num_epochs

            # Save the current epoch to CSV nee
            df = DataFrame(e = [epoch])
            CSV.write(out_path_epoch, df)

            #for logging
            losses = []
            times = []

            #getting learning rate from learning rate scheduler
            eta = scheduler(epoch)
            loss_namee = get_applicable_loss(hp_dict, epoch)
            # loss_f=select_loss(get_applicable_loss(hp_dict, epoch))
            #selecting loss function
            is_point_per_triangle=hp_dict["is_point_per_triangle"]


            # loss_f=loss_function_border_loss
            if (is_sinusoid_loss)
                loss_f = loss_sinusoids

            else
                if loss_namee == "border_loss"
                    loss_f = loss_function_border_loss

                elseif loss_namee == "sv_variance_loss"
                    _loss_f = loss_function_sv_variance_loss

                elseif loss_namee == "var-border"
                    loss_f = loss_function_var_border

                elseif loss_namee == "varsq_div_border"
                    loss_f = loss_simple

                end
            end

            #loop in epoch

            index = 1
            for imagee in train_dataloader
                index = index + 1
                # if(index==1)
                #     imm_val=Array(imagee)
                # end
                current_time = Dates.now()
                if(opt_str!="NAdam" && opt_str!="AdaDelta")
                    tstate = Optimisers.adjust!(tstate, eta)    
                end

                # ## We can compute the gradients using Training.compute_gradients
                # gs, loss, stats, tstate = Lux.Training.compute_gradients(vjp, loss_f, imagee, tstate)

                # ## Optimization
                # tstate = Training.apply_gradients!(tstate, gs) 
                print("\n ssssss $(size(imagee)) \n")
                _, loss, _, tstate = Lux.Training.single_train_step!(vjp, loss_f, imagee, tstate)

                seconds_diff = Dates.value(Dates.now() - current_time) / 1000

                push!(times, seconds_diff)
                @info "time_iter" times = round(seconds_diff; digits=2) log_step_increment = 1
                @info "loss_iter" loss = loss log_step_increment = 1
                println(" ** $index : $seconds_diff sec** ")

                # imagee = Float32.(imagee)
                # cpu_state=cpu(tstate)
                # @save "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/checkpoints/trained_model.jld2" cpu_state
                push!(losses, loss)

            end

            #saving state after epoch 
            cpu_train_state = cpu_device()(tstate)
            ps=cpu_train_state.parameters
            st=cpu_train_state.states
            @save path_check_p ps st


            #logging images
           
            if (epoch % 10 == 0)


            if(show_visualization)
                    imm_val = Float32.(get_sample_image_batched(f, batch_size))

                    if (is_sinusoid_loss)
                        big_mean_weighted, weighted_sin_loss, tetr_dat, sin_p, texture_bank_p = infer_model(tstate, model, Float32.(CuArray(imm_val)))
                    else

                        out_sampled_points, tetr_dat,b_loss = infer_model(tstate, model, Float32.(CuArray(imm_val)))
                        big_mean_weighted = get_sv_mean(out_sampled_points,is_point_per_triangle)
                    end
                    big_mean_weighted = Array(big_mean_weighted)

                    tetr_dat = Array(tetr_dat)
                    plane_dist = 50

                    im_name = "case_1"
                    axis = 3

                    # Put arguments into the render_channel
                    process_and_render(im_name, axis, plane_dist, tetr_dat, radiuss, f, imm_val, lg,
                        epoch, windowWidth, windowHeight, texture_width, texture_height, window,
                        big_mean_weighted[:, 1, 1],is_point_per_triangle)
                    # put!(render_channel,(im_name, axis, plane_dist, tetr_dat, radiuss, f, imm_val, lg,
                    #     epoch, windowWidth, windowHeight, texture_width, texture_height, window,
                    #     big_mean_weighted[:, 1, 1],is_point_per_triangle))

                    # Second rendering task
                    plane_dist = 60
                    im_name = "case_2"

                    # Put arguments into the render_channel
                    process_and_render(im_name, axis, plane_dist, tetr_dat, radiuss, f, imm_val, lg,
                        epoch, windowWidth, windowHeight, texture_width, texture_height, window,
                        big_mean_weighted[:, 1, 1],is_point_per_triangle)
                    # put!(render_channel,(im_name, axis, plane_dist, tetr_dat, radiuss, f, imm_val, lg,
                    #     epoch, windowWidth, windowHeight, texture_width, texture_height, window,
                    #     big_mean_weighted[:, 1, 1],is_point_per_triangle))


                    process_and_render_full_image(im_name, axis,  tetr_dat, radiuss, f, imm_val,  epoch
                    , windowWidth, windowHeight, texture_width, texture_height, window, big_mean_weighted[:, 1, 1],out_path,is_point_per_triangle)
                end
            

            @info "Completed epoch: $epoch"


            end
            curr_loss = mean(losses)

            #setting patience
            if ((curr_loss > minimum(losses_all)))
                patience_count = patience_count + 1
                print("loss not improved from $(minimum(losses_all)) to $(curr_loss) patience_count $(patience_count) \n")

                if (patience_count > hp_dict["patience"])
                    print("patience reached $(patience_count) breaking \n")
                    break
                end
            else
                print("loss improved from $(minimum(losses_all)) to $(curr_loss) \n")
                patience_count = 0
            end
            push!(losses_all, curr_loss)
            @info "loss" loss = round(curr_loss; digits=2) log_step_increment = 1
            @info "time_batch" times = round(sum(times); digits=2) log_step_increment = 1
            @info "lr" eta = eta log_step_increment = 1
            print("\n  *********** epoch $(epoch) $(round(curr_loss; digits = 2)) \n")
        end
    end

    return tstate, mean(losses_all[end-1:end])
end



function get_init_consts(batch_size)
    imagee = get_sample_image_batched(f, batch_size)
    image_size = size(imagee)
    return image_size

end