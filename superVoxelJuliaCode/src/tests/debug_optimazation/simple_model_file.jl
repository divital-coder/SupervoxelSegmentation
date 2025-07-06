using LinearAlgebra
using Meshes
using GLMakie
using Statistics
using LinearAlgebra
using Random,Test
using LinearAlgebra,KernelAbstractions,CUDA
using Revise
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
using HDF5,LuxCUDA
using Plots,JLD2
using Wavelets, ParameterSchedulers, NNlib, LuxCUDA, JLD2
using Meshes
using TensorBoardLogger
using DataFrames, CSV
using Lux.Training

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_per_triangle.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/dif_points_per_triangle.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/main_loop.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")


function count_zeros_r(arr)
    num_zeros = count(x -> x == 0.0, arr)
    num_entries = length(arr)
    percentt = (num_zeros / num_entries) * 100
    return percentt
end

function infer_model(tstate_glob, model, imagee)
    y_pred, st = Lux.apply(model, CuArray(imagee), tstate_glob.parameters, tstate_glob.states)
    out_sampled_points, tetr_dat = y_pred
    return y_pred
end


radiuss = (Float32(3.5), Float32(3.5), Float32(3.5))
batch_size=3
pad_voxels=true
image_shape=(128, 128, 128)

global const len_get_random_point_in_tetrs_kern=Int(floor(256/3))
global const is_point_per_triangle=true

add_triangle_per_point=is_point_per_triangle


conv5_a = (in, out) -> Lux.Conv((7, 7, 7), in => out, NNlib.gelu, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)
conv5_s = (in, out) -> Lux.Conv((7, 7, 7), in => out, NNlib.sigmoid, stride=1, pad=Lux.SamePad(), init_weight=glorot_uniform)

ch=168
radiuss= (Float32(3.5), Float32(3.5), Float32(3.5))
spacing=(Float32(1.0), Float32(1.0), Float32(1.0))
model=Lux.Chain(conv5_a(ch,ch),conv5_a(ch,ch),conv5_a(ch,ch),conv5_a(ch,ch) ,conv5_s(ch,ch),Points_weights_str(radiuss,batch_size,pad_voxels,image_shape,add_triangle_per_point))
# weights=CuArray(rand(Float32,20,20,20,168,3))
# weights=CuArray(rand(Float32,22,22,22,ch,batch_size))
weights=CuArray(ones(Float32,22,22,22,ch,batch_size))
minimum(weights)
maximum(weights)

dev= gpu_device() 
rng = Random.default_rng()
vjp = Lux.Experimental.ADTypes.AutoZygote()
ps, st = Lux.setup(rng, model) |> dev
hp_path="/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_opt_hp/a"

# Create TensorBoard logger
tb_logger = TBLogger(hp_path)



learning_rate_start = parse(Float32, get(ENV, "LEARNING_RATE_START", "0.001"))
learning_rate_end = parse(Float32, get(ENV, "LEARNING_RATE_END", "0.0001"))
add_gradient_accum = parse(Bool, get(ENV, "ADD_GRADIENT_ACCUM", "true"))
add_gradient_norm = parse(Bool, get(ENV, "ADD_GRADIENT_NORM", "true"))
is_WeightDecay = parse(Bool, get(ENV, "IS_WEIGHT_DECAY", "true"))
grad_accum_val = parse(Int, get(ENV, "GRAD_ACCUM_VAL", "4"))
clip_norm_val = parse(Float32, get(ENV, "CLIP_NORM_VAL", "1.0"))
lr_cycle_len = parse(Int, get(ENV, "lr_cycle_len", "10"))
opt_type = get(ENV,"opt_type","OAdam")
global const closeness_to_discr=parse(Float32, get(ENV, "closeness_to_discr", "1000.0"))


opt = Optimisers.OAdam(learning_rate_start)

#configuring optimizer from hyperparameters
if(opt_type=="OAdam")
    opt = Optimisers.OAdam(learning_rate_start)
elseif(opt_type=="Lion")
    opt = Optimisers.Lion(learning_rate_start)
elseif(opt_type=="AdamW")
    opt = Optimisers.AdamW(learning_rate_start)
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

scheduler = ParameterSchedulers.CosAnneal(learning_rate_start, learning_rate_end, lr_cycle_len)

# Declare globals before use


function loss_function_dummy(model, ps, st, data)
    y_pred, st = Lux.apply(model, data, ps, st)
    #y_pred[1] (17, 17, 17, 31, 3, 3)
    #y_pred[2]  (18, 18, 18, 3, 3)
    sv_centers_out=y_pred[1][2:end,2:end,2:end,:,:,:]
    if(is_point_per_triangle)
    sv_centers_expanded = repeat(sv_centers_out, 1, 1, 1, 1, 1,31)
    else
        sv_centers_expanded = repeat(sv_centers_out, 1, 1, 1, 1, 1,7)
    end
    sv_centers_expanded=permutedims(sv_centers_expanded, [1, 2, 3, 6, 4, 5]) #(17, 17, 17, 31, 3, 3)
    b_loss= sv_centers_expanded-y_pred[2]
    b_loss=b_loss.^2
    return mean(b_loss), st, ()
end

function main_train(patience::Int=100)
    tstate = Training.TrainState(model, ps, st, opt)
    fin_loss = 0.0
    best_loss = Inf
    epochs_without_improvement = 0

    for epoch in 1:1400
        eta = scheduler(epoch)
        tstate = Optimisers.adjust!(tstate, eta)
        gs, loss, stats, tstate = Training.single_train_step!(vjp, loss_function_dummy, weights, tstate)
        fin_loss = loss
        
        with_logger(tb_logger) do
            @info "training" loss=loss step=epoch
        end

        # Early stopping logic
        if loss < best_loss
            best_loss = loss
            epochs_without_improvement = 0
        else
            epochs_without_improvement += 1
        end

        if epochs_without_improvement >= patience
            @info "Early stopping triggered at epoch $epoch"
            break
        end
    end
    
    return fin_loss
end
fin_loss = main_train()

# Log final metrics and hyperparameters
# with_logger(tb_logger) do
#     @info "hparams" hparams=hparams metrics=Dict("final_loss" => fin_loss, "metric" => metricc)
# end

metricc = fin_loss

with_logger(tb_logger) do
    @info "metricc" metricc=metricc step=1000
end

# Create hyperparameter dictionary
hparams = Dict(
    "learning_rate_start" => learning_rate_start,
    "learning_rate_end" => learning_rate_end,
    "add_gradient_accum" => add_gradient_accum,
    "add_gradient_norm" => add_gradient_norm,
    "is_WeightDecay" => is_WeightDecay,
    "grad_accum_val" => grad_accum_val,
    "clip_norm_val" => clip_norm_val,
    "lr_cycle_len" => lr_cycle_len,
    "opt_type" => opt_type,
    "closeness_to_discr" => closeness_to_discr,
    "fin_loss" => fin_loss
)



write_hparams!(tb_logger, hparams, ["metricc"])

out_path = get(ENV, "out_path", "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_directions")
out_csv = "$(out_path)/res.csv"
df = DataFrame(res = [metricc])

# Save the DataFrame to a CSV file
CSV.write(out_csv, df)