using ChainRulesCore, Zygote, CUDA, Enzyme
using KernelAbstractions
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote
using LinearAlgebra
using Revise,HDF5
using Plots
using cuTENSOR,TensorOperations,VectorInterface,Test,Lux
using Statistics,LuxCUDA

const num_sinusoids_per_bank=4
const num_texture_banks=32
const is_point_per_triangle=true

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sinusoid_loss.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")

h5_path_b = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/locc.h5"
fb = h5open(h5_path_b, "r")
out_sampled_points=fb["out_sampled_points"][:,:,:,:]
tetr_dat=fb["tetr_dat"][:,:,:,:]
keys(fb)
close(fb)



num_base_samp_points=3
num_additional_samp_points=2



max_wavelength=9.0
max_amplitude=5.0
min_value=1.0



sizz_out=size(out_sampled_points)
batch_size=sizz_out[end]
n_tetr=get_num_tetr_in_sv()
num_sv = Int(round(sizz_out[1] / n_tetr))

#sin_p first index is index of sv last is batch size on second we  store main info x 
# entries 1) offset_x, 2) offset_y, 3) offset_z  4)bias 5) multiplier (should be from 0.5 to 2 - arbitrary choice to enable strengthening or reducing signal in this particulat sv ) 
#and from 6 to  num_texture_banks +6  weight for each texture bank all of those weights need to be after soft max so that they sum to 1
sin_p = CUDA.rand(Float32,sizz_out[1],num_texture_banks+6,batch_size).*2
sin_p_a=sin_p[:,1:5,:]
sin_p_b=softmax(sin_p[:,6:end,:],dims=2)
sin_p=cat(sin_p_a,sin_p_b,dims=2)
#bank store 1) alpha 2) beta 3) gamma 4) wavelength 5) amplitude
texture_bank_p = CUDA.rand(Float32,num_texture_banks,num_sinusoids_per_bank,5)
out_sampled_points=CuArray(out_sampled_points)
n_points_per_tetr=num_base_samp_points + (3 * num_additional_samp_points)

#reshape so that we can use it in the kernel we need to have index of supervoxel on first dimension 
#then index of point in supervoxel on second dimension we need to concatenta all points in supervoxel in this dimension
#then the point koordinates its value and weight
#and then we need to have batch size on the last dimension
out_sampled_points_reshaped = reshape(out_sampled_points[:, :, :, :], (get_num_tetr_in_sv(), Int(round(sizz_out[1] / get_num_tetr_in_sv())), sizz_out[2], 5, batch_size))
out_sampled_points_reshaped = permutedims(out_sampled_points_reshaped, [2, 1, 3, 4, 5])
out_sampled_points_reshaped_c=copy(out_sampled_points_reshaped)

new_siz=size(out_sampled_points_reshaped)
out_sampled_points_reshaped=reshape(out_sampled_points_reshaped,new_siz[1],new_siz[2]*new_siz[3],5,new_siz[end])
out_sampled_points_reshaped=CuArray(out_sampled_points_reshaped)

dimm=(80,80,80)
base_arr=get_base_indicies_arr((dimm[1],dimm[2],dimm[3]))   
base_arr=reshape(base_arr,dimm[1]*dimm[2]*dimm[3],3)

b_sh=size(base_arr)
zerr=zeros(b_sh[1],2)
base_arr=cat(zerr,base_arr,dims=2)
b_sh=size(base_arr)

dummy_out_sampled = reshape(base_arr, (1,b_sh[1],5 , 1))
dummy_out_sampled=CuArray(dummy_out_sampled)
# aa=out_sampled_points_reshaped[78,:,:,:,1]
# aa=reshape(aa,144*9,5)
# bb=out_sampled_points_reshaped_b[78,:,:,1]
# aa==bb
texture_bank_begin_index=0
res=call_get_sinusoid_loss(out_sampled_points_reshaped,sin_p,texture_bank_p,max_wavelength,max_amplitude,min_value,texture_bank_begin_index)

res=call_get_sinusoid_loss(dummy_out_sampled,sin_p,texture_bank_p,max_wavelength,max_amplitude,min_value,texture_bank_begin_index)


rrrr=Array(res)[1,:,1]
rrrr=reshape(rrrr,dimm[1],dimm[2],dimm[3])
sl_num=20
slice= rrrr[sl_num,:,:]

### visualization test
heatmap(slice, color=:viridis, title="Heatmap of Slice", xlabel="X-axis", ylabel="Y-axis")


out_sampled_points=out_sampled_points_reshaped
out_res = CUDA.zeros(size(out_sampled_points,1), size(out_sampled_points,2),batch_size)#first and last dimension where last dimension is batch

d_out_res = CUDA.ones(size(out_res)...)
d_out_sampled_points = CUDA.zeros(size(out_sampled_points)...)

d_sin_p = CUDA.zeros(size(sin_p)...)
d_texture_bank_p = CUDA.zeros(size(texture_bank_p)...)

blocks_x = Int(ceil(size(out_sampled_points,2)/256))
max_index=size(out_sampled_points,2)
batch_size = size(out_sampled_points)[end]

Duplicated(out_sampled_points, d_out_sampled_points)
Duplicated(sin_p, d_sin_p)
Duplicated(texture_bank_p, d_texture_bank_p)
Duplicated(out_res, d_out_res)

blocks = (blocks_x, size(out_sampled_points,1),batch_size) 
# blocks = (1, 1,1) 
threads = (256,)


###### differentiation test - does it compile

# @cuda threads = threads blocks = blocks  get_sinusoid_loss_deff(out_sampled_points, d_out_sampled_points, sin_p,d_sin_p
# ,texture_bank_p, d_texture_bank_p
# ,out_res, d_out_res ,max_wavelength,max_amplitude,min_value,max_index)
# mean(d_out_res)
# mean(d_out_sampled_points)





#### pipeline test - doeas it compile


mmm=get_sinusoid_loss_layers(num_base_samp_points
                                ,num_additional_samp_points
                                ,size(tetr_dat)[1]
                                ,batch_size
                                ,Float32(max_wavelength)
                                ,Float32(max_amplitude)
                                ,Float32(min_value)
                                ,num_texture_banks
                                ,num_sinusoids_per_bank
                            
                                )


mmm=Lux.Chain(mmm...)
rng=Random.GLOBAL_RNG
dev = gpu_device()
ps, st = Lux.setup(rng, mmm)|> dev
opt = Optimisers.AdamW(0.0001)|> dev
tstate_glob = Lux.Training.TrainState(mmm,ps, st, opt)|> dev


y_pred, st = Lux.apply(mmm, (CuArray(out_sampled_points_reshaped_c),CuArray(tetr_dat)), tstate_glob.parameters, tstate_glob.states)


big_mean_weighted,weighted_sin_loss,tetr_dat,sin_p,texture_bank_p=y_pred
print("\n big_mean_weighted $(size(big_mean_weighted)), $(mean(big_mean_weighted)); weighted_sin_loss $(size(weighted_sin_loss)) $(mean(weighted_sin_loss))  tetr_dat $(size(tetr_dat)) $(mean(tetr_dat))  \n")
