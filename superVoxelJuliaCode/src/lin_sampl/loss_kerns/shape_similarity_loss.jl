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


"""
we Want to calculate the similarity between the shapes of the superVoxels that are neighbours in the image
Generally we want to avoid situation that all the superVoxels have the same shape
reshaped_tetr_dat has shape (x,y,z, tetr_index_in_sv, point_index_in_tetr, point_coord, batch_size) 
we want to measure the cosine similarity between each corresponding point between neighbouring supervoxels
"""
function get_sv_shape_similarity(reshaped_tetr_dat, out_res,reshaped_tetr_dat_size,batch_size)

    shared_arr = CuStaticSharedArray(Float32, (48, 3, 4))
    # index = (1 + ((blockIdx().x - 1) * CUDA.blockDim_x()))
    # reshaped_tetr_dat[blockIdx().x,blockIdx().x,blockIdx().x, threadIdx().x, threadIdx().y, threadIdx().z,blockIdx().y]
    shared_arr[threadIdx().x,threadIdx().y,1]=0.0
    shared_arr[threadIdx().x,threadIdx().y,2]=0.0
    shared_arr[threadIdx().x,threadIdx().y,3]=0.0
    shared_arr[threadIdx().x,threadIdx().y,4]=0.0

    for x_change in -1:1
        for y_change in -1:1
            for z_change in -1:1
                if(((Int((blockIdx().x% batch_size)+1)+x_change)>0) && ((Int((blockIdx().x% batch_size)+1)+x_change)<=reshaped_tetr_dat_size[1]) && ((blockIdx().y+y_change)>0) && ((blockIdx().y+y_change)<=reshaped_tetr_dat_size[2]) &&  ((blockIdx().z+z_change)>0) && ((blockIdx().z+z_change)<=reshaped_tetr_dat_size[3]))
                    for i in 1:3
                        #we multiplied the values of the points in the neighbouring superVoxels - so done first part of dot product 
                        shared_arr[threadIdx().x,threadIdx().y,1]+=((reshaped_tetr_dat[
                                Int((blockIdx().x% batch_size)+1)#x
                                ,blockIdx().y#y
                                ,blockIdx().z#z
                                , threadIdx().x#tetr_index_in_sv
                                , threadIdx().y+1#point_index_in_tetr
                                ,i
                                ,Int(ceil(blockIdx().x/batch_size))#batch_size 
                                #we subtract sv center
                                ]-reshaped_tetr_dat[
                                    Int((blockIdx().x% batch_size)+1)#x
                                    ,blockIdx().y#y
                                    ,blockIdx().z#z
                                    , threadIdx().x#tetr_index_in_sv
                                    , 1#point_index_in_tetr
                                    ,i
                                    ,Int(ceil(blockIdx().x/batch_size))#batch_size 
                                    ])                                
                                *(reshaped_tetr_dat[
                                    Int((blockIdx().x% batch_size)+1)+x_change#x
                                    ,blockIdx().y+y_change#y
                                    ,blockIdx().z+z_change#z
                                    , threadIdx().x#tetr_index_in_sv
                                    , threadIdx().y#point_index_in_tetr
                                    ,i
                                    ,Int(ceil(blockIdx().x/batch_size))#batch_size 
                                ]-reshaped_tetr_dat[
                                    Int((blockIdx().x% batch_size)+1)+x_change#x
                                    ,blockIdx().y+y_change#y
                                    ,blockIdx().z+z_change#z
                                    , threadIdx().x#tetr_index_in_sv
                                    , 1#point_index_in_tetr
                                    ,i
                                    ,Int(ceil(blockIdx().x/batch_size))#batch_size 
                                ]
                                ))
                                #saved dot product
 
                        end  


                        #in the first channel of shared memory we have the dot product now we need to divide by distance
                        for i in 1:3
                            #we multiplied the values of the points in the neighbouring superVoxels - so done first part of distance 
                            shared_arr[threadIdx().x,threadIdx().y,2]+=((reshaped_tetr_dat[
                                    Int((blockIdx().x% batch_size)+1)#x
                                    ,blockIdx().y#y
                                    ,blockIdx().z#z
                                    , threadIdx().x#tetr_index_in_sv
                                    , threadIdx().y+1#point_index_in_tetr
                                    ,i
                                    ,Int(ceil(blockIdx().x/batch_size))#batch_size 
                                    #we subtract sv center
                                    ]-reshaped_tetr_dat[
                                        Int((blockIdx().x% batch_size)+1)#x
                                        ,blockIdx().y#y
                                        ,blockIdx().z#z
                                        , threadIdx().x#tetr_index_in_sv
                                        , 1#point_index_in_tetr
                                        ,i
                                        ,Int(ceil(blockIdx().x/batch_size))#batch_size 
                                        ])^2)                                
                                    

                            end  
                            for i in 1:3
                                #we multiplied the values of the points in the neighbouring superVoxels - so done first part of distance 
                                shared_arr[threadIdx().x,threadIdx().y,3]+=((reshaped_tetr_dat[
                                    Int((blockIdx().x% batch_size)+1)+x_change#x
                                    ,blockIdx().y+y_change#y
                                    ,blockIdx().z+z_change#z
                                    , threadIdx().x#tetr_index_in_sv
                                    , threadIdx().y#point_index_in_tetr
                                    ,i
                                    ,Int(ceil(blockIdx().x/batch_size))#batch_size 
                                ]-reshaped_tetr_dat[
                                    Int((blockIdx().x% batch_size)+1)+x_change#x
                                    ,blockIdx().y+y_change#y
                                    ,blockIdx().z+z_change#z
                                    , threadIdx().x#tetr_index_in_sv
                                    , 1#point_index_in_tetr
                                    ,i
                                    ,Int(ceil(blockIdx().x/batch_size))#batch_size 
                                ]
                                )^2)                            
                                    
                                end 
                            #get cosine similarity and accumulate it over neighbours
                            shared_arr[threadIdx().x,threadIdx().y,4]+=shared_arr[threadIdx().x,threadIdx().y,1]/(sqrt(shared_arr[threadIdx().x,threadIdx().y,2])*sqrt(shared_arr[threadIdx().x,threadIdx().y,3]))

                       end
                end
        end
    end
    sync_threads()
    #we have the cosine similarity between the neighbouring superVoxels per point in sv now we should reduce it to one value
    #reduce over y dimension - diffrent points in sv
    if(threadIdx().y==1)
        shared_arr[threadIdx().x,1,4]=shared_arr[threadIdx().x,1,4]+shared_arr[threadIdx().x,2,4]+shared_arr[threadIdx().x,3,4]
    end
    sync_threads()
    #reducing in x dimension - diffrent tetrahedrons in sv - performing binary reduction
    if(threadIdx().y==1)
    sync_threads()

        ii=1
        if ((threadIdx().x-1)%(2*ii)==0)
            if ((threadIdx().x+(ii))<=CUDA.blockDim_x())
            shared_arr[threadIdx().x,1,4]=((shared_arr[threadIdx().x,1,4]+shared_arr[threadIdx().x+ii,1,4]))
            end
        end
        sync_threads()
        ii=2
        if ((threadIdx().x-1)%(2*ii)==0)
            if ((threadIdx().x+(ii))<=CUDA.blockDim_x())
            shared_arr[threadIdx().x,1,4]=((shared_arr[threadIdx().x,1,4]+shared_arr[threadIdx().x+ii,1,4]))
            end
        end
        sync_threads()
        ii=4
        if ((threadIdx().x-1)%(2*ii)==0)
            if ((threadIdx().x+(ii))<=CUDA.blockDim_x())
            shared_arr[threadIdx().x,1,4]=((shared_arr[threadIdx().x,1,4]+shared_arr[threadIdx().x+ii,1,4]))
            end
        end
        sync_threads()
        ii=8
        if ((threadIdx().x-1)%(2*ii)==0)
            if ((threadIdx().x+(ii))<=CUDA.blockDim_x())
            shared_arr[threadIdx().x,1,4]=((shared_arr[threadIdx().x,1,4]+shared_arr[threadIdx().x+ii,1,4]))
            end
        end
        sync_threads()
        ii=16
        if ((threadIdx().x-1)%(2*ii)==0)
            if ((threadIdx().x+(ii))<=CUDA.blockDim_x())
            shared_arr[threadIdx().x,1,4]=((shared_arr[threadIdx().x,1,4]+shared_arr[threadIdx().x+ii,1,4]))
            end
        end
        sync_threads()
        ii=32
        if ((threadIdx().x-1)%(2*ii)==0)
            if ((threadIdx().x+(ii))<=CUDA.blockDim_x())
            shared_arr[threadIdx().x,1,4]=((shared_arr[threadIdx().x,1,4]+shared_arr[threadIdx().x+ii,1,4]))
            end
        end
    end
    sync_threads()

    
    
    #in shared_arr[threadIdx().x,threadIdx().y,4] we have the cosine similarity between the neighbouring superVoxels per point in sv
    # out_res[Int(ceil(blockIdx().x/batch_size)),blockIdx().y,blockIdx().z,Int((blockIdx().x% batch_size)+1)]=shared_arr[1,1,4]

    return nothing
end


function get_sv_shape_similarity_deff(reshaped_tetr_dat, d_reshaped_tetr_dat
    , out_res, d_out_res,reshaped_tetr_dat_size,batch_size)

    Enzyme.autodiff_deferred(Enzyme.Reverse
    , Enzyme.Const(get_sv_shape_similarity), Const
    , Duplicated(reshaped_tetr_dat, d_reshaped_tetr_dat)
    , Duplicated(out_res, d_out_res)
    ,Const(reshaped_tetr_dat_size),
    Const(batch_size)
    )
    return nothing
end




function call_get_sv_shape_similarity(reshaped_tetr_dat)

    blocks_x = size(reshaped_tetr_dat)[1]
    blocks_y = size(reshaped_tetr_dat)[2]
    blocks_z = size(reshaped_tetr_dat)[3]
    batch_size = size(reshaped_tetr_dat)[end]
    out_res = CUDA.zeros(blocks_x, blocks_y, blocks_z,batch_size)
    reshaped_tetr_dat_size=size(reshaped_tetr_dat)

    @cuda threads = (48,3) blocks = (blocks_x*batch_size, blocks_y, blocks_z) get_sv_shape_similarity(reshaped_tetr_dat, out_res,reshaped_tetr_dat_size,batch_size)

    return out_res
end


function ChainRulesCore.rrule(::typeof(call_get_sv_shape_similarity), reshaped_tetr_dat)

    out_res = call_get_sv_shape_similarity(reshaped_tetr_dat)

    function call_test_kernel1_pullback(d_out_res)
        #@device_code_warntype 

        d_out_res = CuArray(Zygote.unthunk(d_out_res))
        d_reshaped_tetr_dat = CUDA.zeros(size(reshaped_tetr_dat)...)
        blocks_x = size(reshaped_tetr_dat)[1]
        blocks_y = size(reshaped_tetr_dat)[2]
        blocks_z = size(reshaped_tetr_dat)[3]
        batch_size = size(reshaped_tetr_dat)[end]
        @cuda threads = (48,3) blocks = (blocks_x*batch_size, blocks_y, blocks_z) get_sv_shape_similarity_deff(reshaped_tetr_dat, d_reshaped_tetr_dat
        , out_res, d_out_res,reshaped_tetr_dat_size,batch_size)

        return NoTangent(), d_reshaped_tetr_dat
    end

    return out_res, call_test_kernel1_pullback

end



#block dim x should be 1 ,block dim y should be 24 and z should be 9