using Revise, CUDA, HDF5,Wavelets
using Images,Random




function get_sample_dat(f::HDF5.File)
    keyss = keys(f)
    imagee = f[keyss[1]][:, :, :]
    # label = f[keyss[1]]["label"][:, :, :]
    imagee=reshape(imagee, (size(imagee)[1], size(imagee)[2], size(imagee)[3], 1, 1))
    return imagee, imagee
end

# function get_sample_image(f::HDF5.File)
#     keyss = keys(f)
#     return f[keyss[1]]["imagee_prep"][:, :, :,:,:]
# end

function pad_to_128(im)
    # Get the current size of the input array
    current_size = size(im)
    res=zeros(128,128,128,1,1)
    # Calculate the amount of padding needed for each dimension
    _x = min(128 ,current_size[1])
    _y = min(128 ,current_size[2])
    _z = min( 128 ,current_size[3])
    
    res[1:_x,1:_y,1:_z,:,:]=im
    res=res./maximum(res)
    res=res.*390
    # res=reshape(im, (size(im)[1], size(im)[2], size(im)[3], 1, 1))

    return res
end 


function get_image_batched(f::HDF5.File,batch_keys)
    images=map(key->pad_to_128(f[key]["image"][:,:,:]),batch_keys)
    res= cat(images...,dims=5)
    return res

end

function get_sample_image_batched(f::HDF5.File,batch_size)
    keyss = keys(f)[1:batch_size]
    return get_image_batched(f,keyss)
end


function divide_keys_into_batches(keyss::Vector{String}, batch_size::Int)
    # Calculate the number of full batches
    num_full_batches = div(length(keyss), batch_size)
    
    # Create sublists of the specified batch size
    batches = [keyss[(i-1)*batch_size+1:i*batch_size] for i in 1:num_full_batches]
    
    return batches
end






struct HDF5_keys_Dataset
    f::HDF5.File
    keys::Vector{String}
end

Base.length(dataset::HDF5_keys_Dataset) = length(dataset.keys)

function Base.getindex(dataset::HDF5_keys_Dataset, indexes::Vector{Int64})
    batch_keys=map(el-> dataset.keys[el],indexes)
    # img = dataset.dataset[keys(dataset.keys[i])]["imagee_prep"][:, :, :,:,1]
    img=get_image_batched(dataset.f,batch_keys)
    img=Float32.(img)

    return img
end

# function Base.getindex(dataset::HDF5_keys_Dataset, i::Int)
#     img = dataset.dataset[keys(dataset.keys[i])]["imagee_prep"][:, :, :,:,1]
#     img=Float32.(img)
#     lab = dataset.dataset[keys(dataset.keys[i])]["label"][:, :, :]
#     return img
# end


function get_train_data_loader(f,batch_size,is_distributed,distributed_backend, max_keys,rng,dev)
    keysss=keys(f)
    keysss_shuffled = shuffle(rng, keysss)
    print("\n num keysssss $(length(keysss_shuffled))\n")
    if(max_keys==-1)
        train_dataset = HDF5_keys_Dataset(f, keysss_shuffled[1:div(length(keysss_shuffled), batch_size) * batch_size])
    else
        train_dataset = HDF5_keys_Dataset(f, keysss_shuffled[1:max_keys])
    end
    # if is_distributed
    #     train_dataset = DistributedUtils.DistributedDataContainer(distributed_backend,
    #         train_dataset)
    # end
    train_loader = DataLoader(train_dataset, batchsize=batch_size, shuffle=true)
    return dev(train_loader)
end    




# keyss = keys(f)
# # aa=f[keyss[1]]
# function add_wtt(f,key)
#     wt = wavelet(WT.haar)
#     imagee = f[key]["image"][:, :, :]
#     wtt = dwt(imagee, wt)
#     wtt = reshape(wtt, size(wtt)..., 1, 1)
#     imagee=reshape(imagee, (size(imagee)[1], size(imagee)[2], size(imagee)[3], 1, 1))
#     imagee_prep = cat(imagee, wtt, dims=4)
#     write(f[key],"imagee_prep",imagee_prep)
#     return imagee
# end

# map(key->add_wtt(f,key),keys(f))
# close(f)