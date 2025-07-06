using KernelAbstractions
using CUDA

using CUDA, KernelAbstractions, Enzyme

c = 1:64
@kernel function square!(x, @Const(c))
    I = @index(Global, Linear)
    @inbounds x[I] = c[I] * x[I] ^ 2
end

function f!(x, backend)
    kernel = square!(backend)
    kernel(x, c, ndrange = size(x))
    # KernelAbstractions.synchronize(backend)
end

x = CUDA.ones(64)
backend = KernelAbstractions.get_backend(x)

∂f_∂x = similar(x)
∂f_∂x .= 1.0
Enzyme.autodiff(
    Reverse, 
    f!, 
    Duplicated(x, ∂f_∂x), 
    Const(backend)
)

∂f_∂x


function set_tetr_dat_kern_deff(tetr_dat, d_tetr_dat, tetr_dat_out, d_tetr_dat_out, source_arr, d_source_arr, control_points, d_control_points, sv_centers, d_sv_centers, max_index, spacing)
    Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(set_tetr_dat_kern_unrolled), Const, Duplicated(tetr_dat, d_tetr_dat), Duplicated(tetr_dat_out, d_tetr_dat_out), Duplicated(source_arr, d_source_arr), Duplicated(control_points, d_control_points), Duplicated(sv_centers, d_sv_centers), Const(max_index), Const(spacing))
    return nothing
end


# maska_testowa = zeros(Int32, 4, 4, 4)
# maska_testowa[1, 1, 1] = 1
# maska_testowa[1, 1, 2] = 0
# maska_testowa[1, 2, 1] = 0
# maska_testowa[2, 1, 1] = 0
# maska_testowa[4, 4, 3] = 1
# maska_testowa[4, 4, 4] = 1
# maska_testowa_3 = zeros(Int32, 6, 6, 6)
# maska_testowa_3[1, 1, 1] = 1
# maska_testowa_3[1, 1, 2] = 1
# maska_testowa_3[1, 1, 3] = 1
# maska_testowa_3[1, 1, 4] = 1
# maska_testowa_3[1, 1, 5] = 1
# maska_testowa_3[1, 1, 6] = 0
# maska_testowa_3[1, 2, 1] = 1
# maska_testowa_3[2, 1, 1] = 1
# maska_testowa_3[6, 6, 6] = 1
# maska_testowa_3[6, 6, 5] = 0



# @kernel function initialize_labels_kernel(mask, labels, width, height, depth)
#     index_cart = @index(Global, Cartesian)

#     if index_cart[1] >= 1 && index_cart[1] <= width && index_cart[2] >= 1 && index_cart[2] <= height && index_cart[3] >= 1 && index_cart[3] <= depth
#         if mask[index_cart[1], index_cart[2], index_cart[3]] == 1
#             labels[index_cart[1], index_cart[2], index_cart[3]] = index_cart[1] + (index_cart[2] - 1) * width + (index_cart[3] - 1) * width * height
#         else
#             labels[index_cart[1], index_cart[2], index_cart[3]] = 0
#         end
#     end
# end

# @kernel function propagate_labels_kernel(mask, labels, width, height, depth)
#     index_cart = @index(Global, Cartesian)


#     if index_cart[1] >= 1 && index_cart[1] <= width && index_cart[2] >= 1 && index_cart[2] <= height && index_cart[3] >= 1 && index_cart[3] <= depth
#         if mask[index_cart[1], index_cart[2], index_cart[3]] == 1
#             current_label = labels[index_cart[1], index_cart[2], index_cart[3]]
#             for di in -1:1
#                 for dj in -1:1
#                     for dk in -1:1
#                         if di == 0 && dj == 0 && dk == 0
#                             continue
#                         end
#                         ni = index_cart[1] + di
#                         nj = index_cart[2] + dj
#                         nk = index_cart[3] + dk
#                         if ni >= 1 && ni <= width && nj >= 1 && nj <= height && nk >= 1 && nk <= depth
#                             if mask[ni, nj, nk] == 1 && labels[ni, nj, nk] < current_label
#                                 labels[index_cart[1], index_cart[2], index_cart[3]] = labels[ni, nj, nk]
#                             end
#                         end
#                     end
#                 end
#             end
#         end
#     end
# end

# # mask=maska_testowa_3
# # width, height, depth = size(mask)
# # labels_gpu = CUDA.fill(0, size(mask))
# # dev = get_backend(labels_gpu)
# # workgroupsize = (3, 3, 3)
# # ndrange = (width, height, depth)

# # # Initialize labels
# # initialize_labels_kernel(dev, workgroupsize)(mask_gpu, labels_gpu, width, height, depth, ndrange = ndrange)
# # CUDA.synchronize()


# function largest_connected_component_v5(mask::Array{Int32, 3})
#     width, height, depth = size(mask)
#     mask_gpu = CuArray(mask)
#     labels_gpu = CUDA.fill(0, size(mask))
#     dev = get_backend(labels_gpu)
#     workgroupsize = (3, 3, 3)
#     ndrange = (width, height, depth)
    
#     # Initialize labels
#     initialize_labels_kernel(dev, workgroupsize)(mask_gpu, labels_gpu, width, height, depth, ndrange = ndrange)
#     CUDA.synchronize()

#     # Propagate labels iteratively
#     for _ in 1:10 
#         propagate_labels_kernel(dev, workgroupsize)(mask_gpu, labels_gpu, width, height, depth, ndrange = ndrange)
#         CUDA.synchronize()
#     end

#     # Download labels back to CPU
#     labels_cpu = Array(labels_gpu)
#     println("labels_cpu: ", labels_cpu)  # Print labels for debugging
    
#     # Debugging: Check if labels_cpu has non-zero values
#     println("Non-zero labels count: ", count(labels_cpu .!= 0))
#     println("Unique labels: ", unique(labels_cpu))
    
#     # Find the largest connected component
#     max_label = maximum(labels_cpu)
#     largest_label = 0
#     largest_size = 0
    
#     for l in 1:max_label
#         size = count(labels_cpu .== l)
#         if size > largest_size
#             largest_size = size
#             largest_label = l
#         end
#     end
    
#     largest_component = labels_cpu .== largest_label
#     return largest_component
# end

# largest_component = largest_connected_component_v5(maska_testowa_3)