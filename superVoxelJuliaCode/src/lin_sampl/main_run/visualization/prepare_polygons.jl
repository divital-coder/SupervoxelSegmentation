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
using Pkg,JLD2
using TensorBoardLogger, Logging, Random
using ParameterSchedulers,GLFW
using MLDataDevices,DelaunayTriangulation
using Combinatorics


includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/render_for_tensor_board.jl")



function get_bool_index_of_edge(tetr_dat_3d, axis, plane_dist,p1,p2 )
    bool_indd_a =  Bool.((tetr_dat_3d[:,:, p1, axis] .< (plane_dist)) .* (tetr_dat_3d[:, :,p2, axis] .> (plane_dist)))
    bool_indd_b =  Bool.((tetr_dat_3d[:,:, p1, axis] .> (plane_dist)) .* (tetr_dat_3d[:, :,p2, axis] .< (plane_dist)))
    bool_indd=bool_indd_a .|| bool_indd_b
    dist_a = tetr_dat_3d[:, :, p1, axis] .- plane_dist
    dist_b = tetr_dat_3d[:, :, p2, axis] .- plane_dist
    t = dist_a ./ (dist_a .- dist_b)
    intersectt= tetr_dat_3d[:, :, p1, 1:3] .+ t .* (tetr_dat_3d[:, :, p2, 1:3] .- tetr_dat_3d[:, :, p1, 1:3])
    # intersectt[repeat(.!bool_indd,1,1,3)].=0.0
    sz=size(intersectt)
    idx1 = reshape(collect(1:sz[1]), sz[1], 1)
    idx1 = repeat(idx1, 1, sz[2])
    intersectt = cat(intersectt, idx1, dims=3)
    return bool_indd,intersectt,(p1,p2)
end


function get_triangle_intersects(edges,intersects,comb)
    e_a=edges[comb[1]]
    e_b=edges[comb[2]]
    e_c=edges[comb[3]]
    i_a=intersects[comb[1]]
    i_b=intersects[comb[2]]
    i_c=intersects[comb[3]]
    e_a,i_a=flatten_arrays(e_a, i_a)
    e_b,i_b=flatten_arrays(e_b, i_b)
    e_c,i_c=flatten_arrays(e_c, i_c)
    indd = [1, 2, 3, 4, 5, 6]
    remaining_indices = setdiff(indd, comb)

    #we need to exclude cases where we have quads to avoid displaying them with incorrect triangulation
    e_a_n=vec(edges[remaining_indices[1]])
    e_b_n=vec(edges[remaining_indices[2]])
    e_c_n=vec(edges[remaining_indices[3]])

    

    bool_loc_index = e_a .& e_b .& e_c
    bool_loc_index_n = e_a_n .& e_b_n .& e_c_n
    #excluding quads
    bool_loc_index=bool_loc_index .& (.!bool_loc_index_n)
    return (hcat(i_a[bool_loc_index, 1:3],i_b[bool_loc_index, 1:3],i_c[bool_loc_index, 1:3]),i_a[bool_loc_index, 4])


end


"""
    get_triangle_info(triangle_inds, all_intersects)

Extract triangle information from the given indices and intersection points.

# Arguments
- `triangle_inds::Array{Bool}`: Boolean array indicating which tetrahedra have 3 intersecting edges.
- `all_intersects::Array{Float32}`: Array of intersection points.

# Returns
- `(triangle_vecs::Vector{Float32}, triangle_sv_inds::Vector{Int})`: Tuple containing triangle vectors and supervoxel indices.
"""
function get_triangle_info(edges,intersects)
    #get all combinations of 3 - we will use it in getting triangle data
    indd = [1, 2, 3, 4, 5, 6]
    combos = generate_combinations(indd, 3)


    tupless=map(c->get_triangle_intersects(edges,intersects,c),combos)
    triangs=map(x->x[1],tupless)
    inds=map(x->x[2],tupless)

    triangs=filter(x->size(x,1)>0,triangs)


    triang_vecs=vcat(triangs...)
    triangle_sv_inds=vcat(inds...)

    # # Flatten the first two dimensions
    # flat_triangle_inds = vec(triangle_inds)
    # sizz = size(all_intersects)
    # flatter_intersects = reshape(all_intersects, (sizz[1] * sizz[2], sizz[3], sizz[4]))
    
    # # Sort last dimension in increasing fashion based on first entry in second dimension
    # sorted_flatter_intersects = sort_flatter_intersects(flatter_intersects)

    # sorted_triangle_inds = sorted_flatter_intersects[flat_triangle_inds, :, :]
    # triangle_vecs = reshape(sorted_triangle_inds[:, 1:3, 4:6], (size(sorted_triangle_inds, 1), 9))
    # triangle_vecs = permutedims(triangle_vecs, [2, 1])
    # triangle_vecs = vec(triangle_vecs)
    # triangle_sv_inds = sorted_triangle_inds[:, 4, 1]
    
    return triang_vecs, triangle_sv_inds
end




#### get quads ####
# Step 1: Flatten edges and intersects
"""
    flatten_arrays(edge, intersectt)

Flatten the first two dimensions of `edge` and `intersectt` arrays.

# Arguments
- `edge::Array{Bool}`: Boolean array indicating edge intersections.
- `intersectt::Array{Float32}`: Array of intersection points.

# Returns
- `(edge_flat, intersectt_flat)`: Tuple containing flattened `edge` and `intersectt` arrays.
"""
function flatten_arrays(edge, intersectt)
    # Flatten edge (Boolean array)
    edge_flat = vec(edge)
    # Reshape intersectt to (num_elements, num_features)
    s = size(intersectt)
    intersectt_flat = reshape(intersectt, (:, s[end]))
    return edge_flat, intersectt_flat
end

# Step 2: Generate combinations
"""
    generate_combinations(indd::Vector{Int}, len::Int)

Generate all valid combinations of the given length from the indices.

# Arguments
- `indd::Vector{Int}`: Vector of indices.
- `len::Int`: Length of combinations.

# Returns
- `combinations::Vector{Vector{Int}}`: Vector of combinations.
"""
function generate_combinations(indd::Vector{Int}, len::Int)
    return collect(combinations(indd, len))
end



function find_common_indices_to(pairs::Vector{Tuple{Int, Int}},current_pair) 
    n = length(pairs) 
    indices = Int[]


    # Check each pair against all other pairs

        
        for j in 1:n
                other_pair = pairs[j]
                if current_pair[1]==other_pair[1] ||current_pair[1]==other_pair[2] || current_pair[2]==other_pair[1] ||current_pair[2]==other_pair[2]
                    push!(indices, j)

                end
        end
        


    return indices
end

# Step 3 and 4: Process combinations to construct triangulations
"""
    process_combinations(combos, edges_flat, intersects_flat, pairs)

Process each combination to construct valid triangulations.

# Arguments
- `combos::Vector{Vector{Int}}`: List of combinations.
- `edges_flat::Vector{Vector{Bool}}`: Flattened edge arrays.
- `intersects_flat::Vector{Matrix{Float32}}`: Flattened intersect arrays.
- `pairs::Vector{Tuple{Int, Int}}`: List of index pairs.

# Returns
- `(all_triangle_vectors, sv_indices)`: Tuple containing triangle vectors and supervoxel indices.
"""
function process_combinations(combos, edges_flat, intersects_flat, pairs)
    all_triangle_vectors = Float32[]
    sv_indices = Int[]

    for combo in combos
        # Extract edges, intersects, and pairs for the current combination
        combo_edges = [edges_flat[i] for i in combo]
        combo_intersects = [intersects_flat[i] for i in combo]
        combo_pairs = [pairs[i] for i in combo]

        # Compute combined boolean index where all edges are true
        bool_loc_index = combo_edges[1]
        for e in combo_edges[2:end]
            bool_loc_index .= bool_loc_index .& e
        end


        # Skip if no valid indices
        if sum(bool_loc_index) == 0
            continue
        end



        # Use bool_loc_index to select valid entries without iteration
        # Extract intersection points for valid entries
        # pts = [intersect[bool_loc_index, :] for intersect in combo_intersects]

        # print("\n iiiiii  combo_pairs $(combo_pairs) bool_loc_index sum $(sum(bool_loc_index))\n")
        tri_b=[1,3,4]

        tri_a=find_common_indices_to(combo_pairs,combo_pairs[1])       
        remaining_indices = setdiff([1, 2, 3, 4], tri_a)
        if(length(remaining_indices)==0)
            remaining_indices=[4]
            tri_a=[1,2,3]
        else
            tri_b=find_common_indices_to(combo_pairs,combo_pairs[remaining_indices[1]])
        end


        # Construct triangles based on pairs
        # Example triangulation:
        # Triangle 1: pts[1], pts[2], pts[3]
        # Triangle 2: pts[1], pts[3], pts[4]

        # Prepare triangle coordinates
        
        tri1_coords = hcat(combo_intersects[tri_a[1]][:, 1:3], combo_intersects[tri_a[2]][:, 1:3], combo_intersects[tri_a[3]][:, 1:3])  # (num_points, 9)
        tri2_coords = hcat(combo_intersects[tri_b[1]][:, 1:3], combo_intersects[tri_b[2]][:, 1:3], combo_intersects[tri_b[3]][:, 1:3])  # (num_points, 9)

        tri1_coords=tri1_coords[bool_loc_index,:]
        tri2_coords=tri2_coords[bool_loc_index,:]

        # Concatenate triangle coordinates
        triangles = vcat(tri1_coords, tri2_coords)  # (2 * num_points, 9)

        # Extract supervoxel indices from the 4th column
        sv_idx1 = Int.(combo_intersects[tri_a[2]][bool_loc_index, 4])
        sv_idx2 = Int.(combo_intersects[tri_b[2]][bool_loc_index, 4])
        sv_indices_tri = vcat(sv_idx1, sv_idx2)

        # Append triangle data and sv indices
        append!(all_triangle_vectors, vec(triangles'))
        append!(sv_indices, sv_indices_tri)
    end

    return all_triangle_vectors, sv_indices
end

# Main function
"""
    construct_quads_and_triangulations(edges, intersects, pairs)

Construct valid triangulations of quads from given edges, intersects, and pairs.

# Arguments
- `edges::Vector{Array{Bool}}`: List of edge boolean arrays.
- `intersects::Vector{Array{Float32}}`: List of intersect arrays.
- `pairs::Vector{Tuple{Int, Int}}`: List of index pairs.

# Returns
- `(triangle_vectors, sv_indices)`: Tuple containing triangle data and supervoxel indices.
"""
function construct_quads_and_triangulations(edges, intersects, pairs)
    # Step 1: Flatten the edges and intersects
    edges_flat = Vector{Vector{Bool}}()
    intersects_flat = Vector{Matrix{Float32}}()
    for i in 1:length(edges)
        edge_flat, intersect_flat = flatten_arrays(edges[i], intersects[i])
        push!(edges_flat, edge_flat)
        push!(intersects_flat, intersect_flat)
    end

    # Step 2: Create vector indd and get all valid combinations
    indd = [1, 2, 3, 4, 5, 6]
    combos = generate_combinations(indd, 4)

    # Step 3 and 4: Process combinations
    triangle_vectors, sv_indices = process_combinations(combos, edges_flat, intersects_flat, pairs)

    return triangle_vectors, sv_indices
end

"""
Get edges created when external triangles of tetrahedrons are intersecting the plane of choice
"""
function get_edges_data(edge_2_3,intersectt_2_3,edge_2_4,intersectt_2_4,edge_3_4,intersectt_3_4)
    edge_2_3_flat, intersectt_2_3_flat = flatten_arrays(edge_2_3, intersectt_2_3)
    edge_2_4_flat, intersectt_2_4_flat = flatten_arrays(edge_2_4, intersectt_2_4)
    edge_3_4_flat, intersectt_3_4_flat = flatten_arrays(edge_3_4, intersectt_3_4)

    #get combinations of valid edges
    a=hcat(intersectt_2_3_flat[:,1:4],intersectt_2_4_flat[:,1:4])
    b=hcat(intersectt_2_3_flat[:,1:4],intersectt_3_4_flat[:,1:4])
    c=hcat(intersectt_2_4_flat[:,1:4],intersectt_3_4_flat[:,1:4])
    # a=hcat(intersectt_2_3_flat[:,1:3],intersectt_2_4_flat[:,1:3])
    # b=hcat(intersectt_2_3_flat[:,1:3],intersectt_3_4_flat[:,1:3])
    # c=hcat(intersectt_2_4_flat[:,1:3],intersectt_3_4_flat[:,1:3])

    a=a[edge_2_3_flat.&(edge_2_4_flat),:]
    b=b[edge_2_3_flat.&(edge_3_4_flat),:]
    c=c[edge_2_4_flat.&(edge_3_4_flat),:]
    res= vcat(a,b,c)
    res[:,3].=0.0
    # line_vertices[:,3].=0.0
    res[:,7].=0.0
    res=hcat(res[:,1:3],res[:,5:7])

    return res
end    


"""
    get_data_to_display(tetr_dat, num_tetr, axis, plane_dist)

Calculate the intersection points of tetrahedra edges with a plane and construct valid triangulations.

# Arguments
- `tetr_dat::Array{Float32}`: Tetrahedra data array.
- `num_tetr::Int`: Number of tetrahedra.
- `axis::Int`: Axis along which the plane is defined.
- `plane_dist::Float32`: Distance of the plane from the origin along the specified axis.

# Returns
- `(triang::Vector{Float32}, sv_i::Vector{Int})`: Tuple containing triangle data and supervoxel indices.
"""
function get_data_to_display(tetr_dat, num_tetr::Int, axis::Int, plane_dist,image_size,imm_val_loc)
 
    #yyyyyyyyyyyyyyy tetr_dat (589824, 5, 5, 4) axis 3 plane_dist 30.0 image_size (128, 128, 128)  imm_val_loc (128, 128, 128) 

    @assert axis == 3 "Only axis 3 is supported currently"
    @assert image_size[1] == image_size[2] "first and second dimension of image size should be equal"

    tetr_dat=tetr_dat[:, 1:4, 1:3, 1]
    tetr_s=size(tetr_dat)
    tetr_dat_3d = reshape(tetr_dat, (num_tetr, Int(round(tetr_s[1] / num_tetr)), tetr_s[2], tetr_s[3]))
    tetr_dat_3d = permutedims(tetr_dat_3d, [2, 1, 3, 4])

    #we are calculating where the edges intersect the plane and we are getting the points of intersection
    #important - points not evaluating to true in bool index are not intersecting the plane and calulated values are wrong
    edge_1_2,intersectt_1_2,pair_1_2 =get_bool_index_of_edge(tetr_dat_3d, axis, plane_dist,1,2 )
    edge_1_3,intersectt_1_3,pair_1_3=get_bool_index_of_edge(tetr_dat_3d, axis, plane_dist,1,3 )
    edge_1_4,intersectt_1_4,pair_1_4=get_bool_index_of_edge(tetr_dat_3d, axis, plane_dist,1,4 )
    edge_2_3,intersectt_2_3,pair_2_3=get_bool_index_of_edge(tetr_dat_3d, axis, plane_dist,2,3 )
    edge_2_4,intersectt_2_4,pair_2_4=get_bool_index_of_edge(tetr_dat_3d, axis, plane_dist,2,4 )
    edge_3_4,intersectt_3_4,pair_3_4=get_bool_index_of_edge(tetr_dat_3d, axis, plane_dist,3,4 )



    edges = [edge_1_2, edge_1_3, edge_1_4, edge_2_3, edge_2_4, edge_3_4]
    intersects = [intersectt_1_2, intersectt_1_3, intersectt_1_4, intersectt_2_3, intersectt_2_4, intersectt_3_4]
    pairs = [pair_1_2, pair_1_3, pair_1_4, pair_2_3, pair_2_4, pair_3_4]


    ### get triangles ###
    triangle_vecs,triangle_sv_inds=get_triangle_info(edges,intersects)


    ### get quads ###
    # Assuming edges, intersects, pairs are lists of arrays obtained from your data

    # Call the main function to get triangle data and supervoxel indices
    triangle_vectors, sv_indices = construct_quads_and_triangulations(edges, intersects, pairs)
    triangle_vecs=vec(permutedims(triangle_vecs, [2, 1]))
    triang_vec=vcat(triangle_vecs,triangle_vectors) 
    triang_vec=Float32.(triang_vec)
    
    sv_i=vcat(triangle_sv_inds,sv_indices)

    #get opengl coordinates
    triang_vec=vec(triang_vec)./image_size[1]
    # triang_vec=vec(permutedims(triang_vec, [2, 1]))./maximum(image_size)
    triang_vec=triang_vec.*2
    triang_vec=triang_vec.-1


    #for pure edge display
    edges_vec=get_edges_data(edge_2_3,intersectt_2_3,edge_2_4,intersectt_2_4,edge_3_4,intersectt_3_4)
    edges_vec=Float32.(edges_vec)

    edges_vec=vec(permutedims(edges_vec, [2, 1]))./image_size[1]
    edges_vec=edges_vec.*2
    edges_vec=edges_vec.-1

    line_indices=UInt32.(collect(0:size(edges_vec,1)))


    if(axis==1)
        imm=imm_val_loc[Int(plane_dist),:,:]
    end
    if(axis==2)
        imm=imm_val_loc[:,Int(plane_dist),:]
    end
    if(axis==3)
        imm=imm_val_loc[:,:,Int(plane_dist)]
    end

    return triang_vec,sv_i,edges_vec,line_indices,imm

end
