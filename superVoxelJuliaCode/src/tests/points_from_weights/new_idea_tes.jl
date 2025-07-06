using Meshes
using Random
using LinearAlgebra
using Printf
using StaticArrays # For SVector used by Meshes.Point and MVector
using Unitful: ustrip

# Makie for visualization (choose one backend)
using GLMakie # or CairoMakie, WGLMakie
# using Pkg
#     Pkg.add("GLMakie")
#     Pkg.add("Meshes")
#     Pkg.add("StaticArrays")
#     Pkg.add("Unitful")



GLMakie.activate!()


# For reproducibility (optional)
# Random.seed!(42)

# Helper function for linear interpolation
function lin_ip(p1::NTuple{3,Float64}, p2::NTuple{3,Float64}, w::Float64)
    return (1.0 - w) .* p1 .+ w .* p2
end

# --- Step 1: Define Initial Grid and Dislocated Points (dv) ---
function define_dv_points(grid_dims::Tuple{Int,Int,Int}, grid_spacings::Tuple{Float64,Float64,Float64}, l_factor::Float64)
    dv_points = Array{Tuple{Float64,Float64,Float64}, 3}(undef, grid_dims)
    rx, ry, rz = grid_spacings

    for i in 1:grid_dims[1]
        for j in 1:grid_dims[2]
            for k in 1:grid_dims[3]
                sv_point = ((i-1) * rx, (j-1) * ry, (k-1) * rz) # 0-indexed base for coordinates
                
                # Ensure weights are strictly > 0 and < 1 for interpolation stability if ever w or 1-w is used directly
                # For the current offset logic, rand() is fine. Adding small epsilon for strict positivity if needed.
                w_dx, w_dy, w_dz = rand(3) # rand() is already in [0,1)
                
                # To ensure weights are (0,1) strictly for some algorithms, one might do:
                # w_dx, w_dy, w_dz = rand(3) .* (1.0 - 2e-9) .+ 1e-9 
                # However, for offsets (2w-1), rand() is perfectly fine.
                                
                offset_x = (2 * w_dx - 1) * l_factor * rx
                offset_y = (2 * w_dy - 1) * l_factor * ry
                offset_z = (2 * w_dz - 1) * l_factor * rz
                
                dv_points[i,j,k] = (sv_point[1] + offset_x, sv_point[2] + offset_y, sv_point[3] + offset_z)
            end
        end
    end
    return dv_points
end

# --- Step 2: Define Linear Control Points (lin) ---
function generate_all_lin_neg_points(dv_points::Array{Tuple{Float64,Float64,Float64}, 3}, grid_dims::Tuple{Int,Int,Int})
    default_pt = (NaN, NaN, NaN)
    all_lin_X_neg = fill(default_pt, grid_dims)
    all_lin_Y_neg = fill(default_pt, grid_dims)
    all_lin_Z_neg = fill(default_pt, grid_dims)

    for i in 1:grid_dims[1]
        for j in 1:grid_dims[2]
            for k in 1:grid_dims[3]
                if i > 1
                    w_lxn = rand() # Weight in [0,1)
                    all_lin_X_neg[i,j,k] = lin_ip(dv_points[i-1,j,k], dv_points[i,j,k], w_lxn)
                end
                if j > 1
                    w_lyn = rand()
                    all_lin_Y_neg[i,j,k] = lin_ip(dv_points[i,j-1,k], dv_points[i,j,k], w_lyn)
                end
                if k > 1
                    w_lzn = rand()
                    all_lin_Z_neg[i,j,k] = lin_ip(dv_points[i,j,k-1], dv_points[i,j,k], w_lzn)
                end
            end
        end
    end
    return all_lin_X_neg, all_lin_Y_neg, all_lin_Z_neg
end

function get_lin_points_for_poly_c(
    all_lin_X_neg::Array{NTuple{3, Float64}, 3}, 
    all_lin_Y_neg::Array{NTuple{3, Float64}, 3}, 
    all_lin_Z_neg::Array{NTuple{3, Float64}, 3}, 
    cx::Int, cy::Int, cz::Int
)
    lin_map = Dict{String, NTuple{3, Float64}}()
    lin_map["L_Xn"] = all_lin_X_neg[cx,  cy,  cz]
    lin_map["L_Xp"] = all_lin_X_neg[cx+1,cy,  cz] 
    lin_map["L_Yn"] = all_lin_Y_neg[cx,  cy,  cz]
    lin_map["L_Yp"] = all_lin_Y_neg[cx,  cy+1,cz]
    lin_map["L_Zn"] = all_lin_Z_neg[cx,  cy,  cz]
    lin_map["L_Zp"] = all_lin_Z_neg[cx,  cy,  cz+1]
    return lin_map
end

# --- Step 3: Define Main Oblique Points (mob) ---
function define_mob_point_from_corners(dv_points::Array{NTuple{3, Float64}, 3}, i::Int, j::Int, k::Int)
    c = Dict{String, NTuple{3, Float64}}()
    c["000"] = dv_points[i-1,j-1,k-1]; c["100"] = dv_points[i  ,j-1,k-1]
    c["010"] = dv_points[i-1,j  ,k-1]; c["001"] = dv_points[i-1,j-1,k  ]
    c["110"] = dv_points[i  ,j  ,k-1]; c["101"] = dv_points[i  ,j-1,k  ]
    c["011"] = dv_points[i-1,j  ,k  ]; c["111"] = dv_points[i  ,j  ,k  ]
    
    w_m = rand(7) # 7 weights in [0,1)
    p_x1 = lin_ip(c["011"], c["111"], w_m[1])
    p_x2 = lin_ip(c["001"], c["101"], w_m[2])
    p_x3 = lin_ip(c["010"], c["110"], w_m[3])
    p_x4 = lin_ip(c["000"], c["100"], w_m[4])
    p_y1 = lin_ip(p_x1, p_x2, w_m[5])
    p_y2 = lin_ip(p_x3, p_x4, w_m[6])
    mob_pt = lin_ip(p_y1, p_y2, w_m[7])
    return mob_pt
end

function generate_all_mob_points(dv_points::Array{NTuple{3, Float64}, 3}, grid_dims::Tuple{Int,Int,Int})
    default_pt = (NaN, NaN, NaN)
    # mob_pts grid is 1-indexed up to grid_dims
    all_mob_pts = fill(default_pt, grid_dims) 
    # mob_pts[i,j,k] calculation needs dv_pts[i-1,j-1,k-1], so loops for i,j,k must start from 2
    for i in 2:grid_dims[1] 
        for j in 2:grid_dims[2]
            for k in 2:grid_dims[3]
                all_mob_pts[i,j,k] = define_mob_point_from_corners(dv_points, i, j, k)
            end
        end
    end
    return all_mob_pts
end

struct MobPointData
    coord::NTuple{3, Float64}
    ijk::NTuple{3, Int} # Global (i,j,k) indices of this mob point in all_mob_pts grid
end

function get_relevant_mob_points_for_poly_c(all_mob_pts::Array{NTuple{3, Float64}, 3}, cx::Int, cy::Int, cz::Int)
    mob_data_map = Dict{String, MobPointData}()
    # cx,cy,cz are the 1-based indices of the dv_C point.
    # mob_ijk indices are relative to the structure of the dv_cell grid.
    # m000 is mob_[cx,cy,cz]
    mob_data_map["m000"] = MobPointData(all_mob_pts[cx  ,cy  ,cz  ], (cx  ,cy  ,cz  ))
    mob_data_map["m100"] = MobPointData(all_mob_pts[cx+1,cy  ,cz  ], (cx+1,cy  ,cz  ))
    mob_data_map["m010"] = MobPointData(all_mob_pts[cx  ,cy+1,cz  ], (cx  ,cy+1,cz  ))
    mob_data_map["m001"] = MobPointData(all_mob_pts[cx  ,cy  ,cz+1], (cx  ,cy  ,cz+1))
    mob_data_map["m110"] = MobPointData(all_mob_pts[cx+1,cy+1,cz  ], (cx+1,cy+1,cz  ))
    mob_data_map["m101"] = MobPointData(all_mob_pts[cx+1,cy  ,cz+1], (cx+1,cy  ,cz+1))
    mob_data_map["m011"] = MobPointData(all_mob_pts[cx  ,cy+1,cz+1], (cx  ,cy+1,cz+1))
    mob_data_map["m111"] = MobPointData(all_mob_pts[cx+1,cy+1,cz+1], (cx+1,cy+1,cz+1))
    return mob_data_map
end
    
# --- Step 4: Define Intermediate Points (int) Between mob Points ---
function generate_global_u_weights_for_int_edges(grid_dims::Tuple{Int,Int,Int})
    global_u_map = Dict{NTuple{2, NTuple{3,Int}}, NTuple{3,Float64}}()
    
    # mob points are calculated for indices 2:grid_dims[d] in each dimension d
    # Iterate through all possible ijk for the first mob point (mA) of an edge
    for i in 2:grid_dims[1]
        for j in 2:grid_dims[2]
            for k in 2:grid_dims[3]
                ijk_A = (i,j,k)
                
                # Potential neighbors (mB) in +X, +Y, +Z directions
                # Check if neighbor is within the bounds where mob points are defined (2 to grid_dims[d])
                potential_neighbors_ijk = []
                if i + 1 <= grid_dims[1]; push!(potential_neighbors_ijk, (i+1, j, k)); end
                if j + 1 <= grid_dims[2]; push!(potential_neighbors_ijk, (i, j+1, k)); end
                if k + 1 <= grid_dims[3]; push!(potential_neighbors_ijk, (i, j, k+1)); end
                
                for ijk_B in potential_neighbors_ijk
                    # Canonical edge key: sort by ijk tuples
                    edge_key_parts = ijk_A < ijk_B ? (ijk_A, ijk_B) : (ijk_B, ijk_A)
                    
                    if !haskey(global_u_map, edge_key_parts)
                        # Generate weights u1,u2,u3 in (0,1) to avoid division by zero if S_AB=0
                        # Using rand() which is [0,1), adding epsilon for strict positivity.
                        global_u_map[edge_key_parts] = Tuple(rand(3) .* (1.0 - 2e-9) .+ 1e-9)
                    end
                end
            end
        end
    end
    return global_u_map
end

function define_int_points_on_edges_for_poly_c(
    mob_data_map::Dict{String, MobPointData}, 
    global_u_weights_map::Dict{NTuple{2, NTuple{3,Int}}, NTuple{3,Float64}}
)
    int_points_on_edges = Dict{Tuple{String,String}, Tuple{NTuple{3,Float64}, NTuple{3,Float64}}}()
    
    unique_edges_local_keys = [
        ("m000", "m100"), ("m010", "m110"), ("m001", "m101"), ("m011", "m111"), 
        ("m000", "m010"), ("m100", "m110"), ("m001", "m011"), ("m101", "m111"), 
        ("m000", "m001"), ("m100", "m101"), ("m010", "m011"), ("m110", "m111")  
    ]

    for (local_key_A, local_key_B) in unique_edges_local_keys
        mob_A_data = mob_data_map[local_key_A]
        mob_B_data = mob_data_map[local_key_B]
        
        mob_A_coord = mob_A_data.coord
        mob_B_coord = mob_B_data.coord
        # These ijk are global indices for mob points
        ijk_A = mob_A_data.ijk 
        ijk_B = mob_B_data.ijk

        # Create canonical key for global_u_weights_map
        u_edge_key_parts = ijk_A < ijk_B ? (ijk_A, ijk_B) : (ijk_B, ijk_A)
        
        if !haskey(global_u_weights_map, u_edge_key_parts)
            # This should not happen if generate_global_u_weights_for_int_edges covers all needed edges
            # based on the range of cx,cy,cz and thus the range of ijk_A, ijk_B.
            error("Missing u_weights for edge between MOB $(ijk_A) and MOB $(ijk_B) (local keys: $(local_key_A)-$(local_key_B))")
        end
        u1, u2, u3 = global_u_weights_map[u_edge_key_parts]
        
        S_AB = u1 + u2 + u3 # S_AB will be > 0 due to u weights in (0,1)
        w_int1_AB = u1 / S_AB
        w_int2_AB = (u1 + u2) / S_AB
        
        int1_AB = lin_ip(mob_A_coord, mob_B_coord, w_int1_AB)
        int2_AB = lin_ip(mob_A_coord, mob_B_coord, w_int2_AB)
        int_points_on_edges[(local_key_A, local_key_B)] = (int1_AB, int2_AB)
        
        # For the reverse edge (B,A), intermediate points are relative to B.
        # int1_BA is closer to B, int2_BA is further from B (closer to A)
        # int1_BA = (1-w_int2_AB) * B + w_int2_AB * A = lin_ip(mob_B_coord, mob_A_coord, 1.0 - w_int2_AB)
        # int2_BA = (1-w_int1_AB) * B + w_int1_AB * A = lin_ip(mob_B_coord, mob_A_coord, 1.0 - w_int1_AB)
        int1_BA = lin_ip(mob_B_coord, mob_A_coord, u3 / S_AB) # Equivalent to 1.0 - w_int2_AB
        int2_BA = lin_ip(mob_B_coord, mob_A_coord, (u2+u3) / S_AB) # Equivalent to 1.0 - w_int1_AB
        int_points_on_edges[(local_key_B, local_key_A)] = (int1_BA, int2_BA)
    end
    return int_points_on_edges
end

# --- Step 5: Define the Star Domain Polyhedron Poly_C ---
function define_poly_c_base_triangles(
    _dv_c_coord::NTuple{3, Float64}, # dv_C is apex, not part of base triangle
    lin_map::Dict{String, NTuple{3, Float64}}, 
    mob_data_map::Dict{String, MobPointData}, 
    int_map::Dict{Tuple{String,String}, Tuple{NTuple{3,Float64}, NTuple{3,Float64}}}
)
    base_triangles_list = Vector{NTuple{3, NTuple{3,Float64}}}()
    
    face_definitions = Dict(
        "-X" => (["m000", "m001", "m011", "m010"], "L_Xn"),
        "+X" => (["m100", "m110", "m111", "m101"], "L_Xp"),
        "-Y" => (["m000", "m100", "m101", "m001"], "L_Yn"),
        "+Y" => (["m010", "m011", "m111", "m110"], "L_Yp"),
        "-Z" => (["m000", "m010", "m110", "m100"], "L_Zn"),
        "+Z" => (["m001", "m101", "m111", "m011"], "L_Zp")
    )

    for (_face_name, (mob_local_key_cycle, lin_key)) in face_definitions
        L_coord = lin_map[lin_key]
        if any(isnan, L_coord)
            @warn "NaN coordinate found for lin point $(lin_key) on face $(_face_name). Skipping face."
            continue
        end

        num_mob_keys = length(mob_local_key_cycle)
        for i in 1:num_mob_keys
            local_k_P = mob_local_key_cycle[i]
            local_k_Q = mob_local_key_cycle[mod1(i + 1, num_mob_keys)]
            
            m_P_data = mob_data_map[local_k_P]
            m_Q_data = mob_data_map[local_k_Q]

            if any(isnan, m_P_data.coord) || any(isnan, m_Q_data.coord)
                @warn "NaN coordinate found for MOB point $(local_k_P) or $(local_k_Q). Skipping segment."
                continue
            end
            m_P_c = m_P_data.coord
            m_Q_c = m_Q_data.coord
            
            # Check if int_map has the edge; if not, mob points might be NaN or from uninitialized region
            if !haskey(int_map, (local_k_P, local_k_Q))
                @warn "Intermediate points not found for edge $(local_k_P) - $(local_k_Q). Skipping segment."
                # This might happen if one of the mob points has NaN ijk or coord, affecting define_int_points_on_edges_for_poly_c
                continue
            end
            I1_c, I2_c = int_map[(local_k_P, local_k_Q)]
            if any(isnan, I1_c) || any(isnan, I2_c)
                @warn "NaN coordinate found for INT point on edge $(local_k_P)-$(local_k_Q). Skipping segment."
                continue
            end
            
            push!(base_triangles_list, (m_P_c, I1_c, L_coord) )
            push!(base_triangles_list, (I1_c, I2_c, L_coord) )
            push!(base_triangles_list, (I2_c, m_Q_c, L_coord) )
        end
    end
    return base_triangles_list
end

# --- Main execution for visualization ---
function generate_and_visualize_poly_c_bases(dv_c_indices::NTuple{3,Int}, 
                                     dv_points_grid, 
                                     all_lin_X_neg, all_lin_Y_neg, all_lin_Z_neg,
                                     all_mob_pts_grid, global_u_map_for_int)
    cx, cy, cz = dv_c_indices # 1-based
    println("\n--- Generating Poly_C base triangles for dv_[$(cx),$(cy),$(cz)] ---")

    dv_c_actual_coord = dv_points_grid[cx,cy,cz]
    
    lin_points_map = get_lin_points_for_poly_c(
        all_lin_X_neg, all_lin_Y_neg, all_lin_Z_neg, cx, cy, cz)
    
    mob_data_map = get_relevant_mob_points_for_poly_c(all_mob_pts_grid, cx, cy, cz)
    
    # Check for NaN coordinates in mob_data_map as this will break int_point calculation
    for (key, mob_data) in mob_data_map
        if any(isnan, mob_data.coord)
            @warn "NaN MOB point $(key) [$(mob_data.ijk)] for dv_C $(dv_c_indices). This Poly_C might be incomplete or invalid."
        end
    end

    int_points_map = define_int_points_on_edges_for_poly_c(mob_data_map, global_u_map_for_int)
    
    curr_base_triangles_coords = define_poly_c_base_triangles(
        dv_c_actual_coord, lin_points_map, mob_data_map, int_points_map)
    
    println("Generated $(length(curr_base_triangles_coords)) base triangles for dv_[$(cx),$(cy),$(cz)]")

    # Visualize all triangles for this polyhedron at once
    trs = Meshes.Triangle[]
    for i in 1:length(curr_base_triangles_coords)
        try
            p1 = Meshes.Point(curr_base_triangles_coords[i][1]...)
            p2 = Meshes.Point(curr_base_triangles_coords[i][2]...)
            p3 = Meshes.Point(curr_base_triangles_coords[i][3]...)
            viz(Meshes.Triangle(p1, p2, p3))
            push!(trs, Meshes.Triangle(p1, p2, p3))
        catch
            println("Error $i")
        end
    end
    

    return trs # Return coordinate list
end


grid_dims_config = (5,5,5) 
grid_spacings_config = (1.0, 1.0, 1.0)
l_factor_config = 0.4 

println("--- Global Point Generation ---")
println("Step 1: Defining dv points...")
dv_points_grid = define_dv_points(grid_dims_config, grid_spacings_config, l_factor_config)

println("Step 2: Defining all lin_neg points...")
all_lin_X_neg, all_lin_Y_neg, all_lin_Z_neg = generate_all_lin_neg_points(dv_points_grid, grid_dims_config)

println("Step 3: Defining all mob points...")
all_mob_pts_grid = generate_all_mob_points(dv_points_grid, grid_dims_config)

println("Step 4: Defining global u_weights for int_point edges...")
# Pass grid_dims to ensure u_weights are generated for all mob points that can exist.
# MOB points are defined from index 2 up to grid_dims.
global_u_map = generate_global_u_weights_for_int_edges(grid_dims_config)
println("Generated u_weights for $(length(global_u_map)) unique mob edges.")

center_cx, center_cy, center_cz = 3,3,3 

# Valid cx,cy,cz for Poly_C: min index 2, max index grid_dim-1
# For 5x5x5, valid range is [2, 4]. Center (3,3,3) is valid.
# Neighbors (2,3,3), (4,3,3) etc. are also valid.
indices_to_process = [
    (center_cx, center_cy, center_cz),      
    (center_cx - 1, center_cy, center_cz), 
    (center_cx + 1, center_cy, center_cz),  
    (center_cx, center_cy - 1, center_cz),  
    (center_cx, center_cy + 1, center_cz),  
    (center_cx, center_cy, center_cz - 1), 
    (center_cx, center_cy, center_cz + 1)   
]
# indices_to_process = [(center_cx, center_cy, center_cz)] # For single Poly_C

# Create a single figure and axis for all polyhedra
fig = Figure(size = (1200, 800))
ax = Axis3(fig[1,1], aspect = :data, title = "Poly_C Visualizations")

trs=generate_and_visualize_poly_c_bases(
    indices_to_process[4], dv_points_grid,
    all_lin_X_neg, all_lin_Y_neg, all_lin_Z_neg,
    all_mob_pts_grid, global_u_map
)
viz(trs, color=1:length(trs))


