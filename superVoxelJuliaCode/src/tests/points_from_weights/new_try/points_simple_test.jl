import Meshes
using Meshes
using StaticArrays # For SVector in is_degenerate_triangle
using LinearAlgebra # For vector operations if needed, Point arithmetic is fine
using Random
using Printf
using Unitful


# --- Robust Helper Functions for Watertightness ---

"""
Deduplicate points within a given tolerance.
Returns a tuple: (unique_points, point_map)
unique_points: Vector{Point}
point_map: Dict{Point, Int} mapping original points to unique indices
"""
function deduplicate_points(points::Vector{<:Meshes.Point}; tol=1e-8)
    unique_points = Meshes.Point[]
    point_map = Dict{Meshes.Point, Int}() # Maps original point object to its index in unique_points
    
    # Temporary map from a representative point (first encountered in a cluster) to its index
    # This helps in mapping subsequent points that are close to an already processed unique point.
    representative_to_index = Dict{Meshes.Point, Int}()
    unique_point_representatives = Meshes.Point[] # Stores the actual unique Point objects

    for pt_original_object in points
        found_match = false
        # Check against already established unique representatives
        for (idx, unique_repr_pt) in enumerate(unique_point_representatives)
            c1_coords = coords(pt_original_object)
            c2_coords = coords(unique_repr_pt)
            
            # Use unitful-aware tolerance for each coordinate
            tol_x = (c1_coords.x isa Unitful.Quantity) ? tol * Unitful.unit(c1_coords.x) : tol
            tol_y = (c1_coords.y isa Unitful.Quantity) ? tol * Unitful.unit(c1_coords.y) : tol
            tol_z = (c1_coords.z isa Unitful.Quantity) ? tol * Unitful.unit(c1_coords.z) : tol

            if isapprox(c1_coords.x, c2_coords.x; atol=tol_x) &&
               isapprox(c1_coords.y, c2_coords.y; atol=tol_y) &&
               isapprox(c1_coords.z, c2_coords.z; atol=tol_z)
                
                point_map[pt_original_object] = idx # Map original object to index of the representative
                found_match = true
                break
            end
        end

        if !found_match
            # This point establishes a new unique representative
            push!(unique_point_representatives, pt_original_object)
            new_idx = length(unique_point_representatives)
            push!(unique_points, pt_original_object) # Add to the final list of unique points to be returned
            point_map[pt_original_object] = new_idx # Map original object to its new unique index
            representative_to_index[pt_original_object] = new_idx
        end
    end
    return unique_points, point_map
end


"""
Check if a triangle is degenerate (area below tolerance).
"""
function is_degenerate_triangle(tri::NTuple{3, Meshes.Point}; tol=1e-12)
    v1, v2, v3 = tri
    
    # Check for coincident points first, as this is a common cause of degeneracy
    # and can be faster than full area calculation if points are identical objects.
    if v1 === v2 || v1 === v3 || v2 === v3
        # Using direct comparison for Point objects if they are literally the same.
        # If they are different objects but same coordinates, deduplicate_points should handle it.
        # However, if deduplicate_points failed or wasn't used, this is a quick check.
        # A more robust check would be `isapprox(coords(v1), coords(v2))`, etc.
        # but given `deduplicate_points`, this might be redundant or point to issues there.
        # For now, let's assume points passed here are already deduplicated representatives
        # or that area check will handle numerical coincidence.
    end

    c1 = coords(v1)
    c2 = coords(v2)
    c3 = coords(v3)
    
    get_xyz(c) = (
        hasproperty(c, :x) ? (try Unitful.ustrip(c.x) catch e; Float64(c.x) end) : 0.0,
        hasproperty(c, :y) ? (try Unitful.ustrip(c.y) catch e; Float64(c.y) end) : 0.0,
        hasproperty(c, :z) ? (try Unitful.ustrip(c.z) catch e; Float64(c.z) end) : 0.0
    )

    p1_xyz = get_xyz(c1)
    p2_xyz = get_xyz(c2)
    p3_xyz = get_xyz(c3)

    # Check for numerical coincidence again after ustrip, more robustly
    if isapprox(p1_xyz, p2_xyz; atol=tol) || isapprox(p1_xyz, p3_xyz; atol=tol) || isapprox(p2_xyz, p3_xyz; atol=tol)
        return true
    end

    # Create SVector for cross product
    sv1 = SVector{3,Float64}(p1_xyz)
    sv2 = SVector{3,Float64}(p2_xyz)
    sv3 = SVector{3,Float64}(p3_xyz)

    edge1 = sv2 - sv1
    edge2 = sv3 - sv1
    
    area_vec = cross(edge1, edge2)
    area = norm(area_vec) / 2.0

    tol_area = if area isa Unitful.Quantity
        tol * Unitful.unit(area) # This case is unlikely if coords are numbers
    else
        tol # Standard float tolerance for area
    end

    return area < tol_area
end

# --- Helper Functions ---

"""
Generates a random weight constrained to be away from 0 and 1.
Default range is [1e-4, 1.0 - 1e-4].
"""
function get_constrained_random_weight(min_w::Float64 = 1e-4)
    # Ensures weight is in [min_w, 1.0 - min_w]
    # rand() gives [0,1)
    return min_w + (1.0 - 2.0 * min_w) * rand()
end


"""
Linear interpolation between two points.
p1: Start point
p2: End point
w: Weight (0.0 to 1.0)
Returns p1 + w * (p2 - p1)
"""
function lin_ip(p1::Meshes.Point, p2::Meshes.Point, w::Float64)
    return p1 + (w * (p2 - p1))
end

"""
Generates the initial displaced grid points (sv_points).
Displacement factor is reduced to < 0.5 to prevent grid tangling.
"""
function generate_sv_points(N::Int, rx::Float64, ry::Float64, rz::Float64)
    sv_points = Array{Meshes.Point, 3}(undef, N, N, N)
    
    # Displacement factor: must be < 0.5 to prevent cell inversion.
    # Original was 0.75, which is too large.
    displacement_coeff = 0.49 # Max displacement is displacement_coeff * grid_spacing

    Threads.@threads for k_idx = 1:N
        for j_idx = 1:N
            for i_idx = 1:N
                initial_x = (i_idx - 1) * rx
                initial_y = (j_idx - 1) * ry
                initial_z = (k_idx - 1) * rz

                # Random displacement in [-displacement_coeff*r, displacement_coeff*r]
                disp_x = (rand() - 0.5) * 2.0 * displacement_coeff * rx
                disp_y = (rand() - 0.5) * 2.0 * displacement_coeff * ry
                disp_z = (rand() - 0.5) * 2.0 * displacement_coeff * rz
                
                sv_points[i_idx, j_idx, k_idx] = Meshes.Point(initial_x + disp_x,
                                                       initial_y + disp_y,
                                                       initial_z + disp_z)
            end
        end
    end
    return sv_points
end

"""
Generates linear control points (lin_x, lin_y, lin_z).
These points lie on segments between sv_points.
Uses constrained random weights.
"""
function generate_lin_points(N::Int, sv_points::Array{Meshes.Point, 3})
    lin_x_points = Array{Meshes.Point, 3}(undef, N, N, N) 
    lin_y_points = Array{Meshes.Point, 3}(undef, N, N, N) 
    lin_z_points = Array{Meshes.Point, 3}(undef, N, N, N) 

    Threads.@threads for k_idx = 1:N
        for j_idx = 1:N
            for i_idx = 2:N
                p1 = sv_points[i_idx-1, j_idx, k_idx]
                p2 = sv_points[i_idx,   j_idx, k_idx]
                lin_x_points[i_idx, j_idx, k_idx] = lin_ip(p1, p2, get_constrained_random_weight())
            end
        end
    end

    Threads.@threads for k_idx = 1:N
        for j_idx = 2:N
            for i_idx = 1:N
                p1 = sv_points[i_idx, j_idx-1, k_idx]
                p2 = sv_points[i_idx, j_idx,   k_idx]
                lin_y_points[i_idx, j_idx, k_idx] = lin_ip(p1, p2, get_constrained_random_weight())
            end
        end
    end

    Threads.@threads for k_idx = 2:N
        for j_idx = 1:N
            for i_idx = 1:N
                p1 = sv_points[i_idx, j_idx, k_idx-1]
                p2 = sv_points[i_idx, j_idx, k_idx  ]
                lin_z_points[i_idx, j_idx, k_idx] = lin_ip(p1, p2, get_constrained_random_weight())
            end
        end
    end
    return lin_x_points, lin_y_points, lin_z_points
end

"""
Generates main oblique points (mob_points) using trilinear interpolation.
Uses constrained random weights.
"""
function generate_mob_points(N::Int, sv_points::Array{Meshes.Point, 3})
    mob_points = Array{Meshes.Point, 3}(undef, N, N, N) 

    Threads.@threads for k_idx = 2:N
        for j_idx = 2:N
            for i_idx = 2:N
                # Weights for interpolation using constrained random values
                w1 = get_constrained_random_weight()
                w2 = get_constrained_random_weight()
                w3 = get_constrained_random_weight()
                w4 = get_constrained_random_weight()
                w5 = get_constrained_random_weight()
                w6 = get_constrained_random_weight()
                w7 = get_constrained_random_weight()

                sv_ijk   = sv_points[i_idx,   j_idx,   k_idx  ]
                sv_im1jk = sv_points[i_idx-1, j_idx,   k_idx  ]
                sv_ijm1k = sv_points[i_idx,   j_idx-1, k_idx  ]
                sv_ijkm1 = sv_points[i_idx,   j_idx,   k_idx-1]
                sv_im1jm1k = sv_points[i_idx-1, j_idx-1, k_idx  ]
                sv_ijm1km1 = sv_points[i_idx,   j_idx-1, k_idx-1]
                sv_im1jkm1 = sv_points[i_idx-1, j_idx,   k_idx-1]
                sv_im1jm1km1 = sv_points[i_idx-1, j_idx-1, k_idx-1]

                x1 = lin_ip(sv_im1jk, sv_ijk, w1)
                x2 = lin_ip(sv_ijm1k, sv_im1jm1k, w2) # Note: Original had sv_ijm1k, sv_im1jm1k. This order matters.
                                                      # Assuming original logic intended interpolation along "edges" or "diagonals"
                                                      # of the source sv_points forming the mob_point's parent cell.
                                                      # For trilinear, it's usually along axes.
                                                      # Let's maintain original structure but with constrained weights.
                x3 = lin_ip(sv_ijkm1, sv_im1jkm1, w3)
                x4 = lin_ip(sv_ijm1km1, sv_im1jm1km1, w4)

                y1 = lin_ip(x1, x2, w5)
                y2 = lin_ip(x3, x4, w6)

                mob_points[i_idx, j_idx, k_idx] = lin_ip(y1, y2, w7)
            end
        end
    end
    return mob_points
end


"""
Defines the 24 base triangles for the polyhedron surface of cell (ic, jc, kc).
"""
function get_polyhedron_base_triangles(ic::Int, jc::Int, kc::Int,
                                       mob_points_all::Array{Meshes.Point, 3},
                                       lin_x_points_all::Array{Meshes.Point, 3},
                                       lin_y_points_all::Array{Meshes.Point, 3},
                                       lin_z_points_all::Array{Meshes.Point, 3})

    # Mob points for the current cell corners (polyhedron vertices)
    m000 = mob_points_all[ic,     jc,     kc    ]
    m100 = mob_points_all[ic+1,   jc,     kc    ]
    m010 = mob_points_all[ic,     jc+1,   kc    ]
    m001 = mob_points_all[ic,     jc,     kc+1  ]
    m110 = mob_points_all[ic+1,   jc+1,   kc    ]
    m101 = mob_points_all[ic+1,   jc,     kc+1  ]
    m011 = mob_points_all[ic,     jc+1,   kc+1  ]
    m111 = mob_points_all[ic+1,   jc+1,   kc+1  ]

    # Lin points for face centers (pyramid apices for faces)
    # These lin_points are indexed to be the "face centers" of the polyhedron for cell (ic,jc,kc)
    lx_neg = lin_x_points_all[ic,     jc,     kc    ] # Face at x_min of cell (ic,jc,kc)
    lx_pos = lin_x_points_all[ic+1,   jc,     kc    ] # Face at x_max of cell (ic,jc,kc)
    ly_neg = lin_y_points_all[ic,     jc,     kc    ] # Face at y_min
    ly_pos = lin_y_points_all[ic,     jc+1,   kc    ] # Face at y_max
    lz_neg = lin_z_points_all[ic,     jc,     kc    ] # Face at z_min
    lz_pos = lin_z_points_all[ic,     jc,     kc+1  ] # Face at z_max

    triangles = Vector{NTuple{3, Meshes.Point}}()
    # Face -Z (bottom face of the polyhedron cell (ic,jc,kc))
    # Apex: lz_neg (center of this bottom face)
    # Base quad: m000, m100, m110, m010 (mob points at z-level kc)
    for tri_pts in [ (lz_neg, m000, m100), (lz_neg, m100, m110), (lz_neg, m110, m010), (lz_neg, m010, m000) ]
        if !is_degenerate_triangle(tri_pts)
            push!(triangles, tri_pts)
        end
    end
    # Face +Z (top face)
    # Apex: lz_pos (center of this top face)
    # Base quad: m001, m101, m111, m011 (mob points at z-level kc+1, but used for cell ic,jc,kc)
    # The base order should be consistent for outward normals relative to apex.
    # If (P,A,B,C,D) makes outward normals for -Z, then for +Z, with P "above", base order should be reversed if view is same.
    # Original code uses same base order: (m001,m101,m111,m011) then (P,A,B), (P,B,C)...
    # This is standard for pyramid construction; HalfEdgeTopology should handle orientation if geometry is sound.
    for tri_pts in [ (lz_pos, m001, m011, m111), (lz_pos, m111, m101, m001) ] # Trying reversed base order for second half.
                                                                    # A common convention: (Apex, v1, v2), (Apex, v2, v3), (Apex, v3, v4), (Apex, v4, v1)
                                                                    # Original: (P,A,B), (P,B,C), (P,C,D), (P,D,A)
                                                                    # Let's stick to original connection style first.
    for tri_pts in [ (lz_pos, m001, m101), (lz_pos, m101, m111), (lz_pos, m111, m011), (lz_pos, m011, m001) ]
        if !is_degenerate_triangle(tri_pts)
            push!(triangles, tri_pts)
        end
    end
    # Face -Y
    # Apex: ly_neg. Base quad: m000, m001, m101, m100
    for tri_pts in [ (ly_neg, m000, m001), (ly_neg, m001, m101), (ly_neg, m101, m100), (ly_neg, m100, m000) ]
        if !is_degenerate_triangle(tri_pts)
            push!(triangles, tri_pts)
        end
    end
    # Face +Y
    # Apex: ly_pos. Base quad: m010, m110, m111, m011
    for tri_pts in [ (ly_pos, m010, m011, m111), (ly_pos, m111, m110, m010) ] # Trying reversed for half
    for tri_pts in [ (ly_pos, m010, m110), (ly_pos, m110, m111), (ly_pos, m111, m011), (ly_pos, m011, m010) ]
        if !is_degenerate_triangle(tri_pts)
            push!(triangles, tri_pts)
        end
    end
    # Face -X
    # Apex: lx_neg. Base quad: m000, m001, m011, m010 (Order implies: (000)-(001)-(011)-(010) )
    # Should be: m000, m010, m011, m001. (counter-clockwise when looking from -X)
    for tri_pts in [ (lx_neg, m000, m010), (lx_neg, m010, m011), (lx_neg, m011, m001), (lx_neg, m001, m000) ]
        if !is_degenerate_triangle(tri_pts)
            push!(triangles, tri_pts)
        end
    end
    # Face +X
    # Apex: lx_pos. Base quad: m100, m101, m111, m110
    for tri_pts in [ (lx_pos, m100, m110, m111), (lx_pos, m111, m101, m100) ] # Trying reversed for half
    for tri_pts in [ (lx_pos, m100, m101), (lx_pos, m101, m111), (lx_pos, m111, m110), (lx_pos, m110, m100) ]
        if !is_degenerate_triangle(tri_pts)
            push!(triangles, tri_pts)
        end
    end
    return triangles
end


"""
Performs watertightness and line crossing checks for a single polyhedron.
"""
function check_single_polyhedron(
    ic::Int, jc::Int, kc::Int,
    sv_center_pt::Meshes.Point,
    base_triangles_points::Vector{NTuple{3, Meshes.Point}}
)
    is_watertight = false

    if isempty(base_triangles_points)
        # No triangles, definitely not watertight and no lines to cross
        return false, false
    end

    all_pts_for_dedup = [pt for tri_tuple in base_triangles_points for pt in tri_tuple]
    unique_poly_vertices, point_instance_to_unique_idx_map = deduplicate_points(all_pts_for_dedup)

    connectivities = Vector{NTuple{3, Int}}()
    valid_triangles_for_mesh = NTuple{3, Meshes.Point}[] # Store triangles that map to valid connectivities

    for tri_pts_tuple in base_triangles_points
        # Map points of this triangle to their unique indices
        # It's possible that after deduplication, points in a triangle become non-distinct
        # e.g., tri_pts_tuple[1] and tri_pts_tuple[2] map to the same unique index.
        idx1 = point_instance_to_unique_idx_map[tri_pts_tuple[1]]
        idx2 = point_instance_to_unique_idx_map[tri_pts_tuple[2]]
        idx3 = point_instance_to_unique_idx_map[tri_pts_tuple[3]]

        # Only add triangle if its vertices are distinct after deduplication
        if idx1 != idx2 && idx1 != idx3 && idx2 != idx3
            push!(connectivities, (idx1, idx2, idx3))
            push!(valid_triangles_for_mesh, tri_pts_tuple) # Keep original points for line crossing
        else
            # This triangle became degenerate after point deduplication, skip it for mesh construction
            # but it was already checked by is_degenerate_triangle using area. This is a stricter check.
        end
    end
    
    base_triangles_points = valid_triangles_for_mesh # Update to only use triangles that form valid connectivities

    if length(unique_poly_vertices) >= 3 && !isempty(connectivities)
        elements = [connect(conn, Triangle) for conn in connectivities]
        try
            # Ensure topology is valid before creating mesh
            # This might require Meshes.jl v0.20+ for direct SimpleMesh(points, topology_vector)
            # For older versions or different API:
            mesh_to_check = SimpleMesh(unique_poly_vertices, elements) # Rebuild with unique vertices
            
            # Check if the mesh is manifold first (optional, but good diagnostic)
            # is_manifold_mesh = ismanifold(mesh_to_check) # Requires Meshes.jl utility

            boundary_of_mesh = boundary(mesh_to_check)
            if nelements(boundary_of_mesh) == 0
                is_watertight = true
            end
        catch e
            # @warn "Watertightness check failed for cell ($ic, $jc, $kc) during mesh construction or boundary check: $e"
            is_watertight = false
        end
    else
        is_watertight = false # Not enough unique points or valid connectivities
    end

    # --- 2. Line Crossing Check ---
    # This part uses the triangles *before* they might be further pruned by connectivity check,
    # as is_degenerate_triangle was the primary filter there.
    # If base_triangles_points was updated, this uses the pruned list.
    has_line_crossing = false
    if isempty(base_triangles_points) # Check again if it became empty after connectivity pruning
        return is_watertight, false # No triangles, no crossings
    end

    num_base_triangles = length(base_triangles_points)
    # Create Meshes.Triangle elements for intersection test
    base_triangles_mesh_elements = [Triangle(pts[1], pts[2], pts[3]) for pts in base_triangles_points]

    for i = 1:num_base_triangles
        tri_i_pts = base_triangles_points[i]
        c1, c2, c3 = coords(tri_i_pts[1]), coords(tri_i_pts[2]), coords(tri_i_pts[3])
        
        bary_x = (c1.x + c2.x + c3.x) / 3.0
        bary_y = (c1.y + c2.y + c3.y) / 3.0
        bary_z = (c1.z + c2.z + c3.z) / 3.0
        barycenter_i = Meshes.Point(bary_x, bary_y, bary_z)

        segment_to_check = Segment(sv_center_pt, barycenter_i)

        for j = 1:num_base_triangles
            if i == j
                continue
            end

            tri_j_mesh_el = base_triangles_mesh_elements[j] # This is a Meshes.Triangle
            
            # intersection can return NoIntersection, Point, Segment, etc.
            # Assuming NoIntersection is a type or `nothing` is returned.
            # Let's use `intersecttype` for clarity if available, or check return type.
            
            intersection_result = intersection(segment_to_check, tri_j_mesh_el)

            if !(intersection_result isa Meshes.NoIntersection) && intersection_result !== nothing
                # An intersection occurred. Check if it's a Point and strictly between segment endpoints.
                intersect_points_to_check = Meshes.Point[]
                if intersection_result isa Meshes.Point
                    push!(intersect_points_to_check, intersection_result)
                elseif intersection_result isa PointSet # Some intersections might return a PointSet
                    for pt_in_set in intersection_result
                        if pt_in_set isa Meshes.Point
                            push!(intersect_points_to_check, pt_in_set)
                        end
                    end
                # Potentially other intersection types like Segment if segment lies in triangle plane.
                # For simplicity, focusing on Point intersections.
                end

                for intersect_pt in intersect_points_to_check
                    # Check if intersection point is strictly between segment endpoints
                    # using a small tolerance for comparison
                    d_p_intersect = norm(coords(intersect_pt) - coords(segment_to_check.p))
                    d_q_intersect = norm(coords(intersect_pt) - coords(segment_to_check.q))
                    d_p_q = norm(coords(segment_to_check.p) - coords(segment_to_check.q))
                    
                    # If intersect_pt is on the segment, d_p_intersect + d_q_intersect approx d_p_q
                    # Check if strictly between: d_p_intersect > tol and d_q_intersect > tol
                    # A simpler check: not isapprox to endpoints
                    # (Original check was !isapprox(pt, p) && !isapprox(pt, q))
                    # This can be sensitive if intersection is very close to an endpoint but not exactly it.
                    
                    # More robust check for "strictly between":
                    # Parameter t for point on segment: P(t) = p + t*(q-p)
                    # If intersect_pt = p + t*(q-p), then t should be in (epsilon, 1-epsilon)
                    vec_pq = segment_to_check.q - segment_to_check.p
                    vec_p_intersect = intersect_pt - segment_to_check.p
                    
                    len_pq_sq = dot(vec_pq, vec_pq)
                    if len_pq_sq > 1e-12 # Avoid division by zero if segment is a point
                        t = dot(vec_p_intersect, vec_pq) / len_pq_sq
                        # Check if t is strictly between 0 and 1 (e.g., within (epsilon, 1-epsilon))
                        epsilon_t = 1e-6 # Tolerance for t
                        if t > epsilon_t && t < (1.0 - epsilon_t)
                            has_line_crossing = true
                            break 
                        end
                    end
                end
            end
            if has_line_crossing; break; end
        end
        if has_line_crossing; break; end
    end

    return is_watertight, has_line_crossing
end


# --- Main Execution ---
function main()
    N = 7 # Grid size
    rx, ry, rz = 1.0, 1.0, 1.0 # Grid cell dimensions

    # For faster testing, reduce N
    # N = 4 
    # result_size_dim for N=4 is 2, so 2*2*2 = 8 polyhedra.

    println("Julia Polyhedron Analysis using Meshes.jl")
    println("Number of threads: $(Threads.nthreads())")
    println("Grid size: $N x $N x $N")
    println("Checking cells from (2,2,2) to ($(N-1),$(N-1),$(N-1))")

    println("Step 1: Generating sv_points...")
    sv_points = generate_sv_points(N, rx, ry, rz)
    println("sv_points generated.")

    println("Step 2: Generating lin_points...")
    lin_x_points, lin_y_points, lin_z_points = generate_lin_points(N, sv_points)
    println("lin_points generated.")

    println("Step 3: Generating mob_points...")
    mob_points_all = generate_mob_points(N, sv_points)
    println("mob_points generated.")

    result_size_dim = N - 2 

    if result_size_dim < 1
        println("Grid size N=$N is too small to perform checks. Needs N>=3. Exiting.")
        return
    end

    watertight_results = Array{Bool, 3}(undef, result_size_dim, result_size_dim, result_size_dim)
    crossing_results   = Array{Bool, 3}(undef, result_size_dim, result_size_dim, result_size_dim)
    
    PolyhedronVisDataType = Vector{NTuple{3, Meshes.Point}}
    visualization_data = Array{PolyhedronVisDataType, 3}(undef, result_size_dim, result_size_dim, result_size_dim)

    println("Step 4: Performing checks for polyhedra...")
    check_range = 2:(N-1) 

    # Using a non-threaded loop for easier debugging if issues persist.
    # Can be changed back to Threads.@threads if performance is critical and logic is stable.
    # Threads.@threads for kc_idx_loop in check_range
    for kc_idx_loop in check_range
        for jc_idx_loop in check_range
            for ic_idx_loop in check_range
                current_ic, current_jc, current_kc = ic_idx_loop, jc_idx_loop, kc_idx_loop
                
                # sv_center_pt for cell (ic,jc,kc) is sv_points[ic,jc,kc]
                sv_center_pt_loop = sv_points[current_ic, current_jc, current_kc]

                base_triangles_pts_loop = get_polyhedron_base_triangles(
                    current_ic, current_jc, current_kc,
                    mob_points_all, lin_x_points, lin_y_points, lin_z_points
                )
                
                is_wt, has_lc = check_single_polyhedron(
                    current_ic, current_jc, current_kc,
                    sv_center_pt_loop,
                    base_triangles_pts_loop 
                )
                
                # Indices for results array (1-based for arrays)
                # If check_range is 2:N-1, then ic_idx_loop-1 maps 2 to 1, etc.
                res_idx_i, res_idx_j, res_idx_k = current_ic-1, current_jc-1, current_kc-1
                
                watertight_results[res_idx_i, res_idx_j, res_idx_k] = is_wt
                crossing_results[res_idx_i, res_idx_j, res_idx_k] = has_lc
                visualization_data[res_idx_i, res_idx_j, res_idx_k] = base_triangles_pts_loop
            end
        end
    end

    println("Checks completed.")

    println("\n--- Results Summary ---")
    total_checked = 0
    total_watertight = 0
    total_no_crossing = 0 # Count of polyhedra with NO line crossings

    if result_size_dim > 0
        for kc_res = 1:result_size_dim
            for jc_res = 1:result_size_dim
                for ic_res = 1:result_size_dim
                    total_checked += 1
                    wt_status = watertight_results[ic_res, jc_res, kc_res]
                    lc_status = crossing_results[ic_res, jc_res, kc_res] # true if crossing occurs

                    if wt_status
                        total_watertight += 1
                    if !lc_status # if has_lc is false (no crossing)
                        total_no_crossing += 1
                    end
                end
            end
        end
    end

    if total_checked > 0
        @printf "Total Polyhedra Checked: %d\n" total_checked
        @printf "Number Watertight: %d (%.2f%%)\n" total_watertight (total_watertight/total_checked*100)
        @printf "Number with No Line Crossings: %d (%.2f%%)\n" total_no_crossing (total_no_crossing/total_checked*100)
    else
        println("No polyhedra were checked (grid size N=$N was too small).")
    end

    println("\nVisualization data is stored in the 'visualization_data' array.")
    # Note: The indices for visualization_data are 1-based from result_size_dim.
    # Cell (ic,jc,kc) (original indices 2:N-1) corresponds to vis_data[ic-1, jc-1, kc-1].
end

main()