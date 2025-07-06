
using LinearAlgebra
using Meshes
using GLMakie
using Statistics
using LinearAlgebra
using Random,Test
using LinearAlgebra,KernelAbstractions,CUDA
using Revise, HDF5

using Revise
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
using Test


using Logging
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")

# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/custom_kern.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/sv_variance_for_loss.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_a.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/dif_custom_kern_point_add_b.jl")
# includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/custom_kern _old.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/utils_lin_sampl.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/sv_centr_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/additional_points_from_weights.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/loss_kerns/dif_custom_kern_tetr.jl")



includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_per_triangle.jl")


"""
find intersection of 2 lines in 3D assuming it exists
"""
function intersect_lines_projected(line1_point1, line1_point2, line2_point1, line2_point2)
   
    # Calculate At * A
    AtA11 = (line1_point2[1] - line1_point1[1]) * (line1_point2[1] - line1_point1[1]) +
            (line1_point2[2] - line1_point1[2]) * (line1_point2[2] - line1_point1[2]) +
            (line1_point2[3] - line1_point1[3]) * (line1_point2[3] - line1_point1[3])
    
    AtA12 = (line1_point2[1] - line1_point1[1]) * -(line2_point2[1] - line2_point1[1]) +
            (line1_point2[2] - line1_point1[2]) * -(line2_point2[2] - line2_point1[2]) +
            (line1_point2[3] - line1_point1[3]) * -(line2_point2[3] - line2_point1[3])
    
    AtA21 = -(line2_point2[1] - line2_point1[1]) * (line1_point2[1] - line1_point1[1]) +
            -(line2_point2[2] - line2_point1[2]) * (line1_point2[2] - line1_point1[2]) +
            -(line2_point2[3] - line2_point1[3]) * (line1_point2[3] - line1_point1[3])
    
    AtA22 = -(line2_point2[1] - line2_point1[1]) * -(line2_point2[1] - line2_point1[1]) +
            -(line2_point2[2] - line2_point1[2]) * -(line2_point2[2] - line2_point1[2]) +
            -(line2_point2[3] - line2_point1[3]) * -(line2_point2[3] - line2_point1[3])
    
    # Calculate At * b
    Atb1 = (line1_point2[1] - line1_point1[1]) * (line2_point1[1] - line1_point1[1]) +
           (line1_point2[2] - line1_point1[2]) * (line2_point1[2] - line1_point1[2]) +
           (line1_point2[3] - line1_point1[3]) * (line2_point1[3] - line1_point1[3])
    
    Atb2 = -(line2_point2[1] - line2_point1[1]) * (line2_point1[1] - line1_point1[1]) +
           -(line2_point2[2] - line2_point1[2]) * (line2_point1[2] - line1_point1[2]) +
           -(line2_point2[3] - line2_point1[3]) * (line2_point1[3] - line1_point1[3])
    
    # Forward elimination
   
    AtA12 /= AtA11
    Atb1 /= AtA11
    AtA11 /= AtA11

    AtA22 -= AtA21 * AtA12
    Atb2 -= AtA21 * Atb1
    AtA21 -= AtA21 * AtA11
    
    # Normalize the second row
    Atb2 /= AtA22
    AtA22 /= AtA22
    
    # Back substitution
    t = Atb1 - AtA12 * Atb2
    
    # Calculate the intersection point
    intersection_point_x = line1_point1[1] + t * (line1_point2[1] - line1_point1[1])
    intersection_point_y = line1_point1[2] + t * (line1_point2[2] - line1_point1[2])
    intersection_point_z = line1_point1[3] + t * (line1_point2[3] - line1_point1[3])
    
    return [intersection_point_x, intersection_point_y, intersection_point_z]
end


"""
we have a triangle defined by 3 three dimensional points (p1,p2,p3) and a point P0 and P1
We want to find weather the distance between the P0 (3D point) and 
the edge of the triangle if it is smaller than the distance between P0 and P1
we are using for this intersect_lines_projected 
    "intersect_lines_projected(line1_point1, line1_point2, line2_point1, line2_point2)"
 that given line points will return the intersection point of the lines that are defined by the points and 
    the denominator of the intersection point   
However this function take into account lines and not 
sections so we are using the function that is calculating the intersection of the lines and then we are checking
if the intersection point is on the section between P0 and P1; and wether we are in the range of the given triangle edge
    Hovewer we need to achive this using differentiable operations and min ,max; we need to avoid using conditionals, less then , greater then
    <,> boolean operations etc
"""

function interpolate_points(P0, P1, weight)
    return (1 - weight) *collect(P0) + weight * collect(P1)
end




function get_random_point_in_triangle(triangle_point1, triangle_point2, triangle_point3,w1, w2, w3)
    # Interpolate the points on the edges of the triangle
    point_on_edge1 = interpolate_points(triangle_point1, triangle_point2, w1)
    point_on_edge2 = interpolate_points(triangle_point2, triangle_point3, w2)
    # Interpolate the points on the edges to get a point inside the triangle
    point_in_triangle = interpolate_points(point_on_edge1, point_on_edge2, w3)

    return point_in_triangle
end


function get_random_point_in_triangleunrolled(triangle_point1, triangle_point2, triangle_point3, w1, w2, w3)
    # Interpolate the points on the edges of the triangle for each coordinate
    point_on_edge1_x = (1 - w1) * triangle_point1[1] + w1 * triangle_point2[1]
    point_on_edge1_y = (1 - w1) * triangle_point1[2] + w1 * triangle_point2[2]
    point_on_edge1_z = (1 - w1) * triangle_point1[3] + w1 * triangle_point2[3]

    point_on_edge2_x = (1 - w2) * triangle_point2[1] + w2 * triangle_point3[1]
    point_on_edge2_y = (1 - w2) * triangle_point2[2] + w2 * triangle_point3[2]
    point_on_edge2_z = (1 - w2) * triangle_point2[3] + w2 * triangle_point3[3]

    # Interpolate the points on the edges to get a point inside the triangle for each coordinate
    point_in_triangle_x = (1 - w3) * point_on_edge1_x + w3 * point_on_edge2_x
    point_in_triangle_y = (1 - w3) * point_on_edge1_y + w3 * point_on_edge2_y
    point_in_triangle_z = (1 - w3) * point_on_edge1_z + w3 * point_on_edge2_z

    return (point_in_triangle_x, point_in_triangle_y, point_in_triangle_z)
end




"""
check is point on a segment
"""
function is_point_on_segment(px, py, pz, x1, y1, z1, x2, y2, z2)
    # print("\n CCCPU px: $px, py: $py, pz: $pz, x1: $x1, y1: $y1, z1: $z1, x2: $x2, y2: $y2, z2: $z2 \n")
    # Initialize variables for direction vector and vector from p1 to p
    line_direction = [0.0, 0.0, 0.0]
    p_to_p1 = [0.0, 0.0, 0.0]
    
    # Calculate the direction vector of the line and the vector from p1 to p
    for i in 1:3
        if i == 1
            line_direction[i] = x2 - x1
            p_to_p1[i] = px - x1
        elseif i == 2
            line_direction[i] = y2 - y1
            p_to_p1[i] = py - y1
        elseif i == 3
            line_direction[i] = z2 - z1
            p_to_p1[i] = pz - z1
        end
    end
    
    # Calculate dot products
    dot_product = 0.0
    segment_length_squared = 0.0
    for i in 1:3
        dot_product += p_to_p1[i] * line_direction[i]
        segment_length_squared += line_direction[i]^2
    end
    
    # Use min and max to avoid boolean operations
    within_segment = (1 - min(1, max(0, dot_product / segment_length_squared))) * (1 - min(1, max(0, (segment_length_squared - dot_product) / segment_length_squared)))
    
    # Ensure the result is either 0 or 1
    return min(1.0,(within_segment*10000000))
end




function intersect_and_clip(triangle_point1, triangle_point2, P0, P1)
    p_edge=intersect_lines_projected(triangle_point1, triangle_point2, P0, P1)

    #check is it on main section
    d_p_edge=is_point_on_segment(p_edge[1], p_edge[2], p_edge[3], P0[1], P0[2], P0[3], P1[1], P1[2], P1[3])

    #check is it on the edge
    d_p_edgeb=is_point_on_segment(p_edge[1], p_edge[2], p_edge[3], triangle_point1[1], triangle_point1[2], triangle_point1[3]
    , triangle_point2[1], triangle_point2[2], triangle_point2[3])

    #d_p_edge and d_p_edgeb now is either 0 or 1
    #so after multiplying if it is 1 we have a point both on a section and on the edge
    d_p_edge_fin = d_p_edge*d_p_edgeb
    #get coordinates
    return collect(p_edge).*d_p_edge_fin
end    


function distance_to_triangle_edge(triangle_point1, triangle_point2, triangle_point3, P0, P1)
    
    #calculate the approximate distance to the intersection of the section and the edge will give 0 if not intersecting

    # we need also to clip to edges of the triangle as we still do not know if the point is on the edge or not
    # we can also recalculate the values - basically when we divide 0 by sth it is still 0 if this is not 0 we can divide by what it is and get 1 just add apsilon to avoid having division by 0
    # not more than one of those will be non zero so we can add them
    p_edge_1=intersect_and_clip(triangle_point1, triangle_point2, P0, P1)
    p_edge_2=intersect_and_clip(triangle_point3, triangle_point2, P0, P1)
    p_edge_3=intersect_and_clip(triangle_point1, triangle_point3, P0, P1)
    p_edge=p_edge_1+p_edge_2+p_edge_3

    # get a distance from P0 to the edge if it intersect otherwise 0
    d_p_edge=LinearAlgebra.norm(collect(p_edge)-collect(P0))

    
    
    #get a point on distance  - distance calculations are not precise 
    # we know just it will be big if the section P0-P1 is intersecting some edge and 0 othewrise
    #so if we will get a point in this direction and clamp we will get a point either at P0 or at edge
    section= P1 - P0
    direction = normalize(section)


    #will give distance to either intersection or P1 (second case if P1 is inside the triangle)
    final_dist=min(d_p_edge,LinearAlgebra.norm(section))  

    # Find the point on the section between P0 and P1 that is at the calculated distance
    point_on_section = P0 + final_dist * direction
    return point_on_section
end






function intersect_line_plane(triangle_point1, triangle_point2, triangle_point3, line_point1, line_point2)
    # Manually calculate the cross product of v1 and v2
    normal = [
        (triangle_point2[2] - triangle_point1[2]) * (triangle_point3[3] - triangle_point1[3]) - (triangle_point2[3] - triangle_point1[3]) * (triangle_point3[2] - triangle_point1[2]),
        (triangle_point2[3] - triangle_point1[3]) * (triangle_point3[1] - triangle_point1[1]) - (triangle_point2[1] - triangle_point1[1]) * (triangle_point3[3] - triangle_point1[3]),
        (triangle_point2[1] - triangle_point1[1]) * (triangle_point3[2] - triangle_point1[2]) - (triangle_point2[2] - triangle_point1[2]) * (triangle_point3[1] - triangle_point1[1])
    ]
    # Normalize the normal vector
    norm_val = 0.0
    for i in 1:3
        norm_val += normal[i]^2
    end
    norm_val = sqrt(norm_val)
    
    for i in 1:3
        normal[i] /= norm_val
    end
    
    # Calculate the dot product manually
    dot_normal_line_dir = 0.0
    dot_normal_diff = 0.0
    for i in 1:3
        line_dir_i = line_point2[i] - line_point1[i]
        dot_normal_line_dir += normal[i] * line_dir_i
        dot_normal_diff += normal[i] * (triangle_point1[i] - line_point1[i])
    end

    # Calculate the parameter t
    t = dot_normal_diff / dot_normal_line_dir
    
    # Calculate the intersection point
    ap_ap_base = []
    for i in 1:3
        line_dir_i = line_point2[i] - line_point1[i]
        push!(ap_ap_base, line_point1[i] + t * line_dir_i)
    end

    return ap_ap_base
end



function elongate_section(apex, in_base, distance)
    # Calculate the direction vector from apex to in_base
    direction_vector = in_base - apex
    
    # Normalize the direction vector
    normalized_direction = normalize(direction_vector)
    
    # Scale the normalized direction vector by the given distance
    scaled_vector = normalized_direction * distance
    
    # Calculate the endpoint of the new section
    end_point = in_base + scaled_vector
    
    return end_point
end


"""
triangle_point1-3 coordinates of the common base of tetrahedrons
apex1,apex2 - coordinates of the apexes of the tetrahedrons
1) we will caclulate the intersection point between a line that connect apicies and a plane of a base
2) we will calculate some random point on the common base of tetrahedrons
3)we will get a point on the edge of the triangle that is closest to the intersection point
now we know that a triangle from pex to the base and point from 3 will be in a range of this apex
in oder to get information weather it is also in a range of the other apex we need to 
get a line from this other apex through the intersection point make it the length of a distance between apex and the intersection point
and find the point that is on the edge of the triangle that is closest to this line 
lets call this calculated points edge_point_ap1 and edge_point_ap2
4) the resulting point will be between edge_point_ap1,edge_point_ap2,base_point,edge_point_ap_common
result should be in the range of both apexes - by range i mean that a line that is getting from 
apex to the point is not intersecting the triangle walls of the tetrahedron
"""
function get_random_point_in_tetrs(triangle_point1,triangle_point2,triangle_point3,apex1,apex2,weights)

    ap_ap_base=intersect_line_plane(triangle_point1, triangle_point2, triangle_point3, apex1, apex2)
    base_point=get_random_point_in_triangle(triangle_point1, triangle_point2, triangle_point3,weights[1], weights[2],weights[3])
    
    edge_point_ap_common=distance_to_triangle_edge(triangle_point1, triangle_point2, triangle_point3, base_point, ap_ap_base)


    vec_ap2=elongate_section(apex2, edge_point_ap_common, norm(apex1-edge_point_ap_common))
    vec_ap1=elongate_section(apex1, edge_point_ap_common, norm(apex2-edge_point_ap_common))


    edge_point_ap2=intersect_lines_projected(apex1, base_point, edge_point_ap_common, apex2)
    edge_point_ap1=intersect_lines_projected(apex2, base_point, edge_point_ap_common, apex1)


    p1=interpolate_points(edge_point_ap2, edge_point_ap1, weights[4])
    p2=interpolate_points(edge_point_ap_common, base_point, weights[5])
    p3=interpolate_points(p1, p2, weights[6])

    return edge_point_ap2, edge_point_ap1,edge_point_ap_common, base_point,p3,ap_ap_base,vec_ap2,vec_ap1

end



"""
we need to test weather the p3 is in the range of both apexes and so
1) check weather point is in one or the other tetrahedron 
if not test failed 
2) check is the line passing from the apex of the other tetrhedron (this that point do not belong to)
intesect the common base it it does test succeeded
"""
function test_viz_main()
    rng=Random.seed!(rand(1:1000))
    # rng=Random.seed!(1234)

    triangle_point1 = [-4.0, -4.0, 0.0]
    triangle_point2 = [4.0, 0.0, 0.0]
    triangle_point3 = [-2.0, -2.0, 0.0]


    # Calculate the current barycenter
    barycenter = [(triangle_point1[i] + triangle_point2[i] + triangle_point3[i]) / 3 for i in 1:3]

    # Determine the translation vector to move the barycenter to [0.0, 0.0, 0.0]
    translation_vector = [-barycenter[i] for i in 1:3]

    # Apply the translation to each vertex
    triangle_point1 = [triangle_point1[i] + translation_vector[i] for i in 1:3]
    triangle_point2 = [triangle_point2[i] + translation_vector[i] for i in 1:3]
    triangle_point3 = [triangle_point3[i] + translation_vector[i] for i in 1:3]

    # triangle_point1 = rand(rng,3).*8
    # triangle_point2 = rand(rng,3).*8
    # triangle_point3 = rand(rng,3).*8
    apex1 = [0.4, 0.4, 2.0]
    apex2 = [0.0, 0.0, -2.0]
    # apex1 = rand(rng,3).*1
    # apex2 = rand(rng,3).*1
    # apex1[3]=apex1[3]*(-1)

    triangle_point1[3]=0
    triangle_point2[3]=0
    triangle_point3[3]=0


    weights=rand(6)

    edge_point_ap2, edge_point_ap1,edge_point_ap_common, base_point,p3,ap_ap_base,vec_ap2,vec_ap1=get_random_point_in_tetrs(triangle_point1,triangle_point2,triangle_point3,apex1,apex2,weights)

    triangle = PolyArea([Meshes.Point(triangle_point1...), Meshes.Point(triangle_point2...), Meshes.Point(triangle_point3...)])
    [ Meshes.Point(edge_point_ap2...)
    ,Meshes.Point(edge_point_ap1...)
    ,Meshes.Point(edge_point_ap_common...)
    ,Meshes.Point(base_point...)]

    p3
    vec_ap1
    apex1

    ob44=[
        # Meshes.Point(triangle_point1...)
        # ,Meshes.Point(triangle_point2...)
        # ,Meshes.Point(triangle_point3...)
        # ,Meshes.Point(ap_ap_base...)
        # ,Meshes.Point(apex1...)
        # ,Meshes.Point(apex2...)
        Meshes.Point(edge_point_ap2...)
        ,Meshes.Point(edge_point_ap1...)
        ,Meshes.Point(p3...)
        ,Meshes.Point(edge_point_ap_common...)
        ,Meshes.Point(base_point...)
        # ,Meshes.Segment(Meshes.Point(vec_ap2...),Meshes.Point(apex2...))
        # ,Meshes.Segment(Meshes.Point(vec_ap1...),Meshes.Point(apex1...))
        ,triangle
        ,Meshes.Segment(Meshes.Point(apex1...) ,Meshes.Point(apex2...))
        ,PolyArea([Meshes.Point(edge_point_ap2...), Meshes.Point(edge_point_ap1...)
        , Meshes.Point(edge_point_ap_common...), Meshes.Point(base_point...) ])

    ]#,Meshes.Point(proj2...),Meshes.Point((proj2+section2)...)

    cc=collect(1:length(ob44))
    # cc[1:4].=1
    # cc[end]=2
    viz(ob44,color=cc,pointsize=10.2,alpha=ones(length(ob44)).*0.6)#  ,alpha=[0.7,0.7,0.7,0.0]  , color=1:length(first_sv_tetrs)
end


"""
for a test we need to establish first weather the point is in the range of the first tetrahedron
if it is we need to check weather the line from the apex of the second tetrahedron to the point is intersecting 
the common base triangle of the tetrahedrons if not we do the same for other tetrahedron
"""
function test_approx()
    ob55=[]
    for i in 1:1000
        triangle_point1 = rand(rng,3).*8
        triangle_point2 = rand(rng,3).*8
        triangle_point3 = rand(rng,3).*8

        apex1 = rand(rng,3).*1
        apex2 = rand(rng,3).*1
        apex1[3]=apex1[3]*(-1)

        triangle_point1[3]=0
        triangle_point2[3]=0
        triangle_point3[3]=0

        weights=rand(6)

        edge_point_ap2, edge_point_ap1,edge_point_ap_common, base_point,p3,ap_ap_base,vec_ap2,vec_ap1=get_random_point_in_tetrs(triangle_point1,triangle_point2,triangle_point3,apex1,apex2,weights)


        triangle = PolyArea([Meshes.Point(triangle_point1...), Meshes.Point(triangle_point2...), Meshes.Point(triangle_point3...)])
        segment_ap1=Meshes.Segment(Meshes.Point(p3...) ,Meshes.Point(apex2...))
        segment_ap2=Meshes.Segment(Meshes.Point(apex1...) ,Meshes.Point(p3...))
        tetrahedron1=Meshes.Tetrahedron(Meshes.Point(triangle_point1...),Meshes.Point(triangle_point2...)
        ,Meshes.Point(triangle_point3...),Meshes.Point(apex1...))
        tetrahedron2=Meshes.Tetrahedron(Meshes.Point(triangle_point1...),Meshes.Point(triangle_point2...)
        ,Meshes.Point(triangle_point3...),Meshes.Point(apex2...))


        ob55=[
            Meshes.Point(p3...)
            ,segment_ap1
            ,segment_ap2
            ,triangle
            # ,tetrahedron1
            # ,tetrahedron2
        ]

        if(Meshes.Point(p3...)∈tetrahedron1)
            @test intersects(segment_ap2, triangle)

        end
        if(Meshes.Point(p3...)∈tetrahedron2)
            @test intersects(segment_ap1, triangle)

        end    

    end

    cc=collect(1:length(ob55))
    # cc[1:4].=1
    # cc[end]=2
    viz(ob55,color=cc,pointsize=10.2,alpha=ones(length(ob55)).*0.6)#  ,alpha=[0.7,0.7,0.7,0.0]  , color=1:length(first_sv_tetrs)

end



################# test distance_to_triangle_edge
function viz_distance_to_triangle_edge()
    triangle_point1 = rand(3).*10
    triangle_point2 = rand(3).*10
    triangle_point3 = rand(3).*10
    triangle_point1[3]=0
    triangle_point2[3]=0
    triangle_point3[3]=0
    P0_base=(triangle_point1+triangle_point2+triangle_point3)/3
    P0 = [P0_base[1], P0_base[2], P0_base[3]]
    P1 = P0.+rand(3).*10
    P1[3]=0

    p1=distance_to_triangle_edge(triangle_point1, triangle_point2, triangle_point3, P0, P1)    # Create the triangle object
    p_curr=Meshes.Point(p1...)

    triangle = PolyArea([Meshes.Point(triangle_point1...), Meshes.Point(triangle_point2...), Meshes.Point(triangle_point3...)])
    ob44=[
        triangle,
        p_curr
    ,Meshes.Point(triangle_point1...)
    ,Meshes.Point(triangle_point2...)
    ,Meshes.Segment(Meshes.Point(P0...) ,Meshes.Point(P1...)),Meshes.Point(P0...)  ]#,Meshes.Point(proj2...),Meshes.Point((proj2+section2)...)


    viz(ob44,color=1:length(ob44),pointsize=10.2)#  ,alpha=[0.7,0.7,0.7,0.0]  , color=1:length(first_sv_tetrs)
end






# triangle_point1 = rand(rng,3).*8
# triangle_point2 = rand(rng,3).*8
# triangle_point3 = rand(rng,3).*8

# apex1 = rand(rng,3).*1
# apex2 = rand(rng,3).*1
# apex1[3]=apex1[3]*(-1)

# triangle_point1[3]=0
# triangle_point2[3]=0
# triangle_point3[3]=0

# barycenter = [(triangle_point1[i] + triangle_point2[i] + triangle_point3[i]) / 3 for i in 1:3]
# weights=rand(6)

# triangle_point1=Float32.(triangle_point1)
# triangle_point2=Float32.(triangle_point2)
# apex1=Float32.(apex1)
# apex2=Float32.(apex2)
# weights=Float32.(weights)


# res=CuArray(Float32.([0.0,0.0,0.0]))
# threads=(1,3)
# blocks=(1,1,1)
# @cuda threads blocks get_random_point_in_tetrs_kern(CuArray(triangle_point1),CuArray(triangle_point2),CuArray(triangle_point3),CuArray(apex1),CuArray(apex2),CuArray(weights),res)
# # CUDA.synchronize()

# # proj=intersect_line_plane(triangle_point1, triangle_point2, triangle_point3, apex1, apex2)
# # base=get_random_point_in_triangle(triangle_point1, triangle_point2, triangle_point3,weights[1], weights[2],weights[3])

# edge_point_ap2, edge_point_ap1,edge_point_ap_common, base_point,p3,ap_ap_base,vec_ap2,vec_ap1=get_random_point_in_tetrs(triangle_point1,triangle_point2,triangle_point3,apex1,apex2,weights)



# p1=interpolate_points(edge_point_ap2, edge_point_ap1, weights[4])
# p2=interpolate_points(edge_point_ap_common, base_point, weights[5])
# p3=interpolate_points(p1, p2, weights[6])

# # a=distance_to_triangle_edge(triangle_point1, triangle_point2, triangle_point3, base, proj)
# b=Array(res)
# p3,b
# p3,b


#######3 test get_plan_tensor_for_points_per_triangle
# function visualize_neighbours_for_get_plan_tensor_for_points_per_triangle()
   
   
   

    image_shape=(60,60,60)
    radiuss=(4.0,4.0,4.0)
    example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
    sv_centers, control_points, tetrs, dims = example_set_of_svs



    # lets reshape tetrs to the 3 dimensional shape to establish which tetrahedrons are neighbours
    dims_tetr=dims.-2
    # dims_tetr=(12,12,12)
    tetr_3d=reshape(tetrs,(48,dims_tetr[1],dims_tetr[2],dims_tetr[3],5,4))
    tetr_3d=permutedims(tetr_3d,(2,3,4,1,5,6))

    tetr_3d[3,3,3,1,1,1:3] #4,4,4
    tetr_3d[2,3,3,1,1,1:3] #3,4,4
    tetr_3d[3,2,3,1,1,1:3] #4,3,4
    tetr_3d[3,3,2,1,1,1:3] #4,4,3

    #now we will get a middle sv and associate with each tetrahedron neighbouring tetrahedron from some of the surrounding voxels
    middle_coords=(3,2,4)

    middle_tetr=tetr_3d[middle_coords[1],middle_coords[2],middle_coords[3],:,2:4,1:4]
    association_dict=Dict()
    for i in 1:48
        association_dict[i]=-1
    end

    middle_tetr[1,:,:]

    
   
    is_noise_both_batch_same=true
    radiuss = (Float32(3.1), Float32(4.3), Float32(4.7))
    num_weights_per_point = 8
    spacing = (Float32(1.0), Float32(1.0), Float32(1.0))
    batch_size = 2
    a = 71
    image_shape = (a, a, a, 2)

    weights_channels=24+24*6
    weights_shape = Int.(round.((a / 2, a / 2, a / 2, weights_channels)))

    example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss,true)
    sv_centers, control_points, tetrs, dims = example_set_of_svs


    # weights = weights .+ 0.5
    # source_arr = rand(image_shape...)

    #batching
    control_points = repeat(control_points, inner=(1, 1, 1, 1, 1, batch_size))
    # sv_centers = repeat(sv_centers, inner=(1, 1, 1, 1, batch_size))
    # tetrs = repeat(tetrs, inner=(1, 1, 1, batch_size))


    if (is_noise_both_batch_same)
        weights_shape = Int.(round.((a / 2, a / 2, a / 2, weights_channels)))
        weights = rand(weights_shape...)
        weights = repeat(weights, inner=(1, 1, 1, 1, batch_size))
        image_shape = (a, a, a, 2)
        source_arr = rand(image_shape...)
        source_arr = repeat(source_arr, inner=(1, 1, 1, 1, batch_size))
    else
        weights_shape = Int.(round.((a / 2, a / 2, a / 2, weights_channels, batch_size)))
        weights = rand(weights_shape...)
        image_shape = (a, a, a, 2, batch_size)
        source_arr = rand(image_shape...)

    end


    weights = Float32.(weights)
    control_points = Float32.(control_points)
    source_arr = Float32.(source_arr)
    sv_centers = Float32.(sv_centers)


    threads_apply_w, blocks_apply_w, num_blocks_z_pure_sv, num_blocks_y_pure_sv = prepare_for_apply_weights_to_locs_kern(size(sv_centers), weights_shape, batch_size)

    sv_centers_out = call_apply_weights_sv(CuArray(sv_centers), CuArray(weights), radiuss, threads_apply_w, blocks_apply_w, num_blocks_z_pure_sv, batch_size, num_blocks_y_pure_sv)

    CUDA.synchronize()

    threads_apply_w, blocks_apply_w, num_blocks_z_pure, num_blocks_y_pure_w = prepare_for_apply_weights_to_locs_kern(size(control_points), weights_shape, batch_size)
    # control_points=Float32.(control_points)
    control_points_out = call_apply_weights_to_locs_kern(sv_centers_out, size(control_points), CuArray(weights), threads_apply_w, blocks_apply_w, num_blocks_z_pure, num_blocks_y_pure_w)
    CUDA.synchronize()


    control_points_out = call_apply_weights_to_locs_kern_add_a(sv_centers_out, control_points_out, CuArray(weights), threads_apply_w, blocks_apply_w, num_blocks_z_pure, num_blocks_y_pure_w)
    CUDA.synchronize()

    control_points_out
    dims.-1






    control_points_curr=Array(control_points_out)[:,:,:,:,:,1]
    sv_centers_out_curr=Array(sv_centers_out)[:,:,:,:,1]
    plan_tuples=check_triangles(middle_coords, tetr_3d, middle_tetr)
    plan_tuples_sorted=sort(plan_tuples, by=x->x["current_tetr"])
    plan_tensor=get_plan_tensor_for_points_per_triangle(plan_tuples)


    base_ind=get_base_indicies_arr(dims.-1)
    flat_ind=reshape(base_ind,(size(base_ind,1)*size(base_ind,2)*size(base_ind,3),size(base_ind,4) ))




    plan_tensor=get_plan_tensor_for_points_per_triangle(plan_tuples)

    plan_tuples=check_triangles(middle_coords, tetr_3d, middle_tetr)
    plan_tuples_sorted=sort(plan_tuples, by=x->x["current_tetr"])


   


    size(sv_centers_out)
    size(control_points_out)
    size(weights)
    
    const len_get_random_point_in_tetrs_kern=Int(floor(256/3))
    siz_c=size(control_points_out)
    siz_plan=size(plan_tensor)
    x_blocks=Int(ceil((siz_c[1]*siz_c[2]*siz_c[3])/len_get_random_point_in_tetrs_kern))
    max_index=siz_c[1]*siz_c[2]*siz_c[3]
    new_control_points_out=copy(control_points_out)
    @cuda threads=(len_get_random_point_in_tetrs_kern,3) blocks=(x_blocks,siz_plan[1],batch_size) get_random_point_in_tetrs_kern(CuArray(plan_tensor),CuArray(weights),control_points_out,sv_centers_out,new_control_points_out,siz_c,max_index)
    control_points_out=new_control_points_out

    sum(new_control_points_out)#2.8841038f6



    plan_tuples_full=check_triangles(middle_coords, tetr_3d, middle_tetr,true)
    #sorted plan tuples is not full - so 24 entries
    plan_tuples=check_triangles(middle_coords, tetr_3d, middle_tetr,false)
    plan_tuples_sorted=sort(plan_tuples, by=x->x["current_tetr"])
    

    # Initialize an empty dictionary to find dictionary based on tetr index
    dict_of_dicts = Dict{Int, Dict}()  
    # Iterate over each dictionary in the list
    for subdict in plan_tuples_full
        # Use the value under "current_tetr" as the key
        key = subdict["current_tetr"]
        # Assign the entire subdictionary to this key
        dict_of_dicts[key] = subdict
    end

    # Initialize an empty dictionary to find position (channel in control points out) based on tetr index 
    pos_index_dict = Dict{Any, Int}()

    # Iterate over each dictionary in the list with an index
    for (index, subdict) in enumerate(plan_tuples_sorted)
        # Add entries for both "current_tetr" and "num_tetr_neigh"
        pos_index_dict[subdict["current_tetr"]] = index
        pos_index_dict[subdict["num_tetr_neigh"]] = index
    end

    #get a list of primary indicies - if index is not here we need to reach out to the neighbour for appropriate point
    prim_indicies=map(el->el["current_tetr"] ,plan_tuples_sorted)

    tetrs_size=size(tetrs)
    new_tetr_size=(tetrs_size[1]*3,tetrs_size[2],tetrs_size[3])
    new_flat_tetrs=zeros(new_tetr_size)
    #iterating over first dimension of the tetrs
    
    
    
    Threads.@threads for ind_prim in 1:tetrs_size[1]
        # print("* $ind_prim *")
        #getting which tetrahedron in sv it is
        ind_tetr=((ind_prim-1)%48)+1
        channel_control_points=pos_index_dict[ind_tetr]
        curr_dict=dict_of_dicts[ind_tetr]
        t1,t2,t3=curr_dict["triangle_points"]

        new_ind_tetr_base=((ind_prim-1)*3)+1
        tetr_curr=tetrs[ind_prim,:,:]
        base_ind=tetr_curr[1,1:3].-1
        

        sv_center=tetr_curr[1,:]
        triang_1=[t1[1]+base_ind[1],t1[2]+base_ind[2],t1[3]+base_ind[3],t1[4]]
        triang_2=[t2[1]+base_ind[1],t2[2]+base_ind[2],t2[3]+base_ind[3],t2[4]]
        triang_3=[t3[1]+base_ind[1],t3[2]+base_ind[2],t3[3]+base_ind[3],t3[4]]
        

        
        dummy=tetr_curr[5,:]
        #weather it is base ind or not depends on weather we are looking on prev or next in axis
        base_ind_p=base_ind
        if(!(ind_tetr in prim_indicies))
            base_ind_p=base_ind+(curr_dict["apex2"].-1)
        end
        #new point we created using get_random_point_in_tetrs_kern
        new_point=[base_ind_p[1],base_ind_p[2],base_ind_p[3],channel_control_points+7]
        # new_point=[1,1,1,2]


        #populating with new data - we will always have the same sv center 
        #and the same 2 old triangle points and a new one 
        #we start from new_ind_tetr_base and we will add 1,2,3
        
        ###1
        to_add=0
        new_flat_tetrs[new_ind_tetr_base+to_add,1,:]=sv_center
        new_flat_tetrs[new_ind_tetr_base+to_add,2,:]=triang_1
        new_flat_tetrs[new_ind_tetr_base+to_add,3,:]=triang_2

        # new_flat_tetrs[new_ind_tetr_base+to_add,4,:]=triang_3
        new_flat_tetrs[new_ind_tetr_base+to_add,4,:]=new_point
        new_flat_tetrs[new_ind_tetr_base+to_add,5,:]=dummy

        to_add=1
        new_flat_tetrs[new_ind_tetr_base+to_add,1,:]=sv_center
        new_flat_tetrs[new_ind_tetr_base+to_add,2,:]=triang_1
        new_flat_tetrs[new_ind_tetr_base+to_add,3,:]=triang_3

        # new_flat_tetrs[new_ind_tetr_base+to_add,4,:]=triang_2
        new_flat_tetrs[new_ind_tetr_base+to_add,4,:]=new_point
        new_flat_tetrs[new_ind_tetr_base+to_add,5,:]=dummy

        to_add=2
        new_flat_tetrs[new_ind_tetr_base+to_add,1,:]=sv_center
        new_flat_tetrs[new_ind_tetr_base+to_add,2,:]=triang_3
        new_flat_tetrs[new_ind_tetr_base+to_add,3,:]=triang_2

        # new_flat_tetrs[new_ind_tetr_base+to_add,4,:]=triang_1
        new_flat_tetrs[new_ind_tetr_base+to_add,4,:]=new_point
        new_flat_tetrs[new_ind_tetr_base+to_add,5,:]=dummy

    end

    iio=initialize_centers_and_control_points(image_shape,radiuss,true)
    
    sv_centerss,cp,flattened_triangles_augmenteds,dimss,plan_tensorss=iio

    threads_tetr_set, blocks_tetr_set = prepare_for_set_tetr_dat(image_shape, size(new_flat_tetrs), batch_size)
    CUDA.synchronize()


    maximum(new_flat_tetrs[:,1,1:3])
    minimum(new_flat_tetrs[:,1,1:3])

    maximum(new_flat_tetrs[:,2,1:3])
    minimum(new_flat_tetrs[:,2,1:3])

    maximum(new_flat_tetrs[:,3,1:3])
    minimum(new_flat_tetrs[:,3,1:3])

    maximum(new_flat_tetrs[:,4,2])
    minimum(new_flat_tetrs[:,4,3])
    size(control_points_out)

    sum(control_points_out[:,:,:,30,:,:])

    tetr_dat_out = call_set_tetr_dat_kern(CuArray(new_flat_tetrs), CuArray(source_arr), control_points_out, sv_centers_out, threads_tetr_set, blocks_tetr_set, spacing, batch_size)
    CUDA.synchronize()
    num_sv_per_tetr=48*3
    obb=[]
    # ind_tetr=2
    # tetr_inf=Array(tetr_dat_out)[(ind_tetr-1)*num_sv_per_tetr+1:ind_tetr*num_sv_per_tetr,:,:,1]
    # for i in 1:num_sv_per_tetr
    #     push!(obb,Meshes.PolyArea(Meshes.Point(tetr_inf[i,2,1:3]...),Meshes.Point(tetr_inf[i,3,1:3]...),Meshes.Point(tetr_inf[i,4,1:3]...)))
    # end 
    ind_tetr=7
    tetr_inf=Array(tetr_dat_out)[(ind_tetr-1)*num_sv_per_tetr+1:ind_tetr*num_sv_per_tetr,:,:,1]
    for i in 1:num_sv_per_tetr
        push!(obb,Meshes.PolyArea(Meshes.Point(tetr_inf[i,2,1:3]...),Meshes.Point(tetr_inf[i,3,1:3]...),Meshes.Point(tetr_inf[i,4,1:3]...)))
    end 
    
    
    cc=collect(1:length(obb))
    viz(obb,color=cc,pointsize=10.2)#  ,alpha=[0.7,0.7,0.7,0.0]  , color=1:length(first_sv_tetrs)

############### test derivative generation

print("*********** start auto diff test ****************")

function get_random_point_in_tetrs_kern_deff(plan_tensor, d_plan_tensor, weights, d_weights,control_points_in, d_control_points_in, sv_centers_out, d_sv_centers_out, control_points_out, d_control_points_out,siz_c, max_index)

    Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(get_random_point_in_tetrs_kern),
    Enzyme.Const, Duplicated(plan_tensor, d_plan_tensor)
    , Duplicated(weights, d_weights)
    , Duplicated(control_points_in, d_control_points_in)
    , Duplicated(sv_centers_out, d_sv_centers_out)
    , Duplicated(control_points_out, d_control_points_out)
    , Const(siz_c), Const(max_index)
    )

    return nothing
end

control_points_in=copy(control_points_out)
d_control_points_in=CUDA.zeros(Float32,size(control_points_out))
d_plan_tensor=CUDA.zeros(Int16,size(plan_tensor))
d_weights=CUDA.zeros(Float32,size(weights))
d_control_points_out=CUDA.ones(Float32,size(control_points_out))
d_sv_centers_out=CUDA.zeros(Float32,size(sv_centers_out))
plan_tensor=CuArray(Int16.(plan_tensor))
weights=CuArray(weights)
@cuda threads=(len_get_random_point_in_tetrs_kern,3) blocks=(x_blocks,siz_plan[1],siz_c[end]) get_random_point_in_tetrs_kern_deff(plan_tensor, d_plan_tensor
, weights, d_weights, control_points_in, d_control_points_in
, sv_centers_out, d_sv_centers_out, 
control_points_out, d_control_points_out,siz_c, max_index)


count_zeros(d_weights,"d_weights")
count_zeros(d_sv_centers_out,"d_sv_centers_out")
count_zeros(d_control_points_in,"d_control_points_in")

size(d_weights)


function count_zeros_r(arr)
    num_zeros = count(x -> x == 0.0, arr)
    num_entries = length(arr)
    percentt = (num_zeros / num_entries) * 100
    return percentt
end

count_zeros_r(d_weights[:,:,:,:,2])


size(d_weights)




percent_zeros = [count_zeros_r(d_weights[:,:,:,i,:]) for i in 1:168]
percent_zeros = [count_zeros_r(d_weights[:,:,:,:,i]) for i in 1:2]

using Plots
bar(percent_zeros, xlabel="Index", ylabel="Percentage of Zeros", title="Percentage of Zeros in d_weights")




print(" *********** end auto diff test   d_weights $(sum(d_weights)) ****************")




########### vizualization of example point and tetragedrons GPU

index_curr=(5,3,4)
plan_index=18
plan_tensor_curr=plan_tensor[plan_index,:,:]

function get_example_point_in_cp(index_curr,plan_tensor_curr,control_points_curr,point_index)
    return control_points_curr[index_curr[1]+plan_tensor_curr[point_index,1]
    ,index_curr[2]+plan_tensor_curr[point_index,2],index_curr[3]+plan_tensor_curr[point_index,3],plan_tensor_curr[point_index,4],:]
end    
function get_example_point_in_sv_out(index_curr,plan_tensor_curr,sv_centers_out_curr,point_index)
    return sv_centers_out_curr[index_curr[1]+plan_tensor_curr[point_index,1]
    ,index_curr[2]+plan_tensor_curr[point_index,2]
    ,index_curr[3]+plan_tensor_curr[point_index,3]
    ,:]
end    

triangle_point1_curr=get_example_point_in_cp(index_curr,plan_tensor_curr,control_points_curr,1)
triangle_point2_curr=get_example_point_in_cp(index_curr,plan_tensor_curr,control_points_curr,2)
triangle_point3_curr=get_example_point_in_cp(index_curr,plan_tensor_curr,control_points_curr,3)
apex1=get_example_point_in_sv_out(index_curr,plan_tensor_curr,sv_centers_out_curr,4)
apex2=get_example_point_in_sv_out(index_curr,plan_tensor_curr,sv_centers_out_curr,5)
# proj=intersect_line_plane(triangle_point1, triangle_point2, triangle_point3, apex1, apex2)
# base=get_random_point_in_triangle(triangle_point1, triangle_point2, triangle_point3,weights[1], weights[2],weights[3])

edge_point_ap2, edge_point_ap1,edge_point_ap_common, base_point,p3,ap_ap_base,vec_ap2,vec_ap1=get_random_point_in_tetrs(triangle_point1_curr
,triangle_point2_curr,triangle_point3_curr,apex1,apex2,weights[index_curr[1],index_curr[2],index_curr[3],((plan_index-1)*6)+24:((plan_index)*6)+24 ,1  ])

# @test isapprox(Array(control_points_out)[index_curr[1],index_curr[2],index_curr[3], 7+plan_index,1:3,1],p3,atol=0.4)

p_new=Array(control_points_out)[index_curr[1],index_curr[2],index_curr[3], 7+plan_index,1:3,1]


triangle = PolyArea([Meshes.Point(triangle_point1_curr...), Meshes.Point(triangle_point2_curr...), Meshes.Point(triangle_point3_curr...)])
segment_ap1=Meshes.Segment(Meshes.Point(p_new...) ,Meshes.Point(apex2...))
segment_ap2=Meshes.Segment(Meshes.Point(apex1...) ,Meshes.Point(p_new...))


segment_ap3=Meshes.Segment(Meshes.Point(apex1...) ,Meshes.Point(triangle_point1_curr...))
segment_ap4=Meshes.Segment(Meshes.Point(apex1...) ,Meshes.Point(triangle_point2_curr...))
segment_ap5=Meshes.Segment(Meshes.Point(apex1...) ,Meshes.Point(triangle_point3_curr...))

segment_ap6=Meshes.Segment(Meshes.Point(apex2...) ,Meshes.Point(triangle_point1_curr...))
segment_ap7=Meshes.Segment(Meshes.Point(apex2...) ,Meshes.Point(triangle_point2_curr...))
segment_ap8=Meshes.Segment(Meshes.Point(apex2...) ,Meshes.Point(triangle_point3_curr...))



ob55=[
    Meshes.Point(p_new...)
    ,segment_ap1
    ,segment_ap2
    ,triangle
    ,segment_ap3
    ,segment_ap4
    ,segment_ap5
    ,segment_ap6
    ,segment_ap7
    ,segment_ap8
    ]

cc=collect(1:length(ob55))
viz(ob55,color=cc,pointsize=10.2,alpha=ones(length(ob55)).*0.8)#  ,alpha=[0.7,0.7,0.7,0.0]  , color=1:length(first_sv_tetrs)

# p1=interpolate_points(edge_point_ap2, edge_point_ap1, weights[4])
# p2=interpolate_points(edge_point_ap_common, base_point, weights[5])
# p3=interpolate_points(p1, p2, weights[6])

# # a=distance_to_triangle_edge(triangle_point1, triangle_point2, triangle_point3, base, proj)
# b=Array(res)
# p3,b
# p3,b







############





    # function get_example_point_in_cp(index_curr,plan_tensor_curr,control_points_curr,point_index)
    #     return control_points_curr[index_curr[1]+plan_tensor_curr[point_index,1]
    #     ,index_curr[2]+plan_tensor_curr[point_index,2],index_curr[3]+plan_tensor_curr[point_index,3],plan_tensor_curr[point_index,4],:]
    # end    
    # function get_example_point_in_sv_out(index_curr,plan_tensor_curr,sv_centers_out_curr,point_index)
    #     return sv_centers_out_curr[index_curr[1]+plan_tensor_curr[point_index,1]
    #     ,index_curr[2]+plan_tensor_curr[point_index,2]
    #     ,index_curr[3]+plan_tensor_curr[point_index,3]
    #     ,:]
    # end    

    # triangle_point1_curr=get_example_point_in_cp(index_curr,plan_tensor_curr,control_points_curr,1)
    # triangle_point2_curr=get_example_point_in_cp(index_curr,plan_tensor_curr,control_points_curr,2)
    # triangle_point3_curr=get_example_point_in_cp(index_curr,plan_tensor_curr,control_points_curr,3)
    # apex1=get_example_point_in_sv_out(index_curr,plan_tensor_curr,sv_centers_out_curr,4)
    # apex2=get_example_point_in_sv_out(index_curr,plan_tensor_curr,sv_centers_out_curr,5)

    # tetrahedrons=[Meshes.Tetrahedron(Meshes.Point(triangle_point1_curr...),Meshes.Point(triangle_point2_curr...)
    # ,Meshes.Point(triangle_point3_curr...),Meshes.Point(apex1...)),
    # Meshes.Tetrahedron(Meshes.Point(triangle_point1_curr...),Meshes.Point(triangle_point2_curr...)
    # ,Meshes.Point(triangle_point3_curr...),Meshes.Point(apex2...))
    # ]

    # viz(tetrahedrons,color=1:2,pointsize=10.2,alpha=ones(2).*0.6)#  ,alpha=[0.7,0.7,0.7,0.0]  , color=1:length(first_sv_tetrs)

# end



# triangle_point1_curr=get_example_point_in_cp(index_curr,plan_tensor_curr,control_points_curr,1)
# triangle_point2_curr=get_example_point_in_cp(index_curr,plan_tensor_curr,control_points_curr,2)
# triangle_point3_curr=get_example_point_in_cp(index_curr,plan_tensor_curr,control_points_curr,3)
# apex1=get_example_point_in_sv_out(index_curr,plan_tensor_curr,sv_centers_out_curr,4)
# apex2=get_example_point_in_sv_out(index_curr,plan_tensor_curr,sv_centers_out_curr,5)


# index_curr=(0,0,1)
# tt=plan_tuples[5]
# triangg=tt["triangle_points_middle"]
# triangle_point1_curr=Int.(triangg[1])
# triangle_point2_curr=Int.(triangg[2])
# triangle_point3_curr=Int.(triangg[3])
# apex1=Int.(tt["apex1_middle"])
# apex2=Int.(tt["apex2_middle"])





# function get_example_point_in_cp_b(index_curr,control_points_curr,point_dir)
#     return control_points_curr[index_curr[1]+point_dir[1]
#     ,index_curr[2]+point_dir[2],index_curr[3]+point_dir[3],point_dir[4],:]
# end    
# function get_example_point_in_sv_out_b(index_curr,sv_centers_out_curr,point_index,point_dir)
#     return sv_centers_out_curr[index_curr[1]+point_dir[1]
#     ,index_curr[2]+point_dir[2]
#     ,index_curr[3]+point_dir[3]
#     ,:]
# end    
# triangle_point1_curr=get_example_point_in_cp_b(index_curr,control_points_curr,triangle_point1_curr)
# triangle_point2_curr=get_example_point_in_cp_b(index_curr,control_points_curr,triangle_point2_curr)
# triangle_point3_curr=get_example_point_in_cp_b(index_curr,control_points_curr,triangle_point3_curr)
# apex1=get_example_point_in_sv_out_b(index_curr,sv_centers_out_curr,4,apex1)
# apex2=get_example_point_in_sv_out_b(index_curr,sv_centers_out_curr,5,apex2)


# tetrahedrons_middle=[Meshes.Tetrahedron(Meshes.Point(triangle_point1_curr...),Meshes.Point(triangle_point2_curr...)
# ,Meshes.Point(triangle_point3_curr...),Meshes.Point(apex1...)),
# Meshes.Tetrahedron(Meshes.Point(triangle_point1_curr...),Meshes.Point(triangle_point2_curr...)
# ,Meshes.Point(triangle_point3_curr...),Meshes.Point(apex2...))
# ]
# viz(tetrahedrons_middle,color=1:2,pointsize=10.2,alpha=ones(2).*0.6)#  ,alpha=[0.7,0.7,0.7,0.0]  , color=1:length(first_sv_tetrs)
