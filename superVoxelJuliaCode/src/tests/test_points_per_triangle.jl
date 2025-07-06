
using LinearAlgebra
using Meshes
using GLMakie
using Statistics
using LinearAlgebra
using Random,Test



using LinearAlgebra


function solve_system(A, b)
    # Calculate the transpose of A
    At = transpose(A)
    
    # Calculate At * A
    AtA = At * A
    
    # Calculate At * b
    Atb = At * b
    
    # Solve the normal equations AtA * x = Atb using Gaussian elimination
    n = size(AtA, 1)
    x = zeros(n)
    
    # Forward elimination
    for i in 1:n
        # Make the diagonal contain all 1's
        factor = AtA[i, i]
        for j in i:n
            AtA[i, j] /= factor
        end
        Atb[i] /= factor
        
        # Make the elements below the pivot elements equal to zero
        for k in i+1:n
            factor = AtA[k, i]
            for j in i:n
                AtA[k, j] -= factor * AtA[i, j]
            end
            Atb[k] -= factor * Atb[i]
        end
    end
    
    # Back substitution
    for i in n:-1:1
        x[i] = Atb[i]
        for j in i+1:n
            x[i] -= AtA[i, j] * x[j]
        end
    end
    
    # Return the solution as a vector of length 2
    return x[1:2]
end

"""
find intersection of 2 lines in 3D assuming it exists
"""
function intersect_lines(line1_point1, line1_point2, line2_point1, line2_point2)
    # Direction vectors of the lines
    d1 = line1_point2 - line1_point1
    d2 = line2_point2 - line2_point1
    
    # Coefficients for the system of equations
    A = [d1 -d2]
    b = line2_point1 - line1_point1
    # Solve the system of equations A * [t; s] = b
    # ts = A \ b
    ts = solve_system(A, b)
    
    # Calculate the intersection point
    intersection_point = line1_point1 + ts[1] * d1
    
    return intersection_point
end

"""
intersect lines that are given by 3D coordinates but are known to be on the same plane
hence first we are performing a projection to 2D and then we are using the 2D intersection function
    and get back to 3D
"""
function intersect_lines_projected(line1_point1, line1_point2, line2_point1, line2_point2)
#     line1_point1_2d, line1_point2_2d, line2_point1_2d, line2_point2_2d, basis1, basis2, normal, origin = transform_to_2d(line1_point1, line1_point2, line2_point1, line2_point2)
#     px,py,denom = intersect_lines_2D(line1_point1_2d, line1_point2_2d, line2_point1_2d, line2_point2_2d)
#     intersection_point_2d=(px,py)
#     intersection_point,intersection_pointt,intersection_pointtt,intersection_pointttt = transform_to_3d(intersection_point_2d, intersection_point_2d, intersection_point_2d, intersection_point_2d, basis1, basis2, normal, origin)
#     return intersection_point,denom
return intersect_lines(line1_point1, line1_point2, line2_point1, line2_point2),0.0
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
function project_point_on_line(point, line_point1, line_point2)
    line_direction = line_point2 - line_point1
    line_direction_normalized = normalize(line_direction)
    point_to_line = point - line_point1
    projection_distance = dot(point_to_line, line_direction_normalized)
    projected_point = line_point1 + projection_distance * line_direction_normalized
    return projected_point
end

function interpolate_points(P0, P1, weight)
    return (1 - weight) *collect(P0) + weight * collect(P1)
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


function get_random_point_in_triangle(triangle_point1, triangle_point2, triangle_point3,w1, w2, w3)
    # Interpolate the points on the edges of the triangle
    point_on_edge1 = interpolate_points(triangle_point1, triangle_point2, w1)
    point_on_edge2 = interpolate_points(triangle_point2, triangle_point3, w2)

    # Interpolate the points on the edges to get a point inside the triangle
    point_in_triangle = interpolate_points(point_on_edge1, point_on_edge2, w3)

    return point_in_triangle
end



function is_point_on_segment(px, py, pz, x1, y1, z1, x2, y2, z2)
    p = [px, py, pz]
    p1 = [x1, y1, z1]
    p2 = [x2, y2, z2]
    
    # Calculate the direction vector of the line
    line_direction = p2 .- p1
    
    # Calculate the vector from p1 to p
    p_to_p1 = p .- p1
    
    # Calculate dot products
    dot_product = dot(p_to_p1, line_direction)
    segment_length_squared = dot(line_direction, line_direction)
    
    # Use min and max to avoid boolean operations
    within_segment = (1 - min(1, max(0, dot_product / segment_length_squared))) * (1 - min(1, max(0, (segment_length_squared - dot_product) / segment_length_squared)))
    
    # Ensure the result is either 0 or 1
    return within_segment
end


function intersect_and_clip(triangle_point1, triangle_point2, P0, P1)
    p_edge,d=intersect_lines_projected(triangle_point1, triangle_point2, P0, P1)

    #check is it on main section
    d_p_edge=is_point_on_segment(p_edge[1], p_edge[2], p_edge[3], P0[1], P0[2], P0[3], P1[1], P1[2], P1[3])
    d_p_edge=d_p_edge*10000000
    d_p_edge=min(1.0,d_p_edge)
    #check is it on the edge
    d_p_edgeb=is_point_on_segment(p_edge[1], p_edge[2], p_edge[3], triangle_point1[1], triangle_point1[2], triangle_point1[3]
    , triangle_point2[1], triangle_point2[2], triangle_point2[3])
    d_p_edgeb=d_p_edgeb*10000000
    d_p_edgeb=min(1.0,d_p_edgeb)
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
    d_p_edge=norm(collect(p_edge)-collect(P0))
    
    #get a point on distance  - distance calculations are not precise 
    # we know just it will be big if the section P0-P1 is intersecting some edge and 0 othewrise
    #so if we will get a point in this direction and clamp we will get a point either at P0 or at edge
    section= P1 - P0
    direction = normalize(section)


    #will give distance to either intersection or P1 (second case if P1 is inside the triangle)
    final_dist=min(d_p_edge,norm(section))  
    # Find the point on the section between P0 and P1 that is at the calculated distance
    point_on_section = P0 + final_dist * direction
    return point_on_section
end




function intersect_line_plane(triangle_point1, triangle_point2, triangle_point3, line_point1, line_point2)
    # Calculate the normal vector of the plane
    v1 = triangle_point2 - triangle_point1
    v2 = triangle_point3 - triangle_point1
    normal = cross(v1, v2)
    normal = normal / norm(normal)  # Normalize the normal vector

    # Parametric equation of the line: P(t) = line_point1 + t * (line_point2 - line_point1)
    line_dir = line_point2 - line_point1

    # Calculate the parameter t where the line intersects the plane
    t = dot(normal, (triangle_point1 - line_point1)) / dot(normal, line_dir)

    # Calculate the intersection point
    intersection_point = line_point1 + t * line_dir

    return intersection_point
end




function get_random_point_in_triangle(triangle_point1, triangle_point2, triangle_point3,w1, w2, w3)
    # Interpolate the points on the edges of the triangle
    point_on_edge1 = interpolate_points(triangle_point1, triangle_point2, w1)
    point_on_edge2 = interpolate_points(triangle_point2, triangle_point3, w2)

    # Interpolate the points on the edges to get a point inside the triangle
    point_in_triangle = interpolate_points(point_on_edge1, point_on_edge2, w3)

    return point_in_triangle
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


    edge_point_ap2=intersect_lines(apex1, base_point, edge_point_ap_common, apex2)
    edge_point_ap1=intersect_lines(apex2, base_point, edge_point_ap_common, apex1)


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
apex1 = [0.0, 0.0, 2.0]
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
    # ,Meshes.Point(edge_point_ap2...)
    # ,Meshes.Point(edge_point_ap1...)
    Meshes.Point(p3...)
    # ,Meshes.Point(edge_point_ap_common...)
    # ,Meshes.Point(base_point...)
    ,Meshes.Segment(Meshes.Point(vec_ap2...),Meshes.Point(apex2...))
    ,Meshes.Segment(Meshes.Point(vec_ap1...),Meshes.Point(apex1...))
    ,triangle
    ,Meshes.Segment(Meshes.Point(apex1...) ,Meshes.Point(apex2...))
    ,PolyArea([Meshes.Point(edge_point_ap2...), Meshes.Point(edge_point_ap1...)
    , Meshes.Point(edge_point_ap_common...), Meshes.Point(base_point...) ])

  ]#,Meshes.Point(proj2...),Meshes.Point((proj2+section2)...)

cc=collect(1:length(ob44))
# cc[1:4].=1
# cc[end]=2
viz(ob44,color=cc,pointsize=10.2,alpha=ones(length(ob44)).*0.6)#  ,alpha=[0.7,0.7,0.7,0.0]  , color=1:length(first_sv_tetrs)



"""
for a test we need to establish first weather the point is in the range of the first tetrahedron
if it is we need to check weather the line from the apex of the second tetrahedron to the point is intersecting 
the common base triangle of the tetrahedrons if not we do the same for other tetrahedron
"""
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





################# test distance_to_triangle_edge
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

P0
p1
P1

triangle = PolyArea([Meshes.Point(triangle_point1...), Meshes.Point(triangle_point2...), Meshes.Point(triangle_point3...)])
ob44=[
    triangle,
    p_curr
,Meshes.Point(triangle_point1...)
,Meshes.Point(triangle_point2...)
,Meshes.Segment(Meshes.Point(P0...) ,Meshes.Point(P1...)),Meshes.Point(P0...)  ]#,Meshes.Point(proj2...),Meshes.Point((proj2+section2)...)


viz(ob44,color=1:length(ob44),pointsize=10.2)#  ,alpha=[0.7,0.7,0.7,0.0]  , color=1:length(first_sv_tetrs)




















# #testing the function
# for i in 1:10000

#     triangle_point1 = rand(3).*10
#     triangle_point2 = rand(3).*10
#     triangle_point3 = rand(3).*10
#     triangle_point1[3]=0
#     triangle_point2[3]=0
#     triangle_point3[3]=0
#     P0_base=(triangle_point1+triangle_point2+triangle_point3)/3
#     P0 = [P0_base[1], P0_base[2], P0_base[3]]
#     P1 = rand(3).*2
#     P1[3]=0
#     p1=distance_to_triangle_edge(triangle_point1, triangle_point2, triangle_point3, P0, P1)    # Create the triangle object
#     p_curr=Meshes.Point(p1...)
#     triangle = PolyArea([Meshes.Point(triangle_point1...), Meshes.Point(triangle_point2...), Meshes.Point(triangle_point3...)])
    
#     if(in(p_curr,triangle))
#         print("*$(norm(p1-P0))*")
#     else
#         print("\n outside triangle_point1 $(triangle_point1) triangle_point2 $(triangle_point2) triangle_point3 $(triangle_point3) P0 $(P0) P1 $(P1) p1 $(p1)\n")
#     end

# end



# ######## not full test of is_point_on_segment
# for i in 1:100
#     P0=rand(3).*10
#     P1=rand(3).*10
#     p_in=interpolate_points(P0, P1, rand())
#     p_out=elongate_section(P0, P1,  rand()*10)
#     @test is_point_on_segment(p_in[1], p_in[2], p_in[3], P0[1], P0[2], P0[3], P1[1], P1[2], P1[3])>1e-8
#     print("out $(is_point_on_segment(p_out[1], p_out[2], p_out[3], P0[1], P0[2], P0[3], P1[1], P1[2], P1[3])<1e-8)  ")
#     # @test is_point_on_segment(p_out[1], p_out[2], p_out[3], P0[1], P0[2], P0[3], P1[1], P1[2], P1[3])<1e-8
# end    



# #in line
# is_point_on_segment(1.0, 0.0,0.0
# , 0.0, 0.0, 0.0
# , 2.0, 0.0, 0.0)

# #outside line
# is_point_on_segment(4.0, 0.0,0.0
# , 0.0, 0.0, 0.0
# , 2.0, 0.0, 0.0)



# function make_lines_parallel(line1_point1, line1_point2)
#     # Calculate the direction vector of the first line
#     direction1 = line1_point2 - line1_point1
    
#     # Choose a scalar multiple for the second line's direction vector
#     scalar = 2.0  # You can choose any non-zero scalar
    
#     # Define the second line's points to make it parallel to the first line
#     line2_point1 = [7.0, 8.0, 9.0]  # Arbitrary starting point for the second line
#     line2_point2 = line2_point1 + scalar * direction1
    
#     return (line2_point1, line2_point2)
# end

# function intersect_lines(line1_point1, line1_point2, line2_point1, line2_point2)
#     # Direction vectors of the lines
#     d1 = line1_point2 - line1_point1
#     d2 = line2_point2 - line2_point1
    
#     # Coefficients for the system of equations
#     A = [d1 -d2]
#     b = line2_point1 - line1_point1
    
#     # Solve the system of equations A * [t; s] = b
#     ts = A \ b
    
#     # Calculate the intersection point
#     intersection_point = line1_point1 + ts[1] * d1
    
#     return intersection_point
# end

# line1_point1 = [1.0, 2.0, 3.0]
# line1_point2 = [4.0, 5.0, 6.0]
# line2_point1, line2_point2 = make_lines_parallel(line1_point1, line1_point2)

# a=Meshes.intersects(Meshes.Segment(Meshes.Point(line1_point1...), Meshes.Point(line1_point2...))
# , Meshes.Segment(Meshes.Point(line2_point1...), Meshes.Point(line2_point2...)))



# intersect_lines(line1_point1, line1_point2, line2_point1, line2_point2)









# ########## test projecting on a plane and back

# # Define 4 points on the same plane
# point1 = (1.0, 2.0, 0.0)
# point2 = (4.0, 5.0, 0.0)
# point3 = (7.0, 8.0, 0.0)
# point4 = (10.0, 11.0, 0.0)

# # Transform to 2D
# p1_2d, p2_2d, p3_2d, p4_2d, basis1, basis2, normal, origin = transform_to_2d(point1, point2, point3, point4)

# println("2D Points: ", p1_2d, ", ", p2_2d, ", ", p3_2d, ", ", p4_2d)

# # Transform back to 3D
# p1_3d, p2_3d, p3_3d, p4_3d = transform_to_3d(p1_2d, p2_2d, p3_2d, p4_2d, basis1, basis2, normal, origin)

# println("3D Points: ", p1_3d, ", ", p2_3d, ", ", p3_3d, ", ", p4_3d)

# # Get random points within a triangle so within the same plane
# for i in 1:300
#     t1,t2,t3=(rand(3).*10).-5, (rand(3).*10).-5, (rand(3).*10).-5
#     p1=get_random_point_in_triangle(t1,t2,t3,rand(),rand(),rand())
#     p2=get_random_point_in_triangle(t1,t2,t3,rand(),rand(),rand())
#     p3=get_random_point_in_triangle(t1,t2,t3,rand(),rand(),rand())
#     p4=get_random_point_in_triangle(t1,t2,t3,rand(),rand(),rand())

#     p1_2d, p2_2d, p3_2d, p4_2d, basis1, basis2, normal, origin = transform_to_2d(p1,p2, p3, p4)
#     p1_3d, p2_3d, p3_3d, p4_3d = transform_to_3d(p1_2d, p2_2d, p3_2d, p4_2d, basis1, basis2, normal, origin)
#     tol = 1e-3

#     # Test if the original points are approximately equal to the transformed points
#     @test isapprox(collect(p1), collect(p1_3d), atol=tol)
#     @test isapprox(collect(p2), collect(p2_3d), atol=tol)
#     @test isapprox(collect(p3), collect(p3_3d), atol=tol)
#     @test isapprox(collect(p4), collect(p4_3d), atol=tol)
# end



# A11=A11+1e-8
# A22=A22+1e-8
# A32=A32+1e-8

# function solve_system(A, b)
#     # Calculate the transpose of A
#     At = transpose(A)
    
#     # Calculate At * A
#     AtA = At * A
    
#     # Calculate At * b
#     Atb = At * b
    
#     # Solve the normal equations AtA * x = Atb using Gaussian elimination
#     n = size(AtA, 1)
#     x = zeros(n)
    
#     # Forward elimination
#     for i in 1:n
#         # Make the diagonal contain all 1's
#         factor = AtA[i, i]
#         for j in i:n
#             AtA[i, j] /= factor
#         end
#         Atb[i] /= factor
        
#         # Make the elements below the pivot elements equal to zero
#         for k in i+1:n
#             factor = AtA[k, i]
#             for j in i:n
#                 AtA[k, j] -= factor * AtA[i, j]
#             end
#             Atb[k] -= factor * Atb[i]
#         end
#     end
    
#     # Back substitution
#     for i in n:-1:1
#         x[i] = Atb[i]
#         for j in i+1:n
#             x[i] -= AtA[i, j] * x[j]
#         end
#     end
    
#     # Return the solution as a vector of length 2
#     return x[1:2]
# # end

# # Example values
# A = [1.1899392927700754 -0.4065505736620587; 2.788036939688229 0.6301590131058837; -0.1679554786293158 0.24056935957014544]
# b = [0.37156938521702476, -0.12740305975032185, -0.16795547862931584]
# # A=[-1.660047357202818 5.568261742494198; -3.5766273182669384 2.640922438248098; 0.0 -0.0] 
# # b=[-0.631845174661458, -1.1728125504432052, 0.0] 
# # Solve the system
# solution = solve_system(A, b)

# ts = A \ b