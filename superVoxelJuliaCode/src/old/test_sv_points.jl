"""
main idea is to check weather a line that goes from sv center to control point is inside the supervoxel
In order to test it we need to define a mesh which surface is defined by the set of control points

Plan is to define in meshes.jl the shape using a set of tetrahydras - we then need to check weather they intersect (they should not)
we can then later also test sampling scheme and check if a sampled point is inside any of the tetrahedron 
meshes additionally supply the visualization functionalities

!! important we assume that weights are in the range between -1 and 1 (so basically after tanh)
"""

using Revise
using Meshes
using LinearAlgebra
using GLMakie
using Combinatorics
using SplitApplyCombine
using CUDA
using Combinatorics


includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/initialize_sv.jl")
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/sv_points/points_from_weights.jl")




# function flip_num(base_ind, tupl, ind)
#     arr = collect(tupl)
#     if (arr[ind] == base_ind[ind])
#         arr[ind] = base_ind[ind] + 1
#     else
#         arr[ind] = base_ind[ind]
#     end
#     return Tuple(arr)
# end

# """
# we can identify the line between two corners that go obliquely through the wall of the cube
# it connects points that has 2 coordinates diffrent and one the same 
# we can also find a point in the middle so it will be in lin_x if this common index is 1 and in lin_y if it is 2 and lin_z if 3
# next if we have 1 it is pre and if 2 post
#     control_points first dimension is lin_x, lin_y, lin_z, oblique
# """
# function get_linear_between(base_ind, ind_1, ind_2)
#     if (ind_1[1] == ind_2[1])
#         return control_points[ind_1[1], base_ind[2], base_ind[3], 1, :]
#     end
#     if (ind_1[2] == ind_2[2])
#         return control_points[base_ind[1], ind_1[2], base_ind[3], 2, :]
#     end

#     return control_points[base_ind[1], base_ind[2], ind_1[3], 3, :]
# end

# """
# get tetrahedron on the basis of the chosen corner (which is defined by some oblique point)
# control_points first dimension is lin_x, lin_y, lin_z, oblique
# """
# function get_tetr_a(control_points, base_ind, corner)
#     sv_center = Meshes.Point3(sv_centers[base_ind[1], base_ind[2], base_ind[3], :])
#     p_a = flip_num(base_ind, corner, 1)
#     p_b = flip_num(base_ind, corner, 2)
#     p_c = flip_num(base_ind, corner, 3)


#     p_ab = get_linear_between(base_ind, p_a, p_b)
#     p_ac = get_linear_between(base_ind, p_a, p_c)
#     p_bc = get_linear_between(base_ind, p_b, p_c)

#     p_a = Meshes.Point3(control_points[p_a[1], p_a[2], p_a[3], 4, :])
#     p_b = Meshes.Point3(control_points[p_b[1], p_b[2], p_b[3], 4, :])
#     p_c = Meshes.Point3(control_points[p_c[1], p_c[2], p_c[3], 4, :])


#     p_ab = Meshes.Point3(p_ab)
#     p_ac = Meshes.Point3(p_ac)
#     p_bc = Meshes.Point3(p_bc)

#     corner = Meshes.Point3(control_points[corner[1], corner[2], corner[3], 4, :])

#     # print("coo corner $(corner) p_a $(p_a) p_b $(p_b) p_c $(p_c) \n ")

#     return [
#         Meshes.Tetrahedron(sv_center, corner, p_a, p_ab), Meshes.Tetrahedron(sv_center, corner, p_ab, p_b)
#         , Meshes.Tetrahedron(sv_center, corner, p_b, p_bc), Meshes.Tetrahedron(sv_center, corner, p_bc, p_c)
#         , Meshes.Tetrahedron(sv_center, corner, p_a, p_ac), Meshes.Tetrahedron(sv_center, corner, p_ac, p_c)]
# end


radiuss = Float32(4.0)
diam = radiuss * 2
num_weights_per_point = 6
a = 36
image_shape = (a, a, a)

example_set_of_svs = initialize_centers_and_control_points(image_shape, radiuss)
sv_centers, control_points, tetrs, dims = example_set_of_svs
size(tetrs)







#### getting data about first supervoxel (first 24 tetrahedrons in tetrs)
function fill_tetrahedron_data(tetr_dat, sv_centers, control_points, index)
    center = map(axis -> sv_centers[Int(tetr_dat[index, 1, 1]), Int(tetr_dat[index, 1, 2]), Int(tetr_dat[index, 1, 3]), axis], [1, 2, 3])
    corners = map(corner_num ->
            map(axis -> control_points[Int(tetr_dat[index, corner_num, 1]), Int(tetr_dat[index, corner_num, 2]), Int(tetr_dat[index, corner_num, 3]), Int(tetr_dat[index, corner_num, 4]), axis], [1, 2, 3]), [2, 3, 4])
    corners = [center, corners...]
    return corners
end

function get_tetrahedrons_from_corners(corners)
    points = map(el -> Meshes.Point((el[1], el[2], el[3])), corners)
    return Meshes.Tetrahedron(points...)
end
first_sv_tetrs = map(index -> fill_tetrahedron_data(tetrs, sv_centers, control_points, index), 1:get_num_tetr_in_sv())
first_sv_tetrs = map(get_tetrahedrons_from_corners, first_sv_tetrs)

viz(first_sv_tetrs, color=1:length(first_sv_tetrs))



# # control_points first dimension is lin_x, lin_y, lin_z, oblique
# # weights=zeros((dims_plus[1],dims_plus[2],dims_plus[3],num_weights_per_point))
# weights = rand(dims_plus[1], dims_plus[2], dims_plus[3], num_weights_per_point)
# weights = weights .- 0.5
# weights = (weights) .* 100
# weights = tanh.(weights * 0.02)


# #reshape for broadcast


# threads=(2,2,2)
# blocks=(2,2,2)

# control_points=call_apply_weights_to_locs_kern(CuArray(control_points),CuArray(weights),radius,threads,blocks)

# control_points=Array(control_points)



function get_tetrahedrons_of_sv(base_ind)
    return [
        get_tetr_a(control_points, base_ind, (base_ind[1], base_ind[2], base_ind[3])), get_tetr_a(control_points, base_ind, (base_ind[1] + 1, base_ind[2] + 1, base_ind[3])), get_tetr_a(control_points, base_ind, (base_ind[1], base_ind[2] + 1, base_ind[3] + 1)), get_tetr_a(control_points, base_ind, (base_ind[1] + 1, base_ind[2], base_ind[3] + 1))
    ]
end #get_tetrahedrons_of_sv

tetrs = [
    get_tetrahedrons_of_sv((1, 1, 1))
    # ,get_tetrahedrons_of_sv((2,1,1))
    # ,get_tetrahedrons_of_sv((1,2,1))
    # ,get_tetrahedrons_of_sv((1,1,2))

    # ,get_tetrahedrons_of_sv((1,2,2))
    # ,get_tetrahedrons_of_sv((2,2,1))
    # ,get_tetrahedrons_of_sv((1,2,1))
    # ,get_tetrahedrons_of_sv((2,2,2))

]
tetrs = collect(Iterators.flatten(tetrs))
tetrs = collect(Iterators.flatten(tetrs))
viz(tetrs, color=1:length(tetrs))

# tetrs=cat(tetrs,tetrs_b,tetrs_c,tetrs_d,dims=1)

function get_faces_verticies(t)
    verts = collect(t.vertices)
    vert_combs = collect(combinations(verts, 3))
    faces = map(pointss -> Meshes.Ngon(pointss[1], pointss[2], pointss[3]), vert_combs)
    return verts, faces
end

all_faces_verticies = map(get_faces_verticies, tetrs)
all_verts, all_faces = invert(all_faces_verticies)
all_verts = collect(Iterators.flatten(all_verts))
all_faces = collect(Iterators.flatten(all_faces))



function isIn_any_vert_or_face(point, verts, faces)
    for v in verts
        if v == point
            return true
        end
    end
    for f in faces
        if point ∈ f
            print("*")
            return true

        end
    end
    return false
end

pp = sv_centers[1, 1, 1, :]
pp = Meshes.Point3(pp[1] + 0.12, pp[2] + 0.1, pp[3] + 0.123)
pp

isIn_any_vert_or_face(pp, all_verts, all_faces)


pp ∈ all_faces[5]


diam = 3
for i in range(Int(diam), stop=Int(diam) * 2, step=0.2)
    for j in range(Int(diam), stop=Int(diam) * 2, step=0.2)
        for k in range(Int(diam), stop=Int(diam) * 2, step=0.2)
            # print("i $(i) j $(j) k $(k) \n")
            pp = Meshes.Point3(1, j, k)

            isIn_any_vert_or_face(pp, all_verts, tetrs)

            # print("is_point_in_array $(is_point_in_array(i,j,k,control_points)) \n")
        end
    end
end


get_faces_verticies(t1)


tetrs = collect(Iterators.flatten(tetrs))
viz(tetrs, color=1:length(tetrs))


trianglee = Meshes.Ngon(Meshes.Point3(5.0, 4.0, 4.0), Meshes.Point3(6.0, 4.0, 4.0), Meshes.Point3(5.0, 7.0, 4.0))




t1 = tetrs[1]


verts = collect(t1.vertices)
vert_combs = collect(combinations(verts, 3))
# combinationss = collect(Combinatorics.combinations([1,2,3,4], 3))


pp1 = Meshes.Point3(5.1, 4.0, 4.0)
sv_center ∈ t1
pp1 ∈ t1
pp1 ∈ trianglee

"""
In order to test weather our tetrahedrons with random weights are still valid we can sample points from the volume
and check if they are inside any of the tetrahedrons; each point should be in just one tetrahedron ; with the exception of their sides and vertices
so we need to 
1) sample points in a volume where all points should be covered by some tetrahedron
2) exclude those points that are either in triangles (walls of tetrahedrons) or in the vertices of the tetrahedrons
3) check if the points are in multiple tetrahedrons - each should be in exactly one - in this case all is covered as it should
"""


"""
we need to check weather points in the cube defined by the oblique points  are all in some tetrahedron
also we need to check weather they are not in multiple tetrahedrons at once; 
Hovewer tetrahedron walls are adjacent so we need to ignore alll points that are on the walls of tetrahedrons
and all control points as those are spetial places where we allow for a point to be assigned to multiple
    tetrahedrons
"""




# function dummy_fun(pa,pb)
#     return pa[1]+1,pb[1,1]
# end

# arra=[[1.0],[1.0],[1.0]]
# arrb=[[[1.0,1.0]],[[1.0,1.0]],[[1.0,1.0]]]

# broadcast((a, b) -> dummy_fun(a, b), arra, arrb)

# weights=reshape(weights,(Int(prod(collect(dims_plus))),num_weights_per_point  ))
# sizz=size(control_points)
# control_points=reshape(control_points,(sizz[1]*sizz[2]*sizz[3],sizz[4],sizz[5]  ))

# combinationss = collect(combinations(points, 3))

# to_go=true

# tetrs=[]

# t1=Meshes.Tetrahedron(sv_center
# ,combinationss[1][1]
# ,combinationss[1][2]
# ,combinationss[1][3]  )
# push!(tetrs,t1)



# while to_go
#     for i_perm in collect(enumerate(combinationss))
#         i,perm=i_perm

#         t2=Meshes.Tetrahedron(sv_center
#         ,perm[1]
#         ,perm[2]
#         ,perm[3]  )
#         interss=false

#         for t in tetrs
#             if Meshes.intersects(t, t2)
#                 interss=true
#             end
#         end

#         if !interss
#             print(t2)
#             push!(tetrs,t2)
#             deleteat!(combinationss, i)  # This will remove the element at index 3
#             to_go=true
#         end
#         push!(tetrs,t2)

#     end
# end


# tetr_1=Meshes.Tetrahedron(Meshes.Point3(1,1,1),Meshes.Point3(1,2,1),Meshes.Point3(1,1,2),Meshes.Point3(2,2,2))
# tetr_2=Meshes.Tetrahedron(Meshes.Point3(1,1,1),Meshes.Point3(1,2,1),Meshes.Point3(2,1,2),Meshes.Point3(2,2,2))

# I = Meshes.intersects(tetr_1, tetr_2)
# tetr_1 ∩ tetr_2
# viz(tetr_2)

"""
idea now that we want to construct our supervoxel volume from tetrahedrons where all tetrahedtons will have an apex
in the sv center it is non obious how to construct the tetrahedrons so they do not intersect - we will need to check it
simple solution (slow but simple) is to get random 3 points and sv center and check if it intersects with any of the tetrahedrons
that are already present in the volume do it so long as we have no longer any tetrahedron that meet our criterion

if we will have futher problems with tetrahedrons intersections we can probably deal with it by checking intersection of their walls if none 
    intersects then tetrahedron do not intersect
"""


# lin_x_point_pre=Meshes.Point3(lin_x[1,1,1,:])
# lin_x_point_post=Meshes.Point3(lin_x[2,1,1,:])

# lin_y_point_pre=Meshes.Point3(lin_y[1,1,1,:])
# lin_y_point_post=Meshes.Point3(lin_y[1,2,1,:])

# lin_z_point_pre=Meshes.Point3(lin_z[1,1,1,:])
# lin_z_point_post=Meshes.Point3(lin_z[1,1,2,:])

# lin_x_point_pre_add=Meshes.Point3(lin_x_add[2,1,1,:])
# # lin_x_point_pre_add=Meshes.Point3(lin_x_add[1,2,2,:])
# lin_x_point_post_add=Meshes.Point3(lin_x_add[2,2,2,:])

# lin_y_point_pre_add=Meshes.Point3(lin_y_add[1,2,1,:])
# lin_y_point_post_add=Meshes.Point3(lin_y_add[2,2,2,:])

# lin_z_point_pre_add=Meshes.Point3(lin_z_add[1,1,2,:])
# lin_z_point_post_add=Meshes.Point3(lin_z_add[2,2,2,:])



# oblique_point_1=Meshes.Point3(oblique[1,1,1,:])
# oblique_point_2=Meshes.Point3(oblique[2,1,1,:])
# oblique_point_3=Meshes.Point3(oblique[1,2,1,:])
# oblique_point_4=Meshes.Point3(oblique[1,1,2,:])
# oblique_point_5=Meshes.Point3(oblique[1,2,2,:])
# oblique_point_6=Meshes.Point3(oblique[2,2,1,:])
# oblique_point_7=Meshes.Point3(oblique[2,1,2,:])
# oblique_point_8=Meshes.Point3(oblique[2,2,2,:])


#take a corner and can modify just one of the coordinates will get 3 combinations


#sv_center,
# points=[lin_x_point_pre,lin_x_point_post
# ,lin_y_point_pre,lin_y_point_post
# ,lin_z_point_pre,lin_z_point_post
# ,oblique_point_1,oblique_point_2,oblique_point_3,oblique_point_4
# ,oblique_point_5,oblique_point_6,oblique_point_7,oblique_point_8]

# # we will create a bunch of tetrahedrons on the basis of the points above
# function is_point_in_array(x,y,z,arr)
#     p=Meshes.Point(x,y,z)
#     sizz=size(arr)
#     ob= reshape(arr,sizz[1]*sizz[2]*sizz[3],3)
#     ob=map(el-> Meshes.Point(el[1],el[2],el[3]),eachrow(ob))
#     # map(el-> print(el),eachrow(ob))
#     res=p in ob
#     return res
# end     

# x=9.0
# y=6.0
# z=9.0
# is_point_in_array(x,y,z,oblique)
# is_point_in_array(x,y,z,lin_x)
# is_point_in_array(x,y,z,lin_y)
# is_point_in_array(x,y,z,lin_z)

# is_point_in_array(x,y,z,lin_z_add)
# is_point_in_array(x,y,z,lin_z_add)
# is_point_in_array(x,y,z,lin_z_add)


function project_point_onto_line(a::Tuple{Float64,Float64,Float64},
    b::Tuple{Float64,Float64,Float64},
    c::Tuple{Float64,Float64,Float64})
    # Convert points to vectors
    a_vec = [a...]
    b_vec = [b...]
    c_vec = [c...]

    # Calculate the vector from b to c
    bc_vec = c_vec - b_vec

    # Calculate the vector from b to a
    ba_vec = a_vec - b_vec

    # Calculate the projection of ba_vec onto bc_vec
    t = dot(ba_vec, bc_vec) / dot(bc_vec, bc_vec)

    # Calculate the coordinates of the projection point
    proj_point = b_vec + t * bc_vec

    # Convert the result to a tuple and return
    return Tuple(proj_point)
end

a = (1.0, 2.0, 3.0)
b = (4.0, 5.0, 6.0)
c = (7.0, 8.0, 9.0)

proj_point = project_point_onto_line(a, b, c)