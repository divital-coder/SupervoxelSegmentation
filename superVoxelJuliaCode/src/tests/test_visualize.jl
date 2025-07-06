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
using Test


using Logging
using Interpolations
using KernelAbstractions, Dates
# using KernelGradients
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote
using Lux, Random, Optimisers, Zygote
using LinearAlgebra, Statistics
using Revise
using Meshes





for i in 1:100

    a = rand(3)
    b = rand(3)
    c = rand(3)
    d = rand(3)
    # a = [0.0,0.0,0.0]
    # b = [1.0,0.0,0.0]
    # c = [0.0,1.0,0.0]
    # d = [0.0,0.0,1.0]

    p = rand(3)
    loc_res = is_point_in_tetrahedron(hcat(a, b, c, d), p)

    points = [Meshes.Point(a...), Meshes.Point(b...), Meshes.Point(c...), Meshes.Point(d...)]
    t = Meshes.Tetrahedron(points...)
    meshes_res = Meshes.Point(p...) ∈ t
    @test meshes_res == loc_res
    # print("\n $(meshes_res) loc_res $(loc_res)  \n")
end

##### testing kernel in typical conditions 

for i in 1:100

    radiuss = 4
    a = [4.0, 4.0, 4.0]
    b = (rand(3) .* 4) .+ 2
    c = (rand(3) .* 4) .+ 2
    d = (rand(3) .* 4) .+ 2
    arr = hcat(a, b, c, d)
    arr = transpose(arr)
    arr = reshape(arr, 1, 4, 3)


    out_image_to_vis = call_are_indicies_in_tetrahedron_kernel(CuArray(arr), (16, 16, 16), radiuss)

    out_image_to_vis = Array(out_image_to_vis)
    cartesian_indices = CartesianIndices(out_image_to_vis)
    for cart in cartesian_indices
        is_ok_gold = is_point_in_tetrahedron(hcat(a, b, c, d), [cart.I...])
        is_ok_my = out_image_to_vis[cart] > 0
        @test is_ok_gold == is_ok_my
    end
end


# a = rand(3)
# b = rand(3)
# c = rand(3)
# d = rand(3)
# points = hcat(a, b, c, d)


# A = rand(3)
# B = rand(3)
# C = rand(3)
# D = rand(3)

# mem_1_b = zeros(Float32, 3)

# mem_1_b[1] = (B[2] - A[2]) * (C[3] - A[3]) - (B[3] - A[3]) * (C[2] - A[2])
# mem_1_b[2] = (B[3] - A[3]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[3] - A[3])
# mem_1_b[3] = (B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1])

# cross(a, b) = [a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1]]

# cross(B - A, C - A)
# mem_1_b






# mem_1_b = zeros(Float32, 3)
# mem_2_b = zeros(Float32, 3)
# mem_3_b = zeros(Float32, 3)
# mem_4_b = zeros(Float32, 3)

# A = rand(3)
# B = rand(3)
# C = rand(3)
# D = rand(3)
# Q = rand(3)


# #cross(B - A, C - A)
# mem_1_b[1] = (B[2] - A[2]) * (C[3] - A[3]) - (B[3] - A[3]) * (C[2] - A[2])
# mem_1_b[2] = (B[3] - A[3]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[3] - A[3])
# mem_1_b[3] = (B[1] - A[1]) * (C[2] - A[2]) - (B[2] - A[2]) * (C[1] - A[1])

# # mem_2_b = cross(B - A, D - A)
# mem_2_b[1] = (B[2] - A[2]) * (D[3] - A[3]) - (B[3] - A[3]) * (D[2] - A[2])
# mem_2_b[2] = (B[3] - A[3]) * (D[1] - A[1]) - (B[1] - A[1]) * (D[3] - A[3])
# mem_2_b[3] = (B[1] - A[1]) * (D[2] - A[2]) - (B[2] - A[2]) * (D[1] - A[1])

# # mem_3_b = cross(C - A, D - A)
# mem_3_b[1] = (C[2] - A[2]) * (D[3] - A[3]) - (C[3] - A[3]) * (D[2] - A[2])
# mem_3_b[2] = (C[3] - A[3]) * (D[1] - A[1]) - (C[1] - A[1]) * (D[3] - A[3])
# mem_3_b[3] = (C[1] - A[1]) * (D[2] - A[2]) - (C[2] - A[2]) * (D[1] - A[1])

# # mem_4_b = cross(C - B, D - B)
# mem_4_b[1] = (C[2] - B[2]) * (D[3] - B[3]) - (C[3] - B[3]) * (D[2] - B[2])
# mem_4_b[2] = (C[3] - B[3]) * (D[1] - B[1]) - (C[1] - B[1]) * (D[3] - B[3])
# mem_4_b[3] = (C[1] - B[1]) * (D[2] - B[2]) - (C[2] - B[2]) * (D[1] - B[1])

# @test isapprox(mem_1_b, cross(B - A, C - A))
# @test isapprox(mem_2_b, cross(B - A, D - A))
# @test isapprox(mem_3_b, cross(C - A, D - A))
# @test isapprox(mem_4_b, cross(C - B, D - B))




# n1 = cross(B - A, C - A)
# n2 = cross(B - A, D - A)
# n3 = cross(C - A, D - A)
# n4 = cross(C - B, D - B)
# G = 0.25 * (A + B + C + D)


# dot(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

# P = [A, A, A, B]
# n = [n1, n2, n3, n4]

# α = [(G - P[i]) ⋅ n[i] for i in 1:4]



# for i in 1:4
#     if α[i] > 0
#         n[i] = -n[i]
#     end
# end


# P = [A, A, A, B]
# n = [mem_1_b, mem_2_b, mem_3_b, mem_4_b]

# α = zeros(Float32, 4)

# α[1] = ((((A[1] + B[1] + C[1] + D[1]) * 0.25) - A[1]) * mem_1_b[1] + (((A[2] + B[2] + C[2] + D[2]) * 0.25) - A[2]) * mem_1_b[2] + (((A[3] + B[3] + C[3] + D[3]) * 0.25) - A[3]) * mem_1_b[3])
# # α[2] = (G - P[2]) ⋅ n[2]
# α[2] = ((((A[1] + B[1] + C[1] + D[1]) * 0.25) - A[1]) * mem_2_b[1] + (((A[2] + B[2] + C[2] + D[2]) * 0.25) - A[2]) * mem_2_b[2] + (((A[3] + B[3] + C[3] + D[3]) * 0.25) - A[3]) * mem_2_b[3])

# # α[3] = (G - P[3]) ⋅ n[3]
# α[3] = ((((A[1] + B[1] + C[1] + D[1]) * 0.25) - A[1]) * mem_3_b[1] + (((A[2] + B[2] + C[2] + D[2]) * 0.25) - A[2]) * mem_3_b[2] + (((A[3] + B[3] + C[3] + D[3]) * 0.25) - A[3]) * mem_3_b[3])

# # α[4] = (G - P[4]) ⋅ n[4]
# α[4] = ((((A[1] + B[1] + C[1] + D[1]) * 0.25) - B[1]) * mem_4_b[1] + (((A[2] + B[2] + C[2] + D[2]) * 0.25) - B[2]) * mem_4_b[2] + (((A[3] + B[3] + C[3] + D[3]) * 0.25) - B[3]) * mem_4_b[3])



# dot(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
# P = [A, A, A, B]
# n = [mem_1_b, mem_2_b, mem_3_b, mem_4_b]
# β = [(Q - P[i]) ⋅ n[i] for i in 1:4]


# β = zeros(Float32, 4)

# β[1] = (Q[1] - A[1]) * mem_1_b[1] + (Q[2] - A[2]) * mem_1_b[2] + (Q[3] - A[3]) * mem_1_b[3]

# # β[2] = (Q - P[2]) ⋅ n[2]
# β[2] = (Q[1] - A[1]) * mem_2_b[1] + (Q[2] - A[2]) * mem_2_b[2] + (Q[3] - A[3]) * mem_2_b[3]

# # β[3] = (Q - P[3]) ⋅ n[3]
# β[3] = (Q[1] - A[1]) * mem_3_b[1] + (Q[2] - A[2]) * mem_3_b[2] + (Q[3] - A[3]) * mem_3_b[3]

# # β[4] = (Q - P[4]) ⋅ n[4]
# β[4] = (Q[1] - B[1]) * mem_4_b[1] + (Q[2] - B[2]) * mem_4_b[2] + (Q[3] - B[3]) * mem_4_b[3]

