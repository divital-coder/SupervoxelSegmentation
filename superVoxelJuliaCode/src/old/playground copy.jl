#take form https://github.com/EnzymeAD/Enzyme.jl/issues/428


using CUDA, Enzyme, Test
using Pkg
# Pkg.add(url="https://github.com/EnzymeAD/Enzyme.jl.git")

function mul_kernel(A)
    i = threadIdx().x
    shared_arr = CuStaticSharedArray(Float32, (256,3))
    A[i] = 1.0

    return nothing
end

function grad_mul_kernel(A, dA)
    Enzyme.autodiff_deferred(Enzyme.Reverse,Enzyme.Const(mul_kernel), Const, Duplicated(A, dA))
    return nothing
end

A = CUDA.ones(64,)
@cuda threads=length(A) mul_kernel(A )
A = CUDA.ones(64,)
dA = similar(A)
dA .= 1
@cuda threads=length(A) grad_mul_kernel(A, dA)
# @test all(dA .== 2)