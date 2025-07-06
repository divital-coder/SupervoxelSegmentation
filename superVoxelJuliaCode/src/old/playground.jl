#take form https://github.com/EnzymeAD/Enzyme.jl/issues/428
using CUDA, Enzyme, Test

function mul_kernel(A, shared)
    i = threadIdx().x
    if i <= length(A)
        shared[i] = A[i] * A[i]
        A[i] = shared[i]
    end
    return nothing
end

function grad_mul_kernel(A, dA,sh,d_sh)
    Enzyme.autodiff_deferred(Enzyme.Const(mul_kernel), Const, Duplicated(A, dA), Duplicated(sh, d_sh))
    return nothing
end

A = CUDA.ones(64,)
@cuda threads=length(A) shmem=64*4 mul_kernel(A,CuDynamicSharedArray(Float32, 64) )
A = CUDA.ones(64,)
dA = similar(A)
dA .= 1
@cuda threads=length(A) shmem=64*4 grad_mul_kernel(A, dA,CuDynamicSharedArray(Float32, 64),CuDynamicSharedArray(Float32, 64))
@test all(dA .== 2)