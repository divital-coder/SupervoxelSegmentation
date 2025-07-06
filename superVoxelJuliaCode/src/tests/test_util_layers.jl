using ChainRulesCore, Zygote, CUDA, Enzyme
using KernelAbstractions
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote
using LinearAlgebra
using Revise
using cuTENSOR, TensorOperations, VectorInterface, Test
import Lux.Experimental.ADTypes  # Add this line

includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/model/util_layers.jl")



function infer_model(tstate, model, data)
    y_pred, st = Lux.apply(model, CuArray(data), tstate.parameters, tstate.states)

    return y_pred, st
end

function test1()
    data = CuArray(rand(Float32, 5, 7, 3, 5,4))

    minimodel = TensorOpLayer_str((7,5,4), :(res[i,k] := x[i,j,k,l,n] * ps.P[j,l,n]))


       
    rng = Random.default_rng()
    opt = Optimisers.Lion(0.000002)
    

    dev = gpu_device()
    ps, st = Lux.setup(rng, minimodel)|> dev
    tstate = Lux.Training.TrainState(minimodel,ps, st, opt)

    
    pred,st=infer_model(tstate, minimodel, data)

    @test size(pred)==(5,3)
end    

function test2()
    data = CuArray(rand(Float32, 5, 4, 3, 5,4))
    minimodel = TensorOpLayer_str((5,4), :(res[i,j,k] := x[i,j,k,l,n] * ps.P[l,n]))
    
       
    rng = Random.default_rng()
    opt = Optimisers.Lion(0.000002)
    dev = gpu_device()
    ps, st = Lux.setup(rng, minimodel)|> dev
    tstate = Lux.Training.TrainState(minimodel,ps, st, opt)
    
    pred,st=infer_model(tstate, minimodel, data)
    @test size(pred)==(5,4,3)
end    

"""
just check is backprop compile
"""
function test3()
    data = CuArray(rand(Float32, 5, 7, 3, 5,4))
    minimodel = TensorOpLayer_str((7,5,4), :(res[i,k] := x[i,j,k,l,n] * ps.P[j,l,n]))
    
    
    
    rng = Random.default_rng()
    opt = Optimisers.Lion(0.000002)
    dev = gpu_device()
    ps, st = Lux.setup(rng, minimodel)|> dev
    tstate = Lux.Training.TrainState(minimodel,ps, st, opt)

    vjp = ADTypes.AutoZygote()  # Modify this line
    function loss_function_dummy(model, ps, st, data)
        y_pred, st = Lux.apply(model, data, ps, st)
        return sum(y_pred), st, ()
    end
    _, loss, _, tstate = Lux.Training.single_train_step!(vjp, loss_function_dummy, CuArray(data), tstate)
    

end

test1()
test2()
test3()


# α = randn()
# A = randn(5, 5, 5, 5, 5, 5)
# B = randn(5, 5, 5)
# C = randn(5, 5, 5)
# D = zeros(5, 5, 5)
# @cutensor begin
#     D[a, b, c] = A[a, e, f, c, f, g] * B[g, b, e] + α * C[c, a, b]
# end
# data = CuArray(rand(Float32, 5, 4, 3, 5,4))
# # minimodel = TensorOpLayer_str((5,4), :(res[i,j,k] := x[i,j,k,l,n] * ps.P[l,n]))

# # data = CuArray(rand(Float32, 5, 7, 3, 5,4))
# # minimodel = TensorOpLayer_str((7,5,4), :(res[i,k] := x[i,j,k,l,n] * ps.P[j,l,n]))



# # rng = Random.default_rng()
# # opt = Optimisers.Lion(0.000002)
# # tstate = Lux.Experimental.TrainState(rng, minimodel, opt)
# # tstate = cu(tstate)#casting to cuda array if we chosen to use nvidia gpu
# # vjp = Lux.Experimental.ADTypes.AutoZygote()
# # function loss_function_dummy(model, ps, st, data)
# #     y_pred, st = Lux.apply(model, data, ps, st)
# #     return sum(y_pred), st, ()
# # end
# # _, loss, _, tstate = Lux.Experimental.single_train_step!(vjp, loss_function_dummy, CuArray(data), tstate)



# # pred,st=infer_model(tstate, minimodel, data)
# # pred
# # pred


# # # size(pred)

# # macro dummy(parsed_ex)
# #     return esc(quote
# #         $parsed_ex
# #         return res
# #     end)
# # end

# # function aaa()
#     P=CuArray(rand(rng,Float32,5,4,4))
#     x=CuArray(rand(rng,Float32,5,4,3,5,4))
#     ex=:(res[i,k] := x[i,j,k,l,n] * P[l,n,j])
#     parser=TensorOperations.tensorparser(ex,:allocator=>TensorOperations.CUDAAllocator(),:backend=>TensorOperations.cuTENSORBackend())
#     parsed_ex=parser(ex)

#     string(parsed_ex.args[2])


#     string(parsed_ex.args[2])


# #res[i,j,k] := x[i,j,k,l,n] * P[l,n])    
# pA=((1, 2, 3), (4, 5))
# pB=((1, 2), ())
# pC=((1, 2, 3), ())


# T_res=Float32
# res = TensorOperations.tensoralloc_contract(T_res, x, ((1, 2, 3), (4, 5)), false, P, ((1, 2), ()), false, ((1, 2, 3), ()), Val{false}(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory, CUDA.DeviceMemory, CUDA.DeviceMemory}())
# res = TensorOperations.tensorcontract!(res, x, ((1, 2, 3), (4, 5)), false, P, ((1, 2), ()), false, ((1, 2, 3), ()), VectorInterface.One(), VectorInterface.Zero(), TensorOperations.cuTENSORBackend(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory, CUDA.DeviceMemory, CUDA.DeviceMemory}())


# #    ex=:(res[i,k] := x[i,j,k,l,n] * P[l,n,j])

# quote
#     var"##T_res#228" = TensorOperations.promote_contract(TensorOperations.scalartype(x), TensorOperations.scalartype(P))
#     res = TensorOperations.tensoralloc_contract(var"##T_res#228", x, ((1, 3), (2, 4, 5)), false, P, ((3, 1, 2), ()), false, ((1, 2), ()), Val{false}(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory, CUDA.DeviceMemory, CUDA.DeviceMemory}())
#     res = TensorOperations.tensorcontract!(res, x, ((1, 3), (2, 4, 5)), false, P, ((3, 1, 2), ()), false, ((1, 2), ()), One(), Zero(), TensorOperations.cuTENSORBackend(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory, CUDA.DeviceMemory, CUDA.DeviceMemory}())
# end

#     # return @dummy(parsed_ex)
# # end

# # aaa()

# function extract_double_brackets(s::String)
#     pattern = r"\(\(.*?\)\)"
#     matches = eachmatch(pattern, s)
#     return [match.match for match in matches]
# end

# # Example usage
# s = """res = TensorOperations.tensoralloc_contract(var"##T_res#228", x, ((1, 3), (2, 4, 5)), false, P, ((3, 1, 2), ()), false, ((1, 2), ()), Val{false}(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory, CUDA.DeviceMemory, CUDA.DeviceMemory}())"""
# extracted_parts = extract_double_brackets(s)
# println(extracted_parts)



# function parse_tuple_from_string(s::String)
#     # Parse the string into an expression
#     expr = Meta.parse(s)
    
#     # Evaluate the expression to get the tuple
#     result = eval(expr)
    
#     return result
# end

# # Example usage
# s = "((1, 3), (2, 4, 5))"
# parsed_tuple = parse_tuple_from_string(s)
# println(parsed_tuple)


# # Function """ function extract_double_brackets(s::String)
# #     pattern = r"\(\(.*?\)\)"
# #     matches = matchall(Regex(pattern), s)
# #     return matches
# # end

# # # Example usage
# # s = """res = TensorOperations.tensoralloc_contract(var"##T_res#228", x, ((1, 3), (2, 4, 5)), false, P, ((3, 1, 2), ()), false, ((1, 2), ()), Val{false}(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory, CUDA.DeviceMemory, CUDA.DeviceMemory}())"""
# # extracted_parts = extract_double_brackets(s)
# # println(extracted_parts)""" should extract all parts of the string that are surrounded by double brackets so "((" from left and "))" from right. but give error """ MethodError: no method matching Regex(::Regex)

# # Closest candidates are:
# #   Regex(::AbstractString)""" in line """matches = matchall(Regex(pattern), s)"""