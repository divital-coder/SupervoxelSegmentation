using ChainRulesCore, Zygote, CUDA, Enzyme
using KernelAbstractions
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote
using LinearAlgebra
using Revise
using cuTENSOR, TensorOperations, VectorInterface, Test

struct TensorOpLayer_str <: Lux.Lux.AbstractLuxLayer
    param_shape
    operation_expression::Expr
    act::String
end
#swish as defoult activation function
TensorOpLayer_str(param_shape,operation_expression) = TensorOpLayer_str(param_shape,operation_expression,"swish")

function Lux.initialparameters(rng::AbstractRNG, l::TensorOpLayer_str)
    P = rand(rng, Float32, l.param_shape...)
    return (P=P,)
end

function extract_double_brackets(s::String)
    pattern = r"\(\(.*?\)\)"
    matches = eachmatch(pattern, s)
    return [match.match for match in matches]
end

function parse_tuple_from_string(s)
    # Parse the string into an expression
    expr = Meta.parse(s)

    # Evaluate the expression to get the tuple
    result = eval(expr)

    return result
end


function Lux.initialstates(::AbstractRNG, l::TensorOpLayer_str)::NamedTuple
    ex = l.operation_expression
    parser = TensorOperations.tensorparser(ex, :allocator => TensorOperations.CUDAAllocator(), :backend => TensorOperations.cuTENSORBackend())
    parsed_ex = parser(ex)
    arg_tuples = extract_double_brackets(string(parsed_ex.args[3]))
    arg_tuples = map(parse_tuple_from_string, arg_tuples)
    return (PA=arg_tuples[1], PB=arg_tuples[2], PC=arg_tuples[3],act=l.act)
    # return (PA=arg_tuples[1], PB=arg_tuples[2], PC=arg_tuples[3],act=l.act,parsed_ex=parsed_ex,ex=ex)
    # return (a=1,b=2,c=3,act=l.act)

end


function (l::TensorOpLayer_str)(x, ps, st::NamedTuple)
    # current_time = Dates.now()
    # res = TensorOperations.tensoralloc_contract(Float32, x, st.PA, false, ps.P, st.PB, false, st.PC, Val{false}(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory,CUDA.DeviceMemory,CUDA.DeviceMemory}())
    
    # print("\n uuuuuuuu x $(size(x)) ps $(size(ps))  st.PA $(st.PA) st.PB $(st.PB)  st.PC $(st.PC) \n")
    res = TensorOperations.tensoralloc_contract(Float32, x, st.PA, false, ps.P, st.PB, false, st.PC, Val{false}(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory, CUDA.DeviceMemory, CUDA.DeviceMemory}())

    # res = TensorOperations.tensorcontract!(res, x, st.PA, false, ps.P, st.PB, false, st.PC, VectorInterface.One(), VectorInterface.Zero(), TensorOperations.cuTENSORBackend(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory,CUDA.DeviceMemory,CUDA.DeviceMemory}())
    res = TensorOperations.tensorcontract!(res, x, st.PA, false, ps.P, st.PB, false, st.PC,VectorInterface.One(), VectorInterface.Zero(), TensorOperations.cuTENSORBackend(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory,CUDA.DeviceMemory,CUDA.DeviceMemory}())
    

    if(st.act=="swish")
        res = swish.(res)
    end
    if(st.act=="sigm")
        res = sigmoid.(res)
    end
    print("\n $(sum(res)) : $(st.ex) \n")
    # ressss=sum(res)
    # seconds_diff = Dates.value(Dates.now() - current_time)/ 1000
    # @info "Time taken for the tensor layer: $(st.ex) ; ressss $(ressss);" times=round(seconds_diff; digits = 2)
    return res, st
end


# function Lux.apply(l::TensorOpLayer_str, x, ps, st::NamedTuple)
#     print("tttttttttttttttttttttt")
#     return Zygote.checkpointed(Lux.apply(l, x, ps, st))
# end

# function ChainRulesCore.rrule(::typeof(Lux.apply), l::Lux.Lux.AbstractLuxLayer, x, ps, st)
#     y = Lux.apply(l, x, ps, st)

#     function pullback_checkpointed(Δy)
#         y, pb =Zygote.pullback(Lux.apply,l, x, ps, st) 
#         return NoTangent(), pb(Δy)
#     end

#     y, pullback_checkpointed
# end


# function Lux.apply(l::TensorOpLayer_str, x, ps, st::NamedTuple)
#     print("tttttttttttttttttttttt")
#     return Zygote.checkpointed(Lux.apply)(l, x, ps, st)
# end

# function (l::TensorOpLayer_str)(x, ps, st::NamedTuple)

#     res = TensorOperations.tensoralloc_contract(Float32, x, st.PA, false, ps.P, st.PB, false, st.PC, Val{false}(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory, CUDA.DeviceMemory, CUDA.DeviceMemory}())
#     res = TensorOperations.tensorcontract!(res, x, st.PA, false, ps.P, st.PB, false, st.PC, VectorInterface.One(), VectorInterface.Zero(), TensorOperations.cuTENSORBackend(), TensorOperations.CUDAAllocator{CUDA.UnifiedMemory, CUDA.DeviceMemory, CUDA.DeviceMemory}())
#     res=swish.(res)

#     return res, st
# end

# # Lux.apply(::TensorOpLayer_str, x::ArrayAndTime, ps, st::NamedTuple)


# function Zygote._pullback(ctx::Zygote.AContext, ::typeof(Lux.apply), l,x, ps, st)
#     y = Lux.apply(l, x, ps, st)
#     function pullback_checkpointed(Δy)
#         y, pb = Zygote._pullback(ctx, Lux.apply, l, x, ps, st)
#         return pb(Δy)
#     end
#     return y, pullback_checkpointed
# end

# function ChainRulesCore.rrule(::typeof(Lux.apply), l::TensorOpLayer_str, x, ps, st)
#     y = Lux.apply(l, x, ps, st)

#     print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

#     function pullback_checkpointed(Δy)
#         y, pb =Zygote.pullback(Lux.apply,l, x, ps, st) 
#         return NoTangent(), pb(Δy)
#     end

#     y, pullback_checkpointed
# end

