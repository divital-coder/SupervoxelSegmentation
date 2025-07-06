using Optimization, Lux, Zygote, MLUtils, Statistics, Plots, Random, ComponentArrays
using CUDA, LuxCUDA,  OptimizationNLopt
using Optimization, OptimizationOptimJL,OptimizationGCMAES
x = Float32.(rand(10, 10))
y = Float32.(sin.(x))
dev = gpu_device()

# data = MLUtils.DataLoader((x, y), batchsize = 1)|> dev
# data = MLUtils.DataLoader((x, y), batchsize = 1)
data = (x, y)

# Define the neural network
model = Chain(Dense(10, 32, tanh), Dense(32, 1))
ps, st = Lux.setup(Random.default_rng(), model)
ps_ca = ComponentArray(ps)
ps_ca = ps_ca |> dev


smodel = StatefulLuxLayer{true}(model, nothing, st)

function callback(state, l)
    state.iter % 25 == 1 && @show "Iteration: %5d, Loss: %.6e\n" state.iter l
    return l < 1e-1 ## Terminate if loss is small
end

function loss(ps, data)
    print("\n dataaaaaaa $(typeof(data)) \n")
    d1 = CuArray(data[1])
    d2 = CuArray(data[2])
    # print("\n typeof(d1) $(typeof(d1)) typeof(d2) $(typeof(d1)) \n")
    ypred = smodel(d1, ps)
    return sum(abs2, ypred .- d2)
end

optf = OptimizationFunction(loss, AutoZygote())
prob = OptimizationProblem(optf, ps_ca, data, lb=((ps_ca.*0)).-100000000, ub=((ps_ca.*0)).+100000000)


CUDA.@allowscalar res = Optimization.solve(prob,GCMAESOpt())

#, callback = callback

#https://docs.sciml.ai/Optimization/stable/optimization_packages/nlopt/