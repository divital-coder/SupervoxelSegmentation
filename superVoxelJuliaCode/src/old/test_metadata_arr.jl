using Pkg
# Pkg.add(url="https://github.com/JuliaArrays/MetadataArrays.jl.git")
using CUDA, MetadataArrays
CUDA.allowscalar(true)
v = CUDA.zeros(3, 3, 3);

mdv = MetadataArray(v, Dict("Id" => 1))

metadata(mdv)

mdv=mdv.+1
v=v

getfield(mdv, :metadata)

julia> metadata(mdv, :groups)
Dict{String, String} with 3 entries:
  "John"   => "Treatment"
  "Jane"   => "Placebo"
  "Louise" => "Placebo"
