cd("/workspaces/superVoxelJuliaCode_lin_sampl")

# In the Julia REPL:
using Pkg

Pkg.activate(".")  # Activate the current directory as project
# Pkg.instantiate()

using Meshes
using JSON3
using Colors
using GLMakie
GLMakie.activate!() # Ensure GLMakie is active

function visualize_meshes_from_json(json_path::String)
    # Load JSON data with String keys, with error handling for invalid JSON
    data = nothing
    try
        data = JSON3.read(open(json_path, "r"), Dict{String, Any}; keytype=String)
    catch e
        # Try to parse as an array and take the first element
        try
            arr = JSON3.read(open(json_path, "r"), Vector{Dict{String, Any}}; keytype=String)
            if length(arr) == 0
                println("JSON file is an empty array. Nothing to visualize.")
                return nothing
            end
            println("Warning: JSON file is an array. Using the first object in the array.")
            data = arr[1]
        catch e2
            println("Failed to parse JSON: ", e)
            println("Also failed to parse as array: ", e2)
            println("Please check that your JSON file is valid. Common issues: trailing commas, incomplete arrays, or empty arrays where triangles are expected.")
            return nothing
        end
    end

    if !haskey(data, "control_points")
        println("Error: JSON object does not contain required key 'control_points'. Keys present: ", collect(keys(data)))
        return nothing
    end
    control_points = data["control_points"]

    # Fix: Sometimes JSON3 returns the key as a Symbol, not String
    triangles = nothing
    if haskey(data, "triangles")
        triangles = data["triangles"]
    elseif haskey(data, :triangles)
        triangles = data[:triangles]
    else
        println("Warning: JSON object does not contain key 'triangles'. Only control points will be visualized. Keys present: ", collect(keys(data)))
        triangles = []
    end

    # Prepare control points (excluding "originalSvCenter")
    points = Dict{String, Meshes.Point}()
    for (name, pt) in control_points
        if name == "originalSvCenter"
            continue
        end
        points[name] = Meshes.Point(pt...)
    end


    # Prepare triangles (if any)
    mesh_tris = Meshes.Triangle[]
    for tri in triangles
        if length(tri) == 3 && all(x -> length(x) == 3, tri)
            p1 = Meshes.Point(tri[1]...)
            p2 = Meshes.Point(tri[2]...)
            p3 = Meshes.Point(tri[3]...)
            push!(mesh_tris, Meshes.Triangle(p1, p2, p3))
        else
            println("Skipping invalid triangle entry: ", tri)
        end
    end


    # Visualize triangles with different colors and legend using viz! and axislegend
    fig = Figure()
    ax = Axis3(fig[1, 1])

    if !isempty(mesh_tris)
        # Plot triangles, each with a different color (no legend for triangles)
        color_palette = distinguishable_colors(length(mesh_tris))
        for (i, tri) in enumerate(mesh_tris)
            viz!(ax, tri, color=color_palette[i], alpha=0.7)
        end
    end

    # Visualize only linX, linY, linZ control points, each with a different color and label
    lin_names = ["linX", "linY", "linZ", "mob", "int1", "int2", "int3", "int4", "int5", "int6", "int7", "int8", "int9", "int10", "int11", "int12"]
    lin_points = [(name, points[name]) for name in lin_names if haskey(points, name)]
    # lin_points=points #TODO remove
    point_palette = distinguishable_colors(length(lin_points))
    for (i, (name, pt)) in enumerate(lin_points)
        viz!(ax, pt, color=point_palette[i], pointsize=12, label=name)
    end

    axislegend(ax)
    println("Legend:")
    for (i, (name, _)) in enumerate(lin_points)
        println("  • ", name, " (color: ", point_palette[i], ")")
    end

    if isempty(mesh_tris)
        println("No triangles to visualize (only control points shown).")
    else
        println("Triangles: each triangle has a unique color.")
    end
    return fig, ax, mesh_tris, points
end

# Example usage:
fig, ax, mesh_tris, points=visualize_meshes_from_json("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/debug_viz/algo_d/sv444.json")
display(fig)