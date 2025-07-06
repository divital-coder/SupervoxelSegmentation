struct Tetrahedron
    vertices::Matrix{Float64}  # 4x3 matrix, each row is a vertex
end

function create_tetrahedron(vertices::Matrix{Float64})
    return Tetrahedron(vertices)
end

struct Plane
    axis::Char
    location::Float64
end

function create_plane(axis::Char, location::Float64)
    return Plane(axis, location)
end


function line_plane_intersection(p1::Vector{Float64}, p2::Vector{Float64}, plane::Plane)
    t = (plane.location - p1[plane.axis == 'x' ? 1 : plane.axis == 'y' ? 2 : 3]) / 
        (p2[plane.axis == 'x' ? 1 : plane.axis == 'y' ? 2 : 3] - p1[plane.axis == 'x' ? 1 : plane.axis == 'y' ? 2 : 3])
    return p1 + t * (p2 - p1)
end

function find_intersections(tetra::Tetrahedron, plane::Plane)
    vertices = tetra.vertices
    intersections = []
    for i in 1:4
        for j in i+1:4
            if (vertices[i, plane.axis == 'x' ? 1 : plane.axis == 'y' ? 2 : 3] - plane.location) *
               (vertices[j, plane.axis == 'x' ? 1 : plane.axis == 'y' ? 2 : 3] - plane.location) < 0
                push!(intersections, line_plane_intersection(vertices[i, :], vertices[j, :], plane))
            end
        end
    end
    return intersections
end

function form_2d_shape(intersections, plane::Plane)
    if plane.axis == 'x'
        return [(p[2], p[3]) for p in intersections]
    elseif plane.axis == 'y'
        return [(p[1], p[3]) for p in intersections]
    else
        return [(p[1], p[2]) for p in intersections]
    end
end

function tetrahedron_cross_section(vertices::Matrix{Float64}, axis::Char, location::Float64)
    tetra = create_tetrahedron(vertices)
    plane = create_plane(axis, location)
    intersections = find_intersections(tetra, plane)
    return form_2d_shape(intersections, plane)
end


vertices = [0.0 0.0 0.0; 1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
axis = 'z'
location = 0.2

cross_section = tetrahedron_cross_section(vertices, axis, location)
println(cross_section)