using ModernGL
using GLFW
using Revise
using TensorBoardLogger, Logging, Random
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/OpenGLUtils.jl")




function upload_data(vertices, indices, polygon_indices)
    VAO = Ref{GLuint}(0)
    VBO = Ref{GLuint}(0)
    EBO = Ref{GLuint}(0)
    PBO = Ref{GLuint}(0)

    glGenVertexArrays(1, VAO)
    glGenBuffers(1, VBO)
    glGenBuffers(1, EBO)
    glGenBuffers(1, PBO)

    glBindVertexArray(VAO[])

    # Vertex positions
    glBindBuffer(GL_ARRAY_BUFFER, VBO[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, C_NULL)
    glEnableVertexAttribArray(0)

    # Polygon indices for color
    glBindBuffer(GL_ARRAY_BUFFER, PBO[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(polygon_indices), polygon_indices, GL_STATIC_DRAW)
    glVertexAttribIPointer(1, 1, GL_INT, 0, C_NULL)
    glEnableVertexAttribArray(1)

    # Element indices
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO[])
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW)

    return VAO[]
end


function render_poli(window, shader_program, VAO, colors, num_indices, tex_1d,tf,windowWidth, windowHeight,im_name,step)
    glUseProgram(shader_program)

    # Set colors uniform
    color_location = glGetUniformLocation(shader_program, "colors")
    glUniform3fv(color_location, length(colors) ÷ 3, colors)

    # Bind the texture and set the uniform
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_1D, tex_1d)
    glUniform1i(glGetUniformLocation(shader_program, "tex_1d"), 0)

    # while !GLFW.WindowShouldClose(window)
        GLFW.PollEvents()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glBindVertexArray(VAO)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, C_NULL)

        GLFW.SwapBuffers(window)
    # end

    #save to tensorboard
    rgb_matrix= readFramebufferAsRGBMatrix(windowWidth, windowHeight)
    colorr=permutedims(Float64.(rgb_matrix), (3, 2, 1))
    colorr=colorr./maximum(colorr)
    log_image(tf,im_name,colorr,CWH,step=step)


    glDeleteVertexArrays(1, Ref(VAO))
    glDeleteProgram(shader_program)
    # GLFW.Terminate()

end


function render_grey_poligons(all_res, sv_means,tf,window,windowWidth, windowHeight,im_name,step)

    shader_program = create_shader_program_loc()
    vertices, indices, colors, polygon_indices = prepare_data(all_res)
    VAO = upload_data(vertices, indices, polygon_indices)
    num_indices = length(indices)

    # Initialize a 1D texture with sv_means data
    tex_1d = Ref{GLuint}()
    glGenTextures(1, tex_1d)
    glBindTexture(GL_TEXTURE_1D, tex_1d[])
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R32F, length(sv_means), 0, GL_RED, GL_FLOAT, pointer(sv_means))

    render_poli(window, shader_program, VAO, colors, num_indices, tex_1d[],tf,windowWidth, windowHeight,im_name,step)
end

function interleave_color(triang_vec, colorss)
    # Number of triangles
    num_triangles = length(colorss)
    
    # Reshape triang_vec to have 3 columns (x, y, z coordinates)
    triang_vec = reshape(triang_vec, 3, :)
    
    # Repeat each color 3 times to match the number of vertices
    colorss_in = repeat(colorss, inner=3)
    
    # Interleave the coordinates with the colors
    vertex_color_data = Vector{Float32}(undef, num_triangles * 12)
    vertex_color_data[1:4:end] = triang_vec[1, :]
    vertex_color_data[2:4:end] = triang_vec[2, :]
    vertex_color_data[3:4:end] = triang_vec[3, :]
    vertex_color_data[4:4:end] = colorss_in
    
    return vertex_color_data
end


function prepare_buffers(triang_vec::Vector{Float32}, colorss::Vector{Float32})
    num_triangles = length(triang_vec) ÷ 9
    @assert length(colorss) == num_triangles "Color array length mismatch"

    # Create and bind VAO
    VAO = Ref{GLuint}(0)
    glGenVertexArrays(1, VAO)
    glBindVertexArray(VAO[])

    # Create buffers
    VBO = Ref{GLuint}(0)
    glGenBuffers(1, VBO)

    # Prepare interleaved vertex and color data
    vertex_color_data = interleave_color(triang_vec, colorss)
    # Upload data
    glBindBuffer(GL_ARRAY_BUFFER, VBO[])
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_color_data), vertex_color_data, GL_STATIC_DRAW)

    # Set vertex attributes
    stride = 4 * sizeof(Float32)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, C_NULL)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, stride, Ptr{Cvoid}(3 * sizeof(Float32)))
    glEnableVertexAttribArray(1)

    return VAO[], num_triangles
end

function render_frame(shader_program::GLuint, VAO::GLuint, num_triangles::Int)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glUseProgram(shader_program)
    glBindVertexArray(VAO)
    glDrawArrays(GL_TRIANGLES, 0, num_triangles * 3)
end

function render_grey_polygons(triang_vec::Vector{Float32}, colorss::Vector{Float32}, 
                            window::GLFW.Window, windowWidth::Int, windowHeight::Int,
                            tf, im_name::String, step)
    # Initialize OpenGL state
    glEnable(GL_DEPTH_TEST)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    
    # Create and set up shaders and buffers
    vertex_shader_source = """
    #version 330 core
    layout(location = 0) in vec3 position;
    layout(location = 1) in float color;
    
    flat out float frag_color;
    
    void main() {
        gl_Position = vec4(position, 1.0);
        frag_color = color;
    }
    """

    fragment_shader_source = """
    #version 330 core
    flat in float frag_color;
    out vec4 FragColor;
    
    void main() {
        FragColor = vec4(frag_color, frag_color, frag_color, 1.0);
    }
    """

    shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
    colorss=colorss.-minimum(colorss)
    colorss=colorss./maximum(colorss)
    VAO, num_triangles = prepare_buffers(triang_vec, colorss)

    # Render frame
    render_frame(shader_program, VAO, num_triangles)
    
    # Swap buffers and handle events
    GLFW.SwapBuffers(window)
    GLFW.PollEvents()

    # Save to tensorboard
    rgb_matrix = readFramebufferAsRGBMatrix(windowWidth, windowHeight)
    colorr = permutedims(Float64.(rgb_matrix), (3, 2, 1))
    colorr = colorr ./ maximum(colorr)
    log_image(tf, "$(im_name)_gp", colorr, step=step)

    # Cleanup
    glDeleteVertexArrays(1, [VAO])
    glDeleteProgram(shader_program)
    return colorr
end


function render_grey_polygons_loc(triang_vec::Vector{Float32}, colorss::Vector{Float32}, 
    window::GLFW.Window, windowWidth::Int, windowHeight::Int,
     im_name::String, step)
# Initialize OpenGL state
glEnable(GL_DEPTH_TEST)
glClearColor(0.0, 0.0, 0.0, 1.0)

# Create and set up shaders and buffers
vertex_shader_source = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in float color;

flat out float frag_color;

void main() {
gl_Position = vec4(position, 1.0);
frag_color = color;
}
"""

fragment_shader_source = """
#version 330 core
flat in float frag_color;
out vec4 FragColor;

void main() {
FragColor = vec4(frag_color, frag_color, frag_color, 1.0);
}
"""

shader_program = create_shader_program(vertex_shader_source, fragment_shader_source)
colorss=colorss.-minimum(colorss)
colorss=colorss./maximum(colorss)
VAO, num_triangles = prepare_buffers(triang_vec, colorss)

# Render frame
render_frame(shader_program, VAO, num_triangles)

# Swap buffers and handle events
GLFW.SwapBuffers(window)
GLFW.PollEvents()

# Save to tensorboard
rgb_matrix = readFramebufferAsRGBMatrix(windowWidth, windowHeight)
colorr = permutedims(Float64.(rgb_matrix), (3, 2, 1))
colorr = colorr ./ maximum(colorr)
# log_image(tf, "$(im_name)_gp", colorr, step=step)

# Cleanup
glDeleteVertexArrays(1, [VAO])
glDeleteProgram(shader_program)
return colorr
end
