

using ModernGL
using GLFW


function create_shader(source::String, shader_type::GLenum)
    shader = glCreateShader(shader_type)
    glShaderSource(shader, 1, [pointer(source)], C_NULL)
    glCompileShader(shader)
    # ...existing code for error checking...
    return shader
end

function create_shader_program(vertex_shader_source, fragment_shader_source)
    vertex_shader = create_shader(vertex_shader_source, GL_VERTEX_SHADER)
    fragment_shader = create_shader(fragment_shader_source, GL_FRAGMENT_SHADER)

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)
    return shader_program
end

function initialize_window(width::Int, height::Int, title::String)
    # ...existing code to initialize GLFW window...
    if !GLFW.Init()
        error("Failed to initialize GLFW")
    end
    window = GLFW.CreateWindow(width, height, title)
    if window == C_NULL
        error("Failed to create GLFW window")
    end
    GLFW.MakeContextCurrent(window)
    glViewport(0, 0, width, height)
    return window
end

