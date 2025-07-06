using Revise
using ModernGL
using GLFW,HDF5
using GeometryTypes
using Images
using TensorBoardLogger
includet("/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/src/lin_sampl/main_run/visualization/initialize_open_gl.jl")

function readFramebufferAsRGBMatrix(windowWidth::Int, windowHeight::Int)
    # Allocate buffer to store pixel data
    pixel_data = Vector{UInt8}(undef, windowWidth * windowHeight * 3)  # 3 channels (RGB)

    # Read pixels from the framebuffer
    glReadPixels(0, 0, windowWidth, windowHeight, GL_RGB, GL_UNSIGNED_BYTE, pixel_data)

    # Convert the raw pixel data into a Julia matrix
    rgb_matrix = reshape(pixel_data, (3, windowWidth, windowHeight))
    rgb_matrix = permutedims(rgb_matrix, (3, 2, 1))  # Reorder dimensions to (height, width, channels)

    return rgb_matrix
end


function render_and_save(tf,im_name,step
    ,windowWidth, windowHeight
    ,texture_width,texture_height
    ,dat
    ,line_vao
    , line_indices
    , line_shader_program
    , rectangle_vao
    , rectangle_shader_program
    , textUreId
    ,window)
    # Render the scene
    xoffset = 0
    yoffset = 0
    glClear(GL_COLOR_BUFFER_BIT)

    # Render the rectangle with texture
    glUseProgram(rectangle_shader_program)

    glBindVertexArray(rectangle_vao[])
    glActiveTexture(textUreId[])
    glBindTexture(GL_TEXTURE_2D, textUreId[])
    glTexSubImage2D(GL_TEXTURE_2D, 0, xoffset, yoffset, texture_width, texture_height, GL_RED, GL_FLOAT, collect(dat))

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
    glBindVertexArray(0)

    
    # Render the lines
    glUseProgram(line_shader_program)
    glBindVertexArray(line_vao[])
    glDrawElements(GL_LINES, Int(round(length(line_indices) )), GL_UNSIGNED_INT, C_NULL)
    # glDrawElements(GL_LINES, 50, GL_UNSIGNED_INT, C_NULL)
    glBindVertexArray(0)
    GLFW.SwapBuffers(window)
    GLFW.PollEvents()

    
    rgb_matrix= readFramebufferAsRGBMatrix(windowWidth, windowHeight)
    colorr=permutedims(Float64.(rgb_matrix), (3, 2, 1))
    colorr=colorr./maximum(colorr)
    log_image(tf,im_name,colorr,CWH,step=step)
    return colorr
end


"""
version where we render separately image and lines and display them separately
"""
function render_and_save_separtated(
    windowWidth, windowHeight
    ,texture_width,texture_height
    ,dat
    ,line_vao
    , line_indices
    , line_shader_program
    , rectangle_vao
    , rectangle_shader_program
    , textUreId
    ,window)
    # Render the scene
    xoffset = 0
    yoffset = 0
    glClear(GL_COLOR_BUFFER_BIT)

    # Render the rectangle with texture
    glUseProgram(rectangle_shader_program)

    glBindVertexArray(rectangle_vao[])
    glActiveTexture(textUreId[])
    glBindTexture(GL_TEXTURE_2D, textUreId[])
    glTexSubImage2D(GL_TEXTURE_2D, 0, xoffset, yoffset, texture_width, texture_height, GL_RED, GL_FLOAT, collect(dat))

    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, C_NULL)
    glBindVertexArray(0)

    GLFW.SwapBuffers(window)
    GLFW.PollEvents()
    
    rgb_matrix= readFramebufferAsRGBMatrix(windowWidth, windowHeight)
    colorr=permutedims(Float64.(rgb_matrix), (3, 2, 1))
    immagee=colorr./maximum(colorr)

    glClear(GL_COLOR_BUFFER_BIT)

    
    # Render the lines
    glUseProgram(line_shader_program)
    glBindVertexArray(line_vao[])
    glDrawElements(GL_LINES, Int(round(length(line_indices) )), GL_UNSIGNED_INT, C_NULL)
    # glDrawElements(GL_LINES, 50, GL_UNSIGNED_INT, C_NULL)
    glBindVertexArray(0)
    GLFW.SwapBuffers(window)
    GLFW.PollEvents()

    
    rgb_matrix= readFramebufferAsRGBMatrix(windowWidth, windowHeight)
    colorr=permutedims(Float64.(rgb_matrix), (3, 2, 1))
    colorr=colorr./maximum(colorr)
    # log_image(tf,im_name,colorr,CWH,step=step)
    return colorr,immagee
end





# texture_width = 180
# texture_height = 180
# windowWidth, windowHeight = 800,800
# window, rectangle_vao, rectangle_vbo, rectangle_ebo, rectangle_shader_program,  line_shader_program, textUreId=initialize_window_etc(windowWidth, windowHeight,texture_width, texture_height)





# texture_width = 180
# texture_height = 180
# windowWidth, windowHeight = 800,800
# im_name
# step

# dat, line_vertices, line_indices=render_first_in_batch(tetr_dat,axis,plane_dist,radiuss)
# line_vao, line_vbo, line_ebo=initialize_lines(line_vertices, line_indices)
# im=render_and_save(tf,im_name,step,windowWidth, windowHeight,texture_width,texture_height,dat,line_vao, line_indices, line_shader_program, rectangle_vao, rectangle_shader_program, textUreId)






#tensorboard --logdir '/home/jakubmitura/projects/MedEye3d.jl/docs/data/hp'
