# get_random_point_in_tetrs
res=""

function intersect_lines_projected(line1_point1_x, line1_point1_y, line1_point1_z, 
    line1_point2_x, line1_point2_y, line1_point2_z, 
    line2_point1_x, line2_point1_y, line2_point1_z, 
    line2_point2_x, line2_point2_y, line2_point2_z, 
    out_shmem)
    return """
    # Calculate At * A
    if(threadIdx().y == 1)
    # AtA11
    shared_mem[threadIdx().x,1,1] = (($line1_point2_x) - ($line1_point1_x)) * (($line1_point2_x) - ($line1_point1_x)) +
    (($line1_point2_y) - ($line1_point1_y)) * (($line1_point2_y) - ($line1_point1_y)) +
    (($line1_point2_z) - ($line1_point1_z)) * (($line1_point2_z) - ($line1_point1_z))

    # AtA12
    shared_mem[threadIdx().x,2,1] = (($line1_point2_x) - ($line1_point1_x)) * -(($line2_point1_x) - ($line2_point2_x)) +
    (($line1_point2_y) - ($line1_point1_y)) * -(($line2_point1_y) - ($line2_point2_y)) +
    (($line1_point2_z) - ($line1_point1_z)) * -(($line2_point1_z) - ($line2_point2_z))
    end
    if(threadIdx().y == 2)
    # AtA21
    shared_mem[threadIdx().x,3,1] = -(($line2_point1_x) - ($line2_point2_x)) * (($line1_point2_x) - ($line1_point1_x)) +
    -(($line2_point1_y) - ($line2_point2_y)) * (($line1_point2_y) - ($line1_point1_y)) +
    -(($line2_point1_z) - ($line2_point2_z)) * (($line1_point2_z) - ($line1_point1_z))

    # AtA22
    shared_mem[threadIdx().x,1,2] = -(($line2_point1_x) - ($line2_point2_x)) * -(($line2_point1_x) - ($line2_point2_x)) +
    -(($line2_point1_y) - ($line2_point2_y)) * -(($line2_point1_y) - ($line2_point2_y)) +
    -(($line2_point1_z) - ($line2_point2_z)) * -(($line2_point1_z) - ($line2_point2_z))
    end
    # Calculate At * b
    if(threadIdx().y == 3)
    # Atb1
    shared_mem[threadIdx().x,2,2] = (($line1_point2_x) - ($line1_point1_x)) * (($line2_point2_x) - ($line1_point1_x)) +
    (($line1_point2_y) - ($line1_point1_y)) * (($line2_point2_y) - ($line1_point1_y)) +
    (($line1_point2_z) - ($line1_point1_z)) * (($line2_point2_z) - ($line1_point1_z))

    # Atb2
    shared_mem[threadIdx().x,3,2] = -(($line2_point1_x) - ($line2_point2_x)) * (($line2_point2_x) - ($line1_point1_x)) +
    -(($line2_point1_y) - ($line2_point2_y)) * (($line2_point2_y) - ($line1_point1_y)) +
    -(($line2_point1_z) - ($line2_point2_z)) * (($line2_point2_z) - ($line1_point1_z))
    end
    # Forward elimination
    sync_threads()
    if(threadIdx().y==1)
        shared_mem[threadIdx().x,2,1] /= (shared_mem[threadIdx().x,1,1]+0.0000000000000001)
    end
    if(threadIdx().y==2)
        shared_mem[threadIdx().x,2,2] /= (shared_mem[threadIdx().x,1,1]+0.0000000000000001)
    end
    if(threadIdx().y==3)
        shared_mem[threadIdx().x,1,1] /= (shared_mem[threadIdx().x,1,1]+0.0000000000000001)
    end
    sync_threads()
    
    if(threadIdx().y==1)
        shared_mem[threadIdx().x,1,2] -= shared_mem[threadIdx().x,3,1] * shared_mem[threadIdx().x,2,1]
        shared_mem[threadIdx().x,3,2] -= shared_mem[threadIdx().x,3,1] * shared_mem[threadIdx().x,2,2]
        shared_mem[threadIdx().x,3,1] -= shared_mem[threadIdx().x,3,1] * shared_mem[threadIdx().x,1,1]
        
        # Normalize the second row
        shared_mem[threadIdx().x,3,2] /= (shared_mem[threadIdx().x,1,2]+0.0000000000000001)
        shared_mem[threadIdx().x,1,2] /= (shared_mem[threadIdx().x,1,2]+0.0000000000000001)
        
        # Back substitution
        shared_mem[threadIdx().x,1,2] = shared_mem[threadIdx().x,2,2] - shared_mem[threadIdx().x,2,1] * shared_mem[threadIdx().x,3,2]
    end
    sync_threads()
    
    sync_threads()

    # Calculate the intersection point
    if(threadIdx().y==1)
        shared_mem[threadIdx().x,1, $out_shmem ] = (($line1_point1_x) + shared_mem[threadIdx().x,1,2] * (($line1_point2_x) - ($line1_point1_x)))
    end
    if(threadIdx().y==2)
        shared_mem[threadIdx().x,2, $out_shmem ] = (($line1_point1_y) + shared_mem[threadIdx().x,1,2] * (($line1_point2_y) - ($line1_point1_y)))
    end
    if(threadIdx().y==3)
        shared_mem[threadIdx().x,3, $out_shmem ] = (($line1_point1_z) + shared_mem[threadIdx().x,1,2] * (($line1_point2_z) - ($line1_point1_z)))
    end

    sync_threads()
    """
end






function is_point_on_segment(px, py, pz, x1, y1, z1, x2, y2, z2,out_s_1,out_s_2)
    return """

shared_mem[threadIdx().x,($out_s_1),($out_s_2)]= (0.5 * (1 + tanh(closeness_to_discr * ((($px) - ($x1)) * (($x2) - ($x1)) + (($py) - ($y1)) * (($y2) - ($y1)) 
    + (($pz) - ($z1)) * (($z2) - ($z1))) /    (((($x2) - ($x1))^2 + (($y2) - ($y1))^2 + (($z2) - ($z1))^2)+0.000000000001) 
    *((((($x2) - ($x1))^2 + (($y2) - ($y1))^2 + (($z2) - ($z1))^2) - 
    ((($px) - ($x1)) * (($x2) - ($x1)) + (($py) - ($y1)) * (($y2) - ($y1)) + (($pz) - ($z1)) * (($z2) - ($z1)))) 
    /     (((($x2) - ($x1))^2 + (($y2) - ($y1))^2 + (($z2) - ($z1))^2)+0.000000000001)) )))
    """
end

function intersect_and_clip(triangle_point1_x, triangle_point1_y, triangle_point1_z, triangle_point2_x, triangle_point2_y, triangle_point2_z)
    return """
        sync_threads()
        if(threadIdx().y == 1)
            $(is_point_on_segment(
                "shared_mem[threadIdx().x,1, 1]"," shared_mem[threadIdx().x,2, 1]"," shared_mem[threadIdx().x,3, 1]",
                "shared_mem[threadIdx().x,1, 3]", "shared_mem[threadIdx().x,2, 3]", "shared_mem[threadIdx().x,3, 3]",
                "shared_mem[threadIdx().x,1, 4]", "shared_mem[threadIdx().x,2, 4]"," shared_mem[threadIdx().x,3, 4]",
                "1", "2"
            ))
        end
        if(threadIdx().y == 2)
            $(is_point_on_segment(
                "shared_mem[threadIdx().x,1, 1]", "shared_mem[threadIdx().x,2, 1]", "shared_mem[threadIdx().x,3, 1]",
                triangle_point1_x, triangle_point1_y, triangle_point1_z,
                triangle_point2_x, triangle_point2_y, triangle_point2_z,
                2, 2
            ))
        end
        sync_threads()

        shared_mem[threadIdx().x,1, 2] = shared_mem[threadIdx().x,1, 2] * shared_mem[threadIdx().x,2, 2]

        sync_threads()
        shared_mem[threadIdx().x,threadIdx().y, 5] += shared_mem[threadIdx().x,threadIdx().y, 1] * shared_mem[threadIdx().x,1, 2]

        sync_threads()
    """
end






res=res*"""
function get_random_point_in_tetrs_kern(plan_tensor,weights,control_points_in,sv_centers_out,control_points_out,dims,max_index)
    
    #we need to check if we are not out of bounds
    if((threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))>max_index)
        return nothing
    end


    # shared_curr_plan=CuStaticSharedArray(Int16, (len_get_random_point_in_tetrs_kern,5,4))

    # if((threadIdx().x<21) && (threadIdx().y==1))
    #     plan_tensor[blockIdx().y, ((threadIdx().x%5)+1) ,Int(ceil(threadIdx().x/5))]=plan_tensor[blockIdx().y,,((threadIdx().x%5)+1),Int(ceil(threadIdx().x/5))]
    # end    
    # sync_threads()
    #calculating 3 dimensional index from linear 
    base_x=Int16(mod((threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) - 1, dims[1]) + 1)
    base_y=Int16(div(mod((threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) - 1, dims[2] * dims[1]), dims[1]) + 1)
    
    base_z=Int16(div((threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x())) - 1, dims[2] * dims[1]) + 1)
    # index=(threadIdx().x + ((blockIdx().x - 1) * CUDA.blockDim_x()))
    


    #ap_ap_base=intersect_line_plane(triangle_point1, triangle_point2, triangle_point3, apex1, apex2)
    # index = @index(Global)
    shared_mem=CuStaticSharedArray(Float64, (len_get_random_point_in_tetrs_kern,3,5))
    shared_mem[threadIdx().x,threadIdx().y,5]=0.0#just reset


    ########intersect_line_plane
    # saving normal
    if(threadIdx().y==1)
        shared_mem[threadIdx().x,1,1]= (control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],2,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],2,blockIdx().z]) * (control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],3,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],3,blockIdx().z]) - (control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],3,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],3,blockIdx().z]) * (control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],2,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],2,blockIdx().z])
    elseif(threadIdx().y==2)
        shared_mem[threadIdx().x,2,1]=(control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],3,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],3,blockIdx().z]) * (control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],1,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],1,blockIdx().z]) - (control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],1,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],1,blockIdx().z]) * (control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],3,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],3,blockIdx().z])   
    elseif(threadIdx().y==3)
        shared_mem[threadIdx().x,3,1]=(control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],1,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],1,blockIdx().z]) * (control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],2,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],2,blockIdx().z]) - (control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],2,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],2,blockIdx().z]) * (control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],1,blockIdx().z] - control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],1,blockIdx().z])
    end    
    # Synchronize threads
    sync_threads()

    if(threadIdx().y==1)
        shared_mem[threadIdx().x,1,2]=sqrt((shared_mem[threadIdx().x,1,1]^2)+(shared_mem[threadIdx().x,2,1]^2)+(shared_mem[threadIdx().x,3,1]^2))
    end
    sync_threads()
    shared_mem[threadIdx().x,threadIdx().y,1]=shared_mem[threadIdx().x,threadIdx().y,1]/(shared_mem[threadIdx().x,1,2]+0.00001)
    sync_threads()


    
    # dot_normal_line_dir is shared_mem[threadIdx().x,2,2] and dot_normal_diff is shared_mem[threadIdx().x,2,3]
    if(threadIdx().y==1)
        shared_mem[threadIdx().x,2,2]=(shared_mem[threadIdx().x,1,1] * (sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
                              plan_tensor[blockIdx().y, 5, 2] + base_y,
                              plan_tensor[blockIdx().y, 5, 3] + base_z,
                              1,blockIdx().z] - sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              1,blockIdx().z]) + shared_mem[threadIdx().x,2,1] * (sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
                              plan_tensor[blockIdx().y, 5, 2] + base_y,
                              plan_tensor[blockIdx().y, 5, 3] + base_z,
                              2,blockIdx().z] - sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              2,blockIdx().z]) + shared_mem[threadIdx().x,3,1] * (sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
                              plan_tensor[blockIdx().y, 5, 2] + base_y,
                              plan_tensor[blockIdx().y, 5, 3] + base_z,
                              3,blockIdx().z] - sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              3,blockIdx().z]))
    end
    if(threadIdx().y==2)
        shared_mem[threadIdx().x,2,3] = shared_mem[threadIdx().x,1,1] * (control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],1,blockIdx().z] - sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              1,blockIdx().z]) + shared_mem[threadIdx().x,2,1] * (control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],2,blockIdx().z] - sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              2,blockIdx().z]) + shared_mem[threadIdx().x,3,1] * (control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],3,blockIdx().z] - sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              3,blockIdx().z])
    end


    sync_threads()
    if(threadIdx().y==1)
        shared_mem[threadIdx().x,2,3]  = shared_mem[threadIdx().x,2,3] / (shared_mem[threadIdx().x,2,2]+0.0000001)
    end
    sync_threads()

    #saving ap_ap_base to the fourth column
    shared_mem[threadIdx().x,threadIdx().y,4]=sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
plan_tensor[blockIdx().y, 4, 2] + base_y,
plan_tensor[blockIdx().y, 4, 3] + base_z,
threadIdx().y] + shared_mem[threadIdx().x,2,3]  * (sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
plan_tensor[blockIdx().y, 5, 2] + base_y,
plan_tensor[blockIdx().y, 5, 3] + base_z,
threadIdx().y] - sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
plan_tensor[blockIdx().y, 4, 2] + base_y,
plan_tensor[blockIdx().y, 4, 3] + base_z,
threadIdx().y]) 

    ##################################get_random_point_in_triangle
    #base_point=get_random_point_in_triangle(triangle_point1, triangle_point2, triangle_point3,weights[1],weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  2 ,blockIdx().z],weights[3])
    #we get a random point on a base triangle that is shared between two tetrahedrons
    

    
    
    shared_mem[threadIdx().x,threadIdx().y,3] = (1 -weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  3 
    ,blockIdx().z]) * ((1 -weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  1 ,blockIdx().z]) * control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],threadIdx().y] +weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  1 ,blockIdx().z] * control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],threadIdx().y]) +weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  3 ,blockIdx().z] * ((1 -weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  2 ,blockIdx().z]) * control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],threadIdx().y] +weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  2 ,blockIdx().z] * control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],threadIdx().y])
    sync_threads()


    ############# ###########################
            #distance_to_triangle_edge(triangle_point1, triangle_point2, triangle_point3, P0, P1) P0 is base_point P1 so [threadIdx().y,3] is ap_ap_base so [threadIdx().y,4]

    ##############################################
    ### intersect_and_clip is a macro that saves and acumulates the result in shared_mem[threadIdx().x,y,5]
   
    # shared_mem[threadIdx().x,1,3] shared_mem[threadIdx().x,1,4]
    $(intersect_lines_projected("control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],3,blockIdx().z]"
    , "control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],3,blockIdx().z]"
    , "shared_mem[threadIdx().x,1,3]","shared_mem[threadIdx().x,2,3]","shared_mem[threadIdx().x,3,3]"
    , "shared_mem[threadIdx().x,1,4]","shared_mem[threadIdx().x,2,4]","shared_mem[threadIdx().x,3,4]","1") ) #save into shared_mem[threadIdx().x,y,1]
    sync_threads()
    $(intersect_and_clip("control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],3,blockIdx().z]"," control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],3,blockIdx().z]"))

    $(intersect_lines_projected("control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],3,blockIdx().z]"
    , "control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],3,blockIdx().z]"
    , "shared_mem[threadIdx().x,1,3]","shared_mem[threadIdx().x,2,3]","shared_mem[threadIdx().x,3,3]"
    , "shared_mem[threadIdx().x,1,4]","shared_mem[threadIdx().x,2,4]","shared_mem[threadIdx().x,3,4]",1)) #save into shared_mem[threadIdx().x,y,1]
    sync_threads()
    $(intersect_and_clip("control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],3,blockIdx().z]"," control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                            plan_tensor[blockIdx().y, 2, 2] + base_y,
                                            plan_tensor[blockIdx().y, 2, 3] + base_z,
                                            plan_tensor[blockIdx().y, 2, 4],3,blockIdx().z]"))

    $(intersect_lines_projected("control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],3,blockIdx().z]"
    , "control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],3,blockIdx().z]"
    , "shared_mem[threadIdx().x,1,3]","shared_mem[threadIdx().x,2,3]","shared_mem[threadIdx().x,3,3]"
    , "shared_mem[threadIdx().x,1,4]","shared_mem[threadIdx().x,2,4]","shared_mem[threadIdx().x,3,4]","1") ) #save into shared_mem[threadIdx().x,y,1]
    sync_threads()
    $(intersect_and_clip("control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                            plan_tensor[blockIdx().y, 1, 2] + base_y,
                                            plan_tensor[blockIdx().y, 1, 3] + base_z,
                                            plan_tensor[blockIdx().y, 1, 4],3,blockIdx().z]"," control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],1,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],2,blockIdx().z]","control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                                            plan_tensor[blockIdx().y, 3, 2] + base_y,
                                            plan_tensor[blockIdx().y, 3, 3] + base_z,
                                            plan_tensor[blockIdx().y, 3, 4],3,blockIdx().z]"))

    #now we have accumulated information about p_edge in shared_mem[threadIdx().x,1,5]

    #get d_p_edge in shared_mem[threadIdx().x,1,2] and norm(section) in shared_mem[threadIdx().x,1,1]
    shared_mem[threadIdx().x,threadIdx().y,2]=(shared_mem[threadIdx().x,threadIdx().y,5]-shared_mem[threadIdx().x,threadIdx().y,3])^2
    shared_mem[threadIdx().x,threadIdx().y,1]=(shared_mem[threadIdx().x,threadIdx().y,4]-shared_mem[threadIdx().x,threadIdx().y,3])^2

    sync_threads()
    if(threadIdx().y==1)
        shared_mem[threadIdx().x,1,2]=sqrt(shared_mem[threadIdx().x,1,2]+shared_mem[threadIdx().x,2,2]+shared_mem[threadIdx().x,3,2])
    end
    if(threadIdx().y==2)
        shared_mem[threadIdx().x,1,1]=sqrt(shared_mem[threadIdx().x,1,1]+shared_mem[threadIdx().x,2,1]+shared_mem[threadIdx().x,3,1])
    end
    sync_threads()
    #d_p_edge in shared_mem[threadIdx().x,1,2] and norm(section) in shared_mem[threadIdx().x,1,1] so we need to get the minimum of them and multiply by normalize (shared_mem[threadIdx().x,4]-shared_mem[threadIdx().x,3] so apapbase - base_point)
    # and add it to point from the base we will save it on the ap ap base place in shared memory as it is not needed anymore
    #we get edge_point_ap_common in shared_mem[threadIdx().x,y,4]
    shared_mem[threadIdx().x,threadIdx().y,4] =shared_mem[threadIdx().x,threadIdx().y,3]+(min(shared_mem[threadIdx().x,1,2],shared_mem[threadIdx().x,1,1])*((shared_mem[threadIdx().x,threadIdx().y,4]-shared_mem[threadIdx().x,threadIdx().y,3])/(shared_mem[threadIdx().x,1,1]+0.000001)))
    # sync_threads()
    sync_threads()
    #we get edge_point_ap2 in shared_mem[threadIdx().x,y,5]
    $(intersect_lines_projected("sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              1,blockIdx().z]","sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              2,blockIdx().z]","sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              3,blockIdx().z]"
    , "shared_mem[threadIdx().x,1,3]","shared_mem[threadIdx().x,2,3]","shared_mem[threadIdx().x,3,3]"
    , "shared_mem[threadIdx().x,1,4]","shared_mem[threadIdx().x,2,4]","shared_mem[threadIdx().x,3,4]"
    ,"sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
                              plan_tensor[blockIdx().y, 5, 2] + base_y,
                              plan_tensor[blockIdx().y, 5, 3] + base_z,
                              1,blockIdx().z]","sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
                              plan_tensor[blockIdx().y, 5, 2] + base_y,
                              plan_tensor[blockIdx().y, 5, 3] + base_z,
                              2,blockIdx().z]","sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
                              plan_tensor[blockIdx().y, 5, 2] + base_y,
                              plan_tensor[blockIdx().y, 5, 3] + base_z,
                              3,blockIdx().z]"
    ,"5"))
    #we get edge_point_ap1 in shared_mem[threadIdx().x,y,1] we do both on the same thread as shared mem column 1 and 2 is used internally in the macro    
    $(intersect_lines_projected("sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
                              plan_tensor[blockIdx().y, 5, 2] + base_y,
                              plan_tensor[blockIdx().y, 5, 3] + base_z,
                              1,blockIdx().z]","sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
                              plan_tensor[blockIdx().y, 5, 2] + base_y,
                              plan_tensor[blockIdx().y, 5, 3] + base_z,
                              2,blockIdx().z]","sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
                              plan_tensor[blockIdx().y, 5, 2] + base_y,
                              plan_tensor[blockIdx().y, 5, 3] + base_z,
                              3,blockIdx().z]"
    , "shared_mem[threadIdx().x,1,3]","shared_mem[threadIdx().x,2,3]","shared_mem[threadIdx().x,3,3]"
    , "shared_mem[threadIdx().x,1,4]","shared_mem[threadIdx().x,2,4]","shared_mem[threadIdx().x,3,4]"
    ,"sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              1,blockIdx().z]","sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              2,blockIdx().z]","sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                              plan_tensor[blockIdx().y, 4, 2] + base_y,
                              plan_tensor[blockIdx().y, 4, 3] + base_z,
                              3,blockIdx().z]"
    ,"1"))
       
    sync_threads()  

    # #p1
    shared_mem[threadIdx().x,threadIdx().y,1]=((1 -weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  4 ,blockIdx().z]) * shared_mem[threadIdx().x,threadIdx().y,3] +weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  4 ,blockIdx().z] * shared_mem[threadIdx().x,threadIdx().y,1])
    # #p2
    shared_mem[threadIdx().x,threadIdx().y,3]=((1 -weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  5 ,blockIdx().z]) * shared_mem[threadIdx().x,threadIdx().y,4] +weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  5 ,blockIdx().z] * shared_mem[threadIdx().x,threadIdx().y,3])

    #using weights we get the final point
    # res[threadIdx().y]=(1 -weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  6 ,blockIdx().z]) * ((1 -weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  4 ,blockIdx().z]) * shared_mem[threadIdx().x,threadIdx().y,3] +weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  4 ,blockIdx().z] * shared_mem[threadIdx().x,threadIdx().y,1]) +weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  6 ,blockIdx().z] * ((1 -weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  5 ,blockIdx().z]) * shared_mem[threadIdx().x,threadIdx().y,4] +weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  4 ,blockIdx().z] * shared_mem[threadIdx().x,threadIdx().y,3])
    
    
#     #x = (1 -weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  6 ,blockIdx().z]) * shared_mem[threadIdx().x,threadIdx().y,1] +weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  6 ,blockIdx().z] * shared_mem[threadIdx().x,threadIdx().y,3]
    


#clamping the value so it would not be futher in any direction than futherest point (rare pathological cases can happen)

shared_mem[threadIdx().x,threadIdx().y,3]=(1 -weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  6 ,blockIdx().z]) * shared_mem[threadIdx().x,threadIdx().y,1] +weights[base_x,base_y,base_z,((blockIdx().y-1)*6)+24+  6 ,blockIdx().z] * shared_mem[threadIdx().x,threadIdx().y,3]

# minimums in shared 1 maximums at shared 2
shared_mem[threadIdx().x,threadIdx().y,1]=min(
    min(
        min(
            min(
                control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                  plan_tensor[blockIdx().y, 1, 2] + base_y,
                                  plan_tensor[blockIdx().y, 1, 3] + base_z,
                                  plan_tensor[blockIdx().y, 1, 4], threadIdx().y, blockIdx().z],
                control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                  plan_tensor[blockIdx().y, 2, 2] + base_y,
                                  plan_tensor[blockIdx().y, 2, 3] + base_z,
                                  plan_tensor[blockIdx().y, 2, 4], threadIdx().y, blockIdx().z]
            ),
            control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                              plan_tensor[blockIdx().y, 3, 2] + base_y,
                              plan_tensor[blockIdx().y, 3, 3] + base_z,
                              plan_tensor[blockIdx().y, 3, 4], threadIdx().y, blockIdx().z]
        ),
        sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                       plan_tensor[blockIdx().y, 4, 2] + base_y,
                       plan_tensor[blockIdx().y, 4, 3] + base_z,
                       threadIdx().y, blockIdx().z]
    ),
    sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
                   plan_tensor[blockIdx().y, 5, 2] + base_y,
                   plan_tensor[blockIdx().y, 5, 3] + base_z,
                   threadIdx().y, blockIdx().z]
)

shared_mem[threadIdx().x,threadIdx().y,2]=max(
    max(
        max(
            max(
                control_points_in[plan_tensor[blockIdx().y, 1, 1] + base_x,
                                  plan_tensor[blockIdx().y, 1, 2] + base_y,
                                  plan_tensor[blockIdx().y, 1, 3] + base_z,
                                  plan_tensor[blockIdx().y, 1, 4], threadIdx().y, blockIdx().z],
                control_points_in[plan_tensor[blockIdx().y, 2, 1] + base_x,
                                  plan_tensor[blockIdx().y, 2, 2] + base_y,
                                  plan_tensor[blockIdx().y, 2, 3] + base_z,
                                  plan_tensor[blockIdx().y, 2, 4], threadIdx().y, blockIdx().z]
            ),
            control_points_in[plan_tensor[blockIdx().y, 3, 1] + base_x,
                              plan_tensor[blockIdx().y, 3, 2] + base_y,
                              plan_tensor[blockIdx().y, 3, 3] + base_z,
                              plan_tensor[blockIdx().y, 3, 4], threadIdx().y, blockIdx().z]
        ),
        sv_centers_out[plan_tensor[blockIdx().y, 4, 1] + base_x,
                       plan_tensor[blockIdx().y, 4, 2] + base_y,
                       plan_tensor[blockIdx().y, 4, 3] + base_z,
                       threadIdx().y, blockIdx().z]
    ),
    sv_centers_out[plan_tensor[blockIdx().y, 5, 1] + base_x,
                   plan_tensor[blockIdx().y, 5, 2] + base_y,
                   plan_tensor[blockIdx().y, 5, 3] + base_z,
                   threadIdx().y, blockIdx().z]
)

shared_mem[threadIdx().x, threadIdx().y, 3] = max(
    shared_mem[threadIdx().x, threadIdx().y, 1],
    min(
        shared_mem[threadIdx().x, threadIdx().y, 3],
        shared_mem[threadIdx().x, threadIdx().y, 2]
    )
)

control_points_out[base_x,base_y,base_z,blockIdx().y+7,threadIdx().y ,blockIdx().z]=shared_mem[threadIdx().x,threadIdx().y,3]


    return nothing
end    
"""






# res=res*"""
# end
# """
filepath = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/get_points_per_triangle.txt"

# Open the file in write mode
file = open(filepath, "w")

# Write the value of res to the file
write(file, res)

# Close the file
close(file)
