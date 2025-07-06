#take form https://github.com/EnzymeAD/Enzyme.jl/issues/428


using CUDA, Enzyme, Test
using Pkg
using ChainRulesCore,Zygote,CUDA,Enzyme

# Pkg.add(url="https://github.com/EnzymeAD/Enzyme.jl.git")


function call_point_info_kern(tetr_dat,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads,blocks,pad_point_info)
    return out_sampled_points
end


# rrule for ChainRules.
function ChainRulesCore.rrule(::typeof(call_point_info_kern),tetr_dat,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads_point_info,blocks_point_info,pad_point_info)
    
    function call_test_kernel1_pullback(d_out_sampled_points)
            # return NoTangent(),tetr_dat,out_sampled_points,source_arr,control_points,sv_centers,num_base_samp_points,num_additional_samp_points,threads_point_info,blocks_point_info,pad_point_info
            return NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent(),NoTangent()
    end


return out_sampled_points, call_test_kernel1_pullback

end




jacobian_result =Zygote.jacobian(call_point_info_kern
                                    ,CUDA.zeros(13000,5,4)#tetrs
                                    ,CUDA.zeros(13000,9,5)#out_sampled_points
                                    ,CUDA.zeros(81,81,81)#source_arr
                                    ,CUDA.zeros(9,9,9,4,3)#control_points
                                    ,CUDA.zeros(8,8,8,3)#sv_centers
                                    ,1#num_base_samp_points
                                    ,1#num_additional_samp_points
                                    ,1#threads_point_info
                                    ,1#blocks_point_info
                                    ,1#pad_point_info
                                    )


                                    tetrs_mem=13000*5*4*4
out_mem=13000*9*5*4
source_mem=81*81*81*4
control_mem=9*9*9*4*3*4
sv_mem=8*8*8*3*4

total_bytes=tetrs_mem+out_mem+source_mem+control_mem+sv_mem
megabytes = total_bytes / 1024 / 1024