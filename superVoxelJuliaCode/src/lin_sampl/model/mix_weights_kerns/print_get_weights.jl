res="""
using Pkg
using ChainRulesCore, Zygote, CUDA, Enzyme
# using CUDAKernels
using KernelAbstractions
# using KernelGradients
using Zygote, Lux
using Lux, Random
import Optimisers, Plots, Random, Statistics, Zygote, LLVMLoopInfo
using LinearAlgebra
using Revise
using Base.Threads


```
first get all points around the sv center in this case for 5 voxels in each direction in 3 axes , then it 
    uses 3x3x convolution to reduce number of points we are intrested in - so in practice after this convolution we can analize every second voxel
   then we are using lrelu to mix information from all loaded convolved points 
   last we are doing reductions - hevewer reductions are parametrirized and puprosfully performed multiple times in paralles so we will get 
   5x5 result from 5x5x5 reduction
   we are performing all operations separately for each parameter set, each channel and each batch
   in order to make it a bit faster we will do 2 parameter sets at once in one kernel
```

function kernel_analysis(source_arr,flat_sv_centers,conv_kernels, param_mixing
        , param_reducing,param_reducing_b, param_reducing_c,result
        ,num_blocks_y_true,dims,half_num_params,beg_axis_pad,source_size)
        
      # Calculate thread and block indices
      # x = threadIdx().x
      # y = threadIdx().y
      # z = threadIdx().z
      # bx = blockIdx().x
      # by = blockIdx().y
      # bz = blockIdx().z
      # Calculate channel and parameter set index
      
      # channel = UInt32(round((blockIdx().y % num_blocks_y_true)+1))
      # param_set_idx = (UInt32(ceil((blockIdx().y / num_blocks_y_true)))+half_num_params*(((threadIdx().x) - 1) ÷ 5))

        shared_mem = CuStaticSharedArray(Float32, (2,10,5,5))
        shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=0.0
         """




"""
KAN inspired but sin cos instead of b spline
"""
function waves_accum_shared_mem(l_relu_weight,channel,param_set_idx,param_tensor
    ,c2,x2,y2,z2
    ,xp,yp,zp
    ,cpx,cpy,cpz)
    return """(sin(shared_mem[$c2,$x2,$y2,$z2]*$param_tensor[1,$cpx,$cpy,$cpz,$xp,$yp ,$zp, $channel, $param_set_idx]+$param_tensor[2,$cpx,$cpy,$cpz,$xp,$yp ,$zp, $channel, $param_set_idx])*$param_tensor[5,$cpx,$cpy,$cpz,$xp,$yp ,$zp, $channel, $param_set_idx]+cos(shared_mem[$c2,$x2,$y2,$z2]*$param_tensor[3,$cpx,$cpy,$cpz,$xp,$yp ,$zp, $channel, $param_set_idx]+$param_tensor[4,$cpx,$cpy,$cpz,$xp,$yp ,$zp, $channel, $param_set_idx])*$param_tensor[6,$cpx,$cpy,$cpz,$xp,$yp ,$zp, $channel, $param_set_idx])
    """


    #     return """
end



# Define a function to trim whitespace if the variable is a string
trim_if_string(var) = isa(var, String) ? strip(var) : var

"""
KAN inspired accumulation but sin cos instead of b spline
"""
function wave_single(l_relu_weight,channel,param_set_idx,param_tensor
    ,c0,x0,y0,z0
    ,c1,x1,y1,z1
    ,c2,x2,y2,z2
    ,xp,yp,zp
    ,cpx,cpy,cpz)
    
    l_relu_weight = trim_if_string(l_relu_weight)
    channel = trim_if_string(channel)
    param_set_idx = trim_if_string(param_set_idx)
    param_tensor = trim_if_string(param_tensor)

    c0 = trim_if_string(c0)
    x0 = trim_if_string(x0)
    y0 = trim_if_string(y0)
    z0 = trim_if_string(z0)

    c1 = trim_if_string(c1)
    x1 = trim_if_string(x1)
    y1 = trim_if_string(y1)
    z1 = trim_if_string(z1)

    c2 = trim_if_string(c2)
    x2 = trim_if_string(x2)
    y2 = trim_if_string(y2)
    z2 = trim_if_string(z2)

    xp = trim_if_string(xp)
    yp = trim_if_string(yp)
    zp = trim_if_string(zp)

    cpx = trim_if_string(cpx)
    cpy = trim_if_string(cpy)
    cpz = trim_if_string(cpz)

    # x=shared_mem[$c1,$x1,$y1 ,$z1]
    # y=shared_mem[$c2,$x2,$y2,$z2]
    # p=$param_tensor[1,$cpx,$cpy,$cpz,$xp,$yp ,$zp, $channel, $param_set_idx]
    

    # shared_mem[$c0,$x0,$y0,$z0]=( sin(x*p[1]+p[2])*p[3] + cos(x*p[4] +p[5] )*p[6] + sin(y*p[7]+p[8])*p[9] + cos(y*p[10] +p[11] )*p[12]   )
    return """

shared_mem[$c0, $x0, $y0, $z0 ] = (
    sin(
        shared_mem[$c1, $x1, $y1, $z1] * $param_tensor[1, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx] +
        $param_tensor[2, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx]
    ) * $param_tensor[3, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx] +
    cos(
        shared_mem[$c1, $x1, $y1, $z1] * $param_tensor[4, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx] +
        $param_tensor[5, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx]
    ) * $param_tensor[6, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx] +
    sin(
        shared_mem[$c2, $x2, $y2, $z2] * $param_tensor[7, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx] +
        $param_tensor[8, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx]
    ) * $param_tensor[9, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx] +
    cos(
        shared_mem[$c2, $x2, $y2, $z2] * $param_tensor[10, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx] +
        $param_tensor[11, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx]
    ) * $param_tensor[12, $cpx, $cpy, $cpz, $xp, $yp, $zp, $channel, $param_set_idx]
)
    """
end    




res=res*"""
shared_mem_conv = CuStaticSharedArray(UInt16, (3,10,5,5))
shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z]=flat_sv_centers[blockIdx().x, 1] + beg_axis_pad + (((mod(threadIdx().x - 1, 5) + 1)-3)*2)
shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]=flat_sv_centers[blockIdx().x, 2] + beg_axis_pad + ((threadIdx().y-3)*2)
shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]=flat_sv_centers[blockIdx().x, 3] + beg_axis_pad + ((threadIdx().z-3)*2)
"""

ttt=[]
for i in 1:3
    for j in 1:3
        for k in 1:3
            push!(ttt, """ (source_arr[
                 (shared_mem_conv[1,threadIdx().x,threadIdx().y,threadIdx().z] +($(i-2))) #ip
                , (shared_mem_conv[2,threadIdx().x,threadIdx().y,threadIdx().z]+($(j-2)))#jp
                , (shared_mem_conv[3,threadIdx().x,threadIdx().y,threadIdx().z]+($(k-2)))#kp
                , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] 
            * conv_kernels[$i,$j,$k,threadIdx().x,threadIdx().y,threadIdx().z
            , UInt32(round((blockIdx().y % num_blocks_y_true)+1))
            ,UInt32(ceil((blockIdx().y / num_blocks_y_true)))])
""")
        end
    end
end


res=res*"""
shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]=($(join(ttt,"+")))

"""

mixx=[]
res=res*""" 
shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=0.0
"""

for i in 1:5

for j in 1:5
    for k in 1:5
        push!(mixx,waves_accum_shared_mem("l_relu_weight"
        ,"UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
        ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_mixing"
        ,"2","$i+((threadIdx().x - 1) ÷ 5 * 5)",j,k#input 2
        ,"threadIdx().x","threadIdx().y","threadIdx().z","$i+((threadIdx().x - 1) ÷ 5 * 5)",j,k))#parameters
    end

end  

res=res*""" 
shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]+(($(join(mixx,"+"))))
"""

mixx=[]

end
res=res*""" 
shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]/250
sync_threads()

"""





res=res*"""


# Synchronize threads
    sync_threads()
    #adding skip connection
    #shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]+shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]

    # shared_mem[2,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]
    sync_threads()
        # Step 4: reduction on x, y, and z axes 
 if ((mod(threadIdx().x - 1, 5) + 1)==1)           
        $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
        ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing"
            ,"1","threadIdx().x","threadIdx().y","threadIdx().z" #output
            ,"1","threadIdx().x","threadIdx().y","threadIdx().z" #input 1
            ,"1","threadIdx().x+1","threadIdx().y","threadIdx().z" #input 2
            ,"threadIdx().x+1","threadIdx().y","threadIdx().z","1","1","1" #parameters
            ))          
        
    end
    if ((mod(threadIdx().x - 1, 5) + 1)==4)           
        $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
        ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing"
            ,"1","threadIdx().x","threadIdx().y","threadIdx().z" #output
            ,"1","threadIdx().x","threadIdx().y","threadIdx().z" #input 1
            ,"1","threadIdx().x+1","threadIdx().y","threadIdx().z" #input 2
            ,"threadIdx().x+1","threadIdx().y","threadIdx().z","1","1","1"#parameters
            ))         
        
    end


        

        if ((mod(threadIdx().x - 1, 5) + 1)==1 )           
            #x3
            $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
            ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing"
            ,"1","threadIdx().x","threadIdx().y","threadIdx().z"#output
            ,"1","threadIdx().x","threadIdx().y","threadIdx().z"#input 1
            ,"1","threadIdx().x+2","threadIdx().y","threadIdx().z"#input 2
            ,"threadIdx().x+2","threadIdx().y","threadIdx().z"#parameters
            ,"1","1","1" #parameters
            )) 
            #x4
            $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
            ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing","1"
            ,"threadIdx().x","threadIdx().y","threadIdx().z"#output
            ,"1","threadIdx().x","threadIdx().y","threadIdx().z"#input 1
            ,"1","threadIdx().x+3","threadIdx().y","threadIdx().z"#input 2
            ,"threadIdx().x+3","threadIdx().y","threadIdx().z"#parameters
            ,"1","1","1" #parameters
            ))        
        end    
        sync_threads()



        #before we will get into y reduction let's populate all shared memory in x with accumulated values
        shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]/8
        
        sync_threads()   
        #y reduction
        if (threadIdx().y==1)           
    
            $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
            ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing_b","1","threadIdx().x","threadIdx().y","threadIdx().z","1","threadIdx().x","threadIdx().y","threadIdx().z",
            "1","threadIdx().x","2","threadIdx().z","threadIdx().x","1","threadIdx().z","1","1","1" #parameters
            ))
        end
        if (threadIdx().y==4)           
            $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
            ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing_b","1","threadIdx().x","threadIdx().y","threadIdx().z","1","threadIdx().x","threadIdx().y","threadIdx().z"
            ,"1","threadIdx().x","5","threadIdx().z","threadIdx().x","2","threadIdx().z","1","1","1" #parameters
            ))
        end
        sync_threads()
        #we have in y1 info from y1 and y2 and in y4 infor from y4 and y5
        #so we just need to get info from y3 and y4 to y1
        if (threadIdx().y==1 )           
            #x3
            $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
            ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing_b","1","threadIdx().x","threadIdx().y","threadIdx().z","1","threadIdx().x","threadIdx().y","threadIdx().z","1","threadIdx().x","3","threadIdx().z","threadIdx().x","3","threadIdx().z","1","1","1" #parameters
            ))
            #x4
            $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
            ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing_b","1","threadIdx().x","threadIdx().y","threadIdx().z","1","threadIdx().x","threadIdx().y","threadIdx().z"
            ,"1","threadIdx().x","4","threadIdx().z"
            ,"threadIdx().x","4","threadIdx().z"
            ,"1","1","1" #parameters
            ))
        end   
        sync_threads()      
        #we had accumulated all info on y dimension on y1 so we want to get it to all y
        shared_mem[1,threadIdx().x,threadIdx().y,threadIdx().z]=shared_mem[1,threadIdx().x,1,threadIdx().z]/8
        sync_threads()      

        # z reducing
        if (threadIdx().z==1 )          
            $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
            ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing_c
            ","1","threadIdx().x","threadIdx().y","threadIdx().z","1","threadIdx().x","threadIdx().y","threadIdx().z"
            ,"1","threadIdx().x","threadIdx().y","2"
            ,"threadIdx().x","threadIdx().y","1"
            ,"1","1","1" #parameters
            ))
        end
        if (threadIdx().z==4 )          
            $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
            ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing_c","1","threadIdx().x","threadIdx().y","threadIdx().z","1","threadIdx().x","threadIdx().y","threadIdx().z"
            ,"1","threadIdx().x","threadIdx().y","5","threadIdx().x","threadIdx().y","2","1","1","1" #parameters
            ))
        end
        sync_threads()      
    
        if (threadIdx().z==1 )          
            $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
            ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing_c","1","threadIdx().x","threadIdx().y","threadIdx().z","1","threadIdx().x","threadIdx().y","threadIdx().z"
            ,"1","threadIdx().x","threadIdx().y","3","threadIdx().x","threadIdx().y","3","1","1","1" #parameters
            ))
            #x4
            $(wave_single("l_relu_weight","UInt32(round((blockIdx().y % num_blocks_y_true)+1))"
            ,"UInt32(ceil((blockIdx().y / num_blocks_y_true)))","param_reducing_c","1","threadIdx().x","threadIdx().y","threadIdx().z","1","threadIdx().x","threadIdx().y","threadIdx().z","1","threadIdx().x","threadIdx().y","4","threadIdx().x","threadIdx().y","4","1","1","1" #parameters
            ) )
        end    
        sync_threads()      
        # we had accumulated values in all axes so we can save it to global memory
        # we had accumulated first x values and later we accumulated y and z block dim X times 
        # so we avoided summarising information from threadblock into single number and now we have for example 5x5 values summarizing whole area 
        # results are in all x and threads but z thread = 1 
    
    
        # Step 5: Organize result
        if (threadIdx().z==1 )       

        result[mod(blockIdx().x - 1, dims[3]) + 1
            ,div(mod(blockIdx().x  - 1, dims[2] * dims[3]), dims[3]) + 1
            ,div(blockIdx().x  - 1, dims[2] * dims[3]) + 1 
          ,(mod(threadIdx().x - 1, 5) + 1)
          , threadIdx().y,(UInt32(ceil((blockIdx().y / num_blocks_y_true)))+half_num_params*(((threadIdx().x) - 1) ÷ 5))
        , UInt32(round((blockIdx().y % num_blocks_y_true)+1)), blockIdx().z] = (shared_mem[1,threadIdx().x, threadIdx().y,1]
        +shared_mem[2,threadIdx().x,threadIdx().y,1] #adding here as skip connection directly from initial convolutions
        +shared_mem[2,threadIdx().x,threadIdx().y,2] 
        +shared_mem[2,threadIdx().x,threadIdx().y,3] 
        +shared_mem[2,threadIdx().x,threadIdx().y,4] 
        +shared_mem[2,threadIdx().x,threadIdx().y,5]         
        )

        end
        return nothing
    end


"""

filepath = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/get_weights_heavy5x5x5.txt"

# Open the file in write mode
file = open(filepath, "w")

# Write the value of res to the file
write(file, res)

# Close the file
close(file)



##################### mix weights kernel

res="""
```
We assume input is n,k,256,batch  where n is x*y*z
we access n by blockIdx().x , k by blockIdx().y and batch by blockIdx().z

and block dim x is 256
output should be the same shape as input
```
function mix_sv_info(input, mix_params, output)
    shared_arr = CuStaticSharedArray(Float32, (2,256))
    shared_arr[1,threadIdx().x] = input[blockIdx().x, blockIdx().y, threadIdx().x, blockIdx().z]
    shared_arr[2,threadIdx().x]=shared_arr[1,threadIdx().x]
    sync_threads()

"""
ttt=[]

for i in 1:256
        push!(ttt, """ (sin(shared_arr[2,$(i)]*mix_params[1,$(i),threadIdx().x,blockIdx().y ]+mix_params[2,$(i),threadIdx().x,blockIdx().y ] )*mix_params[3,$(i),threadIdx().x,blockIdx().y ]+
        cos(shared_arr[2,$(i)]*mix_params[4,$(i),threadIdx().x,blockIdx().y ]+mix_params[5,$(i),threadIdx().x,blockIdx().y ] )*mix_params[6,$(i),threadIdx().x,blockIdx().y ])
        """)
end    

for i in 1:8
   res=res*"""

   shared_arr[1,threadIdx().x]=shared_arr[1,threadIdx().x]+($(join(ttt[(i-1)*32+1:i*32], " + ")))
   
   """
end

res=res*"""
shared_arr[1,threadIdx().x]=(shared_arr[1,threadIdx().x])/256 
#mixing accumulated information from all entries and current entry
shared_arr[1,threadIdx().x]=((sin(shared_arr[1,threadIdx().x]*mix_params[1,257,threadIdx().x,blockIdx().y ]+mix_params[2,257,threadIdx().x,blockIdx().y ] )*mix_params[3,257,threadIdx().x,blockIdx().y ]+
        cos(shared_arr[1,threadIdx().x]*mix_params[4,257,threadIdx().x,blockIdx().y ]+mix_params[5,257,threadIdx().x,blockIdx().y ] )*mix_params[6,257,threadIdx().x,blockIdx().y ])
        
        +(sin(shared_arr[2,threadIdx().x]*mix_params[1,258,threadIdx().x,blockIdx().y ]+mix_params[2,258,threadIdx().x,blockIdx().y ] )*mix_params[3,258,threadIdx().x,blockIdx().y ]+
        cos(shared_arr[2,threadIdx().x]*mix_params[4,258,threadIdx().x,blockIdx().y ]+mix_params[5,258,threadIdx().x,blockIdx().y ] )*mix_params[6,258,threadIdx().x,blockIdx().y ])
        )

output[blockIdx().x, blockIdx().y, threadIdx().x, blockIdx().z] = shared_arr[1,threadIdx().x]
return nothing
end

"""


filepath = "/workspaces/superVoxelJuliaCode_lin_sampl/superVoxelJuliaCode/data/mix_w_new.txt"

# Open the file in write mode
file = open(filepath, "w")

# Write the value of res to the file
write(file, res)

# Close the file
close(file)
