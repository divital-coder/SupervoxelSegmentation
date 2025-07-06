"""
We assume input is n,k,256,batch  where n is x*y*z
we access n by blockIdx().x , k by blockIdx().y and batch by blockIdx().z

and block dim x is 256
output should be the same shape as input
"""
function mix_sv_info(input, mix_params, output)
    shared_arr = CuStaticSharedArray(Float32, (2,256))
    shared_arr[1,threadIdx().x] = input[blockIdx().x, blockIdx().y, threadIdx().x, blockIdx().z]
    shared_arr[2,threadIdx().x]=shared_arr[1,threadIdx().x]
    sync_threads()


shared_arr[1,threadIdx().x]=shared_arr[1,threadIdx().x]+( (sin(shared_arr[2,1]*mix_params[1,1,threadIdx().x,blockIdx().y ]+mix_params[2,1,threadIdx().x,blockIdx().y ] )*mix_params[3,1,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,1]*mix_params[4,1,threadIdx().x,blockIdx().y ]+mix_params[5,1,threadIdx().x,blockIdx().y ] )*mix_params[6,1,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,2]*mix_params[1,2,threadIdx().x,blockIdx().y ]+mix_params[2,2,threadIdx().x,blockIdx().y ] )*mix_params[3,2,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,2]*mix_params[4,2,threadIdx().x,blockIdx().y ]+mix_params[5,2,threadIdx().x,blockIdx().y ] )*mix_params[6,2,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,3]*mix_params[1,3,threadIdx().x,blockIdx().y ]+mix_params[2,3,threadIdx().x,blockIdx().y ] )*mix_params[3,3,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,3]*mix_params[4,3,threadIdx().x,blockIdx().y ]+mix_params[5,3,threadIdx().x,blockIdx().y ] )*mix_params[6,3,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,4]*mix_params[1,4,threadIdx().x,blockIdx().y ]+mix_params[2,4,threadIdx().x,blockIdx().y ] )*mix_params[3,4,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,4]*mix_params[4,4,threadIdx().x,blockIdx().y ]+mix_params[5,4,threadIdx().x,blockIdx().y ] )*mix_params[6,4,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,5]*mix_params[1,5,threadIdx().x,blockIdx().y ]+mix_params[2,5,threadIdx().x,blockIdx().y ] )*mix_params[3,5,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,5]*mix_params[4,5,threadIdx().x,blockIdx().y ]+mix_params[5,5,threadIdx().x,blockIdx().y ] )*mix_params[6,5,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,6]*mix_params[1,6,threadIdx().x,blockIdx().y ]+mix_params[2,6,threadIdx().x,blockIdx().y ] )*mix_params[3,6,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,6]*mix_params[4,6,threadIdx().x,blockIdx().y ]+mix_params[5,6,threadIdx().x,blockIdx().y ] )*mix_params[6,6,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,7]*mix_params[1,7,threadIdx().x,blockIdx().y ]+mix_params[2,7,threadIdx().x,blockIdx().y ] )*mix_params[3,7,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,7]*mix_params[4,7,threadIdx().x,blockIdx().y ]+mix_params[5,7,threadIdx().x,blockIdx().y ] )*mix_params[6,7,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,8]*mix_params[1,8,threadIdx().x,blockIdx().y ]+mix_params[2,8,threadIdx().x,blockIdx().y ] )*mix_params[3,8,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,8]*mix_params[4,8,threadIdx().x,blockIdx().y ]+mix_params[5,8,threadIdx().x,blockIdx().y ] )*mix_params[6,8,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,9]*mix_params[1,9,threadIdx().x,blockIdx().y ]+mix_params[2,9,threadIdx().x,blockIdx().y ] )*mix_params[3,9,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,9]*mix_params[4,9,threadIdx().x,blockIdx().y ]+mix_params[5,9,threadIdx().x,blockIdx().y ] )*mix_params[6,9,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,10]*mix_params[1,10,threadIdx().x,blockIdx().y ]+mix_params[2,10,threadIdx().x,blockIdx().y ] )*mix_params[3,10,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,10]*mix_params[4,10,threadIdx().x,blockIdx().y ]+mix_params[5,10,threadIdx().x,blockIdx().y ] )*mix_params[6,10,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,11]*mix_params[1,11,threadIdx().x,blockIdx().y ]+mix_params[2,11,threadIdx().x,blockIdx().y ] )*mix_params[3,11,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,11]*mix_params[4,11,threadIdx().x,blockIdx().y ]+mix_params[5,11,threadIdx().x,blockIdx().y ] )*mix_params[6,11,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,12]*mix_params[1,12,threadIdx().x,blockIdx().y ]+mix_params[2,12,threadIdx().x,blockIdx().y ] )*mix_params[3,12,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,12]*mix_params[4,12,threadIdx().x,blockIdx().y ]+mix_params[5,12,threadIdx().x,blockIdx().y ] )*mix_params[6,12,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,13]*mix_params[1,13,threadIdx().x,blockIdx().y ]+mix_params[2,13,threadIdx().x,blockIdx().y ] )*mix_params[3,13,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,13]*mix_params[4,13,threadIdx().x,blockIdx().y ]+mix_params[5,13,threadIdx().x,blockIdx().y ] )*mix_params[6,13,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,14]*mix_params[1,14,threadIdx().x,blockIdx().y ]+mix_params[2,14,threadIdx().x,blockIdx().y ] )*mix_params[3,14,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,14]*mix_params[4,14,threadIdx().x,blockIdx().y ]+mix_params[5,14,threadIdx().x,blockIdx().y ] )*mix_params[6,14,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,15]*mix_params[1,15,threadIdx().x,blockIdx().y ]+mix_params[2,15,threadIdx().x,blockIdx().y ] )*mix_params[3,15,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,15]*mix_params[4,15,threadIdx().x,blockIdx().y ]+mix_params[5,15,threadIdx().x,blockIdx().y ] )*mix_params[6,15,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,16]*mix_params[1,16,threadIdx().x,blockIdx().y ]+mix_params[2,16,threadIdx().x,blockIdx().y ] )*mix_params[3,16,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,16]*mix_params[4,16,threadIdx().x,blockIdx().y ]+mix_params[5,16,threadIdx().x,blockIdx().y ] )*mix_params[6,16,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,17]*mix_params[1,17,threadIdx().x,blockIdx().y ]+mix_params[2,17,threadIdx().x,blockIdx().y ] )*mix_params[3,17,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,17]*mix_params[4,17,threadIdx().x,blockIdx().y ]+mix_params[5,17,threadIdx().x,blockIdx().y ] )*mix_params[6,17,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,18]*mix_params[1,18,threadIdx().x,blockIdx().y ]+mix_params[2,18,threadIdx().x,blockIdx().y ] )*mix_params[3,18,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,18]*mix_params[4,18,threadIdx().x,blockIdx().y ]+mix_params[5,18,threadIdx().x,blockIdx().y ] )*mix_params[6,18,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,19]*mix_params[1,19,threadIdx().x,blockIdx().y ]+mix_params[2,19,threadIdx().x,blockIdx().y ] )*mix_params[3,19,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,19]*mix_params[4,19,threadIdx().x,blockIdx().y ]+mix_params[5,19,threadIdx().x,blockIdx().y ] )*mix_params[6,19,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,20]*mix_params[1,20,threadIdx().x,blockIdx().y ]+mix_params[2,20,threadIdx().x,blockIdx().y ] )*mix_params[3,20,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,20]*mix_params[4,20,threadIdx().x,blockIdx().y ]+mix_params[5,20,threadIdx().x,blockIdx().y ] )*mix_params[6,20,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,21]*mix_params[1,21,threadIdx().x,blockIdx().y ]+mix_params[2,21,threadIdx().x,blockIdx().y ] )*mix_params[3,21,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,21]*mix_params[4,21,threadIdx().x,blockIdx().y ]+mix_params[5,21,threadIdx().x,blockIdx().y ] )*mix_params[6,21,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,22]*mix_params[1,22,threadIdx().x,blockIdx().y ]+mix_params[2,22,threadIdx().x,blockIdx().y ] )*mix_params[3,22,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,22]*mix_params[4,22,threadIdx().x,blockIdx().y ]+mix_params[5,22,threadIdx().x,blockIdx().y ] )*mix_params[6,22,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,23]*mix_params[1,23,threadIdx().x,blockIdx().y ]+mix_params[2,23,threadIdx().x,blockIdx().y ] )*mix_params[3,23,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,23]*mix_params[4,23,threadIdx().x,blockIdx().y ]+mix_params[5,23,threadIdx().x,blockIdx().y ] )*mix_params[6,23,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,24]*mix_params[1,24,threadIdx().x,blockIdx().y ]+mix_params[2,24,threadIdx().x,blockIdx().y ] )*mix_params[3,24,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,24]*mix_params[4,24,threadIdx().x,blockIdx().y ]+mix_params[5,24,threadIdx().x,blockIdx().y ] )*mix_params[6,24,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,25]*mix_params[1,25,threadIdx().x,blockIdx().y ]+mix_params[2,25,threadIdx().x,blockIdx().y ] )*mix_params[3,25,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,25]*mix_params[4,25,threadIdx().x,blockIdx().y ]+mix_params[5,25,threadIdx().x,blockIdx().y ] )*mix_params[6,25,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,26]*mix_params[1,26,threadIdx().x,blockIdx().y ]+mix_params[2,26,threadIdx().x,blockIdx().y ] )*mix_params[3,26,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,26]*mix_params[4,26,threadIdx().x,blockIdx().y ]+mix_params[5,26,threadIdx().x,blockIdx().y ] )*mix_params[6,26,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,27]*mix_params[1,27,threadIdx().x,blockIdx().y ]+mix_params[2,27,threadIdx().x,blockIdx().y ] )*mix_params[3,27,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,27]*mix_params[4,27,threadIdx().x,blockIdx().y ]+mix_params[5,27,threadIdx().x,blockIdx().y ] )*mix_params[6,27,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,28]*mix_params[1,28,threadIdx().x,blockIdx().y ]+mix_params[2,28,threadIdx().x,blockIdx().y ] )*mix_params[3,28,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,28]*mix_params[4,28,threadIdx().x,blockIdx().y ]+mix_params[5,28,threadIdx().x,blockIdx().y ] )*mix_params[6,28,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,29]*mix_params[1,29,threadIdx().x,blockIdx().y ]+mix_params[2,29,threadIdx().x,blockIdx().y ] )*mix_params[3,29,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,29]*mix_params[4,29,threadIdx().x,blockIdx().y ]+mix_params[5,29,threadIdx().x,blockIdx().y ] )*mix_params[6,29,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,30]*mix_params[1,30,threadIdx().x,blockIdx().y ]+mix_params[2,30,threadIdx().x,blockIdx().y ] )*mix_params[3,30,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,30]*mix_params[4,30,threadIdx().x,blockIdx().y ]+mix_params[5,30,threadIdx().x,blockIdx().y ] )*mix_params[6,30,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,31]*mix_params[1,31,threadIdx().x,blockIdx().y ]+mix_params[2,31,threadIdx().x,blockIdx().y ] )*mix_params[3,31,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,31]*mix_params[4,31,threadIdx().x,blockIdx().y ]+mix_params[5,31,threadIdx().x,blockIdx().y ] )*mix_params[6,31,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,32]*mix_params[1,32,threadIdx().x,blockIdx().y ]+mix_params[2,32,threadIdx().x,blockIdx().y ] )*mix_params[3,32,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,32]*mix_params[4,32,threadIdx().x,blockIdx().y ]+mix_params[5,32,threadIdx().x,blockIdx().y ] )*mix_params[6,32,threadIdx().x,blockIdx().y ])
)


shared_arr[1,threadIdx().x]=shared_arr[1,threadIdx().x]+( (sin(shared_arr[2,33]*mix_params[1,33,threadIdx().x,blockIdx().y ]+mix_params[2,33,threadIdx().x,blockIdx().y ] )*mix_params[3,33,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,33]*mix_params[4,33,threadIdx().x,blockIdx().y ]+mix_params[5,33,threadIdx().x,blockIdx().y ] )*mix_params[6,33,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,34]*mix_params[1,34,threadIdx().x,blockIdx().y ]+mix_params[2,34,threadIdx().x,blockIdx().y ] )*mix_params[3,34,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,34]*mix_params[4,34,threadIdx().x,blockIdx().y ]+mix_params[5,34,threadIdx().x,blockIdx().y ] )*mix_params[6,34,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,35]*mix_params[1,35,threadIdx().x,blockIdx().y ]+mix_params[2,35,threadIdx().x,blockIdx().y ] )*mix_params[3,35,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,35]*mix_params[4,35,threadIdx().x,blockIdx().y ]+mix_params[5,35,threadIdx().x,blockIdx().y ] )*mix_params[6,35,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,36]*mix_params[1,36,threadIdx().x,blockIdx().y ]+mix_params[2,36,threadIdx().x,blockIdx().y ] )*mix_params[3,36,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,36]*mix_params[4,36,threadIdx().x,blockIdx().y ]+mix_params[5,36,threadIdx().x,blockIdx().y ] )*mix_params[6,36,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,37]*mix_params[1,37,threadIdx().x,blockIdx().y ]+mix_params[2,37,threadIdx().x,blockIdx().y ] )*mix_params[3,37,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,37]*mix_params[4,37,threadIdx().x,blockIdx().y ]+mix_params[5,37,threadIdx().x,blockIdx().y ] )*mix_params[6,37,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,38]*mix_params[1,38,threadIdx().x,blockIdx().y ]+mix_params[2,38,threadIdx().x,blockIdx().y ] )*mix_params[3,38,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,38]*mix_params[4,38,threadIdx().x,blockIdx().y ]+mix_params[5,38,threadIdx().x,blockIdx().y ] )*mix_params[6,38,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,39]*mix_params[1,39,threadIdx().x,blockIdx().y ]+mix_params[2,39,threadIdx().x,blockIdx().y ] )*mix_params[3,39,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,39]*mix_params[4,39,threadIdx().x,blockIdx().y ]+mix_params[5,39,threadIdx().x,blockIdx().y ] )*mix_params[6,39,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,40]*mix_params[1,40,threadIdx().x,blockIdx().y ]+mix_params[2,40,threadIdx().x,blockIdx().y ] )*mix_params[3,40,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,40]*mix_params[4,40,threadIdx().x,blockIdx().y ]+mix_params[5,40,threadIdx().x,blockIdx().y ] )*mix_params[6,40,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,41]*mix_params[1,41,threadIdx().x,blockIdx().y ]+mix_params[2,41,threadIdx().x,blockIdx().y ] )*mix_params[3,41,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,41]*mix_params[4,41,threadIdx().x,blockIdx().y ]+mix_params[5,41,threadIdx().x,blockIdx().y ] )*mix_params[6,41,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,42]*mix_params[1,42,threadIdx().x,blockIdx().y ]+mix_params[2,42,threadIdx().x,blockIdx().y ] )*mix_params[3,42,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,42]*mix_params[4,42,threadIdx().x,blockIdx().y ]+mix_params[5,42,threadIdx().x,blockIdx().y ] )*mix_params[6,42,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,43]*mix_params[1,43,threadIdx().x,blockIdx().y ]+mix_params[2,43,threadIdx().x,blockIdx().y ] )*mix_params[3,43,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,43]*mix_params[4,43,threadIdx().x,blockIdx().y ]+mix_params[5,43,threadIdx().x,blockIdx().y ] )*mix_params[6,43,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,44]*mix_params[1,44,threadIdx().x,blockIdx().y ]+mix_params[2,44,threadIdx().x,blockIdx().y ] )*mix_params[3,44,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,44]*mix_params[4,44,threadIdx().x,blockIdx().y ]+mix_params[5,44,threadIdx().x,blockIdx().y ] )*mix_params[6,44,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,45]*mix_params[1,45,threadIdx().x,blockIdx().y ]+mix_params[2,45,threadIdx().x,blockIdx().y ] )*mix_params[3,45,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,45]*mix_params[4,45,threadIdx().x,blockIdx().y ]+mix_params[5,45,threadIdx().x,blockIdx().y ] )*mix_params[6,45,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,46]*mix_params[1,46,threadIdx().x,blockIdx().y ]+mix_params[2,46,threadIdx().x,blockIdx().y ] )*mix_params[3,46,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,46]*mix_params[4,46,threadIdx().x,blockIdx().y ]+mix_params[5,46,threadIdx().x,blockIdx().y ] )*mix_params[6,46,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,47]*mix_params[1,47,threadIdx().x,blockIdx().y ]+mix_params[2,47,threadIdx().x,blockIdx().y ] )*mix_params[3,47,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,47]*mix_params[4,47,threadIdx().x,blockIdx().y ]+mix_params[5,47,threadIdx().x,blockIdx().y ] )*mix_params[6,47,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,48]*mix_params[1,48,threadIdx().x,blockIdx().y ]+mix_params[2,48,threadIdx().x,blockIdx().y ] )*mix_params[3,48,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,48]*mix_params[4,48,threadIdx().x,blockIdx().y ]+mix_params[5,48,threadIdx().x,blockIdx().y ] )*mix_params[6,48,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,49]*mix_params[1,49,threadIdx().x,blockIdx().y ]+mix_params[2,49,threadIdx().x,blockIdx().y ] )*mix_params[3,49,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,49]*mix_params[4,49,threadIdx().x,blockIdx().y ]+mix_params[5,49,threadIdx().x,blockIdx().y ] )*mix_params[6,49,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,50]*mix_params[1,50,threadIdx().x,blockIdx().y ]+mix_params[2,50,threadIdx().x,blockIdx().y ] )*mix_params[3,50,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,50]*mix_params[4,50,threadIdx().x,blockIdx().y ]+mix_params[5,50,threadIdx().x,blockIdx().y ] )*mix_params[6,50,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,51]*mix_params[1,51,threadIdx().x,blockIdx().y ]+mix_params[2,51,threadIdx().x,blockIdx().y ] )*mix_params[3,51,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,51]*mix_params[4,51,threadIdx().x,blockIdx().y ]+mix_params[5,51,threadIdx().x,blockIdx().y ] )*mix_params[6,51,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,52]*mix_params[1,52,threadIdx().x,blockIdx().y ]+mix_params[2,52,threadIdx().x,blockIdx().y ] )*mix_params[3,52,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,52]*mix_params[4,52,threadIdx().x,blockIdx().y ]+mix_params[5,52,threadIdx().x,blockIdx().y ] )*mix_params[6,52,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,53]*mix_params[1,53,threadIdx().x,blockIdx().y ]+mix_params[2,53,threadIdx().x,blockIdx().y ] )*mix_params[3,53,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,53]*mix_params[4,53,threadIdx().x,blockIdx().y ]+mix_params[5,53,threadIdx().x,blockIdx().y ] )*mix_params[6,53,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,54]*mix_params[1,54,threadIdx().x,blockIdx().y ]+mix_params[2,54,threadIdx().x,blockIdx().y ] )*mix_params[3,54,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,54]*mix_params[4,54,threadIdx().x,blockIdx().y ]+mix_params[5,54,threadIdx().x,blockIdx().y ] )*mix_params[6,54,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,55]*mix_params[1,55,threadIdx().x,blockIdx().y ]+mix_params[2,55,threadIdx().x,blockIdx().y ] )*mix_params[3,55,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,55]*mix_params[4,55,threadIdx().x,blockIdx().y ]+mix_params[5,55,threadIdx().x,blockIdx().y ] )*mix_params[6,55,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,56]*mix_params[1,56,threadIdx().x,blockIdx().y ]+mix_params[2,56,threadIdx().x,blockIdx().y ] )*mix_params[3,56,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,56]*mix_params[4,56,threadIdx().x,blockIdx().y ]+mix_params[5,56,threadIdx().x,blockIdx().y ] )*mix_params[6,56,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,57]*mix_params[1,57,threadIdx().x,blockIdx().y ]+mix_params[2,57,threadIdx().x,blockIdx().y ] )*mix_params[3,57,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,57]*mix_params[4,57,threadIdx().x,blockIdx().y ]+mix_params[5,57,threadIdx().x,blockIdx().y ] )*mix_params[6,57,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,58]*mix_params[1,58,threadIdx().x,blockIdx().y ]+mix_params[2,58,threadIdx().x,blockIdx().y ] )*mix_params[3,58,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,58]*mix_params[4,58,threadIdx().x,blockIdx().y ]+mix_params[5,58,threadIdx().x,blockIdx().y ] )*mix_params[6,58,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,59]*mix_params[1,59,threadIdx().x,blockIdx().y ]+mix_params[2,59,threadIdx().x,blockIdx().y ] )*mix_params[3,59,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,59]*mix_params[4,59,threadIdx().x,blockIdx().y ]+mix_params[5,59,threadIdx().x,blockIdx().y ] )*mix_params[6,59,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,60]*mix_params[1,60,threadIdx().x,blockIdx().y ]+mix_params[2,60,threadIdx().x,blockIdx().y ] )*mix_params[3,60,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,60]*mix_params[4,60,threadIdx().x,blockIdx().y ]+mix_params[5,60,threadIdx().x,blockIdx().y ] )*mix_params[6,60,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,61]*mix_params[1,61,threadIdx().x,blockIdx().y ]+mix_params[2,61,threadIdx().x,blockIdx().y ] )*mix_params[3,61,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,61]*mix_params[4,61,threadIdx().x,blockIdx().y ]+mix_params[5,61,threadIdx().x,blockIdx().y ] )*mix_params[6,61,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,62]*mix_params[1,62,threadIdx().x,blockIdx().y ]+mix_params[2,62,threadIdx().x,blockIdx().y ] )*mix_params[3,62,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,62]*mix_params[4,62,threadIdx().x,blockIdx().y ]+mix_params[5,62,threadIdx().x,blockIdx().y ] )*mix_params[6,62,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,63]*mix_params[1,63,threadIdx().x,blockIdx().y ]+mix_params[2,63,threadIdx().x,blockIdx().y ] )*mix_params[3,63,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,63]*mix_params[4,63,threadIdx().x,blockIdx().y ]+mix_params[5,63,threadIdx().x,blockIdx().y ] )*mix_params[6,63,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,64]*mix_params[1,64,threadIdx().x,blockIdx().y ]+mix_params[2,64,threadIdx().x,blockIdx().y ] )*mix_params[3,64,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,64]*mix_params[4,64,threadIdx().x,blockIdx().y ]+mix_params[5,64,threadIdx().x,blockIdx().y ] )*mix_params[6,64,threadIdx().x,blockIdx().y ])
)


shared_arr[1,threadIdx().x]=shared_arr[1,threadIdx().x]+( (sin(shared_arr[2,65]*mix_params[1,65,threadIdx().x,blockIdx().y ]+mix_params[2,65,threadIdx().x,blockIdx().y ] )*mix_params[3,65,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,65]*mix_params[4,65,threadIdx().x,blockIdx().y ]+mix_params[5,65,threadIdx().x,blockIdx().y ] )*mix_params[6,65,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,66]*mix_params[1,66,threadIdx().x,blockIdx().y ]+mix_params[2,66,threadIdx().x,blockIdx().y ] )*mix_params[3,66,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,66]*mix_params[4,66,threadIdx().x,blockIdx().y ]+mix_params[5,66,threadIdx().x,blockIdx().y ] )*mix_params[6,66,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,67]*mix_params[1,67,threadIdx().x,blockIdx().y ]+mix_params[2,67,threadIdx().x,blockIdx().y ] )*mix_params[3,67,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,67]*mix_params[4,67,threadIdx().x,blockIdx().y ]+mix_params[5,67,threadIdx().x,blockIdx().y ] )*mix_params[6,67,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,68]*mix_params[1,68,threadIdx().x,blockIdx().y ]+mix_params[2,68,threadIdx().x,blockIdx().y ] )*mix_params[3,68,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,68]*mix_params[4,68,threadIdx().x,blockIdx().y ]+mix_params[5,68,threadIdx().x,blockIdx().y ] )*mix_params[6,68,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,69]*mix_params[1,69,threadIdx().x,blockIdx().y ]+mix_params[2,69,threadIdx().x,blockIdx().y ] )*mix_params[3,69,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,69]*mix_params[4,69,threadIdx().x,blockIdx().y ]+mix_params[5,69,threadIdx().x,blockIdx().y ] )*mix_params[6,69,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,70]*mix_params[1,70,threadIdx().x,blockIdx().y ]+mix_params[2,70,threadIdx().x,blockIdx().y ] )*mix_params[3,70,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,70]*mix_params[4,70,threadIdx().x,blockIdx().y ]+mix_params[5,70,threadIdx().x,blockIdx().y ] )*mix_params[6,70,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,71]*mix_params[1,71,threadIdx().x,blockIdx().y ]+mix_params[2,71,threadIdx().x,blockIdx().y ] )*mix_params[3,71,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,71]*mix_params[4,71,threadIdx().x,blockIdx().y ]+mix_params[5,71,threadIdx().x,blockIdx().y ] )*mix_params[6,71,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,72]*mix_params[1,72,threadIdx().x,blockIdx().y ]+mix_params[2,72,threadIdx().x,blockIdx().y ] )*mix_params[3,72,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,72]*mix_params[4,72,threadIdx().x,blockIdx().y ]+mix_params[5,72,threadIdx().x,blockIdx().y ] )*mix_params[6,72,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,73]*mix_params[1,73,threadIdx().x,blockIdx().y ]+mix_params[2,73,threadIdx().x,blockIdx().y ] )*mix_params[3,73,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,73]*mix_params[4,73,threadIdx().x,blockIdx().y ]+mix_params[5,73,threadIdx().x,blockIdx().y ] )*mix_params[6,73,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,74]*mix_params[1,74,threadIdx().x,blockIdx().y ]+mix_params[2,74,threadIdx().x,blockIdx().y ] )*mix_params[3,74,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,74]*mix_params[4,74,threadIdx().x,blockIdx().y ]+mix_params[5,74,threadIdx().x,blockIdx().y ] )*mix_params[6,74,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,75]*mix_params[1,75,threadIdx().x,blockIdx().y ]+mix_params[2,75,threadIdx().x,blockIdx().y ] )*mix_params[3,75,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,75]*mix_params[4,75,threadIdx().x,blockIdx().y ]+mix_params[5,75,threadIdx().x,blockIdx().y ] )*mix_params[6,75,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,76]*mix_params[1,76,threadIdx().x,blockIdx().y ]+mix_params[2,76,threadIdx().x,blockIdx().y ] )*mix_params[3,76,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,76]*mix_params[4,76,threadIdx().x,blockIdx().y ]+mix_params[5,76,threadIdx().x,blockIdx().y ] )*mix_params[6,76,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,77]*mix_params[1,77,threadIdx().x,blockIdx().y ]+mix_params[2,77,threadIdx().x,blockIdx().y ] )*mix_params[3,77,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,77]*mix_params[4,77,threadIdx().x,blockIdx().y ]+mix_params[5,77,threadIdx().x,blockIdx().y ] )*mix_params[6,77,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,78]*mix_params[1,78,threadIdx().x,blockIdx().y ]+mix_params[2,78,threadIdx().x,blockIdx().y ] )*mix_params[3,78,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,78]*mix_params[4,78,threadIdx().x,blockIdx().y ]+mix_params[5,78,threadIdx().x,blockIdx().y ] )*mix_params[6,78,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,79]*mix_params[1,79,threadIdx().x,blockIdx().y ]+mix_params[2,79,threadIdx().x,blockIdx().y ] )*mix_params[3,79,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,79]*mix_params[4,79,threadIdx().x,blockIdx().y ]+mix_params[5,79,threadIdx().x,blockIdx().y ] )*mix_params[6,79,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,80]*mix_params[1,80,threadIdx().x,blockIdx().y ]+mix_params[2,80,threadIdx().x,blockIdx().y ] )*mix_params[3,80,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,80]*mix_params[4,80,threadIdx().x,blockIdx().y ]+mix_params[5,80,threadIdx().x,blockIdx().y ] )*mix_params[6,80,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,81]*mix_params[1,81,threadIdx().x,blockIdx().y ]+mix_params[2,81,threadIdx().x,blockIdx().y ] )*mix_params[3,81,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,81]*mix_params[4,81,threadIdx().x,blockIdx().y ]+mix_params[5,81,threadIdx().x,blockIdx().y ] )*mix_params[6,81,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,82]*mix_params[1,82,threadIdx().x,blockIdx().y ]+mix_params[2,82,threadIdx().x,blockIdx().y ] )*mix_params[3,82,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,82]*mix_params[4,82,threadIdx().x,blockIdx().y ]+mix_params[5,82,threadIdx().x,blockIdx().y ] )*mix_params[6,82,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,83]*mix_params[1,83,threadIdx().x,blockIdx().y ]+mix_params[2,83,threadIdx().x,blockIdx().y ] )*mix_params[3,83,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,83]*mix_params[4,83,threadIdx().x,blockIdx().y ]+mix_params[5,83,threadIdx().x,blockIdx().y ] )*mix_params[6,83,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,84]*mix_params[1,84,threadIdx().x,blockIdx().y ]+mix_params[2,84,threadIdx().x,blockIdx().y ] )*mix_params[3,84,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,84]*mix_params[4,84,threadIdx().x,blockIdx().y ]+mix_params[5,84,threadIdx().x,blockIdx().y ] )*mix_params[6,84,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,85]*mix_params[1,85,threadIdx().x,blockIdx().y ]+mix_params[2,85,threadIdx().x,blockIdx().y ] )*mix_params[3,85,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,85]*mix_params[4,85,threadIdx().x,blockIdx().y ]+mix_params[5,85,threadIdx().x,blockIdx().y ] )*mix_params[6,85,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,86]*mix_params[1,86,threadIdx().x,blockIdx().y ]+mix_params[2,86,threadIdx().x,blockIdx().y ] )*mix_params[3,86,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,86]*mix_params[4,86,threadIdx().x,blockIdx().y ]+mix_params[5,86,threadIdx().x,blockIdx().y ] )*mix_params[6,86,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,87]*mix_params[1,87,threadIdx().x,blockIdx().y ]+mix_params[2,87,threadIdx().x,blockIdx().y ] )*mix_params[3,87,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,87]*mix_params[4,87,threadIdx().x,blockIdx().y ]+mix_params[5,87,threadIdx().x,blockIdx().y ] )*mix_params[6,87,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,88]*mix_params[1,88,threadIdx().x,blockIdx().y ]+mix_params[2,88,threadIdx().x,blockIdx().y ] )*mix_params[3,88,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,88]*mix_params[4,88,threadIdx().x,blockIdx().y ]+mix_params[5,88,threadIdx().x,blockIdx().y ] )*mix_params[6,88,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,89]*mix_params[1,89,threadIdx().x,blockIdx().y ]+mix_params[2,89,threadIdx().x,blockIdx().y ] )*mix_params[3,89,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,89]*mix_params[4,89,threadIdx().x,blockIdx().y ]+mix_params[5,89,threadIdx().x,blockIdx().y ] )*mix_params[6,89,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,90]*mix_params[1,90,threadIdx().x,blockIdx().y ]+mix_params[2,90,threadIdx().x,blockIdx().y ] )*mix_params[3,90,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,90]*mix_params[4,90,threadIdx().x,blockIdx().y ]+mix_params[5,90,threadIdx().x,blockIdx().y ] )*mix_params[6,90,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,91]*mix_params[1,91,threadIdx().x,blockIdx().y ]+mix_params[2,91,threadIdx().x,blockIdx().y ] )*mix_params[3,91,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,91]*mix_params[4,91,threadIdx().x,blockIdx().y ]+mix_params[5,91,threadIdx().x,blockIdx().y ] )*mix_params[6,91,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,92]*mix_params[1,92,threadIdx().x,blockIdx().y ]+mix_params[2,92,threadIdx().x,blockIdx().y ] )*mix_params[3,92,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,92]*mix_params[4,92,threadIdx().x,blockIdx().y ]+mix_params[5,92,threadIdx().x,blockIdx().y ] )*mix_params[6,92,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,93]*mix_params[1,93,threadIdx().x,blockIdx().y ]+mix_params[2,93,threadIdx().x,blockIdx().y ] )*mix_params[3,93,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,93]*mix_params[4,93,threadIdx().x,blockIdx().y ]+mix_params[5,93,threadIdx().x,blockIdx().y ] )*mix_params[6,93,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,94]*mix_params[1,94,threadIdx().x,blockIdx().y ]+mix_params[2,94,threadIdx().x,blockIdx().y ] )*mix_params[3,94,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,94]*mix_params[4,94,threadIdx().x,blockIdx().y ]+mix_params[5,94,threadIdx().x,blockIdx().y ] )*mix_params[6,94,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,95]*mix_params[1,95,threadIdx().x,blockIdx().y ]+mix_params[2,95,threadIdx().x,blockIdx().y ] )*mix_params[3,95,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,95]*mix_params[4,95,threadIdx().x,blockIdx().y ]+mix_params[5,95,threadIdx().x,blockIdx().y ] )*mix_params[6,95,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,96]*mix_params[1,96,threadIdx().x,blockIdx().y ]+mix_params[2,96,threadIdx().x,blockIdx().y ] )*mix_params[3,96,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,96]*mix_params[4,96,threadIdx().x,blockIdx().y ]+mix_params[5,96,threadIdx().x,blockIdx().y ] )*mix_params[6,96,threadIdx().x,blockIdx().y ])
)


shared_arr[1,threadIdx().x]=shared_arr[1,threadIdx().x]+( (sin(shared_arr[2,97]*mix_params[1,97,threadIdx().x,blockIdx().y ]+mix_params[2,97,threadIdx().x,blockIdx().y ] )*mix_params[3,97,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,97]*mix_params[4,97,threadIdx().x,blockIdx().y ]+mix_params[5,97,threadIdx().x,blockIdx().y ] )*mix_params[6,97,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,98]*mix_params[1,98,threadIdx().x,blockIdx().y ]+mix_params[2,98,threadIdx().x,blockIdx().y ] )*mix_params[3,98,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,98]*mix_params[4,98,threadIdx().x,blockIdx().y ]+mix_params[5,98,threadIdx().x,blockIdx().y ] )*mix_params[6,98,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,99]*mix_params[1,99,threadIdx().x,blockIdx().y ]+mix_params[2,99,threadIdx().x,blockIdx().y ] )*mix_params[3,99,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,99]*mix_params[4,99,threadIdx().x,blockIdx().y ]+mix_params[5,99,threadIdx().x,blockIdx().y ] )*mix_params[6,99,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,100]*mix_params[1,100,threadIdx().x,blockIdx().y ]+mix_params[2,100,threadIdx().x,blockIdx().y ] )*mix_params[3,100,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,100]*mix_params[4,100,threadIdx().x,blockIdx().y ]+mix_params[5,100,threadIdx().x,blockIdx().y ] )*mix_params[6,100,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,101]*mix_params[1,101,threadIdx().x,blockIdx().y ]+mix_params[2,101,threadIdx().x,blockIdx().y ] )*mix_params[3,101,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,101]*mix_params[4,101,threadIdx().x,blockIdx().y ]+mix_params[5,101,threadIdx().x,blockIdx().y ] )*mix_params[6,101,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,102]*mix_params[1,102,threadIdx().x,blockIdx().y ]+mix_params[2,102,threadIdx().x,blockIdx().y ] )*mix_params[3,102,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,102]*mix_params[4,102,threadIdx().x,blockIdx().y ]+mix_params[5,102,threadIdx().x,blockIdx().y ] )*mix_params[6,102,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,103]*mix_params[1,103,threadIdx().x,blockIdx().y ]+mix_params[2,103,threadIdx().x,blockIdx().y ] )*mix_params[3,103,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,103]*mix_params[4,103,threadIdx().x,blockIdx().y ]+mix_params[5,103,threadIdx().x,blockIdx().y ] )*mix_params[6,103,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,104]*mix_params[1,104,threadIdx().x,blockIdx().y ]+mix_params[2,104,threadIdx().x,blockIdx().y ] )*mix_params[3,104,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,104]*mix_params[4,104,threadIdx().x,blockIdx().y ]+mix_params[5,104,threadIdx().x,blockIdx().y ] )*mix_params[6,104,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,105]*mix_params[1,105,threadIdx().x,blockIdx().y ]+mix_params[2,105,threadIdx().x,blockIdx().y ] )*mix_params[3,105,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,105]*mix_params[4,105,threadIdx().x,blockIdx().y ]+mix_params[5,105,threadIdx().x,blockIdx().y ] )*mix_params[6,105,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,106]*mix_params[1,106,threadIdx().x,blockIdx().y ]+mix_params[2,106,threadIdx().x,blockIdx().y ] )*mix_params[3,106,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,106]*mix_params[4,106,threadIdx().x,blockIdx().y ]+mix_params[5,106,threadIdx().x,blockIdx().y ] )*mix_params[6,106,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,107]*mix_params[1,107,threadIdx().x,blockIdx().y ]+mix_params[2,107,threadIdx().x,blockIdx().y ] )*mix_params[3,107,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,107]*mix_params[4,107,threadIdx().x,blockIdx().y ]+mix_params[5,107,threadIdx().x,blockIdx().y ] )*mix_params[6,107,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,108]*mix_params[1,108,threadIdx().x,blockIdx().y ]+mix_params[2,108,threadIdx().x,blockIdx().y ] )*mix_params[3,108,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,108]*mix_params[4,108,threadIdx().x,blockIdx().y ]+mix_params[5,108,threadIdx().x,blockIdx().y ] )*mix_params[6,108,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,109]*mix_params[1,109,threadIdx().x,blockIdx().y ]+mix_params[2,109,threadIdx().x,blockIdx().y ] )*mix_params[3,109,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,109]*mix_params[4,109,threadIdx().x,blockIdx().y ]+mix_params[5,109,threadIdx().x,blockIdx().y ] )*mix_params[6,109,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,110]*mix_params[1,110,threadIdx().x,blockIdx().y ]+mix_params[2,110,threadIdx().x,blockIdx().y ] )*mix_params[3,110,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,110]*mix_params[4,110,threadIdx().x,blockIdx().y ]+mix_params[5,110,threadIdx().x,blockIdx().y ] )*mix_params[6,110,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,111]*mix_params[1,111,threadIdx().x,blockIdx().y ]+mix_params[2,111,threadIdx().x,blockIdx().y ] )*mix_params[3,111,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,111]*mix_params[4,111,threadIdx().x,blockIdx().y ]+mix_params[5,111,threadIdx().x,blockIdx().y ] )*mix_params[6,111,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,112]*mix_params[1,112,threadIdx().x,blockIdx().y ]+mix_params[2,112,threadIdx().x,blockIdx().y ] )*mix_params[3,112,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,112]*mix_params[4,112,threadIdx().x,blockIdx().y ]+mix_params[5,112,threadIdx().x,blockIdx().y ] )*mix_params[6,112,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,113]*mix_params[1,113,threadIdx().x,blockIdx().y ]+mix_params[2,113,threadIdx().x,blockIdx().y ] )*mix_params[3,113,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,113]*mix_params[4,113,threadIdx().x,blockIdx().y ]+mix_params[5,113,threadIdx().x,blockIdx().y ] )*mix_params[6,113,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,114]*mix_params[1,114,threadIdx().x,blockIdx().y ]+mix_params[2,114,threadIdx().x,blockIdx().y ] )*mix_params[3,114,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,114]*mix_params[4,114,threadIdx().x,blockIdx().y ]+mix_params[5,114,threadIdx().x,blockIdx().y ] )*mix_params[6,114,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,115]*mix_params[1,115,threadIdx().x,blockIdx().y ]+mix_params[2,115,threadIdx().x,blockIdx().y ] )*mix_params[3,115,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,115]*mix_params[4,115,threadIdx().x,blockIdx().y ]+mix_params[5,115,threadIdx().x,blockIdx().y ] )*mix_params[6,115,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,116]*mix_params[1,116,threadIdx().x,blockIdx().y ]+mix_params[2,116,threadIdx().x,blockIdx().y ] )*mix_params[3,116,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,116]*mix_params[4,116,threadIdx().x,blockIdx().y ]+mix_params[5,116,threadIdx().x,blockIdx().y ] )*mix_params[6,116,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,117]*mix_params[1,117,threadIdx().x,blockIdx().y ]+mix_params[2,117,threadIdx().x,blockIdx().y ] )*mix_params[3,117,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,117]*mix_params[4,117,threadIdx().x,blockIdx().y ]+mix_params[5,117,threadIdx().x,blockIdx().y ] )*mix_params[6,117,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,118]*mix_params[1,118,threadIdx().x,blockIdx().y ]+mix_params[2,118,threadIdx().x,blockIdx().y ] )*mix_params[3,118,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,118]*mix_params[4,118,threadIdx().x,blockIdx().y ]+mix_params[5,118,threadIdx().x,blockIdx().y ] )*mix_params[6,118,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,119]*mix_params[1,119,threadIdx().x,blockIdx().y ]+mix_params[2,119,threadIdx().x,blockIdx().y ] )*mix_params[3,119,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,119]*mix_params[4,119,threadIdx().x,blockIdx().y ]+mix_params[5,119,threadIdx().x,blockIdx().y ] )*mix_params[6,119,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,120]*mix_params[1,120,threadIdx().x,blockIdx().y ]+mix_params[2,120,threadIdx().x,blockIdx().y ] )*mix_params[3,120,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,120]*mix_params[4,120,threadIdx().x,blockIdx().y ]+mix_params[5,120,threadIdx().x,blockIdx().y ] )*mix_params[6,120,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,121]*mix_params[1,121,threadIdx().x,blockIdx().y ]+mix_params[2,121,threadIdx().x,blockIdx().y ] )*mix_params[3,121,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,121]*mix_params[4,121,threadIdx().x,blockIdx().y ]+mix_params[5,121,threadIdx().x,blockIdx().y ] )*mix_params[6,121,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,122]*mix_params[1,122,threadIdx().x,blockIdx().y ]+mix_params[2,122,threadIdx().x,blockIdx().y ] )*mix_params[3,122,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,122]*mix_params[4,122,threadIdx().x,blockIdx().y ]+mix_params[5,122,threadIdx().x,blockIdx().y ] )*mix_params[6,122,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,123]*mix_params[1,123,threadIdx().x,blockIdx().y ]+mix_params[2,123,threadIdx().x,blockIdx().y ] )*mix_params[3,123,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,123]*mix_params[4,123,threadIdx().x,blockIdx().y ]+mix_params[5,123,threadIdx().x,blockIdx().y ] )*mix_params[6,123,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,124]*mix_params[1,124,threadIdx().x,blockIdx().y ]+mix_params[2,124,threadIdx().x,blockIdx().y ] )*mix_params[3,124,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,124]*mix_params[4,124,threadIdx().x,blockIdx().y ]+mix_params[5,124,threadIdx().x,blockIdx().y ] )*mix_params[6,124,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,125]*mix_params[1,125,threadIdx().x,blockIdx().y ]+mix_params[2,125,threadIdx().x,blockIdx().y ] )*mix_params[3,125,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,125]*mix_params[4,125,threadIdx().x,blockIdx().y ]+mix_params[5,125,threadIdx().x,blockIdx().y ] )*mix_params[6,125,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,126]*mix_params[1,126,threadIdx().x,blockIdx().y ]+mix_params[2,126,threadIdx().x,blockIdx().y ] )*mix_params[3,126,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,126]*mix_params[4,126,threadIdx().x,blockIdx().y ]+mix_params[5,126,threadIdx().x,blockIdx().y ] )*mix_params[6,126,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,127]*mix_params[1,127,threadIdx().x,blockIdx().y ]+mix_params[2,127,threadIdx().x,blockIdx().y ] )*mix_params[3,127,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,127]*mix_params[4,127,threadIdx().x,blockIdx().y ]+mix_params[5,127,threadIdx().x,blockIdx().y ] )*mix_params[6,127,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,128]*mix_params[1,128,threadIdx().x,blockIdx().y ]+mix_params[2,128,threadIdx().x,blockIdx().y ] )*mix_params[3,128,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,128]*mix_params[4,128,threadIdx().x,blockIdx().y ]+mix_params[5,128,threadIdx().x,blockIdx().y ] )*mix_params[6,128,threadIdx().x,blockIdx().y ])
)


shared_arr[1,threadIdx().x]=shared_arr[1,threadIdx().x]+( (sin(shared_arr[2,129]*mix_params[1,129,threadIdx().x,blockIdx().y ]+mix_params[2,129,threadIdx().x,blockIdx().y ] )*mix_params[3,129,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,129]*mix_params[4,129,threadIdx().x,blockIdx().y ]+mix_params[5,129,threadIdx().x,blockIdx().y ] )*mix_params[6,129,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,130]*mix_params[1,130,threadIdx().x,blockIdx().y ]+mix_params[2,130,threadIdx().x,blockIdx().y ] )*mix_params[3,130,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,130]*mix_params[4,130,threadIdx().x,blockIdx().y ]+mix_params[5,130,threadIdx().x,blockIdx().y ] )*mix_params[6,130,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,131]*mix_params[1,131,threadIdx().x,blockIdx().y ]+mix_params[2,131,threadIdx().x,blockIdx().y ] )*mix_params[3,131,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,131]*mix_params[4,131,threadIdx().x,blockIdx().y ]+mix_params[5,131,threadIdx().x,blockIdx().y ] )*mix_params[6,131,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,132]*mix_params[1,132,threadIdx().x,blockIdx().y ]+mix_params[2,132,threadIdx().x,blockIdx().y ] )*mix_params[3,132,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,132]*mix_params[4,132,threadIdx().x,blockIdx().y ]+mix_params[5,132,threadIdx().x,blockIdx().y ] )*mix_params[6,132,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,133]*mix_params[1,133,threadIdx().x,blockIdx().y ]+mix_params[2,133,threadIdx().x,blockIdx().y ] )*mix_params[3,133,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,133]*mix_params[4,133,threadIdx().x,blockIdx().y ]+mix_params[5,133,threadIdx().x,blockIdx().y ] )*mix_params[6,133,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,134]*mix_params[1,134,threadIdx().x,blockIdx().y ]+mix_params[2,134,threadIdx().x,blockIdx().y ] )*mix_params[3,134,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,134]*mix_params[4,134,threadIdx().x,blockIdx().y ]+mix_params[5,134,threadIdx().x,blockIdx().y ] )*mix_params[6,134,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,135]*mix_params[1,135,threadIdx().x,blockIdx().y ]+mix_params[2,135,threadIdx().x,blockIdx().y ] )*mix_params[3,135,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,135]*mix_params[4,135,threadIdx().x,blockIdx().y ]+mix_params[5,135,threadIdx().x,blockIdx().y ] )*mix_params[6,135,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,136]*mix_params[1,136,threadIdx().x,blockIdx().y ]+mix_params[2,136,threadIdx().x,blockIdx().y ] )*mix_params[3,136,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,136]*mix_params[4,136,threadIdx().x,blockIdx().y ]+mix_params[5,136,threadIdx().x,blockIdx().y ] )*mix_params[6,136,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,137]*mix_params[1,137,threadIdx().x,blockIdx().y ]+mix_params[2,137,threadIdx().x,blockIdx().y ] )*mix_params[3,137,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,137]*mix_params[4,137,threadIdx().x,blockIdx().y ]+mix_params[5,137,threadIdx().x,blockIdx().y ] )*mix_params[6,137,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,138]*mix_params[1,138,threadIdx().x,blockIdx().y ]+mix_params[2,138,threadIdx().x,blockIdx().y ] )*mix_params[3,138,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,138]*mix_params[4,138,threadIdx().x,blockIdx().y ]+mix_params[5,138,threadIdx().x,blockIdx().y ] )*mix_params[6,138,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,139]*mix_params[1,139,threadIdx().x,blockIdx().y ]+mix_params[2,139,threadIdx().x,blockIdx().y ] )*mix_params[3,139,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,139]*mix_params[4,139,threadIdx().x,blockIdx().y ]+mix_params[5,139,threadIdx().x,blockIdx().y ] )*mix_params[6,139,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,140]*mix_params[1,140,threadIdx().x,blockIdx().y ]+mix_params[2,140,threadIdx().x,blockIdx().y ] )*mix_params[3,140,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,140]*mix_params[4,140,threadIdx().x,blockIdx().y ]+mix_params[5,140,threadIdx().x,blockIdx().y ] )*mix_params[6,140,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,141]*mix_params[1,141,threadIdx().x,blockIdx().y ]+mix_params[2,141,threadIdx().x,blockIdx().y ] )*mix_params[3,141,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,141]*mix_params[4,141,threadIdx().x,blockIdx().y ]+mix_params[5,141,threadIdx().x,blockIdx().y ] )*mix_params[6,141,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,142]*mix_params[1,142,threadIdx().x,blockIdx().y ]+mix_params[2,142,threadIdx().x,blockIdx().y ] )*mix_params[3,142,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,142]*mix_params[4,142,threadIdx().x,blockIdx().y ]+mix_params[5,142,threadIdx().x,blockIdx().y ] )*mix_params[6,142,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,143]*mix_params[1,143,threadIdx().x,blockIdx().y ]+mix_params[2,143,threadIdx().x,blockIdx().y ] )*mix_params[3,143,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,143]*mix_params[4,143,threadIdx().x,blockIdx().y ]+mix_params[5,143,threadIdx().x,blockIdx().y ] )*mix_params[6,143,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,144]*mix_params[1,144,threadIdx().x,blockIdx().y ]+mix_params[2,144,threadIdx().x,blockIdx().y ] )*mix_params[3,144,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,144]*mix_params[4,144,threadIdx().x,blockIdx().y ]+mix_params[5,144,threadIdx().x,blockIdx().y ] )*mix_params[6,144,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,145]*mix_params[1,145,threadIdx().x,blockIdx().y ]+mix_params[2,145,threadIdx().x,blockIdx().y ] )*mix_params[3,145,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,145]*mix_params[4,145,threadIdx().x,blockIdx().y ]+mix_params[5,145,threadIdx().x,blockIdx().y ] )*mix_params[6,145,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,146]*mix_params[1,146,threadIdx().x,blockIdx().y ]+mix_params[2,146,threadIdx().x,blockIdx().y ] )*mix_params[3,146,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,146]*mix_params[4,146,threadIdx().x,blockIdx().y ]+mix_params[5,146,threadIdx().x,blockIdx().y ] )*mix_params[6,146,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,147]*mix_params[1,147,threadIdx().x,blockIdx().y ]+mix_params[2,147,threadIdx().x,blockIdx().y ] )*mix_params[3,147,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,147]*mix_params[4,147,threadIdx().x,blockIdx().y ]+mix_params[5,147,threadIdx().x,blockIdx().y ] )*mix_params[6,147,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,148]*mix_params[1,148,threadIdx().x,blockIdx().y ]+mix_params[2,148,threadIdx().x,blockIdx().y ] )*mix_params[3,148,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,148]*mix_params[4,148,threadIdx().x,blockIdx().y ]+mix_params[5,148,threadIdx().x,blockIdx().y ] )*mix_params[6,148,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,149]*mix_params[1,149,threadIdx().x,blockIdx().y ]+mix_params[2,149,threadIdx().x,blockIdx().y ] )*mix_params[3,149,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,149]*mix_params[4,149,threadIdx().x,blockIdx().y ]+mix_params[5,149,threadIdx().x,blockIdx().y ] )*mix_params[6,149,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,150]*mix_params[1,150,threadIdx().x,blockIdx().y ]+mix_params[2,150,threadIdx().x,blockIdx().y ] )*mix_params[3,150,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,150]*mix_params[4,150,threadIdx().x,blockIdx().y ]+mix_params[5,150,threadIdx().x,blockIdx().y ] )*mix_params[6,150,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,151]*mix_params[1,151,threadIdx().x,blockIdx().y ]+mix_params[2,151,threadIdx().x,blockIdx().y ] )*mix_params[3,151,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,151]*mix_params[4,151,threadIdx().x,blockIdx().y ]+mix_params[5,151,threadIdx().x,blockIdx().y ] )*mix_params[6,151,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,152]*mix_params[1,152,threadIdx().x,blockIdx().y ]+mix_params[2,152,threadIdx().x,blockIdx().y ] )*mix_params[3,152,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,152]*mix_params[4,152,threadIdx().x,blockIdx().y ]+mix_params[5,152,threadIdx().x,blockIdx().y ] )*mix_params[6,152,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,153]*mix_params[1,153,threadIdx().x,blockIdx().y ]+mix_params[2,153,threadIdx().x,blockIdx().y ] )*mix_params[3,153,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,153]*mix_params[4,153,threadIdx().x,blockIdx().y ]+mix_params[5,153,threadIdx().x,blockIdx().y ] )*mix_params[6,153,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,154]*mix_params[1,154,threadIdx().x,blockIdx().y ]+mix_params[2,154,threadIdx().x,blockIdx().y ] )*mix_params[3,154,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,154]*mix_params[4,154,threadIdx().x,blockIdx().y ]+mix_params[5,154,threadIdx().x,blockIdx().y ] )*mix_params[6,154,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,155]*mix_params[1,155,threadIdx().x,blockIdx().y ]+mix_params[2,155,threadIdx().x,blockIdx().y ] )*mix_params[3,155,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,155]*mix_params[4,155,threadIdx().x,blockIdx().y ]+mix_params[5,155,threadIdx().x,blockIdx().y ] )*mix_params[6,155,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,156]*mix_params[1,156,threadIdx().x,blockIdx().y ]+mix_params[2,156,threadIdx().x,blockIdx().y ] )*mix_params[3,156,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,156]*mix_params[4,156,threadIdx().x,blockIdx().y ]+mix_params[5,156,threadIdx().x,blockIdx().y ] )*mix_params[6,156,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,157]*mix_params[1,157,threadIdx().x,blockIdx().y ]+mix_params[2,157,threadIdx().x,blockIdx().y ] )*mix_params[3,157,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,157]*mix_params[4,157,threadIdx().x,blockIdx().y ]+mix_params[5,157,threadIdx().x,blockIdx().y ] )*mix_params[6,157,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,158]*mix_params[1,158,threadIdx().x,blockIdx().y ]+mix_params[2,158,threadIdx().x,blockIdx().y ] )*mix_params[3,158,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,158]*mix_params[4,158,threadIdx().x,blockIdx().y ]+mix_params[5,158,threadIdx().x,blockIdx().y ] )*mix_params[6,158,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,159]*mix_params[1,159,threadIdx().x,blockIdx().y ]+mix_params[2,159,threadIdx().x,blockIdx().y ] )*mix_params[3,159,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,159]*mix_params[4,159,threadIdx().x,blockIdx().y ]+mix_params[5,159,threadIdx().x,blockIdx().y ] )*mix_params[6,159,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,160]*mix_params[1,160,threadIdx().x,blockIdx().y ]+mix_params[2,160,threadIdx().x,blockIdx().y ] )*mix_params[3,160,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,160]*mix_params[4,160,threadIdx().x,blockIdx().y ]+mix_params[5,160,threadIdx().x,blockIdx().y ] )*mix_params[6,160,threadIdx().x,blockIdx().y ])
)


shared_arr[1,threadIdx().x]=shared_arr[1,threadIdx().x]+( (sin(shared_arr[2,161]*mix_params[1,161,threadIdx().x,blockIdx().y ]+mix_params[2,161,threadIdx().x,blockIdx().y ] )*mix_params[3,161,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,161]*mix_params[4,161,threadIdx().x,blockIdx().y ]+mix_params[5,161,threadIdx().x,blockIdx().y ] )*mix_params[6,161,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,162]*mix_params[1,162,threadIdx().x,blockIdx().y ]+mix_params[2,162,threadIdx().x,blockIdx().y ] )*mix_params[3,162,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,162]*mix_params[4,162,threadIdx().x,blockIdx().y ]+mix_params[5,162,threadIdx().x,blockIdx().y ] )*mix_params[6,162,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,163]*mix_params[1,163,threadIdx().x,blockIdx().y ]+mix_params[2,163,threadIdx().x,blockIdx().y ] )*mix_params[3,163,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,163]*mix_params[4,163,threadIdx().x,blockIdx().y ]+mix_params[5,163,threadIdx().x,blockIdx().y ] )*mix_params[6,163,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,164]*mix_params[1,164,threadIdx().x,blockIdx().y ]+mix_params[2,164,threadIdx().x,blockIdx().y ] )*mix_params[3,164,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,164]*mix_params[4,164,threadIdx().x,blockIdx().y ]+mix_params[5,164,threadIdx().x,blockIdx().y ] )*mix_params[6,164,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,165]*mix_params[1,165,threadIdx().x,blockIdx().y ]+mix_params[2,165,threadIdx().x,blockIdx().y ] )*mix_params[3,165,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,165]*mix_params[4,165,threadIdx().x,blockIdx().y ]+mix_params[5,165,threadIdx().x,blockIdx().y ] )*mix_params[6,165,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,166]*mix_params[1,166,threadIdx().x,blockIdx().y ]+mix_params[2,166,threadIdx().x,blockIdx().y ] )*mix_params[3,166,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,166]*mix_params[4,166,threadIdx().x,blockIdx().y ]+mix_params[5,166,threadIdx().x,blockIdx().y ] )*mix_params[6,166,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,167]*mix_params[1,167,threadIdx().x,blockIdx().y ]+mix_params[2,167,threadIdx().x,blockIdx().y ] )*mix_params[3,167,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,167]*mix_params[4,167,threadIdx().x,blockIdx().y ]+mix_params[5,167,threadIdx().x,blockIdx().y ] )*mix_params[6,167,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,168]*mix_params[1,168,threadIdx().x,blockIdx().y ]+mix_params[2,168,threadIdx().x,blockIdx().y ] )*mix_params[3,168,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,168]*mix_params[4,168,threadIdx().x,blockIdx().y ]+mix_params[5,168,threadIdx().x,blockIdx().y ] )*mix_params[6,168,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,169]*mix_params[1,169,threadIdx().x,blockIdx().y ]+mix_params[2,169,threadIdx().x,blockIdx().y ] )*mix_params[3,169,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,169]*mix_params[4,169,threadIdx().x,blockIdx().y ]+mix_params[5,169,threadIdx().x,blockIdx().y ] )*mix_params[6,169,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,170]*mix_params[1,170,threadIdx().x,blockIdx().y ]+mix_params[2,170,threadIdx().x,blockIdx().y ] )*mix_params[3,170,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,170]*mix_params[4,170,threadIdx().x,blockIdx().y ]+mix_params[5,170,threadIdx().x,blockIdx().y ] )*mix_params[6,170,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,171]*mix_params[1,171,threadIdx().x,blockIdx().y ]+mix_params[2,171,threadIdx().x,blockIdx().y ] )*mix_params[3,171,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,171]*mix_params[4,171,threadIdx().x,blockIdx().y ]+mix_params[5,171,threadIdx().x,blockIdx().y ] )*mix_params[6,171,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,172]*mix_params[1,172,threadIdx().x,blockIdx().y ]+mix_params[2,172,threadIdx().x,blockIdx().y ] )*mix_params[3,172,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,172]*mix_params[4,172,threadIdx().x,blockIdx().y ]+mix_params[5,172,threadIdx().x,blockIdx().y ] )*mix_params[6,172,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,173]*mix_params[1,173,threadIdx().x,blockIdx().y ]+mix_params[2,173,threadIdx().x,blockIdx().y ] )*mix_params[3,173,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,173]*mix_params[4,173,threadIdx().x,blockIdx().y ]+mix_params[5,173,threadIdx().x,blockIdx().y ] )*mix_params[6,173,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,174]*mix_params[1,174,threadIdx().x,blockIdx().y ]+mix_params[2,174,threadIdx().x,blockIdx().y ] )*mix_params[3,174,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,174]*mix_params[4,174,threadIdx().x,blockIdx().y ]+mix_params[5,174,threadIdx().x,blockIdx().y ] )*mix_params[6,174,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,175]*mix_params[1,175,threadIdx().x,blockIdx().y ]+mix_params[2,175,threadIdx().x,blockIdx().y ] )*mix_params[3,175,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,175]*mix_params[4,175,threadIdx().x,blockIdx().y ]+mix_params[5,175,threadIdx().x,blockIdx().y ] )*mix_params[6,175,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,176]*mix_params[1,176,threadIdx().x,blockIdx().y ]+mix_params[2,176,threadIdx().x,blockIdx().y ] )*mix_params[3,176,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,176]*mix_params[4,176,threadIdx().x,blockIdx().y ]+mix_params[5,176,threadIdx().x,blockIdx().y ] )*mix_params[6,176,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,177]*mix_params[1,177,threadIdx().x,blockIdx().y ]+mix_params[2,177,threadIdx().x,blockIdx().y ] )*mix_params[3,177,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,177]*mix_params[4,177,threadIdx().x,blockIdx().y ]+mix_params[5,177,threadIdx().x,blockIdx().y ] )*mix_params[6,177,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,178]*mix_params[1,178,threadIdx().x,blockIdx().y ]+mix_params[2,178,threadIdx().x,blockIdx().y ] )*mix_params[3,178,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,178]*mix_params[4,178,threadIdx().x,blockIdx().y ]+mix_params[5,178,threadIdx().x,blockIdx().y ] )*mix_params[6,178,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,179]*mix_params[1,179,threadIdx().x,blockIdx().y ]+mix_params[2,179,threadIdx().x,blockIdx().y ] )*mix_params[3,179,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,179]*mix_params[4,179,threadIdx().x,blockIdx().y ]+mix_params[5,179,threadIdx().x,blockIdx().y ] )*mix_params[6,179,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,180]*mix_params[1,180,threadIdx().x,blockIdx().y ]+mix_params[2,180,threadIdx().x,blockIdx().y ] )*mix_params[3,180,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,180]*mix_params[4,180,threadIdx().x,blockIdx().y ]+mix_params[5,180,threadIdx().x,blockIdx().y ] )*mix_params[6,180,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,181]*mix_params[1,181,threadIdx().x,blockIdx().y ]+mix_params[2,181,threadIdx().x,blockIdx().y ] )*mix_params[3,181,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,181]*mix_params[4,181,threadIdx().x,blockIdx().y ]+mix_params[5,181,threadIdx().x,blockIdx().y ] )*mix_params[6,181,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,182]*mix_params[1,182,threadIdx().x,blockIdx().y ]+mix_params[2,182,threadIdx().x,blockIdx().y ] )*mix_params[3,182,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,182]*mix_params[4,182,threadIdx().x,blockIdx().y ]+mix_params[5,182,threadIdx().x,blockIdx().y ] )*mix_params[6,182,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,183]*mix_params[1,183,threadIdx().x,blockIdx().y ]+mix_params[2,183,threadIdx().x,blockIdx().y ] )*mix_params[3,183,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,183]*mix_params[4,183,threadIdx().x,blockIdx().y ]+mix_params[5,183,threadIdx().x,blockIdx().y ] )*mix_params[6,183,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,184]*mix_params[1,184,threadIdx().x,blockIdx().y ]+mix_params[2,184,threadIdx().x,blockIdx().y ] )*mix_params[3,184,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,184]*mix_params[4,184,threadIdx().x,blockIdx().y ]+mix_params[5,184,threadIdx().x,blockIdx().y ] )*mix_params[6,184,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,185]*mix_params[1,185,threadIdx().x,blockIdx().y ]+mix_params[2,185,threadIdx().x,blockIdx().y ] )*mix_params[3,185,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,185]*mix_params[4,185,threadIdx().x,blockIdx().y ]+mix_params[5,185,threadIdx().x,blockIdx().y ] )*mix_params[6,185,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,186]*mix_params[1,186,threadIdx().x,blockIdx().y ]+mix_params[2,186,threadIdx().x,blockIdx().y ] )*mix_params[3,186,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,186]*mix_params[4,186,threadIdx().x,blockIdx().y ]+mix_params[5,186,threadIdx().x,blockIdx().y ] )*mix_params[6,186,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,187]*mix_params[1,187,threadIdx().x,blockIdx().y ]+mix_params[2,187,threadIdx().x,blockIdx().y ] )*mix_params[3,187,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,187]*mix_params[4,187,threadIdx().x,blockIdx().y ]+mix_params[5,187,threadIdx().x,blockIdx().y ] )*mix_params[6,187,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,188]*mix_params[1,188,threadIdx().x,blockIdx().y ]+mix_params[2,188,threadIdx().x,blockIdx().y ] )*mix_params[3,188,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,188]*mix_params[4,188,threadIdx().x,blockIdx().y ]+mix_params[5,188,threadIdx().x,blockIdx().y ] )*mix_params[6,188,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,189]*mix_params[1,189,threadIdx().x,blockIdx().y ]+mix_params[2,189,threadIdx().x,blockIdx().y ] )*mix_params[3,189,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,189]*mix_params[4,189,threadIdx().x,blockIdx().y ]+mix_params[5,189,threadIdx().x,blockIdx().y ] )*mix_params[6,189,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,190]*mix_params[1,190,threadIdx().x,blockIdx().y ]+mix_params[2,190,threadIdx().x,blockIdx().y ] )*mix_params[3,190,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,190]*mix_params[4,190,threadIdx().x,blockIdx().y ]+mix_params[5,190,threadIdx().x,blockIdx().y ] )*mix_params[6,190,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,191]*mix_params[1,191,threadIdx().x,blockIdx().y ]+mix_params[2,191,threadIdx().x,blockIdx().y ] )*mix_params[3,191,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,191]*mix_params[4,191,threadIdx().x,blockIdx().y ]+mix_params[5,191,threadIdx().x,blockIdx().y ] )*mix_params[6,191,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,192]*mix_params[1,192,threadIdx().x,blockIdx().y ]+mix_params[2,192,threadIdx().x,blockIdx().y ] )*mix_params[3,192,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,192]*mix_params[4,192,threadIdx().x,blockIdx().y ]+mix_params[5,192,threadIdx().x,blockIdx().y ] )*mix_params[6,192,threadIdx().x,blockIdx().y ])
)


shared_arr[1,threadIdx().x]=shared_arr[1,threadIdx().x]+( (sin(shared_arr[2,193]*mix_params[1,193,threadIdx().x,blockIdx().y ]+mix_params[2,193,threadIdx().x,blockIdx().y ] )*mix_params[3,193,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,193]*mix_params[4,193,threadIdx().x,blockIdx().y ]+mix_params[5,193,threadIdx().x,blockIdx().y ] )*mix_params[6,193,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,194]*mix_params[1,194,threadIdx().x,blockIdx().y ]+mix_params[2,194,threadIdx().x,blockIdx().y ] )*mix_params[3,194,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,194]*mix_params[4,194,threadIdx().x,blockIdx().y ]+mix_params[5,194,threadIdx().x,blockIdx().y ] )*mix_params[6,194,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,195]*mix_params[1,195,threadIdx().x,blockIdx().y ]+mix_params[2,195,threadIdx().x,blockIdx().y ] )*mix_params[3,195,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,195]*mix_params[4,195,threadIdx().x,blockIdx().y ]+mix_params[5,195,threadIdx().x,blockIdx().y ] )*mix_params[6,195,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,196]*mix_params[1,196,threadIdx().x,blockIdx().y ]+mix_params[2,196,threadIdx().x,blockIdx().y ] )*mix_params[3,196,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,196]*mix_params[4,196,threadIdx().x,blockIdx().y ]+mix_params[5,196,threadIdx().x,blockIdx().y ] )*mix_params[6,196,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,197]*mix_params[1,197,threadIdx().x,blockIdx().y ]+mix_params[2,197,threadIdx().x,blockIdx().y ] )*mix_params[3,197,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,197]*mix_params[4,197,threadIdx().x,blockIdx().y ]+mix_params[5,197,threadIdx().x,blockIdx().y ] )*mix_params[6,197,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,198]*mix_params[1,198,threadIdx().x,blockIdx().y ]+mix_params[2,198,threadIdx().x,blockIdx().y ] )*mix_params[3,198,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,198]*mix_params[4,198,threadIdx().x,blockIdx().y ]+mix_params[5,198,threadIdx().x,blockIdx().y ] )*mix_params[6,198,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,199]*mix_params[1,199,threadIdx().x,blockIdx().y ]+mix_params[2,199,threadIdx().x,blockIdx().y ] )*mix_params[3,199,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,199]*mix_params[4,199,threadIdx().x,blockIdx().y ]+mix_params[5,199,threadIdx().x,blockIdx().y ] )*mix_params[6,199,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,200]*mix_params[1,200,threadIdx().x,blockIdx().y ]+mix_params[2,200,threadIdx().x,blockIdx().y ] )*mix_params[3,200,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,200]*mix_params[4,200,threadIdx().x,blockIdx().y ]+mix_params[5,200,threadIdx().x,blockIdx().y ] )*mix_params[6,200,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,201]*mix_params[1,201,threadIdx().x,blockIdx().y ]+mix_params[2,201,threadIdx().x,blockIdx().y ] )*mix_params[3,201,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,201]*mix_params[4,201,threadIdx().x,blockIdx().y ]+mix_params[5,201,threadIdx().x,blockIdx().y ] )*mix_params[6,201,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,202]*mix_params[1,202,threadIdx().x,blockIdx().y ]+mix_params[2,202,threadIdx().x,blockIdx().y ] )*mix_params[3,202,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,202]*mix_params[4,202,threadIdx().x,blockIdx().y ]+mix_params[5,202,threadIdx().x,blockIdx().y ] )*mix_params[6,202,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,203]*mix_params[1,203,threadIdx().x,blockIdx().y ]+mix_params[2,203,threadIdx().x,blockIdx().y ] )*mix_params[3,203,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,203]*mix_params[4,203,threadIdx().x,blockIdx().y ]+mix_params[5,203,threadIdx().x,blockIdx().y ] )*mix_params[6,203,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,204]*mix_params[1,204,threadIdx().x,blockIdx().y ]+mix_params[2,204,threadIdx().x,blockIdx().y ] )*mix_params[3,204,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,204]*mix_params[4,204,threadIdx().x,blockIdx().y ]+mix_params[5,204,threadIdx().x,blockIdx().y ] )*mix_params[6,204,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,205]*mix_params[1,205,threadIdx().x,blockIdx().y ]+mix_params[2,205,threadIdx().x,blockIdx().y ] )*mix_params[3,205,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,205]*mix_params[4,205,threadIdx().x,blockIdx().y ]+mix_params[5,205,threadIdx().x,blockIdx().y ] )*mix_params[6,205,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,206]*mix_params[1,206,threadIdx().x,blockIdx().y ]+mix_params[2,206,threadIdx().x,blockIdx().y ] )*mix_params[3,206,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,206]*mix_params[4,206,threadIdx().x,blockIdx().y ]+mix_params[5,206,threadIdx().x,blockIdx().y ] )*mix_params[6,206,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,207]*mix_params[1,207,threadIdx().x,blockIdx().y ]+mix_params[2,207,threadIdx().x,blockIdx().y ] )*mix_params[3,207,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,207]*mix_params[4,207,threadIdx().x,blockIdx().y ]+mix_params[5,207,threadIdx().x,blockIdx().y ] )*mix_params[6,207,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,208]*mix_params[1,208,threadIdx().x,blockIdx().y ]+mix_params[2,208,threadIdx().x,blockIdx().y ] )*mix_params[3,208,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,208]*mix_params[4,208,threadIdx().x,blockIdx().y ]+mix_params[5,208,threadIdx().x,blockIdx().y ] )*mix_params[6,208,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,209]*mix_params[1,209,threadIdx().x,blockIdx().y ]+mix_params[2,209,threadIdx().x,blockIdx().y ] )*mix_params[3,209,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,209]*mix_params[4,209,threadIdx().x,blockIdx().y ]+mix_params[5,209,threadIdx().x,blockIdx().y ] )*mix_params[6,209,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,210]*mix_params[1,210,threadIdx().x,blockIdx().y ]+mix_params[2,210,threadIdx().x,blockIdx().y ] )*mix_params[3,210,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,210]*mix_params[4,210,threadIdx().x,blockIdx().y ]+mix_params[5,210,threadIdx().x,blockIdx().y ] )*mix_params[6,210,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,211]*mix_params[1,211,threadIdx().x,blockIdx().y ]+mix_params[2,211,threadIdx().x,blockIdx().y ] )*mix_params[3,211,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,211]*mix_params[4,211,threadIdx().x,blockIdx().y ]+mix_params[5,211,threadIdx().x,blockIdx().y ] )*mix_params[6,211,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,212]*mix_params[1,212,threadIdx().x,blockIdx().y ]+mix_params[2,212,threadIdx().x,blockIdx().y ] )*mix_params[3,212,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,212]*mix_params[4,212,threadIdx().x,blockIdx().y ]+mix_params[5,212,threadIdx().x,blockIdx().y ] )*mix_params[6,212,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,213]*mix_params[1,213,threadIdx().x,blockIdx().y ]+mix_params[2,213,threadIdx().x,blockIdx().y ] )*mix_params[3,213,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,213]*mix_params[4,213,threadIdx().x,blockIdx().y ]+mix_params[5,213,threadIdx().x,blockIdx().y ] )*mix_params[6,213,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,214]*mix_params[1,214,threadIdx().x,blockIdx().y ]+mix_params[2,214,threadIdx().x,blockIdx().y ] )*mix_params[3,214,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,214]*mix_params[4,214,threadIdx().x,blockIdx().y ]+mix_params[5,214,threadIdx().x,blockIdx().y ] )*mix_params[6,214,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,215]*mix_params[1,215,threadIdx().x,blockIdx().y ]+mix_params[2,215,threadIdx().x,blockIdx().y ] )*mix_params[3,215,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,215]*mix_params[4,215,threadIdx().x,blockIdx().y ]+mix_params[5,215,threadIdx().x,blockIdx().y ] )*mix_params[6,215,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,216]*mix_params[1,216,threadIdx().x,blockIdx().y ]+mix_params[2,216,threadIdx().x,blockIdx().y ] )*mix_params[3,216,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,216]*mix_params[4,216,threadIdx().x,blockIdx().y ]+mix_params[5,216,threadIdx().x,blockIdx().y ] )*mix_params[6,216,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,217]*mix_params[1,217,threadIdx().x,blockIdx().y ]+mix_params[2,217,threadIdx().x,blockIdx().y ] )*mix_params[3,217,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,217]*mix_params[4,217,threadIdx().x,blockIdx().y ]+mix_params[5,217,threadIdx().x,blockIdx().y ] )*mix_params[6,217,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,218]*mix_params[1,218,threadIdx().x,blockIdx().y ]+mix_params[2,218,threadIdx().x,blockIdx().y ] )*mix_params[3,218,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,218]*mix_params[4,218,threadIdx().x,blockIdx().y ]+mix_params[5,218,threadIdx().x,blockIdx().y ] )*mix_params[6,218,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,219]*mix_params[1,219,threadIdx().x,blockIdx().y ]+mix_params[2,219,threadIdx().x,blockIdx().y ] )*mix_params[3,219,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,219]*mix_params[4,219,threadIdx().x,blockIdx().y ]+mix_params[5,219,threadIdx().x,blockIdx().y ] )*mix_params[6,219,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,220]*mix_params[1,220,threadIdx().x,blockIdx().y ]+mix_params[2,220,threadIdx().x,blockIdx().y ] )*mix_params[3,220,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,220]*mix_params[4,220,threadIdx().x,blockIdx().y ]+mix_params[5,220,threadIdx().x,blockIdx().y ] )*mix_params[6,220,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,221]*mix_params[1,221,threadIdx().x,blockIdx().y ]+mix_params[2,221,threadIdx().x,blockIdx().y ] )*mix_params[3,221,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,221]*mix_params[4,221,threadIdx().x,blockIdx().y ]+mix_params[5,221,threadIdx().x,blockIdx().y ] )*mix_params[6,221,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,222]*mix_params[1,222,threadIdx().x,blockIdx().y ]+mix_params[2,222,threadIdx().x,blockIdx().y ] )*mix_params[3,222,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,222]*mix_params[4,222,threadIdx().x,blockIdx().y ]+mix_params[5,222,threadIdx().x,blockIdx().y ] )*mix_params[6,222,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,223]*mix_params[1,223,threadIdx().x,blockIdx().y ]+mix_params[2,223,threadIdx().x,blockIdx().y ] )*mix_params[3,223,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,223]*mix_params[4,223,threadIdx().x,blockIdx().y ]+mix_params[5,223,threadIdx().x,blockIdx().y ] )*mix_params[6,223,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,224]*mix_params[1,224,threadIdx().x,blockIdx().y ]+mix_params[2,224,threadIdx().x,blockIdx().y ] )*mix_params[3,224,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,224]*mix_params[4,224,threadIdx().x,blockIdx().y ]+mix_params[5,224,threadIdx().x,blockIdx().y ] )*mix_params[6,224,threadIdx().x,blockIdx().y ])
)


shared_arr[1,threadIdx().x]=shared_arr[1,threadIdx().x]+( (sin(shared_arr[2,225]*mix_params[1,225,threadIdx().x,blockIdx().y ]+mix_params[2,225,threadIdx().x,blockIdx().y ] )*mix_params[3,225,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,225]*mix_params[4,225,threadIdx().x,blockIdx().y ]+mix_params[5,225,threadIdx().x,blockIdx().y ] )*mix_params[6,225,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,226]*mix_params[1,226,threadIdx().x,blockIdx().y ]+mix_params[2,226,threadIdx().x,blockIdx().y ] )*mix_params[3,226,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,226]*mix_params[4,226,threadIdx().x,blockIdx().y ]+mix_params[5,226,threadIdx().x,blockIdx().y ] )*mix_params[6,226,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,227]*mix_params[1,227,threadIdx().x,blockIdx().y ]+mix_params[2,227,threadIdx().x,blockIdx().y ] )*mix_params[3,227,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,227]*mix_params[4,227,threadIdx().x,blockIdx().y ]+mix_params[5,227,threadIdx().x,blockIdx().y ] )*mix_params[6,227,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,228]*mix_params[1,228,threadIdx().x,blockIdx().y ]+mix_params[2,228,threadIdx().x,blockIdx().y ] )*mix_params[3,228,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,228]*mix_params[4,228,threadIdx().x,blockIdx().y ]+mix_params[5,228,threadIdx().x,blockIdx().y ] )*mix_params[6,228,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,229]*mix_params[1,229,threadIdx().x,blockIdx().y ]+mix_params[2,229,threadIdx().x,blockIdx().y ] )*mix_params[3,229,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,229]*mix_params[4,229,threadIdx().x,blockIdx().y ]+mix_params[5,229,threadIdx().x,blockIdx().y ] )*mix_params[6,229,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,230]*mix_params[1,230,threadIdx().x,blockIdx().y ]+mix_params[2,230,threadIdx().x,blockIdx().y ] )*mix_params[3,230,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,230]*mix_params[4,230,threadIdx().x,blockIdx().y ]+mix_params[5,230,threadIdx().x,blockIdx().y ] )*mix_params[6,230,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,231]*mix_params[1,231,threadIdx().x,blockIdx().y ]+mix_params[2,231,threadIdx().x,blockIdx().y ] )*mix_params[3,231,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,231]*mix_params[4,231,threadIdx().x,blockIdx().y ]+mix_params[5,231,threadIdx().x,blockIdx().y ] )*mix_params[6,231,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,232]*mix_params[1,232,threadIdx().x,blockIdx().y ]+mix_params[2,232,threadIdx().x,blockIdx().y ] )*mix_params[3,232,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,232]*mix_params[4,232,threadIdx().x,blockIdx().y ]+mix_params[5,232,threadIdx().x,blockIdx().y ] )*mix_params[6,232,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,233]*mix_params[1,233,threadIdx().x,blockIdx().y ]+mix_params[2,233,threadIdx().x,blockIdx().y ] )*mix_params[3,233,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,233]*mix_params[4,233,threadIdx().x,blockIdx().y ]+mix_params[5,233,threadIdx().x,blockIdx().y ] )*mix_params[6,233,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,234]*mix_params[1,234,threadIdx().x,blockIdx().y ]+mix_params[2,234,threadIdx().x,blockIdx().y ] )*mix_params[3,234,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,234]*mix_params[4,234,threadIdx().x,blockIdx().y ]+mix_params[5,234,threadIdx().x,blockIdx().y ] )*mix_params[6,234,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,235]*mix_params[1,235,threadIdx().x,blockIdx().y ]+mix_params[2,235,threadIdx().x,blockIdx().y ] )*mix_params[3,235,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,235]*mix_params[4,235,threadIdx().x,blockIdx().y ]+mix_params[5,235,threadIdx().x,blockIdx().y ] )*mix_params[6,235,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,236]*mix_params[1,236,threadIdx().x,blockIdx().y ]+mix_params[2,236,threadIdx().x,blockIdx().y ] )*mix_params[3,236,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,236]*mix_params[4,236,threadIdx().x,blockIdx().y ]+mix_params[5,236,threadIdx().x,blockIdx().y ] )*mix_params[6,236,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,237]*mix_params[1,237,threadIdx().x,blockIdx().y ]+mix_params[2,237,threadIdx().x,blockIdx().y ] )*mix_params[3,237,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,237]*mix_params[4,237,threadIdx().x,blockIdx().y ]+mix_params[5,237,threadIdx().x,blockIdx().y ] )*mix_params[6,237,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,238]*mix_params[1,238,threadIdx().x,blockIdx().y ]+mix_params[2,238,threadIdx().x,blockIdx().y ] )*mix_params[3,238,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,238]*mix_params[4,238,threadIdx().x,blockIdx().y ]+mix_params[5,238,threadIdx().x,blockIdx().y ] )*mix_params[6,238,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,239]*mix_params[1,239,threadIdx().x,blockIdx().y ]+mix_params[2,239,threadIdx().x,blockIdx().y ] )*mix_params[3,239,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,239]*mix_params[4,239,threadIdx().x,blockIdx().y ]+mix_params[5,239,threadIdx().x,blockIdx().y ] )*mix_params[6,239,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,240]*mix_params[1,240,threadIdx().x,blockIdx().y ]+mix_params[2,240,threadIdx().x,blockIdx().y ] )*mix_params[3,240,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,240]*mix_params[4,240,threadIdx().x,blockIdx().y ]+mix_params[5,240,threadIdx().x,blockIdx().y ] )*mix_params[6,240,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,241]*mix_params[1,241,threadIdx().x,blockIdx().y ]+mix_params[2,241,threadIdx().x,blockIdx().y ] )*mix_params[3,241,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,241]*mix_params[4,241,threadIdx().x,blockIdx().y ]+mix_params[5,241,threadIdx().x,blockIdx().y ] )*mix_params[6,241,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,242]*mix_params[1,242,threadIdx().x,blockIdx().y ]+mix_params[2,242,threadIdx().x,blockIdx().y ] )*mix_params[3,242,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,242]*mix_params[4,242,threadIdx().x,blockIdx().y ]+mix_params[5,242,threadIdx().x,blockIdx().y ] )*mix_params[6,242,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,243]*mix_params[1,243,threadIdx().x,blockIdx().y ]+mix_params[2,243,threadIdx().x,blockIdx().y ] )*mix_params[3,243,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,243]*mix_params[4,243,threadIdx().x,blockIdx().y ]+mix_params[5,243,threadIdx().x,blockIdx().y ] )*mix_params[6,243,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,244]*mix_params[1,244,threadIdx().x,blockIdx().y ]+mix_params[2,244,threadIdx().x,blockIdx().y ] )*mix_params[3,244,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,244]*mix_params[4,244,threadIdx().x,blockIdx().y ]+mix_params[5,244,threadIdx().x,blockIdx().y ] )*mix_params[6,244,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,245]*mix_params[1,245,threadIdx().x,blockIdx().y ]+mix_params[2,245,threadIdx().x,blockIdx().y ] )*mix_params[3,245,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,245]*mix_params[4,245,threadIdx().x,blockIdx().y ]+mix_params[5,245,threadIdx().x,blockIdx().y ] )*mix_params[6,245,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,246]*mix_params[1,246,threadIdx().x,blockIdx().y ]+mix_params[2,246,threadIdx().x,blockIdx().y ] )*mix_params[3,246,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,246]*mix_params[4,246,threadIdx().x,blockIdx().y ]+mix_params[5,246,threadIdx().x,blockIdx().y ] )*mix_params[6,246,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,247]*mix_params[1,247,threadIdx().x,blockIdx().y ]+mix_params[2,247,threadIdx().x,blockIdx().y ] )*mix_params[3,247,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,247]*mix_params[4,247,threadIdx().x,blockIdx().y ]+mix_params[5,247,threadIdx().x,blockIdx().y ] )*mix_params[6,247,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,248]*mix_params[1,248,threadIdx().x,blockIdx().y ]+mix_params[2,248,threadIdx().x,blockIdx().y ] )*mix_params[3,248,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,248]*mix_params[4,248,threadIdx().x,blockIdx().y ]+mix_params[5,248,threadIdx().x,blockIdx().y ] )*mix_params[6,248,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,249]*mix_params[1,249,threadIdx().x,blockIdx().y ]+mix_params[2,249,threadIdx().x,blockIdx().y ] )*mix_params[3,249,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,249]*mix_params[4,249,threadIdx().x,blockIdx().y ]+mix_params[5,249,threadIdx().x,blockIdx().y ] )*mix_params[6,249,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,250]*mix_params[1,250,threadIdx().x,blockIdx().y ]+mix_params[2,250,threadIdx().x,blockIdx().y ] )*mix_params[3,250,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,250]*mix_params[4,250,threadIdx().x,blockIdx().y ]+mix_params[5,250,threadIdx().x,blockIdx().y ] )*mix_params[6,250,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,251]*mix_params[1,251,threadIdx().x,blockIdx().y ]+mix_params[2,251,threadIdx().x,blockIdx().y ] )*mix_params[3,251,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,251]*mix_params[4,251,threadIdx().x,blockIdx().y ]+mix_params[5,251,threadIdx().x,blockIdx().y ] )*mix_params[6,251,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,252]*mix_params[1,252,threadIdx().x,blockIdx().y ]+mix_params[2,252,threadIdx().x,blockIdx().y ] )*mix_params[3,252,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,252]*mix_params[4,252,threadIdx().x,blockIdx().y ]+mix_params[5,252,threadIdx().x,blockIdx().y ] )*mix_params[6,252,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,253]*mix_params[1,253,threadIdx().x,blockIdx().y ]+mix_params[2,253,threadIdx().x,blockIdx().y ] )*mix_params[3,253,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,253]*mix_params[4,253,threadIdx().x,blockIdx().y ]+mix_params[5,253,threadIdx().x,blockIdx().y ] )*mix_params[6,253,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,254]*mix_params[1,254,threadIdx().x,blockIdx().y ]+mix_params[2,254,threadIdx().x,blockIdx().y ] )*mix_params[3,254,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,254]*mix_params[4,254,threadIdx().x,blockIdx().y ]+mix_params[5,254,threadIdx().x,blockIdx().y ] )*mix_params[6,254,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,255]*mix_params[1,255,threadIdx().x,blockIdx().y ]+mix_params[2,255,threadIdx().x,blockIdx().y ] )*mix_params[3,255,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,255]*mix_params[4,255,threadIdx().x,blockIdx().y ]+mix_params[5,255,threadIdx().x,blockIdx().y ] )*mix_params[6,255,threadIdx().x,blockIdx().y ])
 +  (sin(shared_arr[2,256]*mix_params[1,256,threadIdx().x,blockIdx().y ]+mix_params[2,256,threadIdx().x,blockIdx().y ] )*mix_params[3,256,threadIdx().x,blockIdx().y ]+
cos(shared_arr[2,256]*mix_params[4,256,threadIdx().x,blockIdx().y ]+mix_params[5,256,threadIdx().x,blockIdx().y ] )*mix_params[6,256,threadIdx().x,blockIdx().y ])
)

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


# cp_x=5
# cp_y=5
# cp_z=5
# batch_size=5
# mixing_param_sets=4
# Res_a=rand(cp_x, cp_y, cp_z,mixing_param_sets*256, batch_size)

# Res_b=reshape(Res_a, cp_x* cp_y* cp_z,mixing_param_sets, 256,batch_size)

# Res_c=reshape(Res_b, cp_x, cp_y, cp_z,mixing_param_sets*256,batch_size)

# Res_a==Res_c