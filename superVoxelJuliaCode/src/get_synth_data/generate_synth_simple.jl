"""
generate simple data for experiments with the differentiable supervoxel framework

"""
module generate_synth_simple

export saveAndVisRandGraphNoAlgoOutPutSeen,getRand_graph_andArr
using EmbeddedGraphs
using Distances
using Graphs
using CUDA
using ParallelStencil
using Distributions
using InferOpt
using Optimisers
using Flux
using Lux
using ForwardDiff
using Pkg
Pkg.add(url="https://github.com/jakubMitura14/MedEye3d.jl.git")
# Pkg.add(url="https://github.com/jakubMitura14/MedPipe3D.jl.git")
import MedEye3d,MedPipe3D
import MedEye3d.ForDisplayStructs
import MedEye3d.ForDisplayStructs.TextureSpec
using ColorTypes
import MedEye3d.SegmentationDisplay
import MedEye3d.DataStructs.ThreeDimRawDat
import MedEye3d.DataStructs.DataToScrollDims
import MedEye3d.DataStructs.FullScrollableDat
import MedEye3d.ForDisplayStructs.KeyboardStruct
import MedEye3d.ForDisplayStructs.MouseStruct
import MedEye3d.ForDisplayStructs.ActorWithOpenGlObjects
import MedEye3d.OpenGLDisplayUtils
import MedEye3d.DisplayWords.textLinesFromStrings
import MedEye3d.StructsManag.getThreeDims
import MedEye3d.DisplayWords.textLinesFromStrings
using HDF5,Colors
using MedPipe3D.LoadFromMonai, MedPipe3D.HDF5saveUtils,MedEye3d.visualizationFromHdf5, MedEye3d.distinctColorsSaved

# patienGroupName="1"
# pathToHDF5="/home/jakub/projects/hdf5Data/forGraphDataSet.hdf5"
# listOfColorUsed= falses(18)

# fid = h5open(pathToHDF5, "r+")
# gr= getGroupOrCreate(fid, patienGroupName)
# addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,0)
# writeGroupAttribute(fid,patienGroupName, "spacing", [1,1,1])

# mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)


@init_parallel_stencil(Threads, Float64, 3);



"""
idea is to create a random 3d Graphs  
    then save it as the 3d image - array
    1)define the gaussian distribution diffrent for each vertex - it will be the basis for texture of the supervoxels
        what is important the verticies connected wih the edge should have higher similarity - smaller kl divergence than those not connected
    2) define the size of the supervoxel in a way that will avoid overlapping between voxels - border should ave some small irregularities
    3) perform some small gaussian smoothing to make the problem haqrder
"""


"""
constructing the distributions with parameters such that shannon jensen divergence between probability distributions is below supplid treshold
and that non connected close elements are above given treshold
matr - matrix of all the gaussian distributions - with length the same as number of nodes first column is representing mean and second variance
"""
function evaluateAreGaussInRange(matr::Matrix,edgeList,minConnectedDiff,maxConnectedDiff )
    #defined actual js div between distributions connected by the edges
    js_divs=map(edg->Distances.js_divergence(rand(Normal(matr[1,[edg[1]]][1], abs(matr[2, [edg[1]]][1])), 300), rand(Normal(matr[1,[edg[2]]][1],   abs(matr[2,[edg[2]]][1]) ), 300)) ,edgeList)
    #defining mean squared error
    jsDivs_below= filter(it-> it<minConnectedDiff , js_divs)|>
        listt->map(it-> minConnectedDiff-it   ,listt) |>
        listt->map(it->it^2  ,listt) 
        
    jsDivs_above= filter(it-> it>maxConnectedDiff , js_divs)|>
    listt->map(it-> it-maxConnectedDiff   ,listt) |>
    listt->map(it->it^2  ,listt) 

    #minus MSE becouse we will maximize
    return mean(vcat(jsDivs_below,jsDivs_above))
end



"""
checks weather we have a graph that comply with set rules
"""
function isGraphValid(edgeList,randGG ,minDist,minEdges)
    if (length(edgeList)>=minEdges )
        distances = map(edg->randGG[edg[1],edg[2]],edgeList)
        minDistt= minimum(distances)
        return (minDistt>= minDist )
    end#if
    return false

end#isGraphValid    

"""
create embeded random graph that meet some rules
"""
function getRandom_graph(numberOfNodes,maxDistToJoin,minDist,minEdges )
    randGG=0
    edgeList=0
        while(true)
            print("create random graph")
            #creating random graph
            randGG=random_geometric_graph(numberOfNodes,maxDistToJoin; dim=3)
            #get all edges and map it so the first element in a tuple is source and second destination although we will treat it as undirected graph
            edgeList=collect(edges(randGG)) |>
            edges->map(edge-> (edge.src,edge.dst)  ,edges)
            #invoke function recursively until set condition is met
            isGraphValid(edgeList,randGG ,minDist,minEdges) || break
        end

        return (randGG, edgeList)
end

"""
given some parameters will return the random graph and gaussian distributions for each node
    those distributions will be similar when connected by an edge in a graph
"""
function get_randomGraph_and_gaus_distribs(numberOfNodes,maxDistToJoin, dim_x,dim_y,dim_z ,minConnectedDiff,maxConnectedDiff,minDist,minEdges )
    
    #creating embedded random graph
    randGG, edgeList=getRandom_graph(numberOfNodes,maxDistToJoin,minDist,minEdges )

    #get x,y,z coordinates of the verticies scale them to chosen size of the embedding space and approximate to closest integer
    x_dims=Int.(ceil.(vertices_loc(randGG,1).*dim_x))
    y_dims=Int.(ceil.(vertices_loc(randGG,2).*dim_y))
    z_dims=Int.(ceil.(vertices_loc(randGG,3).*dim_z))
    #get full coordinates of each vertex
    coords=collect(zip(x_dims,y_dims,z_dims))

    
    #just partially apply in order to make easir to put into optimazation
    function evaluateAreGaussInRangeInner(matr::Matrix)
        evaluateAreGaussInRange(matr::Matrix,edgeList,minConnectedDiff,maxConnectedDiff )
    end#evaluateAreGaussInRangeInner    
    


    #making sure that the evaluated gaussians are similar when connected by the edge
    #making sure that function will be differentiable using combinatorial optimazation methods
    regularized_predictor = PerturbedAdditive(evaluateAreGaussInRangeInner; ε=1.0, nb_samples=5);

    params=rand(1.0:500.0,2,numberOfNodes)
    rule = Optimisers.Adam()
    opt_state = Optimisers.setup(rule, params);  # optimiser state based on model parameters

    #we iterate until we will get needed similarity
    indd=0
    currLoss=100
    print(" currLoss $currLoss  ")

    while (true)
        currLoss=regularized_predictor(params)
        indd=indd+1
        #if it takes to long we will restart parameters with random values
        if(indd>20000000)
            indd=0
            params=rand(1.0:500.0,2,numberOfNodes)
        end#if 
        print(" currLoss $currLoss  ")
        #making shure no Nan are present
        replace(params, NaN => 1.0)
        grad = ForwardDiff.gradient(regularized_predictor, params)
        opt_state, params = Optimisers.update!(opt_state, params, grad);
        #stop if the target is acheived
        currLoss<10.0 || break
    end



    final_cost = regularized_predictor(params)
    print("final cost $final_cost ")
    gausses= map(indexx->Normal(params[1,indexx][1], abs(params[2,indexx][1]))   ,1:numberOfNodes)

    return (coords, gausses,edgeList ,randGG )
end#get_randomGraph

"""
having supplied the coordinates of the nodes we will iteratively enlarge the area occupied by each node 
    given this are is not occupied already 
"""

@parallel_indices (ix,iy,iz) function dilatate(In,Out)
    # 7-point Neuman stencil set to maximum of the neghbouring labels - importantly we will only do this if the entry is equal to 0
    if (ix>1 && iy>1 && iz>1 &&      ix<(size(In,1))&& iy<(size(In,2)) && iz<(size(In,3)) && In[ix,iy,iz]==0 )
        Out[ix,iy,iz] = maximum( [In[ix-1,iy  ,iz  ] , In[ix-1,iy  ,iz  ], In[ix+1,iy  ,iz  ] ,In[ix  ,iy-1,iz  ],In[ix  ,iy+1,iz  ]
        ,In[ix  ,iy  ,iz-1], In[ix  ,iy  ,iz+1]])     
    # elseif (ix>1 && iy>1 && iz>1 &&      ix<(size(In,1))&& iy<(size(In,2)) && iz<(size(In,3)) && In[ix,iy,iz]!=0)
    #     Out[ix,iy,iz] =  In[ix,iy,iz] 
    end
    return
end


"""
given indicies of the nodes in the graph iteratively dilatates them without overwriting othe nodes
the bigger the number_iters the bigger would be the fileds occupied by the node
"""
@views function iter_dilatate(minDist,dim_x,dim_y,dim_z ,number_iters,coords)
    #first we create zeros array for indicies of nodes
    base=zeros(Int,(dim_x,dim_y,dim_z)) 
    out=zeros(Int,(dim_x,dim_y,dim_z)) 

    #set all spots where the verticies are to the value of their indexx
    for (indexx,coord) in enumerate(coords)
        base[coord[1],coord[2],coord[3]]=indexx
    end#for    

    for i in 1:number_iters
        print("  dilatate ")
        @parallel dilatate(base,out)
        base[:,:,:]=copy(out)
    end#for
    return out
end #iter_dilatate



"""
gets random graph according to set rules and return 
3d arrays that represents embedding of this graph
"""
function getRand_graph_andArr(numberOfNodes,maxDistToJoin, dim_x,dim_y,dim_z ,minConnectedDiff,maxConnectedDiff,minDist,minEdges,number_iters)
    #get graph and associated gaussian distributions
    coords, gausses,edgeList ,randGG =get_randomGraph_and_gaus_distribs(numberOfNodes,maxDistToJoin, dim_x,dim_y,dim_z ,minConnectedDiff,maxConnectedDiff,minDist,minEdges )
    #create array marking which voxel is owned by which node
    base_arr_ind=iter_dilatate(minDist,dim_x,dim_y,dim_z,number_iters,coords )
    #populate the voxels owned by each vertex with the numbers from the respective gaussian distribution
    base_arr= zeros(Float32,(dim_x,dim_y,dim_z)) 
    for x in 1:dim_x, y in 1:dim_y, z in 1:dim_z
        indd= base_arr_ind[x,y,z]
        if(indd>0)
            distrib = gausses[ indd ]
            base_arr[x,y,z]= rand(distrib,1)[1]
        end#if
    end#for    
    return (base_arr,randGG ,coords, gausses,edgeList)
end#getRand_graph_andArr


# ##some constants defining possible discrepancies between distributions defined in termes of js divergence
# #for two connected distributions
# minConnectedDiff=5.0
# maxConnectedDiff=5.5

# ###parameters
# #how many nodes will create a random graph
# numberOfNodes=25
# #maximum distance between 2 verticies that will lead to creation of the edge between them
# maxDistToJoin=0.35
# #dimensions of the array in which the image will be EmbeddedGraphs
# dim_x,dim_y,dim_z =100,110,120
# #minimum allowed distance between two nodes must be number between 0 and 1 
# minDist=0.2
# #minimum number of edges
# minEdges=13
# #set how big at the maximum should be spheres occupied by a node
# number_iters = Int(round(minDist*minimum([dim_x,dim_y,dim_z])))

# pathToHDF5="/home/jakub/projects/hdf5Data/forGraphDataSet.hdf5"
# base_arr,randGG=getRand_graph_andArr(numberOfNodes,maxDistToJoin, dim_x,dim_y,dim_z ,minConnectedDiff,maxConnectedDiff,minDist,minEdges,number_iters)
# patienGroupName="1"
# fid = h5open(pathToHDF5, "w")
# keys(fid)

function saveAndVisRandGraphNoAlgoOutPutSeen(base_arr,fid,patienGroupName,dim_x,dim_y,dim_z)
    
    if( !(patienGroupName in keys(fid)))
        algoOutput=zeros(Float32,(dim_x,dim_y,dim_z)) 
        base_arr= base_arr./maximum(base_arr)
        maximum(base_arr)
        listOfColorUsed= falses(18)
        gr= getGroupOrCreate(fid, patienGroupName)
        saveMaskBeforeVisualization(fid,patienGroupName,base_arr,"base_arr", "contLabel" )
        saveMaskBeforeVisualization(fid,patienGroupName,algoOutput,"algoOutput", "contLabel" )
        writeGroupAttribute(fid,patienGroupName, "spacing", [1,1,1])
    end


    addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,2)
    base_arrTex = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(
        name = "base_arr",
        # we point out that we will supply multiple colors
        isContinuusMask=true,
        colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
        ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
    )

    algoOutputTex = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(
        name = "algoOutput",
        # we point out that we will supply multiple colors
        isContinuusMask=true,
        #isVisible=false,
        colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
        ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
        )   
    addTextSpecs[1]=base_arrTex
    addTextSpecs[2]=algoOutputTex

    mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)
    base_arr= getArrByName("base_arr" ,mainScrollDat)
    #base_arr[:,:,:]=map(el-> (el<0.1 && el!=0) ? 0.1 : el  ,base_arr)
    #base_arr[:,:,:]=map(el-> (el<0.9 && el!=0) ? 0.9 : el  ,base_arr)
    visualizationFromHdf5.refresh(MedEye3d.SegmentationDisplay.mainActor)
    return (mainScrollDat,algoOutput )
end #saveAndVisRandGraphNoAlgoOutPutSeen

end #generate_synth_simple



