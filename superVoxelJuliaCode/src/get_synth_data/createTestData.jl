using Distributions

# nums=Float32.(rand(1.0:1000000,3,3,3))

"""
get test data for differentiable clustering
"""
function createTestDataForDiffusion(Nx, Ny, Nz, oneSidePad, crossBorderWhere)
    totalPad = oneSidePad * 2
    nums = Float32.(reshape(collect(1:Nx*Ny*Nz), (Nx, Ny, Nz)))#./100
    #nums = Float32.(rand(1.0:1000000,Nx,Ny,Nz))

    withPad = Float32.(zeros(Nx + totalPad, Ny + totalPad, Nz + totalPad))
    withPad[(oneSidePad+1):((oneSidePad+Nx)), (oneSidePad+1):(oneSidePad+Ny), (oneSidePad+1):(oneSidePad+Nz)] = nums
    A = CuArray(withPad)
    dA = similar(A)
    probs = Float32.(ones(Nx, Ny, Nz)) .* 0.1

    probs[crossBorderWhere, :, :] .= 0.9
    probs[:, crossBorderWhere, :] .= 0.9
    probs[:, :, crossBorderWhere] .= 0.9
    probsB = ones(Nx, Ny, Nz)
    probs = probsB .- probs#so we will keep low probability on edges
    withPadp = Float32.(zeros(Nx + totalPad, Ny + totalPad, Nz + totalPad))
    withPadp[(oneSidePad+1):((oneSidePad+Nx)), (oneSidePad+1):(oneSidePad+Ny), (oneSidePad+1):(oneSidePad+Nz)] = probs
    dp = CUDA.ones(Nx + totalPad, Ny + totalPad, Nz + totalPad)
    Aout = CUDA.zeros(Nx + totalPad, Ny + totalPad, Nz + totalPad)
    dAout = CUDA.ones(Nx + totalPad, Ny + totalPad, Nz + totalPad)
    dA .= 1
    p = CuArray(withPadp)
    return (A, dA, p, dp, Aout, dAout)

end

"""
we get as the output two arrays of the same size (Nx,Ny,Nz) plus padding on all saveAndVisRandGraphNoAlgoOutPutSeen
origArr - imitating the image in each corner modeled by some diffrent gaussian
indArr - just the set of consequtive indexes normalize - so we can be sure that each voxel has unique number for start
"""
function createTestDataFor_Clustering(Nx, Ny, Nz, oneSidePad, crossBorderWhere)
   
    totalPad = oneSidePad * 2
    nums = Float32.(reshape(collect(1:Nx*Ny*Nz), (Nx, Ny, Nz)))#./100
    nums=nums./(maximum(nums))
    #nums = Float32.(rand(1.0:1000000,Nx,Ny,Nz))

    indArr = Float32.(zeros(Nx + totalPad, Ny + totalPad, Nz + totalPad))
    indArr[(oneSidePad+1):((oneSidePad+Nx)), (oneSidePad+1):(oneSidePad+Ny), (oneSidePad+1):(oneSidePad+Nz)] = nums

    origArr=ones(Float32,Nx+ totalPad, Ny+ totalPad, Nz+ totalPad)

    
    tops=1+Int(oneSidePad)
    tope=Int(crossBorderWhere)+ Int(oneSidePad)
    bottoms=crossBorderWhere
    bottome=Nz+ oneSidePad
    lefts=1+oneSidePad
    lefte=crossBorderWhere+ oneSidePad
    rights=crossBorderWhere+ oneSidePad+1
    righte=Nx+ oneSidePad
    anteriors=crossBorderWhere+ oneSidePad+1
    anteriore=Ny+ oneSidePad
    posteriors=1+oneSidePad
    posteriore=crossBorderWhere+ oneSidePad


    top_left_post =view(origArr,tops:tope,lefts:lefte, posteriors:posteriore )
    top_right_post =view(origArr,tops:tope,rights:righte, posteriors:posteriore)
    
    top_left_ant =view(origArr,tops:tope,lefts:lefte, anteriors:anteriore )
    top_right_ant =view(origArr,tops:tope,rights:righte, anteriors:anteriore ) 
    
    bottom_left_post =view(origArr,bottoms:bottome,lefts:lefte, posteriors:posteriore ) 
    bottom_right_post =view(origArr,bottoms:bottome,rights:righte, posteriors:posteriore ) 
    
    bottom_left_ant =view(origArr,bottoms:bottome,lefts:lefte, anteriors:anteriore )
    bottom_right_ant =view(origArr,bottoms:bottome,rights:righte, anteriors:anteriore ) 
    

    indexxx=0
    function setRandValues(vieww)
        indexxx=indexxx+1
        sizz=size(vieww)
        # vieww[:,:,:]=rand(Normal(meanDiff*indexxx,10.0), sizz)
        vieww[:,:,:]=rand(Normal(rand(0.0:10.0,1)[1],rand(0.0:10.0,1)[1]), sizz)
    end
    
    setRandValues(top_left_post)
    setRandValues(top_right_post)
    setRandValues(top_left_ant)
    setRandValues(bottom_left_post)
    setRandValues(bottom_right_post)
    setRandValues(bottom_left_ant)
    setRandValues(bottom_right_ant)
    origArr./maximum(origArr)
    
    return origArr,indArr 
end


function get_example_image(pathToHDF5,data_dir )
    fid = h5open(pathToHDF5, "w")
    patientNum = 1#6
    patienGroupName=string(patientNum)
    listOfColorUsed= falses(18)
    train_labels = map(fileEntry-> joinpath(data_dir,"labels",fileEntry),readdir(joinpath(data_dir,"labels"); sort=true))
    train_images = map(fileEntry-> joinpath(data_dir,"volumes",fileEntry),readdir(joinpath(data_dir,"volumes"); sort=true))
    zipped= collect(zip(train_images,train_labels))
    tupl=zipped[patientNum]
    #proper loading using some utility function
    targetSpacing=(1,1,1)
    loaded = LoadFromMonai.loadBySitkromImageAndLabelPaths(tupl[1],tupl[2],targetSpacing)
    #CTORG
    #for this particular example we are intrested only in liver so we will keep only this label
    labelArr=map(entry-> UInt32(entry==1),loaded[2])
    imageArr=Float32.(loaded[1])
    gr= getGroupOrCreate(fid, patienGroupName)
    #we save loaded and trnsformed data into HDF5 to avoid doing preprocessing every time
    saveMaskBeforeVisualization(fid,patienGroupName,imageArr,"image", "CT" )
    saveMaskBeforeVisualization(fid,patienGroupName,labelArr,"labelSet", "boolLabel" )
    writeGroupAttribute(fid,patienGroupName, "spacing", [1,1,1])
    #manual Modification array
    manualModif = MedEye3d.ForDisplayStructs.TextureSpec{UInt32}(# choosing number type manually to reduce memory usage
        name = "manualModif",
        isVisible=false,
        color = getSomeColor(listOfColorUsed)# automatically choosing some contrasting color
        ,minAndMaxValue= UInt32.([0,1]) #important to keep the same number type as chosen at the bagining
        ,isEditable = true ) # we will be able to manually modify this array in a viewer

    algoVisualization = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(
        name = "algoOutput",
        # we point out that we will supply multiple colors
        isContinuusMask=true,
        colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
        ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
    )

        addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,2)
        addTextSpecs[1]=manualModif
        addTextSpecs[2]=algoVisualization

    mainScrollDat= loadFromHdf5Prim(fid,patienGroupName,addTextSpecs,listOfColorUsed)
    return mainScrollDat,tupl
end #get_example_image