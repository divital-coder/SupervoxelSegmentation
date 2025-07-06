using MedEye3d
import MedEye3d.DisplayWords.textLinesFromStrings
import MedEye3d
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
using MedEye3d.visualizationFromHdf5


"""
using medeye display control points locations
"""

# function displ_contr_point()

    
arr=zeros(Float32,128,128,128)
spacing=(1.0,1.0,1.0)

datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(arr) ,voxelSize=spacing, dimensionToScroll = 3 );
listOfColorUsed= falses(18)
fractionOfMainIm= Float32(0.8);

CTIm=TextureSpec{Int16}(
    name= "CTIm",
    numb= Int32(3),
    isMainImage = true,
    minAndMaxValue= Int16.([0,100]))

addTextSpecs=Vector{MedEye3d.ForDisplayStructs.TextureSpec}(undef,2)

base_arr_tex = MedEye3d.ForDisplayStructs.TextureSpec{Float32}(
    name = "base_arr",
    # we point out that we will supply multiple colors
    isContinuusMask=true,
    colorSet = [getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed),getSomeColor(listOfColorUsed)]
    ,minAndMaxValue= Float32.([0,1])# values between 0 and 1 as this represent probabilities
)

addTextSpecs[2]=base_arr_tex
addTextSpecs[1]=CTIm
CTIm=zeros(Int16,128,128,128)

mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(arr)[3] );
tupleVect = [("CTIm",CTIm),("base_arr",arr)]

slicesDat= getThreeDims(tupleVect )

mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
,dataToScroll= slicesDat
,mainTextToDisp= mainLines
,sliceTextToDisp=supplLines );
SegmentationDisplay.coordinateDisplay(addTextSpecs ,fractionOfMainIm ,datToScrollDimsB ,1000);
SegmentationDisplay.passDataForScrolling(mainScrollDat);


# end#displ_contr_point    


# displ_contr_point()





# we need to pass some metadata about image array size and voxel dimensions to enable proper display
datToScrollDimsB= MedEye3d.ForDisplayStructs.DataToScrollDims(imageSize=  size(ctPixels) ,voxelSize=PurePetSpacing, dimensionToScroll = 3 );
# example of texture specification used - we need to describe all arrays we want to display, to see all possible configurations look into TextureSpec struct docs .
textureSpecificationsPETCT = [
  TextureSpec{Float32}(
      name = "PET",
      isNuclearMask=true,
      # we point out that we will supply multiple colors
      isContinuusMask=true,
      #by the number 1 we will reference this data by for example making it visible or not
      numb= Int32(1),
      colorSet = [RGB(0.0,0.0,0.0),RGB(1.0,1.0,0.0),RGB(1.0,0.5,0.0),RGB(1.0,0.0,0.0) ,RGB(1.0,0.0,0.0)]
      #display cutoff all values below 200 will be set 2000 and above 8000 to 8000 but only in display - source array will not be modified
      ,minAndMaxValue= Float32.([200,8000])
     ),
  TextureSpec{UInt8}(
      name = "manualModif",
      numb= Int32(2),
      color = RGB(0.0,1.0,0.0)
      ,minAndMaxValue= UInt8.([0,1])
      ,isEditable = true
     ),

     TextureSpec{Int16}(
      name= "CTIm",
      numb= Int32(3),
      isMainImage = true,
      minAndMaxValue= Int16.([0,100]))  
];
# We need also to specify how big part of the screen should be occupied by the main image and how much by text fractionOfMainIm= Float32(0.8);
fractionOfMainIm= Float32(0.8);
"""
If we want to display some text we need to pass it as a vector of SimpleLineTextStructs - utility function to achieve this is 
textLinesFromStrings() where we pass resies of strings, if we want we can also edit those structs to controll size of text and space to next line look into SimpleLineTextStruct doc
mainLines - will be displayed over all slices
supplLines - will be displayed over each slice where is defined - below just dummy data
"""
import MedEye3d.DisplayWords.textLinesFromStrings

mainLines= textLinesFromStrings(["main Line1", "main Line 2"]);
supplLines=map(x->  textLinesFromStrings(["sub  Line 1 in $(x)", "sub  Line 2 in $(x)"]), 1:size(ctPixels)[3] );

import MedEye3d.StructsManag.getThreeDims

tupleVect = [("PET",petPixels) ,("CTIm",ctPixels),("manualModif",zeros(UInt8,size(petPixels)) ) ]
slicesDat= getThreeDims(tupleVect )
"""
Holds data necessary to display scrollable data
"""
mainScrollDat = FullScrollableDat(dataToScrollDims =datToScrollDimsB
                                 ,dimensionToScroll=1 # what is the dimension of plane we will look into at the beginning for example transverse, coronal ...
                                 ,dataToScroll= slicesDat
                                 ,mainTextToDisp= mainLines
                                 ,sliceTextToDisp=supplLines );


SegmentationDisplay.coordinateDisplay(textureSpecificationsPETCT ,fractionOfMainIm ,datToScrollDimsB ,1000);
SegmentationDisplay.passDataForScrolling(mainScrollDat);