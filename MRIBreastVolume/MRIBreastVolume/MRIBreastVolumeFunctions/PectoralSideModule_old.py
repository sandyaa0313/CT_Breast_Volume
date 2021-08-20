import slicer
import sitkUtils
import SimpleITK as sitk
import numpy as np
import copy

def EvaluatePectoralSide(inputImage, smoothingIterations):
    #模糊化初始影像，可以降噪、邊緣平滑化
    smoothedImage = AnisotropicDiffusion(sitk.Cast(inputImage, sitk.sitkFloat32))
    CreateNewVolumeNode(smoothedImage, "default")

    #計算胸腔與軀幹分塊
    fullBody, trunk, lungs= OptimizeTrunkAndLung(smoothedImage)
    CreateNewVolumeNode(trunk, "s4_trunk")
    CreateNewVolumeNode(lungs, "Lungs")

    GetBinaryBoundingBox(lungs)

    #取得軀幹外層一圈作為表皮並在後續步驟去除
    skin = sitk.And(fullBody, sitk.BinaryNot(Erode(fullBody, [6, 6, 0])))

    #取得非脂肪部位
    organe = sitk.And(fullBody, sitk.Greater(smoothedImage, 0))
    organe = sitk.And(organe, sitk.BinaryNot(skin))
    CreateNewVolumeNode(organe, "Organe")

    dilated_lung = Dilate(lungs, [12, 10, 5])
    minus = sitk.And(organe, sitk.BinaryNot(dilated_lung))
    bounding = GetBinaryBoundingBox(dilated_lung)
    half_lund = copy.deepcopy(dilated_lung)
    half_lund[bounding[0]:bounding[0]+bounding[3], bounding[1]:bounding[1]+bounding[4]//2, bounding[2]:bounding[5]] = 0
    CreateNewVolumeNode(minus, "minus")
    add_half = sitk.Or(half_lund, minus)
    add_half = Dilate(add_half, [4, 4, 0])
    add_half = Erode(add_half, [4, 4, 0])
    add_half = SlicewiseFillHole(add_half, 2, [0, 0, 1], [0, 0, 1])
    add_half = SlicewiseKeepLargestComponent(add_half, 2)
    add_half = Dilate(add_half, [10, 10, 5])
    add_half = SlicewiseFillHole(add_half, 2, [0, 0, 1], [0, 0, 1])
    add_half = Erode(add_half, [10, 10, 5])
    CreateNewVolumeNode(add_half, "add_Half")

    dilated_lung = Dilate(lungs, [10, 8, 5])
    filled = sitk.Or(organe, dilated_lung)
    filled = sitk.Or(filled, add_half)
    CreateNewVolumeNode(filled, "test1")
    filled = Erode(filled, [6, 6, 0])
    CreateNewVolumeNode(filled, "test2")
    filled = SlicewiseKeepLargestComponent(filled, 2)
    CreateNewVolumeNode(filled, "test3")
    filled = Dilate(filled, [8, 8, 0])
    CreateNewVolumeNode(filled, "test4")
    filled = SlicewiseFillHole(filled, 2, [0, 0, 1], [0, 0, 1])
    filled = Erode(filled, [2, 2, 0])
    CreateNewVolumeNode(filled, "test5")

    t6 = sitk.Or(organe, filled)
    CreateNewVolumeNode(t6, "test6")

    return filled

def CreateNewVolumeNode(image, name):
    volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    volumeNode.SetName(name)
    volumeNode.CreateDefaultDisplayNodes()
    sitkUtils.PushVolumeToSlicer(image, volumeNode)

def AnisotropicDiffusion(inputImage):
    smoothed = sitk.CurvatureAnisotropicDiffusion(
        image1 = inputImage,
        timeStep = 0.0625,
        conductanceParameter = 3.0,
        conductanceScalingUpdateInterval = 1,
        numberOfIterations = 8)
    return smoothed

def RemoveUpperNeck(image):
    imageSize = image.GetSize()

    image = Erode(image, [20, 20, 0])
    boundary = imageSize[2] - 1

    for k in reversed(range(imageSize[2])): #從頭部往下開始
        axialSlice = image[0:imageSize[0], 0:imageSize[1], k]
        components = sitk.ConnectedComponent(axialSlice)
        labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
        labelShapeStatistics.Execute(components)

        if len(labelShapeStatistics.GetLabels()) == 1:
            boundary = k
            break

    image = Dilate(image, [20, 20, 0])
    image[0:imageSize[0], 0:imageSize[1], boundary:imageSize[2]] = 0
    
    return image, boundary

def GetBinaryBoundingBox(image):
    labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelShapeStatistics.Execute(image)
    print(labelShapeStatistics.GetBoundingBox(1))

    return list(labelShapeStatistics.GetBoundingBox(1))

def OptimizeTrunkAndLung(image):
    imageSize = image.GetSize()

    notBackground = sitk.Greater(image, -300)

    fullBody = SlicewiseFillHole(notBackground, 2, [0, 0, 1], [0, 0, 1])
    fullBody, boundary = RemoveUpperNeck(fullBody)
    CreateNewVolumeNode(fullBody, "fullBody")

    notBackground[0:imageSize[0], 0:imageSize[1], boundary:imageSize[2]] = 0

    bone = sitk.And(notBackground, sitk.Greater(image, 250))
    bone = Dilate(bone, [3, 3, 10])
    bone = Erode(bone, [3, 3, 10])
    bone = sitk.And(bone, fullBody)
    bone = SlicewiseFillHole(bone, 2, [0, 0, 1], [0, 0, 1])
    CreateNewVolumeNode(bone, "bone")

    hole = KeepLargestComponent(notBackground)
    hole = sitk.And(fullBody, sitk.BinaryNot(hole))
    hole = Dilate(hole, [5, 5, 0])
    hole = Erode(hole, [5, 5, 0])
    CreateNewVolumeNode(hole, "hole")

    trunk = sitk.And(KeepLargestComponent(notBackground), sitk.BinaryNot(bone))
    CreateNewVolumeNode(trunk, "s0_Trunk")
    trunk = SlicewiseKeepLargestComponent(trunk, 2)
    CreateNewVolumeNode(trunk, "s0.5_Trunk")

    trunk = Dilate(trunk, [2, 2, 0])
    trunk = Erode(trunk, [2, 2, 0])
    CreateNewVolumeNode(trunk, "s1_Trunk")

    #######################################################
    #斷開小型連結
    kernel = [4, 4, 0]
    trunk = Erode(trunk, kernel)
    #可能會因為病人過瘦而斷開 因此在外層補上一層連結
    shell = sitk.And(fullBody, sitk.BinaryNot(Erode(fullBody, kernel)))
    trunk = sitk.Or(trunk, shell)
    CreateNewVolumeNode(trunk, "s2_Trunk")

    #slicewise keep largest component
    #將被刪除的部分於下一層中也刪除
    fragMask = sitk.Image(imageSize[0], imageSize[1], trunk.GetPixelID())

    for k in reversed(range(imageSize[2])): #從頭部往下開始
        axialSlice = trunk[0:imageSize[0], 0:imageSize[1], k]
        fragMask.CopyInformation(axialSlice)
        culled = sitk.And(axialSlice, sitk.BinaryNot(fragMask))

        cleaned = KeepLargestComponent(culled)

        fragMask = sitk.Or(fragMask, sitk.And(culled, sitk.BinaryNot(cleaned)))

        trunk = sitk.Paste(destinationImage = trunk,
                            sourceImage = cleaned,
                            sourceSize = [imageSize[0], imageSize[1], 1],
                            sourceIndex = [0, 0, 0],
                            destinationIndex = [0, 0, k])

    trunk = Dilate(trunk, kernel)
    trunk = sitk.And(trunk, fullBody)
    trunk = sitk.Or(trunk, bone)
    trunk = SlicewiseKeepLargestComponent(trunk, 2)
    trunk = Dilate(trunk, [3, 3, 3])
    trunk = Erode(trunk, [3, 3, 3])
    CreateNewVolumeNode(trunk, "s3_Trunk")

    #######################################################
    lungs = sitk.And(fullBody, sitk.BinaryNot(trunk))
    lungs = sitk.Or(lungs, hole)
    CreateNewVolumeNode(lungs, "raw_Lung")
    lungs = Dilate(lungs, [15, 15, 0])
    CreateNewVolumeNode(lungs, "dl_Lung")
    lungs = SlicewiseFillHole(lungs, 2, [0, 0, 1], [0, 0, 1])
    lungs = Erode(lungs, [15, 15, 0])

    #######################################################

    trunk = sitk.And(fullBody, sitk.BinaryNot(lungs))

    return fullBody, trunk, lungs

def GetRoughPectoralShape(value1Region, smoothingIterations):
    cropped = RoughCrop(value1Region)
    shape = SmoothByCurvatureFlow(cropped, smoothingIterations)
    return shape

def SlicewiseFillHole(image, axis, lowerPadding, upperPadding): #axis : i, j, k = 0, 1, 2
    imageSize = image.GetSize()

    direction = [0, 0, 0]
    boundary = [[0, imageSize[0]], [0, imageSize[1]], [0, imageSize[2]]]
    sliceSize = list(imageSize)
    sliceSize[axis] = 1

    for k in range(imageSize[axis]):
        boundary[axis][0] = k
        boundary[axis][1] = k+1
        direction[axis] = k

        axialSlice = image[boundary[0][0]:boundary[0][1], boundary[1][0]:boundary[1][1], boundary[2][0]:boundary[2][1]]

        padded = sitk.ConstantPad(axialSlice, lowerPadding, upperPadding, 1)
        filled = sitk.BinaryFillhole(padded)
        filled = sitk.Crop(filled, lowerPadding, upperPadding)

        image = sitk.Paste(destinationImage = image,
                            sourceImage = filled,
                            sourceSize = sliceSize,
                            sourceIndex = [0, 0, 0],
                            destinationIndex = direction)
    return image

def SlicewiseKeepLargestComponent(image, axis): #axis : i, j, k = 0, 1, 2
    imageSize = image.GetSize()

    direction = [0, 0, 0]
    boundary = [[0, imageSize[0]], [0, imageSize[1]], [0, imageSize[2]]]
    sliceSize = list(imageSize)
    sliceSize[axis] = 1

    for k in range(imageSize[axis]):
        boundary[axis][0] = k
        boundary[axis][1] = k+1
        direction[axis] = k

        axialSlice = image[boundary[0][0]:boundary[0][1], boundary[1][0]:boundary[1][1], boundary[2][0]:boundary[2][1]]

        cleaned = KeepLargestComponent(axialSlice)

        image = sitk.Paste(destinationImage = image,
                            sourceImage = cleaned,
                            sourceSize = sliceSize,
                            sourceIndex = [0, 0, 0],
                            destinationIndex = direction)

    return image

def Erode(image, erodeRadiusInMm):
    imageSpacing = image.GetSpacing()

    erodeRadius = []
    for dim in range(image.GetDimension()):
        erodeRadius.append(int(round(erodeRadiusInMm[dim] / imageSpacing[dim])))
    
    result = sitk.BinaryErode(image, erodeRadius, sitk.sitkBall)
    return result

def Dilate(image, dilateRadiusInMm):
    imageSpacing = image.GetSpacing()

    dilateRadius = []
    for dim in range(image.GetDimension()):
        dilateRadius.append(int(round(dilateRadiusInMm[dim] / imageSpacing[dim])))

    result = sitk.BinaryDilate(image, dilateRadius, sitk.sitkBall)
    return result

def KeepLargestComponent(image):
    components = sitk.ConnectedComponent(image)
    labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelShapeStatistics.Execute(components)
    maximumLabel = 0
    maximumSize = 0
    for label in labelShapeStatistics.GetLabels():
        size = labelShapeStatistics.GetPhysicalSize(label)
        if size > maximumSize:
            maximumLabel = label
            maximumSize = size
    return sitk.Mask(image, sitk.Equal(components, maximumLabel))

def FillNarrowHoles(image):
    imageSpacing = image.GetSpacing()
    dilateRadiusInMm = [2.5, 2.5, 2.5]
    dilateRadius = [int(round(dilateRadiusInMm[0] / imageSpacing[0])),\
                    int(round(dilateRadiusInMm[1] / imageSpacing[1])),\
                    int(round(dilateRadiusInMm[2] / imageSpacing[2]))]
    dilated = sitk.BinaryDilate(image, dilateRadius, sitk.sitkBall)

    padded = sitk.ConstantPad(dilated, [0, 0, 1], [0, 0, 1], 1)
    filled = sitk.BinaryFillhole(padded)
    filled = sitk.Crop(filled, [0, 0, 1], [0, 0, 1])

    erodeRadiusInMm = [dilateRadiusInMm[0] * 2.0, dilateRadiusInMm[1] * 2.0, dilateRadiusInMm[2] * 2.0]
    erodeRadius = [int(round(erodeRadiusInMm[0] / imageSpacing[0])),\
                    int(round(erodeRadiusInMm[1] / imageSpacing[1])),\
                    int(round(erodeRadiusInMm[2] / imageSpacing[2]))]
    eroded = sitk.BinaryErode(filled, erodeRadius, sitk.sitkBall)
    return sitk.Or(image, eroded)

def RoughCrop(shape):
    imageSize = shape.GetSize()
    cropped = sitk.Image(imageSize, shape.GetPixelID())
    cropped.CopyInformation(shape)

    tmp = int(int(imageSize[0])/2)

    for k in range(imageSize[2]):
        for j in range(1, imageSize[1]):
            if shape.GetPixel([tmp, j - 1, k]) == 0 and\
               shape.GetPixel([tmp, j    , k]) > 0:
                break

        cropOffsetInMm = 6
        cropOffset = int(round(cropOffsetInMm / shape.GetSpacing()[1]))
        cropJ = max(j - cropOffset, 0)

        cropped = sitk.Paste(destinationImage = cropped,
                            sourceImage = shape,
                            sourceSize = [imageSize[0], imageSize[1] - cropJ, imageSize[2]],
                            sourceIndex = [0, cropJ, 0],
                            destinationIndex = [0, cropJ, 0])
    return cropped

def SmoothByCurvatureFlow(shape, iterations):
    levelSet = sitk.SignedDanielssonDistanceMap(shape, True, False, True)
    levelSet = sitk.ZeroFluxNeumannPad(levelSet, [0, 0, 1], [0, 0, 1])

    potential = sitk.Image(levelSet.GetSize(), sitk.sitkFloat32)
    potential.CopyInformation(levelSet)
    potential = sitk.Add(potential, 1.0)

    levelSet = sitk.ShapeDetectionLevelSet(levelSet, potential, 0.0, 0.0, 1.0, iterations)
    levelSet = sitk.Crop(levelSet, [0, 0, 1], [0, 0, 1])

    return sitk.Greater(levelSet, 0)

def GenerateEdgeMap(otsuMultiple):
    value2 = sitk.Equal(otsuMultiple, 2)

    kernel = sitk.Image([1, 3, 1], sitk.sitkInt8)
    kernel.SetPixel([0, 0, 0], 0)
    kernel.SetPixel([0, 1, 0], -1)
    kernel.SetPixel([0, 2, 0], 1)

    edgeMap = sitk.Convolution(sitk.Cast(value2, sitk.sitkInt8), kernel)
    edgeMap = sitk.Maximum(edgeMap, 0)
    return sitk.Cast(edgeMap, sitk.sitkUInt8)

def PruneEdgeMap(edgeMap, value1Region):
    imageSpacing = edgeMap.GetSpacing()
    erodeRadiusInMm = [2, 2, 0]
    erodeRadius = [int(round(erodeRadiusInMm[0] / imageSpacing[0])),\
                    int(round(erodeRadiusInMm[1] / imageSpacing[1])),\
                    int(round(erodeRadiusInMm[2] / imageSpacing[2]))]
    mask = sitk.BinaryErode(value1Region, erodeRadius, sitk.sitkBall)
    return sitk.MaskNegated(edgeMap, mask)

def SnapToEdge(pectoralShape, edgeMap):
    distanceMap = sitk.DanielssonDistanceMap(edgeMap, True, False, True)
    distanceThreshold = 3 # mm
    potential = sitk.Clamp(distanceMap, distanceMap.GetPixelID(), 0, distanceThreshold)
    potential = sitk.Divide(potential, distanceThreshold)
    potential = sitk.ZeroFluxNeumannPad(potential, [0, 0, 1], [0, 0, 1])

    levelSet = sitk.SignedDanielssonDistanceMap(pectoralShape, True, False, True)
    levelSet = sitk.ZeroFluxNeumannPad(levelSet, [0, 0, 1], [0, 0, 1])

    levelSet = sitk.GeodesicActiveContourLevelSet(levelSet, potential, 0.001, 0.0, 0.5, 1.0, 1500)
    levelSet = sitk.Crop(levelSet, [0, 0, 1], [0, 0, 1])

    return sitk.Greater(levelSet, 0)
