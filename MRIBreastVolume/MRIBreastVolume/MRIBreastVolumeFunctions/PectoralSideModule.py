from logging import fatal
import slicer
import sitkUtils
import SimpleITK as sitk
import numpy as np
import copy

CREATE_VOLUME_NODE_FOR_DEBUG = True

def EvaluatePectoralSide(inputImage, smoothingIterations):
    #前處理
    fullBody, notBackground, organe, bone = Preprocessing(inputImage)
    CreateNewVolumeNode(bone, "bone")

    #計算胸腔與軀幹區塊
    trunk, lungs = OptimizeTrunkAndLung(fullBody, notBackground, bone)
    CreateNewVolumeNode(trunk, "Trunk")
    CreateNewVolumeNode(lungs, "Lungs")

    GetBinaryBoundingBox(lungs)

    CreateNewVolumeNode(organe, "Organe")
    CreateNewVolumeNode(BinaryMinus(trunk, organe), "trunk-organe")

    add_half = OptimizeBackPart(lungs, organe)

    edge, edge_raw = GeneratePectoralBoundaryEdge(organe, lungs, add_half)

    snap = SnapToEdge(lungs, edge)
    CreateNewVolumeNode(snap, "snap")

    snap = Dilate(snap, [2, 2, 2])
    snap = SlicewiseFillHole(snap, 2, [0, 0, 1], [0, 0, 1])
    snap = Erode(snap, [3, 3, 2])
    CreateNewVolumeNode(snap, "snap_clean")

    edge = BinaryMinus(sitk.BinaryDilate(edge_raw, [2, 1, 6], sitk.sitkBall), snap)
    CreateNewVolumeNode(edge, "edge_DL_M_snap")

    filled = sitk.Or(organe, add_half)
    filled = sitk.Or(filled, snap)
    CreateNewVolumeNode(filled, "test1")

    filled = Dilate(filled, [3, 3, 3])
    filled = SlicewiseFillHole(filled, 2, [0, 0, 1], [0, 0, 1])
    filled = SlicewiseFillHole(filled, 0, [1, 0, 0], [1, 0, 0])
    filled = Erode(filled, [3, 3, 3])
    CreateNewVolumeNode(filled, "test1_fill")
    filled = BinaryMinus(filled,  sitk.BinaryDilate(edge, [2, 1, 6], sitk.sitkBall))
    CreateNewVolumeNode(filled, "test1_bdcut")
    filled = SlicewiseFillHole(filled, 2, [0, 0, 1], [0, 0, 1])
    filled = SlicewiseKeepLargestComponent(filled, 2)
    CreateNewVolumeNode(filled, "test1_klc_1")
    filled = SlicewiseKeepLargestComponent(filled, 0)
    CreateNewVolumeNode(filled, "test1_klc_2")
    filled = SlicewiseKeepLargestComponent(filled, 2)
    CreateNewVolumeNode(filled, "test1_klc_3")

    filled = Dilate(filled, [4, 4, 4])
    filled = SlicewiseFillHole(filled, 2, [0, 0, 1], [0, 0, 1])
    filled = Erode(filled, [4, 4, 4])
    CreateNewVolumeNode(filled, "test1_clean")

    filled = sitk.BinaryDilate(filled, [2, 2, 6], sitk.sitkBall)
    CreateNewVolumeNode(filled, "test1_blob")

    t6 = BinaryMinus(organe, filled)
    CreateNewVolumeNode(t6, "diff")

    return filled

def Preprocessing(inputImage):
    imageSize = inputImage.GetSize()

    #模糊化初始影像，可以降噪、邊緣平滑化
    baseImage = AnisotropicDiffusion(sitk.Cast(inputImage, sitk.sitkFloat32))
    CreateNewVolumeNode(baseImage, "default")

    #取得非背景、空洞部分
    notBackground = sitk.Greater(baseImage, -500)
    CreateNewVolumeNode(notBackground, "notBG")

    #取得整身遮罩(避免後續取到CT背板)
    fullBody = SlicewiseFillHole(notBackground, 2, [0, 0, 1], [0, 0, 1])
    fullBody = KeepLargestComponent(fullBody)
    fullBody = sitk.BinaryMorphologicalOpening(fullBody, [2, 2, 2], sitk.sitkBall)
    CreateNewVolumeNode(fullBody, "fullbody")

    #斷開肩膀以上的部位(雙手舉高時，CT上連結較弱)(無法處理雙手下垂)
    #保留肩膀以下部位，供後續model對齊使用
    disconnect = Erode(fullBody, [25, 25, 0])
    CreateNewVolumeNode(disconnect, "Disconnect")

    #沿z方向找到最先連成一塊(肩膀以下)的z值
    for k in reversed(range(imageSize[2])): #從頭部往下開始
        axialSlice = disconnect[0:imageSize[0], 0:imageSize[1], k]
        components = sitk.ConnectedComponent(axialSlice)
        labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
        labelShapeStatistics.Execute(components)

        if len(labelShapeStatistics.GetLabels()) == 1:
            boundary = k
            break
    
    #將肩膀以上的部分移除
    fullBody[0:imageSize[0], 0:imageSize[1], boundary:imageSize[2]] = 0
    #下巴可能還是會出現在這層，清除
    fullBody = KeepLargestComponent(fullBody)

    #使用遮罩與threshold生成需要的資料
    notBackground = sitk.And(fullBody, notBackground)
    bone = sitk.And(fullBody, sitk.Greater(baseImage, 250))

    #選擇一定厚度的整身外層作為皮膚遮罩
    skin = BinaryMinus(fullBody, Erode(fullBody, [6, 6, 0]))
    organe = BinaryMinus(sitk.And(fullBody, sitk.Greater(baseImage, 0)), skin)

    return fullBody, notBackground, organe, bone

def AnisotropicDiffusion(inputImage):
    smoothed = sitk.CurvatureAnisotropicDiffusion(
        image1 = inputImage,
        timeStep = 0.0625,
        conductanceParameter = 3.0,
        conductanceScalingUpdateInterval = 1,
        numberOfIterations = 8)
    return smoothed

def OptimizeTrunkAndLung(fullBody, notBackground, bone):
    imageSize = fullBody.GetSize()

    #取得肋骨部位(通常不會與肩胛骨有連結)
    bone = KeepLargestComponent(bone)
    CreateNewVolumeNode(bone, "LeGu")
    #放大肋骨作為後續切割依據
    bone = Dilate(bone, [3, 3, 15])
    CreateNewVolumeNode(bone, "LeGu_DL")
    bone = Erode(bone, [3, 3, 5])
    bone = SlicewiseFillHole(bone, 2, [0, 0, 1], [0, 0, 1])
    CreateNewVolumeNode(bone, "LeGu_result")

    #保留最原始的體內空洞
    hole = BinaryMinus(fullBody, SlicewiseKeepLargestComponent(notBackground, 2))
    hole = sitk.BinaryMorphologicalClosing(hole, [3, 3, 3], sitk.sitkBall)
    CreateNewVolumeNode(hole, "hole")

    #尋找肺部頂點z值
    #移除微小空洞
    hole_refine = sitk.BinaryMorphologicalOpening(hole, [1, 1, 1], sitk.sitkBall)
    #沿z方向找到最先有三塊(食道與兩個肺部)的z值
    for k in reversed(range(imageSize[2])): #從頭部往下開始
        axialSlice = hole_refine[0:imageSize[0], 0:imageSize[1], k]
        components = sitk.ConnectedComponent(axialSlice)
        labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
        labelShapeStatistics.Execute(components)

        if len(labelShapeStatistics.GetLabels()) == 3:
            lungs_top = k
            break

    #軀幹
    trunk = BinaryMinus(notBackground, bone)
    trunk_clean = SlicewiseKeepLargestComponent(trunk, 2)
    CreateNewVolumeNode(trunk_clean, "trunk_clean")
    debri = BinaryMinus(trunk, trunk_clean)
    CreateNewVolumeNode(debri, "trunk_debri")
    debri[0:imageSize[0], 0:imageSize[1], lungs_top:imageSize[2]] = 0
    CreateNewVolumeNode(debri, "trunk_debri_underlung")

    #有些人的軀幹在很微妙的地方會有破洞，先補起來
    trunk = sitk.BinaryMorphologicalClosing(trunk_clean, [5, 5, 0], sitk.sitkBall)
    CreateNewVolumeNode(trunk, "trunk_raw_fillsamll")

    #######################################################
    #斷開小型連結
    kernel = [4, 4, 0]
    trunk = Erode(trunk, kernel)
    #可能會因為病人過瘦而斷開 因此在外層補上一層外緣
    shell = BinaryMinus(fullBody, Erode(fullBody, kernel))
    trunk = sitk.Or(trunk, shell)
    CreateNewVolumeNode(trunk, "trunk_1")

    #每層保留軀幹(應該要是最大的區塊)
    #將被刪除的部分於下一層中也刪除(頗危險的假設)
    cumuDebriMask = sitk.Image(imageSize[0], imageSize[1], trunk.GetPixelID())

    for k in reversed(range(imageSize[2])): #從頭部往下開始
        axialSlice = trunk[0:imageSize[0], 0:imageSize[1], k]
        debriSlice = debri[0:imageSize[0], 0:imageSize[1], k]
        cumuDebriMask.CopyInformation(axialSlice)

        culled = BinaryMinus(axialSlice, cumuDebriMask)

        cleaned = KeepLargestComponent(culled)
        cleaned = sitk.BinaryDilate(cleaned, [4, 4])
        cleaned = sitk.BinaryErode(cleaned, [4, 4])

        cumuDebriMask = sitk.Or(sitk.Or(cumuDebriMask, BinaryMinus(culled, cleaned)), debriSlice)

        trunk = sitk.Paste(destinationImage = trunk,
                            sourceImage = cleaned,
                            sourceSize = [imageSize[0], imageSize[1], 1],
                            sourceIndex = [0, 0, 0],
                            destinationIndex = [0, 0, k])
    
    CreateNewVolumeNode(trunk, "trunk_2")

    #側面清除碎屑
    #上下加蓋，避免因移除脊椎導致前後斷開
    trunk = sitk.ConstantPad(trunk, [0, 0, 1], [0, 0, 1], 1)
    trunk = SlicewiseKeepLargestComponent(trunk, 0)
    trunk = sitk.Crop(trunk, [0, 0, 1], [0, 0, 1])
    CreateNewVolumeNode(trunk, "trunk_3")

    trunk = Dilate(trunk, kernel)
    trunk = sitk.Or(trunk, bone)
    trunk = sitk.And(trunk, fullBody)
    trunk = SlicewiseKeepLargestComponent(trunk, 2)
    trunk = Dilate(trunk, [3, 3, 3])
    trunk = Erode(trunk, [3, 3, 3])
    CreateNewVolumeNode(trunk, "trunk_4")

    #######################################################
    lungs = BinaryMinus(fullBody, trunk)
    lungs = sitk.Or(lungs, hole)
    CreateNewVolumeNode(lungs, "lung_1")
    #lungs = KeepLargestComponent(lungs)
    lungs = Dilate(lungs, [15, 15, 5])
    lungs = SlicewiseFillHole(lungs, 2, [0, 0, 1], [0, 0, 1])
    lungs = Erode(lungs, [15, 15, 5])
    CreateNewVolumeNode(lungs, "lung_2")

    #找到肺部與氣管開始相連的z值
    #若上面曾經屬於肺部，則假設正下方皆為肺部範圍(假設人體下半必定比上半大)
    mark = False
    for k in reversed(range(imageSize[2])): #從頭部往下開始
        axialSlice = lungs[0:imageSize[0], 0:imageSize[1], k]
        if not mark:
            maxSize = GetLargestComponentSize(axialSlice)
            #print(k, maxSize)
            mark = (maxSize > 450)
        else:
            accumulate = sitk.Or(axialSlice, lungs[0:imageSize[0], 0:imageSize[1], k+1])
            lungs = sitk.Paste(destinationImage = lungs,
                            sourceImage = accumulate,
                            sourceSize = [imageSize[0], imageSize[1], 1],
                            sourceIndex = [0, 0, 0],
                            destinationIndex = [0, 0, k])
        
    #######################################################

    #直接將整身扣掉肺部作為軀幹區域
    lungs = KeepLargestComponent(lungs)
    trunk = BinaryMinus(fullBody, lungs)

    return trunk, lungs

def OptimizeBackPart(lungs, organe):
    dilated_lung = Dilate(lungs, [12, 10, 5])
    minus = BinaryMinus(organe, dilated_lung)
    bounding = GetBinaryBoundingBox(dilated_lung)
    half_lund = copy.deepcopy(dilated_lung)
    half_lund[bounding[0]:bounding[0]+bounding[3], bounding[1]:bounding[1]+bounding[4]//2, bounding[2]:bounding[5]] = 0
    add_half = sitk.Or(half_lund, minus)

    add_half = Dilate(add_half, [5, 5, 0])
    add_half = SlicewiseKeepLargestComponent(add_half, 2)
    add_half = SlicewiseFillHole(add_half, 2, [0, 0, 1], [0, 0, 1])
    add_half = Erode(add_half, [5, 5, 0])
    add_half = SlicewiseKeepLargestComponent(add_half, 2)

    return add_half

def GeneratePectoralBoundaryEdge(organe, lungs, addHalf):
    filled = sitk.Or(organe, addHalf)

    edge = GenerateEdgeMap(filled)
    edge = BinaryMinus(edge, lungs)
    CreateNewVolumeNode(edge, "edge")
    edge_raw = edge
    edge = sitk.BinaryDilate(edge, [2, 1, 6], sitk.sitkBall)
    CreateNewVolumeNode(edge, "edge_dl")

    imageSize = edge.GetSize()
    for k in range(imageSize[2]):
        axialSlice = edge[0:imageSize[0], 0:imageSize[1], k]

        cleaned = RemoveSmallComponent(axialSlice, 50)

        edge = sitk.Paste(destinationImage = edge,
                            sourceImage = cleaned,
                            sourceSize = [imageSize[0], imageSize[1], 1],
                            sourceIndex = [0, 0, 0],
                            destinationIndex = [0, 0, k])
    
    CreateNewVolumeNode(edge, "edge_cl")

    edge = sitk.PermuteAxes(edge, [2, 1, 0])
    imageSize = edge.GetSize()
    for k in range(imageSize[2]):
        axialSlice = edge[0:imageSize[0], 0:imageSize[1], k]

        cleaned = RemoveSmallComponent(axialSlice, 50)

        edge = sitk.Paste(destinationImage = edge,
                            sourceImage = cleaned,
                            sourceSize = [imageSize[0], imageSize[1], 1],
                            sourceIndex = [0, 0, 0],
                            destinationIndex = [0, 0, k])
    edge = sitk.PermuteAxes(edge, [2, 1, 0])

    CreateNewVolumeNode(edge, "edge_cl_2")

    edge = sitk.BinaryErode(edge, [1, 1, 0], sitk.sitkBall)
    CreateNewVolumeNode(edge, "edge_er")
    
    edge = sitk.BinaryThinning(edge)
    CreateNewVolumeNode(edge, "edge_thin")

    return edge, edge_raw

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

def GetLargestComponentSize(image):
    components = sitk.ConnectedComponent(image)
    labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelShapeStatistics.Execute(components)

    maximumSize = 0
    for label in labelShapeStatistics.GetLabels():
        size = labelShapeStatistics.GetPhysicalSize(label)
        if size > maximumSize:
            maximumSize = size

    return maximumSize

def RemoveSmallComponent(image, minimumSize):
    components = sitk.ConnectedComponent(image)
    labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelShapeStatistics.Execute(components)

    imageSize = image.GetSize()
    mask = sitk.Image(imageSize[0], imageSize[1], image.GetPixelID())
    mask.CopyInformation(image)

    for label in labelShapeStatistics.GetLabels():
        size = labelShapeStatistics.GetPhysicalSize(label)
        if size > minimumSize:
            mask = sitk.Or(mask, sitk.Equal(components, label))
            
    return sitk.Mask(image, mask)

def GenerateEdgeMap(image):
    kernel = sitk.Image([1, 3, 1], sitk.sitkInt8)
    kernel.SetPixel([0, 0, 0],  0)
    kernel.SetPixel([0, 1, 0],  1)
    kernel.SetPixel([0, 2, 0], -1)

    edgeMap = sitk.Convolution(sitk.Cast(image, sitk.sitkInt8), kernel)
    edgeMap = sitk.Maximum(edgeMap, 0)
    return sitk.Cast(edgeMap, sitk.sitkUInt8)

def SnapToEdge(pectoralShape, edgeMap):
    distanceMap = sitk.DanielssonDistanceMap(edgeMap, True, False, True)
    distanceThreshold = 3 # mm
    potential = sitk.Clamp(distanceMap, distanceMap.GetPixelID(), 0, distanceThreshold)
    potential = sitk.Divide(potential, distanceThreshold)
    potential = sitk.ZeroFluxNeumannPad(potential, [0, 0, 1], [0, 0, 1])

    levelSet = sitk.SignedDanielssonDistanceMap(pectoralShape, True, False, True)
    levelSet = sitk.ZeroFluxNeumannPad(levelSet, [0, 0, 1], [0, 0, 1])

    levelSet = sitk.GeodesicActiveContourLevelSet(levelSet, potential, 0.001, -0.125, 0.75, 1.0, 1000)
    levelSet = sitk.Crop(levelSet, [0, 0, 1], [0, 0, 1])

    return sitk.Greater(levelSet, 0)

def BinaryMinus(imageA, imageB):
    return sitk.And(imageA, sitk.BinaryNot(imageB))

def GetBinaryBoundingBox(image):
    labelShapeStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelShapeStatistics.Execute(image)

    return list(labelShapeStatistics.GetBoundingBox(1))

def CreateNewVolumeNode(image, name):
    if not CREATE_VOLUME_NODE_FOR_DEBUG:
        return

    volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
    volumeNode.SetName(name)
    volumeNode.CreateDefaultDisplayNodes()
    sitkUtils.PushVolumeToSlicer(image, volumeNode)