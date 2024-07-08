"""
将label-studio组件 标记的图片资源导入到目标目录内
"""
import shutil

import cv2
import os
import numpy as np

"""
重命名path目录下的jpg,png所有文件
"""
def renameImgFiles(path: str):
    print("Dir", path)
    paths = os.listdir(path)
    print(type(paths), len(paths))
    print(paths[0:3])
    index = 0
    for tmpName in paths:
        if tmpName.endswith(".jpg") or tmpName.endswith(".png"):
            # prefix = tmpName[0 : len(tmpName) - 4]
            suffix = tmpName[len(tmpName) - 4:]
            # print(prefix)
            # print(suffix)
            oldName = path + "\\" + tmpName
            newName = path + "\\" + ("wiko_box_%03d" % index) + suffix
            # print(oldName, newName)
            os.rename(oldName, newName)
            print("rename:", index)
            index += 1


"""
将path目录下jpg文件和png文件缩放处理
"""
def scaleImgFiles(path: str, savePath: str):
    paths = os.listdir(path)
    for tmpName in paths:
        if tmpName.endswith(".jpg") or tmpName.endswith(".png"):
            imgPath = path + "\\" + tmpName
            imgSrc = cv2.imread(imgPath)
            imgRlt = cv2.resize(imgSrc, (imgSrc.shape[1] // 10, imgSrc.shape[0] // 10))
            saveName = savePath + "\\" + tmpName
            cv2.imwrite(saveName, imgRlt)
            print(saveName)


"""
将标记好的annoPath目录下的文件,放到训练目录中去
annoPath 标记好的数据目录
trainPath 目标训练目录
"""
def fetchToTrain(annoPath: str, trainPath: str):
    if not (os.path.exists(annoPath) and os.path.exists(trainPath)):
        print("错误:关联目录不存在...")
        return False
    imagesPath = annoPath + "\\images\\"
    labelsPath = annoPath + "\\labels\\"
    lstImgNames = os.listdir(imagesPath)
    lstLblNames = os.listdir(labelsPath)
    imgTotal = len(lstImgNames)
    if imgTotal != len(lstLblNames):
        print("描述文件与资源文件的数量不相等")
        return False
    if imgTotal == 0:
        print("资源文件为空")
        return False
    # 复制文件到目标目录(官网建议train目录占80的资源,验证目录占20%的资源)
    # 备注：Yolov3建议train目录占80的资源,验证目录占20%的资源 Yolov8不知道是否需要尊守此建议
    desTrainImgPath = trainPath + "\\images\\train\\"
    desTrainAnnoPath = trainPath + "\\labels\\train\\"
    desValImgPath = trainPath + "\\images\\val\\"
    desValAnnoPath = trainPath + "\\labels\\val\\"
    # 训练数量与验证数量比为4:1
    trainCnt = int(1.0 * imgTotal * 0.8)
    validationCnt = imgTotal - trainCnt
    for i in range(imgTotal):
        srcImgName = imagesPath + lstImgNames[i]
        srcLblName = labelsPath + lstLblNames[i]
        print(i, srcImgName, srcLblName)
        if i < trainCnt:
            # 训练目录
            # 复制文件到目标目录
            desTrainImgName = desTrainImgPath + lstImgNames[i]
            desTrainAnnoName = desTrainAnnoPath + lstLblNames[i]
            shutil.copyfile(srcImgName, desTrainImgName)
            shutil.copyfile(srcLblName, desTrainAnnoName)
        else:
            # 验证目录
            # 复制文件到目标目录
            desValImgName = desValImgPath + lstImgNames[i]
            desValAnnoName = desValAnnoPath + lstLblNames[i]
            shutil.copyfile(srcImgName, desValImgName)
            shutil.copyfile(srcLblName, desValAnnoName)


# 复制训练数据
fetchToTrain( r"C:\Users\admin\Desktop\temp", r"E:\python\yolo_analyze\tws-anker\train\datasets\tws-anker")