
import os.path
import shutil

import numpy as np
import cv2

"""
src_path: 指定源文件路径
dest_path: 目标目录路径
"""
def copy_file(src_path, dest_path):
    try:
        # 复制文件
        shutil.copy(src_path, dest_path)
        print(f"文件从 {src_path} 复制到 {dest_path}")
    except IOError as e:
        print(f"复制文件时发生错误：{e}")

"""
file_path: 指定文件路径
content: 要写入的字符串
"""
def write_to_txt_file(file_path, content):
    try:
        # 打开文件，如果文件不存在会被创建
        with open(file_path, 'w') as file:
            # 写入字符串到文件
            file.write(content)
        print(f"字符串写入到 {file_path} 成功")
    except IOError as e:
        print(f"写入文件时发生错误：{e}")

"""
boxes框 数据转换
centerX,centerY,W,H to leftop, right-bottom
"""
def __xywhn2xyxy(nlabel, width, height):
    """
    Transformed label from normalized center_x, center_y, width, height to
    x_1, y_1, x_2, y_2
    """
    # label = [0.0, 0.0, 0.0, 0.0, 0.0]
    label = np.array(nlabel, np.float64)
    label[1] = (nlabel[1] - (nlabel[3] / 2)) * width
    label[2] = (nlabel[2] - (nlabel[4] / 2)) * height
    label[3] = (nlabel[1] + (nlabel[3] / 2)) * width
    label[4] = (nlabel[2] + (nlabel[4] / 2)) * height

    return label.astype(int)

"""
数据转换与复制
dataDir: 源数据目录
trainDri: 要保存的训练数据目录
"""
def converter_to_yolov8(dataDir, trainDir):
    # print(dataDir, trainDir)
    dtImagesDir = os.path.join(dataDir, "images")
    dtLabelsDir = os.path.join(dataDir, "labels")
    # print(dtImagesDir, dtLabelsDir)

    dtImgfiles = os.listdir(dtImagesDir)
    # 训练数据占:80% 验证数据占:20%
    totalCnt = len(dtImgfiles)
    trainCnt = int(totalCnt * 0.8)
    valCnt = totalCnt - trainCnt
    # print(f"totalCnt={totalCnt} trainCnt={trainCnt} valCnt={valCnt}")

    dataIndex = 0
    for dtImgF in dtImgfiles:
        dtLblF = dtImgF.replace(".jpg", ".txt")
        dtImgPath = os.path.join(dtImagesDir, dtImgF)
        dtLblPath = os.path.join(dtLabelsDir, dtLblF)
        # print(dtImgPath, dtLblPath)
        if not os.path.exists(dtLblPath):
            print("error", f"Can't find the file: {dtLblF}")
            continue

        # v3与v8数据格式一样,box数据不需要转换
        # # label-studio 导出 Yolo数据类型是:
        # # x(box框中心点x), y(box框中心点y), w, h
        # # 转化为 x(box框左上角x), y(box框中左上角y), w, h
        # datas = np.loadtxt(dtLblPath)
        # # 数据格式不对(可能一个图存在多个box标记)
        # if datas.shape[0] != 5:
        #     print("Data error:", dtLblPath, dtImgPath)
        #     continue
        # # # x,y坐标转换
        # # datas[1] = datas[1] - datas[3] / 2
        # # datas[2] = datas[2] - datas[4] / 2
        # # content = "%d %.8f %.8f %.8f %.8f" % (int(datas[0]), datas[1], datas[2], datas[3], datas[4])
        # # print(datas, content)

        # 保存到目标目录
        if dataIndex < trainCnt:
            desImgPath = os.path.join(trainDir, "images\\train")
            desLblPath = os.path.join(trainDir, f"labels\\train")
        else:
            desImgPath = os.path.join(trainDir, "images\\val")
            desLblPath = os.path.join(trainDir, f"labels\\val")
        dataIndex += 1
        # 复制jpg文件
        copy_file(dtImgPath, desImgPath)
        # 复制label文件
        copy_file(dtLblPath, desLblPath)
        # testMarkBox(dtImgPath, datas)

"""
测试图片数据标记框
srcPath: 原图路径
box: 标记区域 0,x,y,w,h
"""
def testMarkBox(srcPath, box):
    try:
        img = cv2.imread(srcPath)
        realH, realW = img.shape[0], img.shape[1]
        color = (0, 255, 0)
        x1y1 = (int(box[1] * realW), int(box[2] * realH))
        x2y2 = (int((box[1] + box[3]) * realW), int((box[2] + box[4]) * realH))
        cv2.rectangle(img, x1y1, x2y2, color, 1)
        cv2.imshow("box", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(repr(e))

if __name__ == "__main__":
    dtDir = r"C:\Users\admin\Desktop\tws-data\yolo"
    # 原始数据目录,由label-studio
    trDir = r"E:\python\yolo_analyze\tws_jbl\train\datasets\tws-jbl"
    converter_to_yolov8(dtDir, trDir)