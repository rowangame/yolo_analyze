import time

import numpy as np
import cv2


class FrameWrapper(object):
    def __init__(self, frame, index):
        self.frame = frame
        self.index = index


class BufferFrameManager:
    buffer_frames = []
    buffer_record_last_tick = 0
    timeUnit = 1000
    delta_time = 200

    main_frame = FrameWrapper(None, 0)
    last_frame = FrameWrapper(None, 0)
    index : int = 0

    # roi区域数值设定
    roi_inbox_left = (406, 152, 35, 38)
    roi_inbox_right = (441, 152, 35, 38)
    roi_open_box = (307, 95, 265, 224)

    @classmethod
    def getFrameIncreaseIndex(cls):
        cls.index += 1
        return cls.index

    @classmethod
    def getNowTimeMs(cls):
        return time.time() * cls.timeUnit

    @classmethod
    def rect_intersecting(cls, rec1, rec2):
        # rect = (x, y, width, height)
        x1, y1, w1, h1 = rec1
        x2, y2, w2, h2 = rec2
        # 判断两个矩形是否相交
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2)

    """
    分析中间帧的图像变化,将图像的变化区域标记出来
    """
    @classmethod
    def markChangedArea(cls, frame: np.ndarray, seq: int):
        if cls.last_frame.frame is None:
            index = cls.getFrameIncreaseIndex()
            cls.last_frame.frame = frame
            cls.last_frame.index = index

            cls.main_frame.frame = frame
            cls.main_frame.index = index

            return frame
        else:
            lastFrame = cls.last_frame.frame
            lastIndex = cls.last_frame.index
            curFrame = frame
            # 记录当前帧
            index = cls.getFrameIncreaseIndex()
            cls.last_frame.frame = frame
            cls.last_frame.index = index

            # 分析当前帧与上一帧是否有差异,并标记区域
            boChanged, areas = cls.frameDiffAnalyze(lastFrame, curFrame)
            if boChanged:
                # 变化区域多,无法分析场景变化区域的内容
                if len(areas) > 2:
                    cls.main_frame.frame = curFrame
                    cls.main_frame.index = index
                    # 清空列表
                    cls.buffer_frames.clear()
                else:
                    # 这里要分析变化的区别内容(是否是ROI区域)
                    for tmpRec in areas:
                        if cls.rect_intersecting(tmpRec, cls.roi_inbox_left) or cls.rect_intersecting(tmpRec, cls.roi_inbox_right):
                            if cls.rect_intersecting(tmpRec, cls.roi_inbox_left):
                                tmpRecEx = cls.roi_inbox_left
                            else:
                                tmpRecEx = cls.roi_inbox_right
                            x, y, w, h = tmpRecEx[0], tmpRecEx[1], tmpRecEx[2],tmpRecEx[3]
                            channel = 3
                            roiLastImg = np.zeros((h, w, channel), dtype=np.uint8)
                            roiLastImg[0:h, 0:w, 0:channel] = lastFrame[y:y+h, x:x+w, 0:channel]
                            roiCurImg = np.zeros((h, w, channel), dtype=np.uint8)
                            roiCurImg[0:h, 0:w, 0:channel] = curFrame[y:y+h, x:x+w, 0:channel]
                            # lightType = TWS_Manager.getRoiEarTwinkleType(roiCurImg, roiLastImg, seq)
                            # print("lightType:", lightType, "seq:", seq)

                # 标记区域
                imgBuffer = curFrame.copy()
                for tmpA in areas:
                    # ROI区域用不同的颜色标记
                    if cls.rect_intersecting(tmpA, cls.roi_inbox_left) or cls.rect_intersecting(tmpA, cls.roi_inbox_right):
                            if cls.rect_intersecting(tmpA, cls.roi_inbox_left):
                                tmpRecEx = cls.roi_inbox_left
                            else:
                                tmpRecEx = cls.roi_inbox_right
                            color = (0, 255, 0)
                            tmpX, tmpY, tmpW, tmpH = tmpRecEx[0], tmpRecEx[1], tmpRecEx[2], tmpRecEx[3]
                            cv2.rectangle(imgBuffer, (tmpX, tmpY), (tmpX + tmpW, tmpY + tmpH), color, 1)
                    else:
                        color = (0, 0, 255)
                        tmpX, tmpY, tmpW, tmpH = tmpA[0], tmpA[1], tmpA[2], tmpA[3]
                        cv2.rectangle(imgBuffer, (tmpX, tmpY), (tmpX + tmpW, tmpY + tmpH), color, 1)
                return imgBuffer
            else:
                # 将变化不大的图片数据放入列表(主要分析TWS持续某种动作的时间)
                cls.buffer_frames.append((curFrame, index))
                return curFrame

    @classmethod
    def startRecord(cls):
        cls.buffer_frames.clear()
        cls.buffer_record_last_tick = time.time() * cls.timeUnit

    @classmethod
    def analyzeChange(cls):
        # 如果图像变化区域很大,说明场景有更新则记录最新的图像数据
        cls.buffer_frames.clear()
        cls.buffer_record_last_tick = time.time() * cls.timeUnit

    @classmethod
    def frameDiffAnalyze(cls, mainFrame, curFrame):
        # 转换为灰度图
        grayMain = cv2.cvtColor(mainFrame, cv2.COLOR_BGR2GRAY)
        grayCur = cv2.cvtColor(curFrame, cv2.COLOR_BGR2GRAY)

        # 计算两张图片的差异
        diff = cv2.absdiff(grayCur, grayMain)

        # 应用阈值，将差异显著的部分提取出来
        _, thresholded = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

        # 查找差异区域的轮廓
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 如果差异区域很多或者差异区域较大,则认为两张图像数据变化很大
        # 则需要分析差异区域内的体现内容
        diffAreaCnt = 0
        diffAreaBigCnt = 0
        diffAreaBigValue = 10
        diffAreaCntTag = 5
        # 记录差异较大的区域
        contourAreas = []
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > diffAreaBigValue or h > diffAreaBigValue:
                diffAreaBigCnt += 1
                contourAreas.append((x, y, w, h))
            diffAreaCnt += 1
        changeState = (diffAreaCnt > 0) or (diffAreaCnt > diffAreaCntTag)
        return changeState, contourAreas

    @classmethod
    def getFrameDiff(cls, frame1, frame2):
        src1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        src2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        return cv2.absdiff(src1, src2)
