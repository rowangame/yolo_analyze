
import cv2
import numpy as np
from ultralytics import YOLO

from tws_anker.color_analyze.roi_color_analyze import Roi_Color_Analyze
from tws_anker.model_test.scene_manager import Scene_Manager


class TWS_Frame_Manager:
    # 双耳入盒后,亮灯ROI区域相关
    # 要保存的ROI区域图像数目记录
    ROI_Ear_Cnt = 0
    # ROI区域宽度
    ROI_WIDTH = 10
    # ROI区域高度
    ROI_HEIGHT = 10
    # ROI区域Y偏移量
    ROI_OFF_SET_Y = 0
    # ROI区域X偏移量
    ROI_OFF_SET_X = 3

    """
    预测场景内容, 并标记边框
    @:return 返回标记的帧数据
    """
    @classmethod
    def markDetectObject(cls, model: YOLO, frame: np.ndarray):
        colors = [(0,0,255), (255,0,0), (0,255,0), (0,255,0), (0,255,0), (255,0,0)]
        try:
            results = model.predict(source=frame, save=False, save_txt=False)
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            cls_es = boxes.cls.cpu().numpy().astype(np.int32)
            xyxy = boxes.xyxy.cpu().numpy()
            index = 0
            for clsId in cls_es:
                # print("name:", results[0].names[clsId], "conf:", confs[index])
                # 匹配度不高的对象过滤掉
                if confs[index] < 0.3:
                    continue
                # 画边框[所有识别的对象]
                tmpXyxy = xyxy[index]
                p1 = (int(tmpXyxy[0]), int(tmpXyxy[1]))
                p2 = (int(tmpXyxy[2]), int(tmpXyxy[3]))
                cv2.rectangle(frame, p1, p2, colors[clsId], 1)

                # 画类型和可信用度
                text = "%s %.2f" % (results[0].names[clsId], confs[index])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text, p1, font, 1.0, colors[clsId], 1)

                index += 1
        except Exception as e:
            print(repr(e))
        return frame

    """
    预测场景内容, 并标记边框
    @:return 返回标记的帧数据, 类型数据, 是否有灯效
    """
    @classmethod
    def markDetectObjectEx(cls, model: YOLO, frame: np.ndarray):
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 0), (0, 255, 0), (255, 0, 0)]
        type_cls = []
        boHasWhiteEffect = False
        try:
            results = model.predict(source=frame, save=False, save_txt=False)
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            cls_es = boxes.cls.cpu().numpy().astype(np.int32)
            xyxy = boxes.xyxy.cpu().numpy()
            index = 0

            valueCnt = 0
            roiRect = None
            for clsId in cls_es:
                # print("name:", results[0].names[clsId], "conf:", confs[index])
                # 匹配度不高的对象过滤掉
                if confs[index] < 0.3:
                    continue

                # 画边框[所有识别的对象]
                tmpXyxy = xyxy[index]
                p1 = (int(tmpXyxy[0]), int(tmpXyxy[1]))
                p2 = (int(tmpXyxy[2]), int(tmpXyxy[3]))
                cv2.rectangle(frame, p1, p2, colors[clsId], 1)

                # 将类型和中心点数据添加到到列表中
                centerPnt = ((p2[0] + p1[0]) // 2, (p2[1] + p1[1]) // 2)
                type_cls.append((clsId, centerPnt))

                # 双耳入盒后，需要分析充电盒是否有亮灯状态(先标记ROI区域)
                if clsId == Scene_Manager.ID_EAR_IN:
                    # cls.copyEarStateData(frame, p1, p2)
                    tmpRoiX, tmpRoiY, tmpRoiW, tmpRoiH = cls.getROIArea(p1, p2)
                    roiRect = (tmpRoiX, tmpRoiY, tmpRoiW, tmpRoiH)
                    # cls.saveROIImage(frame, roiRect)
                    cv2.rectangle(frame, roiRect, (255, 0, 255), 1)

                if clsId == Scene_Manager.ID_SCREEN_VALUE:
                    valueCnt = valueCnt + 1

                # 画类型和可信用度
                text = "%s %.2f" % (cls.getObjectName(clsId), confs[index])
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, text, p1, font, 0.6, colors[clsId], 1)

                index += 1

            # 如果计数器两个都有值,则要分析是否亮白灯(如果没有亮白灯,则认为此时为异常情况)
            # 需要考虑灯效与计电器的时延问题。存在计电器有数值,灯效不一定马上亮的问题
            # 因此只需要判断某个时刻的一帧图像满足要求就行(相当于开启了灯效效果)
            if (valueCnt == 2) and (roiRect is not None):
                roiImg = cls.getROIImage(frame, roiRect)
                boWhite = Roi_Color_Analyze.isWhiteLightType(roiImg)
                boHasWhiteEffect = boWhite
                # 画灯效标记,在耳机中心显示数值
                for tmpData in type_cls:
                    if tmpData[0] == Scene_Manager.ID_EAR_IN:
                        if boWhite:
                            text = "effect=1"
                        else:
                            text = "effect=0"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(frame, text, tmpData[1], font, 0.6, (255,0,255), 1)
                        break
        except Exception as e:
            print(repr(e))
        return frame, type_cls, boHasWhiteEffect

    """
    得到对象名称
    """
    @classmethod
    def getObjectName(cls, clsId):
        # 对像名(与datasets/tws-anker/notes.json文件类型一致)
        obj_names = ['dev', 'ear-in', 'ear-out', 'v-value', 'v-zero']
        return obj_names[clsId]

    """
    复制耳机入盒出盒状态的图像数据
    """
    @classmethod
    def copyEarStateData(cls, aFrame, p1, p2):
        x, y = p1[0], p1[1]
        w, h = p2[0] - p1[0], p2[1] - p1[1]
        channel = 3
        desImg = np.zeros((h, w, channel), dtype=np.uint8)
        desImg[0:h, 0:w, 0:channel] = aFrame[y:y + h, x:x + w, 0:channel]

        roiX = w // 2 - cls.ROI_WIDTH // 2 + cls.ROI_OFF_SET_X
        roiY = h - cls.ROI_HEIGHT + cls.ROI_OFF_SET_Y
        roiW = cls.ROI_WIDTH
        roiH = cls.ROI_HEIGHT
        cv2.rectangle(desImg, (roiX, roiY, roiW, roiH), (255, 0, 255), 1)

        if cls.ROI_Ear_Cnt < 10:
            fileName = "./ear_in_img/%03d.png" % (cls.ROI_Ear_Cnt + 1)
            cv2.imwrite(fileName, desImg)
            cls.ROI_Ear_Cnt = cls.ROI_Ear_Cnt + 1

    """
    得到ROI(亮灯)的矩形区域
    """
    @classmethod
    def getROIArea(cls, p1, p2):
        x, y = p1[0], p1[1]
        w, h = p2[0] - p1[0], p2[1] - p1[1]

        roiX = w // 2 - cls.ROI_WIDTH // 2 + cls.ROI_OFF_SET_X
        roiY = h - cls.ROI_HEIGHT + cls.ROI_OFF_SET_Y
        roiW = cls.ROI_WIDTH
        roiH = cls.ROI_HEIGHT

        return x + roiX, y + roiY, roiW, roiH

    """
    得到ROI图像数据
    """
    @classmethod
    def getROIImage(cls, aFrame, roiArea):
        roiX = roiArea[0]
        roiY = roiArea[1]
        roiW = roiArea[2]
        roiH = roiArea[3]
        channel = 3
        roiImg = np.zeros((roiH, roiW, channel), dtype=np.uint8)
        roiImg[0:roiH, 0:roiW, 0:channel] = aFrame[roiY:roiY + roiH, roiX:roiX + roiW, 0:channel]
        return roiImg

    """
    保存ROI图像数据
    """
    @classmethod
    def saveROIImage(cls, aFrame, roiArea):
        if cls.ROI_Ear_Cnt < 25:
            roiX = roiArea[0]
            roiY = roiArea[1]
            roiW = roiArea[2]
            roiH = roiArea[3]
            channel = 3
            roiImg = np.zeros((roiH, roiW, channel), dtype=np.uint8)
            roiImg[0:roiH, 0:roiW, 0:channel] = aFrame[roiY:roiY + roiH, roiX:roiX + roiW, 0:channel]

            fileName = "./roi_img/%03d.png" % (cls.ROI_Ear_Cnt + 1)
            cv2.imwrite(fileName, roiImg)
            cls.ROI_Ear_Cnt = cls.ROI_Ear_Cnt + 1
