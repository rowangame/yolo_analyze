
"""
TWS动作管理类
"""

import cv2
import numpy as np
from PIL import ImageFont,ImageDraw,Image
from ultralytics import YOLO

from hsv_manager import HSV_Manager
from tws_roi_define import Tws_Roi_Define

roi_save_count = 0

class TWS_Manager:
    """
    当前设备是否是开盒状态
    """
    @classmethod
    def isBoxOpen(cls, roiImg):
        return False

    """
    当前设备是否是关盒状态
    """
    @classmethod
    def isBoxClosed(cls, roiImg):
        return False

    """
    左耳是否是入盒状态
    """
    @classmethod
    def isLeftEarInBox(cls, roiImg):
        return False

    """
    左耳是否是出盒状态
    """
    @classmethod
    def isLeftEarOutBox(cls, roiImg):
        return False

    """
    右耳是否是入盒状态
    """
    @classmethod
    def isRightEarInBox(cls, roiImg):
        return False

    """
    右耳是否是出盒状态
    """
    @classmethod
    def isRightEarOutBox(cls, roiImg):
        return False

    """
    得到当前ROI区域闪灯类型
    """
    @classmethod
    def getRoiEarTwinkleType(cls, roiCurImg, roiLastImg, seq):
        # 保存测试文件
        # global roi_save_count
        # if roi_save_count < 10:
        #     cv2.imwrite("./out/%d_1_roi_cur.png" % roi_save_count, roiCurImg)
        #     cv2.imwrite("./out/%d_0_roi_last.png" % roi_save_count, roiLastImg)
        #     roi_save_count += 1
        return 0

    """
    得到当前亮灯类型：
    0: 没有亮灯 1: 白灯 2: 红灯 3: 蓝灯
    """
    @classmethod
    def getEarLightType(cls, hsvImg, threshold=0.5):
        # 统计颜色数值与阈值对比
        ratioBlue = HSV_Manager.statisticsColor(hsvImg, HSV_Manager.HSV_BLUE)
        ratioRed1 = HSV_Manager.statisticsColor(hsvImg, HSV_Manager.HSV_RED1)
        ratioRed2 = HSV_Manager.statisticsColor(hsvImg, HSV_Manager.HSV_RED2)
        ratioRed = max(ratioRed1, ratioRed2)
        ratioWhite = HSV_Manager.statisticsColor(hsvImg, HSV_Manager.NG_HSV_WHITE)
        ratioBlack = HSV_Manager.statisticsColor(hsvImg, HSV_Manager.NG_HSV_BLACK)
        ratioTuple = (ratioBlack, ratioWhite, ratioRed, ratioBlue)

        # 使用 max 和 enumerate 获取最大值元素的索引
        maxIndex, maxValue = max(enumerate(ratioTuple), key=lambda x: x[1])
        # print(ratioTuple, maxIndex, maxValue)

        # 没有达到阈值,则认为没有灯效
        boReal = maxValue > threshold
        if not boReal:
            maxIndex = 0
        return maxIndex, boReal


    """
    标记roi区域
    """
    @classmethod
    def markRoiArea(cls, imgSrc: np.ndarray, color: tuple[int,int,int]):
        tmpRec = Tws_Roi_Define.roi_inbox_left
        tmpX, tmpY, tmpW, tmpH = tmpRec[0], tmpRec[1], tmpRec[2], tmpRec[3]
        cv2.rectangle(imgSrc, (tmpX, tmpY), (tmpX + tmpW, tmpY + tmpH), color, 1)

        tmpRec = Tws_Roi_Define.roi_inbox_right
        tmpX, tmpY, tmpW, tmpH = tmpRec[0], tmpRec[1], tmpRec[2], tmpRec[3]
        cv2.rectangle(imgSrc, (tmpX, tmpY), (tmpX + tmpW, tmpY + tmpH), color, 1)

        tmpRec = Tws_Roi_Define.roi_open_box
        tmpX, tmpY, tmpW, tmpH = tmpRec[0], tmpRec[1], tmpRec[2], tmpRec[3]
        cv2.rectangle(imgSrc, (tmpX, tmpY), (tmpX + tmpW, tmpY + tmpH), color, 1)

        return imgSrc

    """
    得到roi区域,亮灯类型
    :return 左耳亮灯类型，右耳亮灯类型
    """
    @classmethod
    def getRoiAreaLightType(cls, imgSrc):
        # 复制roi区域数据
        srcShape = imgSrc.shape
        channel = srcShape[2]

        tmpRec = Tws_Roi_Define.roi_inbox_left
        x, y, w, h = tmpRec[0], tmpRec[1], tmpRec[2], tmpRec[3]
        lInboxImg = np.zeros((h, w, channel), dtype=np.uint8)
        lInboxImg[0:h, 0:w, 0:channel] = imgSrc[y:y + h, x:x + w, 0:channel]

        tmpRec = Tws_Roi_Define.roi_inbox_right
        x, y, w, h = tmpRec[0], tmpRec[1], tmpRec[2], tmpRec[3]
        rInboxImg = np.zeros((h, w, channel), dtype=np.uint8)
        rInboxImg[0:h, 0:w, 0:channel] = imgSrc[y:y + h, x:x + w, 0:channel]

        lHsvImg = cv2.cvtColor(lInboxImg, cv2.COLOR_BGR2HSV)
        lType, real = cls.getEarLightType(lHsvImg)

        rHsvImg = cv2.cvtColor(rInboxImg, cv2.COLOR_BGR2HSV)
        rType, real = cls.getEarLightType(rHsvImg)

        return lType, rType

    """
    标记左右耳入盒后闪灯类型
    """
    @classmethod
    def markLRInBoxLightType(cls, lType, rType, imgSrc):
        """
        txt_color = (255, 255, 0)
        fscale = 0.6
        thickness = 1

        tmpRec = Tws_Roi_Define.roi_inbox_left
        x, y, w, h = tmpRec[0], tmpRec[1], tmpRec[2], tmpRec[3]
        # pos = (x + w // 2, y + h // 2)
        pos = (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imgSrc, f"{lType}", pos, font, fscale, txt_color, thickness)

        tmpRec = Tws_Roi_Define.roi_inbox_right
        x, y, w, h = tmpRec[0], tmpRec[1], tmpRec[2], tmpRec[3]
        pos = (x, y)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(imgSrc, f"{rType}", pos, font, fscale, txt_color, thickness)
        """

        # 绘制中文
        text_tpl = ("关", "白灯", "红灯", "蓝灯")
        fntPath = "./ttf/myfont.ttf"
        fntSize = 15

        # 设置字体的颜色
        alpha = 255
        bgra_lst = [(255, 255, 255, alpha), (255, 255, 255, alpha), (0, 0, 255, alpha), (255, 0, 0, alpha)]

        # 设置字体大小
        font = ImageFont.truetype(fntPath, fntSize)
        # 将numpy array的图片格式转为PIL的图片格式
        img_pil = Image.fromarray(imgSrc)
        # 创建画板
        draw = ImageDraw.Draw(img_pil)

        # 在图片上绘制中文
        tmpRec = Tws_Roi_Define.roi_inbox_left
        x, y, w, h = tmpRec[0], tmpRec[1], tmpRec[2], tmpRec[3]
        pos = (x, y - fntSize)
        draw.text(pos, text_tpl[lType], font=font, fill=bgra_lst[lType])

        tmpRec = Tws_Roi_Define.roi_inbox_right
        x, y, w, h = tmpRec[0], tmpRec[1], tmpRec[2], tmpRec[3]
        pos = (x, y - fntSize)
        draw.text(pos, text_tpl[rType], font=font, fill=bgra_lst[rType])

        # 将图片转为numpy array的数据格式
        imgRlt = np.array(img_pil)
        return imgRlt

    """
    预测场景内容,是否是tws盒子的内容,并标记边框
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