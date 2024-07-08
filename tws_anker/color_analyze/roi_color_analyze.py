
import cv2

from tws_anker.color_analyze.hsv_manager import HSV_Manager


class Roi_Color_Analyze:
    # ROI图像二值化(阈值)
    ROI_WHITE_SHRESHOLD_MIN_VALUE = 90

    # ROI图像亮灯像素数量(阈值)
    ROI_WHITE_SHRESHOLD_PIXEL_COUNT = 2

    @classmethod
    def loadRoiImgs(cls):
        imgRlts = []
        imgCnt = 25
        for i in range(1, imgCnt + 1):
            fileName = "./roi_img/%03d.png" % (i)
            print(fileName)
            tmpImg = cv2.imread(fileName)
            imgRlts.append(tmpImg)
        return imgRlts

    @classmethod
    def analyzeColor(cls, imgs):
        count = 1
        for tmpImg in imgs:
            tmpHsvImg = cv2.cvtColor(tmpImg, cv2.COLOR_BGR2HSV)
            cmnCnt, WhiteCnt = HSV_Manager.statisticsColor(tmpHsvImg, HSV_Manager.NG_HSV_WHITE)
            print("count=%d cmnCnt=%d WhiteCnt=%d" % (count, cmnCnt, WhiteCnt))
            count = count + 1

    @classmethod
    def analyzeColorEx(cls, imgs):
        count = 1
        for tmpImg in imgs:
            # r, tmpRltImg = cv2.threshold(tmpImg, 90, 255, cv2.THRESH_BINARY)
            # fileName = "./roi_img_ex/%03d.png" % count
            # cv2.imwrite(fileName, tmpRltImg)
            # count = count + 1
            # print(tmpRltImg.shape)
            r, tmpRltImg = cv2.threshold(tmpImg, cls.ROI_WHITE_SHRESHOLD_MIN_VALUE, 255, cv2.THRESH_BINARY)
            tmpHsvImg = cv2.cvtColor(tmpRltImg, cv2.COLOR_BGR2HSV)
            cmnCnt, WhiteCnt = HSV_Manager.statisticsColor(tmpHsvImg, HSV_Manager.NG_HSV_WHITE)
            print("count=%d cmnCnt=%d WhiteCnt=%d" % (count, cmnCnt, WhiteCnt))
            count = count + 1

    """
    roiImg图像是否有亮白灯
    分析原理(因相机分辨率不高(640x480）,白灯区区域只有2-6个像素),需要加强亮度处理:
        1.将原图数据二值化
        2.将二值化的数据进行HSV格式转换
        3.统计白色像素个数
        4.白色像素个数大于阈值后,则认为有亮白灯效果
    """
    @classmethod
    def isWhiteLightType(cls, roiImg):
        r, tmpRltImg = cv2.threshold(roiImg, cls.ROI_WHITE_SHRESHOLD_MIN_VALUE, 255, cv2.THRESH_BINARY)
        tmpHsvImg = cv2.cvtColor(tmpRltImg, cv2.COLOR_BGR2HSV)
        cmnCnt, WhiteCnt = HSV_Manager.statisticsColor(tmpHsvImg, HSV_Manager.NG_HSV_WHITE)
        return WhiteCnt >= cls.ROI_WHITE_SHRESHOLD_PIXEL_COUNT

if __name__ == "__main__":
    imgs = Roi_Color_Analyze.loadRoiImgs()
    Roi_Color_Analyze.analyzeColorEx(imgs)