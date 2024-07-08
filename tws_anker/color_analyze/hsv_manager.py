
import cv2
import numpy as np

class HSV_Manager:
    """
    常见颜色的HSV范围和分类:
    refer: https://blog.csdn.net/Taily_Duan/article/details/51506776
    """
    """
    黑色（Black）:
    色相范围（Hue）: 0-180
    饱和度范围（Saturation）: 0-255
    亮度范围（Value）: 0-46
    """
    NG_HSV_BLACK = [(0,180), (0,255), (0, 46)]

    """
    灰色（Gray）:
    色相范围（Hue）: 0-180
    饱和度范围（Saturation）: 0-43
    亮度范围（Value）: 46-220
    """
    NG_HSV_Gray = [(0, 180), (0, 255), (46, 220)]

    """
    白色（White）:
    色相范围（Hue）: 0-180
    饱和度范围（Saturation）: 0-30
    亮度范围（Value）: 221-255
    """
    NG_HSV_WHITE = [(0, 180), (0, 30), (221, 255)]

    """
    红色（Red）:
    色相范围（Hue）: 0-10和156-180
    饱和度范围（Saturation）: 43-255
    亮度范围（Value）: 46-255
    """
    HSV_RED1 = [(0, 1), (43, 255), (46, 255)]
    HSV_RED2 = [(156, 180), (43, 255), (46, 255)]

    """
    橙色（Orange）:
    色相范围（Hue）: 11-25 
    饱和度范围（Saturation）: 43-255
    亮度范围（Value）: 46-255
    """
    HSV_ORANGE = [(11, 25), (43, 255), (46, 255)]

    """
    黄色（Yellow）:
    色相范围（Hue）: 26-34
    饱和度范围（Saturation）: 43-255
    亮度范围（Value）: 46-255
    """
    HSV_YELLOW = [(26, 34), (43, 255), (46, 255)]

    """
    绿色（Green）:
    色相范围（Hue）: 35-77 
    饱和度范围（Saturation）: 43-255
    亮度范围（Value）: 46-255
    """
    HSV_GREEN = [(35, 77), (43, 255), (46, 255)]

    """
    青色（Cyan）:
    色相范围（Hue）: 78-99
    饱和度范围（Saturation）: 43-255
    亮度范围（Value）: 46-255
    """
    HSV_CYAN = [(78, 99), (43, 255), (46, 255)]

    """
    蓝色（Blue）:
    色相范围（Hue）: 100-124
    饱和度范围（Saturation）: 43-255
    亮度范围（Value）: 46-255
    """
    HSV_BLUE = [(100, 124), (43, 255), (46, 255)]

    """
    紫色（Purple）:
    色相范围（Hue）: 125-155
    饱和度范围（Saturation）: 43-255
    亮度范围（Value）: 46-255
    """
    HSV_PURPLE = [(125, 155), (43, 255), (46, 255)]

    """
    与cv2中一致的转换
    refer: https://blog.csdn.net/zsc201825/article/details/89919496
    """
    @classmethod
    def bgrToHSV(cls, color):
        b, g, r = color[0], color[1], color[2]
        cMax = max(b, g, r)
        cMin = min(b, g, r)

        delta = cMax - cMin
        # 亮度值就等于 max(b, g, r)
        Value = cMax

        # 饱和度
        if Value != 0:
            Saturation = delta / Value
            Saturation = int(Saturation * 255)
        else:
            Saturation = 0

        # 色相
        if cMax == r:  # 最大值 == 红色
            Hue = 0 + 60 * (g - b) / delta
        elif cMax == g:  # 最大值  == 绿色
            Hue = 120 + 60 * (b - r) / delta
        else:  # 最大值  == 蓝色
            Hue = 240 + 60 * (r - g) / delta
        if Hue < 0:
            Hue = Hue + 360
        Hue = int(Hue / 2)
        return [Hue, Saturation, Value]

    """
    统计颜色数值
    img: 原数据
    hsvType: 常用颜色数值定义,参考HSV_Manager的常用颜色数值定义
    """
    @classmethod
    def statisticsColor(cls, imgHsvSrc, hsvType):
        # 分析颜色范围
        hLH = hsvType[0]
        sLH = hsvType[1]
        vLH = hsvType[2]

        lower = np.array([hLH[0], sLH[0], vLH[0]])
        upper = np.array([hLH[1], sLH[1], vLH[1]])
        imgSrc_mask = cv2.inRange(imgHsvSrc, lower, upper)

        # 数据值计: 2个类型,0,255两个数值
        hist_mask = cv2.calcHist([imgSrc_mask], [0], None, [2], [0, 256])

        # 返值颜色分量计数值
        return hist_mask[0,0], hist_mask[1,0]