

import cv2
import matplotlib.pyplot as plt
import numpy as np
from hsv_manager import HSV_Manager
from tws_manager import TWS_Manager


def hsvTest():
    h, w = 300, 300
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # # 设置为蓝色
    color = [120, 0, 0]
    img[:] = color
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 验证转化值
    hsv = imgHsv[0,0]
    hsvEx = HSV_Manager.bgrToHSV(color)
    print(hsv, hsvEx, f"validate:{hsv==hsvEx}")

    # 显示结果
    cv2.imshow('Blue Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def colorSpaceTest():
    height, width, channel = 100, 100, 3

    # 颜色空间列表
    hsvSpace = []
    titles = []
    dits = HSV_Manager.__dict__
    for tmpV in dits:
        if tmpV.startswith("HSV_"):
            hsvSpace.append(getattr(HSV_Manager, tmpV))
            titles.append(tmpV)
    # print(hsvSpace)
    # print(titles)

    # 初始化颜色数据
    rgbImgs = []
    for hsv in hsvSpace:
        # 取hsv上限值
        h = hsv[0][1]
        s = hsv[1][1]
        v = hsv[2][1]
        img = np.zeros((height, width, channel), dtype=np.uint8)
        img[:, :, 0:channel] = [h, s, v]
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        rgbImgs.append(img)

    # 显示图像
    index = 1
    for tmpImg in rgbImgs:
        plt.subplot(3, 3, index)
        plt.imshow(tmpImg, "gray")
        plt.title(titles[index-1])
        plt.xticks([])
        plt.yticks([])
        index += 1
    plt.show()


def testRoi():
    rois = []
    count = 9
    for i in range(count + 1):
        fname0 = "./out/%d_0_roi_last.png" % i
        fname1 = "./out/%d_1_roi_cur.png" % i
        img0 = cv2.imread(fname0)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2HSV)
        rois.append((img0, fname0))

        img1 = cv2.imread(fname1)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        rois.append((img1, fname1))

    """
    # 分析颜色范围
    # 定义蓝色的HSV范围
    hLH = HSV_Manager.HSV_BLUE[0]
    sLH = HSV_Manager.HSV_BLUE[1]
    vLH = HSV_Manager.HSV_BLUE[2]

    lower_blue = np.array([hLH[0], sLH[0], vLH[0]])
    upper_blue = np.array([hLH[1], sLH[1], vLH[1]])

    # 创建蓝色的掩码
    count = 0
    for hsv_image, fname in rois:
        blue_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)
        histBoBlue = cv2.calcHist([blue_mask], [0], None, [2], [0, 256])
        # print(f"count={count}, 0={histBoBlue[0]}, 255={histBoBlue[1]}")
        # print(f"total1:{histBoBlue[0] + histBoBlue[1]}, total2:{blue_mask.shape[0] * blue_mask.shape[1]}")
        ratio = histBoBlue[1] / (histBoBlue[0] + histBoBlue[1])
        print(f"count={count}, name={fname[fname.rfind('/') + 1:]} isBlue->ratio={ratio} state={ratio>0.7} ")
        # print(blue_mask)
        count += 1
    """

    # count = 0
    # for hsv_image, fname in rois:
    #     ratio = HSV_Manager.statisticsColor(hsv_image, HSV_Manager.HSV_BLUE)
    #     print(f"count={count}, name={fname[fname.rfind('/') + 1:]} isBlue->ratio={ratio} state={ratio > 0.7} ")
    #     count += 1

    light_name = ("黑", "白灯", "红灯", "蓝灯")

    count = 0
    for hsv_image, fname in rois:
        print("\n")
        type, state = TWS_Manager.getEarLightType(hsv_image)
        print(f"name={fname[fname.rfind('/') + 1:]} type={type} name={light_name[type]}, state={state}")
        count += 1

def colorSpaceTestEx():
    bgr_s = [(0,0,255), (0,255,0), (255,0,0), (0,255,255), (255,255,0), (255,0,255), (0,0,0), (120,120,120), (255,255,255)]
    titles = ("Red",      "Green",    "Blue",  "Yellow",    "Cyan",        "Purple",     "Black",  "Gray",      "White")
    imgs = []
    h, w, c = 10, 10, 3

    index = 0
    for clr in bgr_s:
        img = np.zeros((h, w, c), dtype=np.uint8)
        img[:,:,0:c] = clr
        hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        imgRgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(imgRgb)
        print(f"{titles[index]}, bgr={clr}, hsv={hsvImg[0,0,:]}")
        index += 1

    index = 1
    for tmpImg in imgs:
        plt.subplot(3, 3, index)
        plt.imshow(tmpImg, "gray")
        plt.title(titles[index-1])
        plt.xticks([])
        plt.yticks([])
        index += 1
    # plt.savefig("out/hsv_space_1.png")
    plt.show()

def colorSpaceTestExW():
    hsv_s = [(0,255,255),(60,255,255),(120,255,255),(30,255,255),(90,255,255),(150,255,255),(0,0,0),(0,0,120),(0,0,255)]
    titles = ("Red", "Green", "Blue", "Yellow", "Cyan", "Purple", "Black", "Gray", "White")
    imgs = []
    h, w, c = 10, 10, 3

    index = 0
    for hsv in hsv_s:
        imgHsv = np.zeros((h, w, c), dtype=np.uint8)
        imgHsv[:, :, 0:c] = hsv
        imgBgr = cv2.cvtColor(imgHsv, cv2.COLOR_HSV2BGR)
        imgRgb = cv2.cvtColor(imgBgr, cv2.COLOR_BGR2RGB)
        imgs.append(imgRgb)
        print(f"{titles[index]}, hsv={imgHsv[0,0,:]} bgr={imgBgr[0,0,:]}")
        index += 1

    index = 1
    for tmpImg in imgs:
        plt.subplot(3, 3, index)
        plt.imshow(tmpImg, "gray")
        plt.title(titles[index - 1])
        plt.xticks([])
        plt.yticks([])
        index += 1
    # plt.savefig("out/hsv_space_2.png")
    plt.show()

# testRoi()
hsvTest()
# colorSpaceTest()
# colorSpaceTestEx()
# colorSpaceTestExW()