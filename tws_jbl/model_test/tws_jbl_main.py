import os
import time

from ultralytics import YOLO
import cv2

from device_manager import Device_Manager
from buffer_frame_manager import BufferFrameManager
from tws_manager import TWS_Manager


def loadFrameTest():
    try:
         frames = []
         for i in range(100):
             imgName = "./res/img%03d.png" % i
             img = cv2.imread(imgName, cv2.IMREAD_COLOR)
             frames.append(img)

         """
         # 标记变化区域
         chgs = [];
         for index in range(0, len(frames) - 1):
             frame1 = frames[index]
             frame2 = frames[index + 1]
             boChanged, areas = BufferFrameManager.frameDiffAnalyze(frame1, frame2)
             chgs.append((index, len(areas)))
             print(f"{index}->{index+1}: area={len(areas)}")
             # 标记区域
             imgBuffer = frame2.copy()
             for tmpA in areas:
                 tmpX, tmpY, tmpW, tmpH = tmpA[0], tmpA[1], tmpA[2], tmpA[3]
                 cv2.rectangle(imgBuffer, (tmpX, tmpY), (tmpX + tmpW, tmpY + tmpH), (0, 255, 0), 1)
             cv2.imshow("dframe", imgBuffer)
             key = cv2.waitKey(1000)
             if key == ord('q'):
                 break
         """

         for index in range(0, len(frames) - 1):
             aFrame = frames[index]

             # 标记变化区域
             # markedFrame = BufferFrameManager.markChangedArea(aFrame, index)
             # cv2.imshow("dframe", markedFrame)

             # 标记roi区域
             roiFrame = TWS_Manager.markRoiArea(aFrame, (0, 255, 0))

             # 得到亮灯类型
             lType, rType = TWS_Manager.getRoiAreaLightType(aFrame)
             roiFrame = TWS_Manager.markLRInBoxLightType(lType, rType, roiFrame)

             cv2.imshow("roiFrame", roiFrame)
             key = cv2.waitKey(1000)
             if key == ord('q'):
                 break
    except Exception as e:
        print(repr(e))
    finally:
        cv2.destroyAllWindows()

def startAnalyze():
    capture = None
    try:
        # 加载对象识别模型
        model = YOLO('tws-jbl-best.pt')

        devId = 1
        capture = Device_Manager.open_device(devId)
        Device_Manager.set_wh(capture, 640, 480)
        # Device_Manager.set_wh(capture, 640, 360)
        # Device_Manager.set_wh(capture, 1280, 720)
        Device_Manager.set_fps(capture, 24)
        count = 0
        index = 0
        while True:
            ret, aFrame = capture.read()
            # if count < 100:
            #     cv2.imwrite("./res/img%03d.png" % (count), frame)
            #     count += 1

            # # 标记roi区域
            # roiFrame = TWS_Manager.markRoiArea(aFrame, (0, 255, 0))
            # # 得到亮灯类型q
            # lType, rType = TWS_Manager.getRoiAreaLightType(aFrame)
            # roiFrame = TWS_Manager.markLRInBoxLightType(lType, rType, roiFrame)
            # cv2.imshow("tws-light", roiFrame)

            # 标记变化区域
            # markedFrame = BufferFrameManager.markChangedArea(aFrame, index)
            # cv2.imshow("dframe", markedFrame)

            # 识别场景对象
            rltFrame = TWS_Manager.markDetectObject(model, aFrame)
            cv2.imshow("dframe", rltFrame)
            index += 1

            key = cv2.waitKey(50)
            if key == ord('q'):  # 判断是哪一个键按下
                break
    except Exception as e:
        print(repr(e))
    finally:
        Device_Manager.close_device(capture)
        cv2.destroyAllWindows()


def startRecordData():
    capture = None
    try:
        devId = 1
        capture = Device_Manager.open_device(devId)
        Device_Manager.set_wh(capture, 640, 480)
        Device_Manager.set_fps(capture, 24)

        # 关盒,开盒,耳机数据集
        data_type = ("close", "open-rlin", "open-rin", "open-lin", "l-ear", "r-ear")
        data_index = 5
        path = "./data/" + data_type[data_index]
        if not os.path.exists(path):
            os.mkdir(path)

        count = 0
        index = 0
        savePerValue = 10
        while True:
            ret, aFrame = capture.read()

            if index % savePerValue == 0:
                fileName = "%s/%s-%03d.png" % (path, data_type[data_index], count)
                print(fileName)
                cv2.imwrite(fileName, aFrame)
                index = 0
                count += 1
                if count > 150:
                    break
            index += 1

            cv2.imshow("data-1", aFrame)
            key = cv2.waitKey(50)
            if key == ord('q'):  # 判断是哪一个键按下
                break
    except Exception as e:
        print(repr(e))
    finally:
        Device_Manager.close_device(capture)
        cv2.destroyAllWindows()

def pngToJpg():
    data_type = ("close", "open-rlin", "open-rin", "open-lin", "l-ear", "r-ear")
    for name in data_type:
        path = "./data-1/" + name
        file_list = os.listdir(path)
        for file in file_list:
            filePath = path + "/" + file
            if not filePath.endswith(".png"):
                continue
            img = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
            jpgPath = filePath.replace(".png", ".jpg")
            # print(filePath)
            os.remove(filePath)
            print(jpgPath)
            # 保存图像(jpg图片质量:0-100)
            cv2.imwrite(jpgPath, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

if __name__ == "__main__":
    # pngToJpg()
    # loadFrameTest()
    # startRecordData()
    startAnalyze()