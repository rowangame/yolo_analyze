import os
import time
import cv2
from device_manager import Device_Manager

def startRecordData():
    capture = None
    try:
        devId = 0
        capture = Device_Manager.open_device(devId)
        Device_Manager.set_wh(capture, 640, 480)
        Device_Manager.set_fps(capture, 24)

        # 数据集
        path = "./data"
        if not os.path.exists(path):
            os.mkdir(path)

        count = 0
        index = 0
        savePerValue = 10
        while True:
            ret, aFrame = capture.read()

            if index % savePerValue == 0:
                fileName = "%s/%03d.png" % (path, count)
                print(fileName)
                cv2.imwrite(fileName, aFrame)
                index = 0
                count += 1
                if count > 300:
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

if __name__ == '__main__':
    startRecordData()