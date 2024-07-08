
import os

from ultralytics import YOLO
import cv2

from tws_anker.data_maker.device_manager import Device_Manager
from tws_anker.file_utils.config_file_manager import Config_File_Manager
from tws_anker.mp4_utils.mp4_writer import Mp4_Writer
from tws_frame_manager import TWS_Frame_Manager
from scene_manager import Scene_Manager

"""
从目录中加载数据
"""
def loadFrames(framePath: str):
    frames = []
    if not os.path.exists(framePath):
        print("错误:关联目录不存在...")
        return frames
    lstImgNames = os.listdir(framePath)
    count = 0
    for tmpFile in lstImgNames:
        if tmpFile.endswith(".png"):
            imgPath = framePath + "\\" + tmpFile
            tmpFrame = cv2.imread(imgPath)
            frames.append(tmpFrame)
        count += 1
        if count >= 300:
            break
    print("Total frames:%d" % len(frames))
    return frames


"""
对数据列表进行预测处理
"""
def predictProcess(frames: list):
    mp4Writer = Mp4_Writer()
    try:
        loadConfigData()

        # 加载对象识别模型
        print("加载对象识别模型...")
        model = YOLO('tws-anker-best.pt')
        wndTitle = "model_test1"

        print("验证对象识别...")
        index = 0
        maxsize = len(frames)
        while index < maxsize:
            aFrame = frames[index]

            # 识别场景对象
            # rltFrame = TWS_Frame_Manager.markDetectObject(model, aFrame)
            rltFrame, typeCls, boEffect = TWS_Frame_Manager.markDetectObjectEx(model, aFrame)
            Scene_Manager.addAScene(rltFrame, typeCls, boEffect)

            cv2.imshow(wndTitle, rltFrame)
            index += 1

            key = cv2.waitKey(1000)
            if key == ord('q'):  # 判断是哪一个键按下
                break
    except Exception as e:
        print(repr(e))
    finally:
        cv2.destroyAllWindows()
        saveConfigData()

def startTest(framePath: str):
    frames = loadFrames(framePath)
    predictProcess(frames)


def predictProcessEx():
    capture = None
    try:
        loadConfigData()

        # 加载对象识别模型
        print("加载对象识别模型...")
        model = YOLO('tws-anker-best.pt')
        wndTitle = "model_test1"

        # 视频设备初始化
        print("视频设备初始化...")

        devId = 0
        capture = Device_Manager.open_device(devId)
        Device_Manager.set_wh(capture, 640, 480)
        Device_Manager.set_fps(capture, 24)

        # 场景分析
        print("场景分析...")
        while True:
            ret, aFrame = capture.read()

            # 识别场景对象
            # # rltFrame = TWS_Frame_Manager.markDetectObject(model, aFrame)
            rltFrame, typeCls, boEffect = TWS_Frame_Manager.markDetectObjectEx(model, aFrame)
            Scene_Manager.addAScene(rltFrame, typeCls, boEffect)
            cv2.imshow(wndTitle, rltFrame)

            key = cv2.waitKey(50)
            if key == ord('q'):  # 判断是哪一个键按下
                break
    except Exception as e:
        print(repr(e))
    finally:
        cv2.destroyAllWindows()
        saveConfigData()


def loadConfigData():
    print("加载配置数据...")
    logPath = Config_File_Manager.getLogPath()
    # 赋值日志文件目录
    Config_File_Manager.Log_Path = logPath

    configFile = logPath + Config_File_Manager.Config_Name
    sCnt, errorId = Config_File_Manager.readConfigData(configFile)
    Scene_Manager.stressTestCnt = sCnt
    Scene_Manager.errorId = errorId


def saveConfigData():
    print("保存配置数据...")
    logPath = Config_File_Manager.getLogPath()
    configFile = logPath + Config_File_Manager.Config_Name
    Config_File_Manager.writeConfigData(configFile,  Scene_Manager.stressTestCnt, Scene_Manager.errorId)


def startTestEx():
    predictProcessEx()


if __name__ == "__main__":
    startTest(r"C:\Users\admin\Desktop\temp\images")
    # startTestEx()