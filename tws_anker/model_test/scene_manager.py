import time

import cv2
import numpy as np

from tws_anker.file_utils.config_file_manager import Config_File_Manager
from tws_anker.file_utils.txt_file_manager import Txt_File_Manager
from tws_anker.mp4_utils.mp4_manager import Mp4_Manager

"""
场景管理对象
"""
class Scene_Manager:
    # 对象id(与datasets/tws-anker/notes.json文件类型一致)
    ID_AMMETER = 0
    ID_EAR_IN = 1
    ID_EAR_OUT = 2
    ID_SCREEN_VALUE = 3
    ID_SCREEN_ZERO = 4

    # 状态值
    STATE_NONE_OBJ = -1  # 场景识别不到对像类型
    STATE_REAL_OBJ = 1  # 场景识别到对象类型
    STATE_V_ZERO = 0  # 计数器数值为零
    STATE_V_VALUE = 1  # 计数器数值非零
    STATE_EAR_OUT = 0  # 耳机出盒状态
    STATE_EAR_IN = 1  # 耳机入盒状态

    SC_MIN_WAIT_TIME = 25  # 场景切换等待的最小时间(s)

    frameBuffers = []  # 帧数据缓冲列表
    stressTestCnt = 0  # 测试次数记录
    errorId = 0  # 出错的ID

    """
    添加一个场景数据。包括耳机状态(入盒还是出盒),左右耳计数器是否有值
    @:param
        aFrame: 场景图像数据
        type_cls: 检测试的对象(类型id,可信度,中以点x,y)
        boEffect: 是否有灯效(亮白灯)
    分析原理:
        1.耳机出盒后，左右耳计数器要为零(数值应该由大到小再零,在相机拍射的时候(fps=24))
        2.耳机入盒后,左右耳计数据非零(数值应该零到小再到大,在相机拍射的时候(fps=24))
        3.当耳机出盒状态到入盒状态后,左右耳计数器有值时,为一个周期.需要记录当前状态并保存（记压测成功一次)
        4.当耳机入盒后,左右耳计数器不为零,需要记录当前状态(记压测异常一次,并保存数据)
    """
    @classmethod
    def addAScene(cls, aFrame: np.ndarray, type_cls: list, boEffect: bool):
        frameData = {}
        # print(type_cls)
        stateValues = cls.getStateValue(type_cls)
        # print(stateValues)
        errorCode = cls.getErrorState(stateValues)
        # print("errorValue:%d" % errorValue)
        # 分析正确的场景状态
        curTime = time.time()

        # 缓存帧数据状态
        frameData["frame"] = aFrame
        frameData["tick"] = curTime
        frameData["sCode"] = errorCode
        frameData["sValue"] = stateValues
        frameData["effect"] = boEffect

        # 添加到缓存列表中
        cls.frameBuffers.append(frameData)

        # 如果左右耳记数器没有计数,分析是否需要保存数据测试记录
        tmpBufCnt = len(cls.frameBuffers)
        if (tmpBufCnt > 1) and (errorCode == 0):
            # 左右耳没记数(分析前面的帧数据是否有记录)
            if (stateValues["lval"] == cls.STATE_V_ZERO) and (stateValues["rval"] == cls.STATE_V_ZERO):
                for tmpIndex in range(tmpBufCnt - 1):
                    tmpBufData = cls.frameBuffers[tmpIndex]
                    # 左右耳有计数有灯效,需要保存一次记录数据
                    if (tmpBufData["sCode"] == 0) \
                        and (tmpBufData["sValue"]["lval"] == cls.STATE_V_VALUE) \
                        and (tmpBufData["sValue"]["rval"] == cls.STATE_V_VALUE) and (tmpBufData["effect"] == True):
                        cls.stressTestCnt = cls.stressTestCnt + 1
                        caseInfo = cls.getCaseInfo(cls.stressTestCnt, 0)
                        Txt_File_Manager.addCase(caseInfo)
                        # 清除缓存并添加当前帧数据
                        cls.frameBuffers.clear()
                        cls.frameBuffers.append(frameData)
                        return

        # 如果超过规定时间还没有识别到一个周期(双耳入盒到出盒的过程),必定是出错了(则保存数据到Mp4文件,并记录出错ID)
        if tmpBufCnt > 1:
            if cls.frameBuffers[tmpBufCnt - 1]["tick"] - cls.frameBuffers[0]["tick"] > cls.SC_MIN_WAIT_TIME:
                cls.errorId = cls.errorId + 1
                cls.stressTestCnt = cls.stressTestCnt + 1

                tInfo = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
                mp4Name = Config_File_Manager.Log_Path + "%s-%d.mp4" % (tInfo, cls.errorId)

                caseInfo = cls.getCaseInfo(cls.stressTestCnt, cls.errorId, mp4Name)
                Txt_File_Manager.addCase(caseInfo)

                # 记录帧数据,并保存mp4文件
                tmpFrames = []
                for tmpIndex in range(tmpBufCnt - 1):
                    tmpBufData = cls.frameBuffers[tmpIndex]
                    tmpFrames.append(tmpBufData["frame"])
                Mp4_Manager.writeMp4File(mp4Name, tmpFrames, 24)

                # 清除缓存并添加当前帧数据
                cls.frameBuffers.clear()
                cls.frameBuffers.append(frameData)
                return

    @classmethod
    def getCaseInfo(cls, stressCnt: int, errorId: int, logFile: str = ""):
        if errorId == 0:
            tInfo = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            return "pass#%s#%05d" % (tInfo, stressCnt)
        else:
            tInfo = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
            return "fail#%s#%05d#errorId=%d#logfile=%s" % (tInfo, stressCnt, errorId,logFile)

    @classmethod
    def toStringByStateValues(cls, sValues: dict, boEffect: bool):
        if sValues["tws"] == cls.STATE_NONE_OBJ:
            earInfo = "盒子状态:无法识别"
        elif sValues["tws"] == cls.STATE_EAR_IN:
            earInfo = "盒子状态:入盒"
        else:
            earInfo = "盒子状态:出盒"

        if sValues["lval"] == cls.STATE_NONE_OBJ:
            lValue = "左耳数值:无法识别"
        elif sValues["lval"] == cls.STATE_V_VALUE:
            lValue = "左耳数值:非零"
        else:
            lValue = "左耳数值:0"

        if sValues["rval"] == cls.STATE_NONE_OBJ:
            rValue = "右耳数值:无法识别"
        elif sValues["rval"] == cls.STATE_V_VALUE:
            rValue = "右耳数值:非零"
        else:
            rValue = "右耳数值:0"

        if boEffect:
            effectValue = "灯效:白灯"
        else:
            effectValue = "灯效:无"
        return "{%s,%s,%s,%s}" % (earInfo, lValue, rValue,effectValue)

    """
    判断数据类型是左边还是右边数据
    根据640x480的中心点区别:
    左边：小于中心点x
    右边: 大于中心点x
    @:return 0:left 1:right
    """
    @classmethod
    def getTypeLeftOrRight(cls, centerPnt):
        if centerPnt[0] < 320:
            return 0
        return 1

    """
    得到场景对象状态与数据
    @:return
        tws: 是否识别到了tws耳机状态
        lval: 左耳是否识别到有数值
        rval: 右耳是否识别到有数值
        ldev: 是否识别到了左耳计量器
        rdev: 是否识别到了右耳计量器
    """
    @classmethod
    def getStateValue(cls, type_cls: list):
        rlts = {"tws": cls.STATE_NONE_OBJ,
                "lval": cls.STATE_NONE_OBJ,
                "rval": cls.STATE_NONE_OBJ,
                "ldev": cls.STATE_NONE_OBJ,
                "rdev": cls.STATE_NONE_OBJ}
        # 分类对象
        twsLst = []
        valueLst = []
        devLst = []
        for tmpData in type_cls:
            clsId = tmpData[0]
            if clsId == cls.ID_AMMETER:
                devLst.append(tmpData)
            if clsId == cls.ID_SCREEN_VALUE or clsId == cls.ID_SCREEN_ZERO:
                valueLst.append(tmpData)
            if clsId == cls.ID_EAR_IN or clsId == cls.ID_EAR_OUT:
                twsLst.append(tmpData)

        # 判断类型与状态
        if len(twsLst) > 0:
            tmpData = twsLst[0]
            clsId = tmpData[0]
            if clsId == cls.ID_EAR_IN:
                rlts["tws"] = cls.STATE_EAR_IN
            else:
                rlts["tws"] = cls.STATE_EAR_OUT

        # 正常数值数目(有左右区分)
        if len(valueLst) == 2:
            data1 = valueLst[0]
            data2 = valueLst[1]
            boLeft = data1[1][0] < data2[1][0] # 比较x坐标
            clsId = data1[0]
            if boLeft:
                if clsId == cls.ID_SCREEN_VALUE:
                    rlts["lval"] = cls.STATE_V_VALUE
                else:
                    rlts["lval"] = cls.STATE_V_ZERO
            else:
                if clsId == cls.ID_SCREEN_VALUE:
                    rlts["rval"] = cls.STATE_V_VALUE
                else:
                    rlts["rval"] = cls.STATE_V_ZERO

            boLeft = data2[1][0] < data1[1][0]
            clsId = data2[0]
            if boLeft:
                if clsId == cls.ID_SCREEN_VALUE:
                    rlts["lval"] = cls.STATE_V_VALUE
                else:
                    rlts["lval"] = cls.STATE_V_ZERO
            else:
                if clsId == cls.ID_SCREEN_VALUE:
                    rlts["rval"] = cls.STATE_V_VALUE
                else:
                    rlts["rval"] = cls.STATE_V_ZERO
        elif len(valueLst) == 1: # 异常数值数目
            tmpData = valueLst[0]
            boLeft = cls.getTypeLeftOrRight(tmpData[1]) == 0
            clsId = tmpData[0]
            if boLeft:
                if clsId == cls.ID_SCREEN_VALUE:
                    rlts["lval"] = cls.STATE_V_VALUE
                else:
                    rlts["lval"] = cls.STATE_V_ZERO
            else:
                if clsId == cls.ID_SCREEN_VALUE:
                    rlts["rval"] = cls.STATE_V_VALUE
                else:
                    rlts["rval"] = cls.STATE_V_ZERO

        if len(devLst) == 2:
            rlts["ldev"] = cls.STATE_REAL_OBJ
            rlts["rdev"] = cls.STATE_REAL_OBJ
        elif len(devLst) == 1:
            tmpData = devLst[0]
            boLeft = cls.getTypeLeftOrRight(tmpData[1]) == 0
            if boLeft:
                rlts["ldev"] = cls.STATE_REAL_OBJ
            else:
                rlts["rdev"] = cls.STATE_REAL_OBJ
        return rlts

    """
    得到异常状态值:
        异常1:场景中无法识别耳机状态(入盒还是出盒状态)
        异常2:场景中识别不到左右耳计数器,计数器数值
        异常3:左右耳入盒后,2秒后至少有一个计量器读取的数值为零
    @return 
        0:场景数据正常
        1:场景中无法识别耳机状态(入盒还是出盒状态)
        2:场景中识别不到左右耳计数器,计数器数值
        3:左右耳入盒后,2秒后至少有一个计量器读取的数值为零
       -1:其它异常情况 
    """
    @classmethod
    def getErrorState(cls, stateValues: dict):
        ERROR_DEFAULT = -1
        # 暂时只验证五个状态是否被识别到
        boYes = True
        for tmpKey in stateValues.keys():
            tmpValue = stateValues[tmpKey]
            if tmpValue == cls.STATE_NONE_OBJ:
                boYes = False
                break
        if boYes:
            return 0
        return ERROR_DEFAULT