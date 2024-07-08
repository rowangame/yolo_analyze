import os


class Config_File_Manager:
    Config_Name = "config.txt"
    Log_Path = ""

    @classmethod
    def readLines(cls, fileName):
        rlts = []
        with open(fileName, 'r', encoding="utf-8") as tmpFile:
            lines = tmpFile.readlines()
            for tmpLine in lines:
                tmpStr = tmpLine.strip()
                if len(tmpStr) > 0:
                    rlts.append(tmpStr)
        return rlts

    @classmethod
    def writeConfigData(cls, fileName: str, stressTestCnt, errorId: int):
        with open(fileName, 'w', encoding="utf-8") as tmpFile:
            tmpFile.write("stressTestCnt=%d\n" % stressTestCnt)
            tmpFile.write("errorId=%d\n" % errorId)

    @classmethod
    def readConfigData(cls, fileName: str):
        lines = cls.readLines(fileName)
        return int(lines[0].split("=")[1]),int(lines[1].split("=")[1])

    @classmethod
    def getLogPath(cls):
        # 得到当前脚本文件的绝对路径
        absPath = os.path.abspath(__file__)
        # 取上一级目录
        pDir = os.path.dirname(absPath)
        # 合成路径
        return os.path.join(os.path.dirname(pDir), "log\\")

    @classmethod
    def testLogFile(cls):
        logPath = Config_File_Manager.getLogPath()
        print("logPath:", logPath)
        logName = logPath + cls.Config_Name
        Config_File_Manager.writeConfigData(logName, 1, 12)

        stressTestCnt, errorId = Config_File_Manager.readConfigData(logName)
        print("stressTestCnt=%d errorId=%d" % (stressTestCnt, errorId))
