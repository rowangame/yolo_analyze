import os
import time

from tws_anker.file_utils.config_file_manager import Config_File_Manager


class Txt_File_Manager:
    FILE_NAME_FMT = "stress_test_%s.txt"

    # 数据结果记录
    SAVE_PER_SIZE = 5
    caseList = []

    """
    追加模式写入文本文件中
    """
    @classmethod
    def writeTxt(cls, filePath: str, lines: list):
        # 以追加模式打开文件并写入新内容
        with open(filePath, 'a', encoding='utf-8') as file:
            for aLine in lines:
                file.write(aLine + "\n")

    @classmethod
    def getFilePath(cls):
        tInfo = time.strftime("%Y-%m-%d", time.localtime())
        filePath = Config_File_Manager.Log_Path + cls.FILE_NAME_FMT % (tInfo)
        # print(filePath)
        return filePath

    @classmethod
    def addCase(cls, info):
        cls.caseList.append(info)
        if len(cls.caseList) >= cls.SAVE_PER_SIZE:
            filePath = cls.getFilePath()
            cls.writeTxt(filePath, cls.caseList)
            cls.caseList.clear()

if __name__ == "__main__":
    Txt_File_Manager.getFilePath()