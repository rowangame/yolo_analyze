
import cv2

"""
mp4文件生成
"""
class Mp4_Writer:
    def __init__(self):
        self.Vedio_Width = 640
        self.Vedio_Height = 480
        self.Fps = 1
        self.mp4Writer = None

    """
    初始化
    """
    def initWriter(self, fileName):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
        self.mp4Writer = cv2.VideoWriter(fileName, fourcc, self.Fps, (self.Vedio_Width, self.Vedio_Height))

    """
    初始化Ex
    """
    def initWriterEx(self, fileName: str, width, height, fps: int):
        self.Vedio_Width = width
        self.Vedio_Height = height
        self.Fps = fps
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4v 编码器
        self.mp4Writer = cv2.VideoWriter(fileName, fourcc, self.Fps, (self.Vedio_Width, self.Vedio_Height))

    """
    写入一帧数像
    """
    def writeAFrame(self, frame):
        # 确保图像大小一致
        # resized_img = cv2.resize(frame, (cls.Vedio_Width, cls.Vedio_Height))
        if self.mp4Writer:
            self.mp4Writer.write(frame)

    """
    写入完全成后，需要释放保存数据
    """
    def saveData(self):
        # 释放 VideoWriter 对象
        if self.mp4Writer:
            self.mp4Writer.release()

