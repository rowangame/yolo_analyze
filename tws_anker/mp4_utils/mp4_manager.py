import threading

from tws_anker.mp4_utils.mp4_writer import Mp4_Writer


class Mp4_Manager:
    @classmethod
    def doProcess(cls, fileName: str, frames: list, fps: int):
        mp4Writer = Mp4_Writer()
        try:
            print("初始化mp4录像器...")
            print("文件名=%s" % fileName)
            mp4Writer.initWriterEx(fileName, 640, 480, fps)

            print("写入帧数据:%d" % len(frames))
            for tmpFrame in frames:
                mp4Writer.writeAFrame(tmpFrame)
        except Exception as e:
            print(repr(e))
        finally:
            print("保存视频文件...")
            mp4Writer.saveData()

    @classmethod
    def writeMp4File(cls, fileName: str, frames: list, fps: int = 12):
        # 使用线程保存mp4文件，防止阻碍主线程运行
        task = threading.Thread(target=cls.doProcess, args=(fileName, frames, fps))
        task.start()