import cv2


class Device_Manager:
    # 参数调节工具下载地址
    # https://www.hikvision.com/cn/support/tools/hitools/cl013a49fa899c9591/
    # 打开指定的摄像头(HIK VISION)
    @classmethod
    def open_device(cls, videoId):
        try:
            return cv2.VideoCapture(videoId)
        except Exception as e:
            print(repr(e))
            return None

    # 关闭设备
    @classmethod
    def close_device(cls, video):
        try:
            video.release()
        except Exception as e:
            print(repr(e))

    # 设置分辨率(设备必须支持的分辨率)
    # 主流分辨率: 640x360 640x480 1280x720 1280x960 1920x1080
    # refer: https://blog.csdn.net/weixin_40922744/article/details/103356458
    @classmethod
    def set_wh(cls, video, width, height):
        try:
            video.set(cv2.CAP_PROP_FRAME_WIDTH, width)  # width
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)  # height
        except Exception as e:
            print(repr(e))

    # 设置fps
    @classmethod
    def set_fps(cls, video, fps=24):
        try:
            video.set(cv2.CAP_PROP_FPS, fps)
        except Exception as e:
            print(repr(e))