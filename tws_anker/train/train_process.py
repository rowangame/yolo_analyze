
from ultralytics import YOLO

def close_error_thread():
    from multiprocessing import freeze_support
    # 运行上方法,防止训练时，异常结束没有中止了线程的问题
    # freeze_support()

def train_model():
    # 开始训练
    model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）

    # 使用1个GPU训练模型
    results = model.train(data='tws_anker.yaml', epochs=300, imgsz=640, device=[0], workers=2)

if __name__ == "__main__":
    train_model()