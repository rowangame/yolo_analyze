
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

def test_seg():
    model = YOLO('tws-jbl-best.pt')
    imgPath = "./datasets/tws-jbl/images/val/ca1d05af-open-rin-001.jpg"
    # image = Image.open(imgPath)
    image = cv2.imread(imgPath, cv2.IMREAD_COLOR)

    """
    boxes: 检测出来物体的矩形框，就是目标检测的框。
    masks: 检测出来的遮罩层，调用图像分割时，这项有数据。
    keypoints: 检测出来的关键点，人体姿势估计时，身体的点就是这项。
    names: 分类数据的名称
    refer: https://docs.ultralytics.com/reference/engine/results/
    """
    results = model.predict(source=image, save=True, save_txt=False)

    # 画边框[所有识别的对象]
    input_image = cv2.imread(imgPath)
    boxes = results[0].boxes
    confs = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(np.int32)
    xyxy = boxes.xyxy.cpu().numpy()
    index = 0
    for clsId in cls:
        print("name:", results[0].names[clsId], "conf:", confs[index])
        tmpXyxy = xyxy[index]
        p1 = (int(tmpXyxy[0]), int(tmpXyxy[1]))
        p2 = (int(tmpXyxy[2]), int(tmpXyxy[3]))
        cv2.rectangle(input_image, p1, p2, (0, 255, 0), 1)
        index += 1
    cv2.imwrite('out/tws-out.jpg', input_image)

test_seg()