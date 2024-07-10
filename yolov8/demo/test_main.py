
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 图像分割[0-99]个对象
def test_seg():
    model = YOLO('yolov8n-seg.pt')
    image = Image.open("./assets/bus.jpg")

    """
    boxes: 检测出来物体的矩形框，就是目标检测的框。
    masks: 检测出来的遮罩层，调用图像分割时，这项有数据。
    keypoints: 检测出来的关键点，人体姿势估计时，身体的点就是这项。
    names: 分类数据的名称，比如{0: 人，1: 狗}这类索引。
    refer: https://docs.ultralytics.com/reference/engine/results/
    """
    results = model.predict(source=image, save=False, save_txt=False)
    # print("boxes:")
    # print(results[0].boxes)
    # print("masks:")
    # print(results[0].masks)

    # results是一个支持批量图片的结果集，因为我们只有一张图像，所以取results[0]
    # masks.xy是一张图里所有物体掩膜的轮廓坐标，我们只取一个，取索引为1的物体。
    pixel_xy = results[0].masks.xy[0]
    points = np.array(pixel_xy, np.int32)
    # print(points)

    # 画轮廓并保存图片
    input_image = cv2.imread('./assets/bus.jpg')
    cv2.drawContours(input_image, [points], -1, (0, 255, 0), 2)
    cv2.imwrite('out/output.jpg', input_image)

    # 画边框[所有识别的对象]
    input_image = cv2.imread('./assets/bus.jpg')
    boxes = results[0].boxes
    cls = boxes.cls.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    index = 0
    for clsId in cls:
        print("name:", results[0].names[clsId])
        tmpXyxy = xyxy[index]
        p1 = (int(tmpXyxy[0]), int(tmpXyxy[1]))
        p2 = (int(tmpXyxy[2]), int(tmpXyxy[3]))
        cv2.rectangle(input_image, p1, p2, (0, (50 * index) % 255, 0), 1)
        index += 1
    cv2.imwrite('out/output2.jpg', input_image)

def testRectangle():
    input_image = cv2.imread('./assets/bus.jpg')
    cv2.rectangle(input_image, (10, 20), (60, 120), (0, 0, 255), 1)
    cv2.imshow("rectangle", input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# refer: https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.BaseTensor.to
# 分类[0-999个类]
def test_cls():
    model = YOLO('yolov8n-cls.pt')
    image = Image.open("./assets/snake.jpg")
    results = model.predict(source=image, save=True, save_txt=False)
    print("results.len=", len(results))
    # for tmp in results:
    #     print(tmp)
    # print(results[0].probs)

    top1 = results[0].probs.top1
    top1conf = results[0].probs.top1conf.cpu().numpy()
    print(top1, top1conf)
    print("top1 name:", results[0].names[top1])
    print("prob:", top1conf)

    top5 = results[0].probs.top5
    top5conf = results[0].probs.top5conf.cpu().numpy()
    print(top5)
    print(top5conf)
    for i in range(len(top5)):
        print("names:", results[0].names[top5[i]], "prob:", top5conf[i])


# testRectangle()
test_seg()
# test_cls()