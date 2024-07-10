
import cv2
import numpy as np

# 读取两张图片
# image1 = cv2.imread('./res/img008.png')
# image2 = cv2.imread('./res/img009.png')
image1 = cv2.imread('./res/img060.png')
image2 = cv2.imread('./res/img064.png')

# 转换为灰度图
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# 计算两张图片的差异
diff = cv2.absdiff(gray1, gray2)

# 应用阈值，将差异显著的部分提取出来
_, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

# 查找差异区域的轮廓
contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 在原始图片上绘制差异区域的矩形
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    if w > 30 or h > 30:
        cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 255, 0), 1)
    else:
        cv2.rectangle(image2, (x, y), (x + w, y + h), (255,255,0), 1)

# 显示结果v
cv2.imshow('Difference', diff)
cv2.imshow('Thresholded Difference', thresholded)
cv2.imshow('Marked Differences', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()