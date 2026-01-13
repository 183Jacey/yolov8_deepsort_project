import os
import cv2
import numpy as np

img_color = cv2.imread("/home/jacey/桌面/yolo_project/python_practice/dataset/img_1.jpg")  # 彩色图（BGR格式，OpenCV默认)
if img_color is None:
    raise FileNotFoundError("图片读取失败，请检查:1.dataset文件夹是否存在;2.img_1.jpg是否在dataset内")

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

# 3. 保存图像（用于后续数据集预处理备份）
cv2.imwrite("/home/jacey/桌面/yolo_project/python_practice/img_1_gray.jpg", img_gray)
print("灰度图已保存为 img_1_gray.jpg")

# 2. 显示图像（CPU环境适配，避免卡顿）
cv2.imshow("Color Image", img_color)
cv2.imshow("Gray Image", img_gray)
cv2.waitKey(0)  # 按任意键关闭窗口
cv2.destroyAllWindows()  # 释放窗口资源，避免内存泄漏

# 1. 图像缩放（YOLOv8默认输入640×480，保持宽高比）
target_size = (640, 480)
img_resized = cv2.resize(img_color, target_size, interpolation=cv2.INTER_AREA)  # 缩小用INTER_AREA，更清晰
print(f"原图像尺寸：{img_color.shape}，缩放后尺寸：{img_resized.shape}")

# 2. 边缘检测（Canny算法，用于后续电子围栏边界识别、目标轮廓提取）
img_blur = cv2.GaussianBlur(img_gray, (7, 7), 0)  # 高斯模糊降噪（核心预处理步骤）
edges = cv2.Canny(img_blur, threshold1=70, threshold2=150)  # 阈值可调整，50/150为通用最优值

# 3. 显示预处理结果
cv2.imshow("Resized (640*640)", img_resized)
cv2.imshow("Canny Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. 保存预处理结果（用于论文实验数据）
cv2.imwrite("img_1_resized.jpg", img_resized)
cv2.imwrite("img_1_edges.jpg", edges)
print("预处理图像已保存")

# 1. 复制缩放后的图像（避免修改原图）
img_visual = img_resized.copy()

# 2. 绘制矩形检测框（模拟YOLOv8检测行人，坐标为示例，可替换为真实检测结果）
# 矩形参数：图像、左上角坐标、右下角坐标、颜色（BGR）、线宽
cv2.rectangle(img_visual, (220, 160), (380, 440), (0, 255, 0), 2)

# 3. 叠加文字标签（行人+置信度）
# 文字参数：图像、文字内容、坐标、字体、字号、颜色、线宽
cv2.putText(img_visual, "Person: 0.92", (220, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# 4. 显示可视化结果
cv2.imshow("YOLO-like Detection Visualization", img_visual)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 5. 保存可视化结果（用于论文图表、答辩演示）
cv2.imwrite("img_1_visualization.jpg", img_visual)
print("可视化结果已保存")

