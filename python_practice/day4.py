import torch
import numpy as np

# 1. 创建随机张量（模拟2张640×640的3通道图像数据，batch=2, channel=3, h=640, w=640）
img_tensor = torch.rand(size=(2, 3, 640, 640), dtype=torch.float32)
print("1. 模拟图像张量：")
print(f"   形状：{img_tensor.shape}，数据类型：{img_tensor.dtype}，设备：{img_tensor.device}")

# 2. 创建零张量（模拟模型权重初始化模板）
weight_tensor = torch.zeros(size=(10, 3, 3, 3), dtype=torch.float32)  # 10个3×3×3卷积核
print("\n2. 卷积核零张量：")
print(f"   形状：{weight_tensor.shape}，数据类型：{weight_tensor.dtype}")

# 3. 从Numpy数组转换（模拟标注数据转换，如2个目标的坐标：[x1,y1,x2,y2,class]）
np_annot = np.array([[100, 200, 300, 400, 0], [50, 150, 250, 350, 1]], dtype=np.float32)
annot_tensor = torch.from_numpy(np_annot)
print("\n3. Numpy转换的标注张量：")
print(f"   形状：{annot_tensor.shape}，数据类型：{annot_tensor.dtype}")
