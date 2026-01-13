import torch

# 1. 逐元素运算（模拟激活函数输入计算，如Conv层输出加偏置）
conv_out = torch.rand((2, 10, 640, 640))  # 模拟2张图经过10通道卷积的输出
bias = torch.rand((10, 1, 1))  # 10个通道的偏置（广播机制适配）
conv_out_with_bias = conv_out + bias  # 逐元素加法（广播：(2,10,640,640)+(10,1,1)）
print("1. 逐元素加法（Conv+偏置）：")
print(f"   输入形状：{conv_out.shape}，输出形状：{conv_out_with_bias.shape}")

# 2. 矩阵乘法（模拟线性层计算，如YOLOv8的检测头线性映射）
flatten_conv = conv_out_with_bias.view(2, -1)  # 展平：(2,10,640,640)→(2, 10×640×640)
linear_weight = torch.rand((10*640*640, 85))  # 检测头权重（85=4坐标+1置信度+80类别）
detection_out = torch.matmul(flatten_conv, linear_weight)  # 矩阵乘法：(2, N) × (N, 85)
print("\n2. 矩阵乘法（检测头计算）：")
print(f"   展平后形状：{flatten_conv.shape}，检测输出形状：{detection_out.shape}")

# 3. 聚合运算（模拟损失计算中的均值/求和）
loss_tensor = torch.rand((2, 100))  # 模拟2个样本的100个预测框损失
avg_loss = loss_tensor.mean()  # 计算平均损失
sum_loss = loss_tensor.sum()    # 计算总损失
print("\n3. 聚合运算（损失计算）：")
print(f"   平均损失：{avg_loss:.4f}，总损失：{sum_loss:.4f}")
