import torch

# 1. view/reshape（展平/调整形状，确保元素总数不变）
img = torch.rand((1, 3, 640, 640))  # 单张图像：(batch, c, h, w)
img_flatten = img.view(1, -1)  # 展平为1维：-1表示自动计算元素数（1×3×640×640=1228800）
img_reshape = img.reshape(1, 3, 320, 1280)  # 调整分辨率（保持c=3，总像素不变）
print("1. view/reshape：")
print(f"   展平后形状：{img_flatten.shape}，调整分辨率后：{img_reshape.shape}")

# 2. squeeze/unsqueeze（增减维度，适配batch或通道）
single_img = torch.rand((3, 640, 640))  # 无batch维度的图像
img_with_batch = single_img.unsqueeze(0)  # 增加batch维度：(1,3,640,640)
img_remove_batch = img_with_batch.squeeze(0)  # 移除batch维度：(3,640,640)
print("\n2. squeeze/unsqueeze：")
print(f"   加batch后：{img_with_batch.shape}，去batch后：{img_remove_batch.shape}")

# 3. transpose/permute（维度交换，如图像格式从CHW转HWC）
img_chw = torch.rand((3, 640, 640))  # PyTorch默认格式：CHW
img_hwc = img_chw.permute(1, 2, 0)  # 交换为HWC（适配OpenCV显示）
print("\n3. permute（CHW→HWC）：")
print(f"   原格式形状：{img_chw.shape}，目标格式：{img_hwc.shape}")
