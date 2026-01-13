import torch

# 1. 定义需求导的张量（模拟模型权重、偏置）
w = torch.tensor([0.5], requires_grad=True)  # 权重，开启求导
b = torch.tensor([1.2], requires_grad=True)  # 偏置，开启求导

# 2. 模拟前向传播（简单线性模型：y = w*x + b，x为输入，y_pred为预测值）
x = torch.tensor([2.0, 3.0, 4.0])  # 3个输入样本（模拟3个预测框的特征）
y_true = torch.tensor([2.2, 3.7, 5.2])  # 真实值（模拟标注）
y_pred = w * x + b  # 线性预测

# 3. 计算损失（均方误差，模拟模型损失函数）
loss = torch.mean((y_pred - y_true) ** 2)
print(f"前向传播：")
print(f"  预测值：{y_pred.detach().numpy()}（detach()屏蔽求导，仅看数值）")
print(f"  损失值：{loss.item():.4f}")

# 4. 反向传播（自动计算梯度：d(loss)/dw、d(loss)/db）
loss.backward()
print(f"\n反向传播（梯度计算）：")
print(f"  权重w的梯度：{w.grad.item():.4f}")  # 预期≈(y_pred-y_true)*x的均值×2
print(f"  偏置b的梯度：{b.grad.item():.4f}")  # 预期≈(y_pred-y_true)的均值×2