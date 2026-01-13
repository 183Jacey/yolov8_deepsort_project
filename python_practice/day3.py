import numpy as np

def numpy_core_operations():
    # 1. 创建3×3随机数组（模拟3个3通道图像的像素块，值范围0-255）
    random_arr = np.random.randint(0, 255, size=(3, 3))
    print("1. 3×3随机像素数组：\n", random_arr)

    # 2. 矩阵乘法（模拟模型中特征映射的计算）
    mat1 = np.array([[1, 2], [3, 4]])  # 模拟特征矩阵1
    mat2 = np.array([[5, 6], [7, 8]])  # 模拟特征矩阵2
    mat_mult = np.dot(mat1, mat2)
    print("\n2. 特征矩阵乘法结果：\n", mat_mult)

    # 3. 计算数组平均值（模拟特征均值归一化）
    arr_avg = np.mean(random_arr)
    print(f"\n3. 随机数组平均值：{arr_avg:.2f}")

    # 4. 数组切片（模拟截取图像ROI区域）
    roi = random_arr[1:, 1:]  # 截取右下角2×2区域（对应目标检测的感兴趣区域）
    print("\n4. 截取的ROI区域（2×2）：\n", roi)


# 直接运行
if __name__ == "__main__":
    numpy_core_operations()
