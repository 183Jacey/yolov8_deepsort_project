import pandas as pd

def pandas_dataset_operations():
    # 1. 创建数据集（模拟5张检测图片的信息：文件名、宽度、高度）
    data = {
        "image_name": ["img1.jpg", "img2.jpg", "img3.jpg", "img4.jpg", "img5.jpg"],
        "width": [1920, 1280, 1080, 800, 1920],  # 图片宽度
        "height": [1080, 720, 720, 600, 1080]    # 图片高度
    }
    df = pd.DataFrame(data)
    print("1. 原始图片数据集：\n", df)

    # 2. 筛选宽度>1000的高分辨率图片（目标检测优先用高分辨率素材）
    high_res_df = df.loc[df["width"] > 1000]
    print("\n2. 宽度>1000的高分辨率图片：\n", high_res_df)

    # 3. 计算所有图片的宽度平均值（统计数据集尺寸分布）
    avg_width = df["width"].mean()
    print(f"\n3. 所有图片宽度平均值：{avg_width:.0f}像素")


# 直接运行
if __name__ == "__main__":
    pandas_dataset_operations()