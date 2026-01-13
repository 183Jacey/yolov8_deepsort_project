import os

def get_all_image_files(folder_path):
    """
    遍历指定文件夹下所有图片文件（支持常见格式：jpg/jpeg/png/bmp）
    :param folder_path: 文件夹路径（支持~、相对/绝对路径）
    :return: 图片文件的绝对路径列表
    """
    # 解析路径中的~为用户主目录（避免路径错误）
    abs_folder = os.path.expanduser(folder_path)
    # 检查路径是否存在
    if not os.path.exists(abs_folder):
        print(f"错误：文件夹[{abs_folder}]不存在")
        return []
    
    # 支持的图片格式（目标检测数据集常用格式）
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_paths = []
    
    # 遍历文件夹（包括子文件夹）
    for root, _, files in os.walk(abs_folder):
        for file in files:
            # 判断文件后缀是否为图片格式
            if file.lower().endswith(image_extensions):
                # 拼接为绝对路径
                image_paths.append(os.path.join(root, file))
    
    return image_paths


# 直接调用示例（替换为你的数据集文件夹路径）
dataset_folder = "~/桌面/yolo_project/python_practice/dataset"  # 你的图片文件夹
all_images = get_all_image_files(dataset_folder)
print(f"找到{len(all_images)}张图片，示例路径：{all_images[:2]}")


def has_target_string(source_str,target_str):
    if not isinstance(source_str,str) or not isinstance(target_str,str):
        return False
    
    return target_str.lower() in source_str.lower()


image_name = ["img_1.jpg","img_person_02.jpg","video_car.mp4","test_PEOPLE_03.png"]
target_keyword = "person"
matched_files = [name for name in image_name if has_target_string(name,target_keyword)]
print(f"包含关键词[{target_keyword}]的文件:{matched_files}")

def calculate_list_average(num_list):
    if not isinstance(num_list,list) or len(num_list) == 0:
        return 0.0
    return round(sum(num_list)/len(num_list),2)


confidence_scores = [0.85,0.72,0.91,0.68,0.88]
print(f"检测置信度平均值：{calculate_list_average(confidence_scores)}")

def batch_process_image_files(folder_path, target_keyword):
    """
    批量处理图片文件夹：筛选含目标关键词的文件 + 统计数量
    （控制流练习：结合循环、条件判断、函数调用）
    """
    # 1. 获取所有图片路径
    image_paths = get_all_image_files(folder_path)
    if len(image_paths) == 0:
        print("无图片文件可处理")
        return
    
    # 2. 筛选含目标关键词的图片
    matched_images = []
    for path in image_paths:
        # 从路径中提取文件名
        file_name = os.path.basename(path)
        if has_target_string(file_name, target_keyword):
            matched_images.append(path)
    
    # 3. 输出统计结果
    print(f"文件夹[{folder_path}]中，含关键词[{target_keyword}]的图片共{len(matched_images)}张：")
    for idx, path in enumerate(matched_images[:5], 1):  # 只显示前5个示例
        print(f"  {idx}. {path}")


# 直接调用示例
batch_process_image_files("~/桌面/yolo_project/python_practice/dataset", "person")