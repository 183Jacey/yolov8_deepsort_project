import os
import cv2

# VisDrone类别映射（仅保留行人，对应YOLO类别ID=0）
VISDRONE_TO_YOLO = {1: 0}

def convert_annotation(ann_path, img_path, save_dir):
    # 读取图片尺寸（用于归一化）
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    # 读取VisDrone标注
    with open(ann_path, 'r') as f:
        lines = f.readlines()
    
    # 转换并保存YOLO标注
    yolo_lines = []
    for line in lines:
        parts = line.strip().split(',')
        # VisDrone标注格式：<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
        obj_cat = int(parts[5])
        if obj_cat not in VISDRONE_TO_YOLO:
            continue  # 过滤非行人类别
        
        # 转换为YOLO格式（中心x, 中心y, 宽, 高，归一化）
        x1 = int(parts[0])
        y1 = int(parts[1])
        bw = int(parts[2])
        bh = int(parts[3])
        
        cx = (x1 + bw/2) / w
        cy = (y1 + bh/2) / h
        nw = bw / w
        nh = bh / h
        
        yolo_lines.append(f"{VISDRONE_TO_YOLO[obj_cat]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
    
    # 保存YOLO标注（与图片同名，存到labels目录）
    img_name = os.path.basename(img_path).replace('.jpg', '.txt')
    save_path = os.path.join(save_dir, img_name)
    with open(save_path, 'w') as f:
        f.writelines(yolo_lines)

def process_dataset(visdrone_root, split, save_root):
    # 创建YOLO格式的目录（images/labels + split）
    img_save_dir = os.path.join(save_root, 'images', split)
    label_save_dir = os.path.join(save_root, 'labels', split)
    os.makedirs(img_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)
    
    # 处理每张图片和标注
    visdrone_img_dir = os.path.join(visdrone_root, 'images')
    visdrone_ann_dir = os.path.join(visdrone_root, 'annotations')
    
    for img_name in os.listdir(visdrone_img_dir):
        if not img_name.endswith('.jpg'):
            continue
        img_path = os.path.join(visdrone_img_dir, img_name)
        ann_path = os.path.join(visdrone_ann_dir, img_name.replace('.jpg', '.txt'))
        
        # 复制图片到YOLO目录
        os.system(f"cp {img_path} {img_save_dir}/")
        # 转换标注
        convert_annotation(ann_path, img_path, label_save_dir)

if __name__ == "__main__":
    # 配置路径（替换为你的实际路径）
    VISDRONE_TRAIN_ROOT = "../data_set/VisDrone2019-DET-train"
    VISDRONE_VAL_ROOT = "../data_set/VisDrone2019-DET-val"
    YOLO_DATA_ROOT = "../data_set/merged_dataset"  # 融合后的数据集根目录
    
    # 处理训练集和验证集
    process_dataset(VISDRONE_TRAIN_ROOT, 'train', YOLO_DATA_ROOT)
    process_dataset(VISDRONE_VAL_ROOT, 'val', YOLO_DATA_ROOT)
    print("VisDrone格式转换完成，已保存到", YOLO_DATA_ROOT)
    