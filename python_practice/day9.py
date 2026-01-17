from ultralytics import YOLO
import cv2
import time
import pandas as pd
from numba import jit
@jit(nopython=True)  # 加速帧处理
def process_frame(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 1. 初始化模型（CPU环境，加载nano版模型提速）
model = YOLO('yolov8n.pt')  # yolov8n.pt轻量，适配CPU推理

# 2. 实验参数配置（3组conf对比）
conf_values = [0.3, 0.5, 0.7]
fixed_params = {
    'imgsz': 640,
    'iou': 0.45,
    'verbose': False,  # 关闭冗余日志（工业级调优关键）
    'augment': False,  # 关闭推理增强（CPU提速核心）
    'plots': False,
    'save': False
}

# 3. 测试数据路径（替换为你的素材路径）
img_path = 'test_data/images/img_1.jpg'  # 单张测试图
video_path = 'test_data/videos/video_1.mp4'  # 测试视频

# 4. 定义测试函数（计算FPS与准确率）
def test_conf(conf):
    # 记录开始时间（单图推理测试）
    start_time = time.time()
    # 执行推理
    results = model.predict(source=img_path, conf=conf, **fixed_params)
    # 计算FPS（CPU推理取10次平均值，减少误差）
    fps_list = []
    for _ in range(10):
        t1 = time.time()
        model.predict(source=img_path, conf=conf, **fixed_params)
        t2 = time.time()
        fps = 1 / (t2 - t1)
        fps_list.append(fps)
    avg_fps = sum(fps_list) / len(fps_list)
    
    # 计算准确率（仅统计"person"类，与人工标注对比）
    # 人工标注：假设img_1.jpg含3个行人（根据实际素材调整）
    gt_person = 6
    pred_person = len([box for box in results[0].boxes if box.cls == 0])  # cls=0为person类
    accuracy = (pred_person / gt_person) * 100 if gt_person != 0 else 0
    
    return {'conf': conf, 'avg_fps': round(avg_fps, 2), 'accuracy': round(accuracy, 2)}

# 5. 执行3组实验并记录结果
results_df = pd.DataFrame(columns=['conf', 'avg_fps', 'accuracy'])
for conf in conf_values:
    res = test_conf(conf)
    results_df = pd.concat([results_df, pd.DataFrame([res])], ignore_index=True,sort=False)

# 6. 保存结果到CSV（便于后续分析）
results_df.to_csv('yolov8_param_results.csv', index=False)
print("实验结果：")
print(results_df)

# 视频推理测试（添加到脚本末尾）
cap = cv2.VideoCapture(video_path)
fps_video = 0
frame_count = 0
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 执行推理
    model.predict(source=frame, conf=0.5, **fixed_params)
    frame_count += 1
cap.release()
fps_video = frame_count / (time.time() - start_time)
print(f"\nconf=0.5时视频推理FPS：{round(fps_video, 2)}")
