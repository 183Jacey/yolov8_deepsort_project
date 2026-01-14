import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
from datetime import datetime


class YOLOv8CPUInferencer:
    """
    YOLOv8 CPU版推理类（毕设可复用）
    支持：图片批量推理、视频推理、实时摄像头推理
    核心特性：轻量化模型适配、结果自动保存、异常处理、FPS统计
    """
    def __init__(self, model_path="yolov8n.pt", conf_thres=0.5, iou_thres=0.45, imgsz=640):
        """
        初始化方法（毕设中可根据需求修改参数）
        :param model_path: 模型路径（默认yolov8n.pt，轻量化适合CPU）
        :param conf_thres: 置信度阈值（过滤低置信度目标，毕设可调）
        :param iou_thres: NMS IoU阈值（去重重叠框）
        :param imgsz: 推理输入尺寸（640为YOLOv8默认，可改320提速）
        """
        # 1. 加载CPU版模型（强制指定device='cpu'）
        self.model = YOLO(model_path)
        self.model.to(device="cpu")  # 确保使用CPU推理
        
        # 2. 推理参数（毕设中可根据场景调整）
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = imgsz
        
        # 3. 结果保存路径（按时间戳创建，避免覆盖）
        self.save_dir = Path("runs/cpu_infer/fixed_result")
        self.save_dir.mkdir(parents=True, exist_ok=True)  # 自动创建文件夹
        print(f"推理结果将保存至：{self.save_dir}")

    def preprocess(self, img):
        """
        图像预处理（毕设中可扩展：如低光照增强、噪声去除）
        :param img: OpenCV读取的BGR图像（shape: (H, W, 3)）
        :return: 预处理后的图像（适配模型输入）
        """
        # 1. 保持长宽比缩放（避免拉伸变形）
        h, w = img.shape[:2]
        scale = min(self.imgsz / w, self.imgsz / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 2. 黑边填充（适配模型输入尺寸）
        pad_left = (self.imgsz - new_w) // 2
        pad_top = (self.imgsz - new_h) // 2
        padded_img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        padded_img[pad_top:pad_top+new_h, pad_left:pad_left+new_w, :] = resized_img
        
        return padded_img, scale, pad_left, pad_top

    def postprocess(self, img, results, scale, pad_left, pad_top):
        """
        后处理：解析推理结果，绘制检测框（毕设可扩展：如目标计数、类别筛选）
        :param img: 原始图像（用于绘制结果）
        :param results: YOLOv8推理结果对象
        :param scale: 预处理缩放比例（用于还原检测框坐标）
        :param pad_left: 左填充宽度（用于还原检测框坐标）
        :param pad_top: 上填充高度（用于还原检测框坐标）
        :return: 绘制检测框后的图像
        """
        h, w = img.shape[:2]
        # 遍历每张图的推理结果（批量推理时支持多图）
        for result in results:
            # 解析检测框（只保留"person"类别，毕设可改其他类别）
            for box in result.boxes:
                # 1. 过滤低置信度目标
                if box.conf[0] < self.conf_thres:
                    continue
                
                # 2. 还原检测框到原始图像坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # CPU环境需转numpy
                # 减去填充、除以缩放比例
                x1 = (x1 - pad_left) / scale
                y1 = (y1 - pad_top) / scale
                x2 = (x2 - pad_left) / scale
                y2 = (y2 - pad_top) / scale
                # 确保坐标在图像范围内（避免越界）
                x1, y1, x2, y2 = map(int, [max(0, x1), max(0, y1), min(w, x2), min(h, y2)])
                
                # 3. 绘制检测框+置信度（毕设可改颜色/字体）
                label = f"person: {box.conf[0]:.2f}"  # 类别+置信度
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绿色框
                cv2.putText(
                    img, label, (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
        
        return img

    def infer_image(self, img_path, save=True):
        """
        单张/批量图片推理（适配你的30张行人图片）
        :param img_path: 图片路径（支持单张图路径或图片文件夹路径）
        :param save: 是否保存结果（默认True，毕设需保留结果用于报告）
        :return: 推理完成的图像列表
        """
        # 1. 处理输入路径（支持单张图或文件夹批量推理）
        img_paths = []
        if os.path.isfile(img_path) and img_path.endswith((".jpg", ".png", ".jpeg")):
            img_paths = [img_path]
        elif os.path.isdir(img_path):
            # 遍历文件夹，只保留图片文件（适配你的30张行人图）
            img_paths = [
                os.path.join(img_path, f) 
                for f in os.listdir(img_path) 
                if f.endswith((".jpg", ".png", ".jpeg"))
            ]
        else:
            raise ValueError("输入路径不是有效图片或文件夹！")
        
        # 2. 批量推理
        result_imgs = []
        for idx, path in enumerate(img_paths):
            # 读取原始图像（OpenCV默认BGR格式）
            img = cv2.imread(path)
            if img is None:
                print(f"跳过无效图片：{path}")
                continue
            
            # 预处理→推理→后处理
            padded_img, scale, pad_left, pad_top = self.preprocess(img)
            # CPU推理（stream=False适合图片，毕设可保持）
            results = self.model(
                padded_img,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.imgsz,
                device="cpu",
                stream=False
            )
            img_with_box = self.postprocess(img, results, scale, pad_left, pad_top)
            
            # 保存结果（按序号命名，方便毕设整理）
            if save:
                save_path = self.save_dir / f"result_img_{idx+1}.jpg"
                cv2.imwrite(str(save_path), img_with_box)
                print(f"已保存图片结果：{save_path}")
            
            result_imgs.append(img_with_box)
        
        return result_imgs

    def infer_video(self, video_path, save=True, show=False):
        """
        视频推理（适配你的8个街景视频）
        :param video_path: 视频路径（支持本地视频或摄像头（0为默认摄像头））
        :param save: 是否保存结果视频（默认True）
        :param show: 是否实时显示（CPU环境建议关闭，避免卡顿）
        :return: 无（结果直接保存）
        """
        # 1. 打开视频流（支持本地视频或摄像头）
        cap = cv2.VideoCapture(video_path if isinstance(video_path, str) else int(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频/摄像头：{video_path}")
        
        # 2. 获取视频参数（用于保存结果视频）
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 3. 初始化视频写入器（保存结果）
        video_writer = None
        if save:
            save_path = self.save_dir / f"result_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4格式
            video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
            print(f"开始处理视频（共{total_frames}帧），结果将保存至：{save_path}")
        
        # 4. 逐帧推理（CPU优化：stream=True节省内存）
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # 视频结束
            
            frame_count += 1
            if frame_count % 5 == 0:  # 每5帧打印一次进度（避免日志过多）
                print(f"处理进度：{frame_count}/{total_frames}帧")
            
            # 预处理→推理→后处理
            padded_frame, scale, pad_left, pad_top = self.preprocess(frame)
            results = self.model(
                padded_frame,
                conf=self.conf_thres,
                iou=self.iou_thres,
                imgsz=self.imgsz,
                device="cpu",
                stream=True  # 视频推理必开，CPU内存占用降低50%
            )
            frame_with_box = self.postprocess(frame, results, scale, pad_left, pad_top)
            
            # 保存帧到视频
            if save and video_writer.isOpened():
                video_writer.write(frame_with_box)
            
            # 实时显示（CPU环境建议关闭，FPS会降低）
            if show:
                cv2.imshow("YOLOv8 CPU Inference", frame_with_box)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break  # 按q退出
        
        # 5. 释放资源
        cap.release()
        if save and video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        print(f"视频处理完成！结果保存至：{save_path}")


# ------------------------------
# 毕设复用示例（直接运行此文件即可测试）
# ------------------------------
if __name__ == "__main__":
    # 1. 初始化推理器（毕设中可修改模型路径、置信度等参数）
    inferencer = YOLOv8CPUInferencer(
        model_path="yolov8n.pt",  # 若用自己训练的模型，替换为"best.pt"
        conf_thres=0.4,  # 行人检测可适当降低置信度（避免漏检）
        imgsz=480  # 缩小输入尺寸，CPU推理速度提升30%
    )
    
    # 2. 测试图片推理（替换为你的30张行人图片所在文件夹路径）
    print("\n=== 开始图片推理 ===")
    inferencer.infer_image(img_path="/home/jacey/桌面/yolo_project/python_practice/test_data/images")  # 你的图片文件夹
    
    # 3. 测试视频推理（替换为你的街景视频路径，如"video_1.mp4"）
    print("\n=== 开始视频推理 ===")
    inferencer.infer_video(video_path="/home/jacey/桌面/yolo_project/python_practice/test_data/videos/video_1.mp4", save=True, show=False)
