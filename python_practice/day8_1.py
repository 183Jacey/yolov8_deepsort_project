import cv2
from ultralytics import YOLO
import os  # 用于判断文件是否存在

class YOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        self.model = self.load_model(model_path)
        self.conf_threshold = conf_threshold

    def load_model(self, model_path):
        try:
            # 异常场景2：模型路径不存在
            if not os.path.exists(model_path) and model_path != "yolov8n.pt":
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = YOLO(model_path)
            print(f"Model loaded successfully from {model_path}")
            return model
        except FileNotFoundError as e:
            print(f"[Error] Model load failed: {str(e)}")
            return None
        except Exception as e:  # 捕获其他未知异常（如模型损坏）
            print(f"[Error] Unexpected error when loading model: {str(e)}")
            return None

    def read_image(self, img_path):
        try:
            # 异常场景3：传入非字符串路径
            if not isinstance(img_path, str):
                raise TypeError("Image path must be a string, not " + str(type(img_path)))
            # 异常场景1：图像路径不存在
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file not found: {img_path}")
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Image is corrupted or unsupported format: {img_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb
        except TypeError as e:
            print(f"[Error] Image read failed: {str(e)}")
            return None
        except FileNotFoundError as e:
            print(f"[Error] Image read failed: {str(e)}")
            return None
        except ValueError as e:
            print(f"[Error] Image read failed: {str(e)}")
            return None

    def detect(self, img_path):
        img = self.read_image(img_path)
        if img is None:
            return None
        print(f"Image read successfully, shape: {img.shape}")
        return img



# 测试代码（添加到文件末尾）
if __name__ == "__main__":
    # 初始化检测器
    detector = YOLODetector()
    
    # 测试1：图像路径不存在（场景1）
    print("\n=== Test 1: Non-existent image path ===")
    detector.detect("non_existent_img.jpg")
    
    # 测试2：传入非字符串路径（场景3）
    print("\n=== Test 2: Non-string image path ===")
    detector.detect(12345)
    
    # 测试3：模型路径错误（场景2，需先准备一个不存在的模型路径）
    print("\n=== Test 3: Invalid model path ===")
    bad_detector = YOLODetector(model_path="bad_model.pt")
    
    # 测试4：正常场景（需准备一张测试图，如~/python_practice/test_img.jpg）
    print("\n=== Test 4: Normal scenario ===")

    # 测试5：若有测试图，替换为实际路径；无则跳过，后续用预热计划收集的素材
    print("\n=== Test 5: Image read successfully ===")
    detector.detect("python_practice/img_1.jpg")