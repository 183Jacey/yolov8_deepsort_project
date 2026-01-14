# yolo_cpu虚拟环境安装命令备份
## 1.创建虚拟环境
python3 -m venv ~/yolo_cpu

## 2. 激活虚拟环境
source ~/yolo_cpu/bin/activate

## 3.安装核心依赖（已验证版本兼容，带阿里源，避免网络问题）
# 安装PyTorch（CPU版）
pip install torch==2.1.0+cpu torchvision==0.16.0+cpu torchaudio==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html -i https://mirrors.aliyun.com/pypi/simple/

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装OpenCV
pip install opencv-python==4.8.0.76 -i https://mirrors.aliyun.com/pypi/simple/

# pip install opencv-python==4.8.0.76 numpy pandas ultralytics==8.0.196 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装labelImg（图像标注工具）
pip install labelImg -i https://mirrors.aliyun.com/pypi/simple/

