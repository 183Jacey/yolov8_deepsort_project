#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main() {
    // 读取图片
    Mat img = imread("dataset/img_1.jpg");  
    if (img.empty()) {
        cout << "图片读取失败！" << endl;
        return -1;
    }

    // 缩放至640×480（YOLOv8默认输入尺寸）
    Mat img_resized;
    resize(img, img_resized, Size(640, 480));

    // 灰度转换
    Mat img_gray;
    cvtColor(img_resized, img_gray, COLOR_BGR2GRAY);

    // 保存结果
    imwrite("cpp_resized.jpg", img_resized);
    imwrite("cpp_gray.jpg", img_gray);
    cout << "图像处理完成，已保存结果" << endl;

    return 0;
}
