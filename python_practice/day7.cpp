#include <iostream>
#include <vector>
#include <string>
using namespace std;

// 类与继承练习（模拟检测器基础类）
class BaseDetector {
public:
    virtual void detect(string img_path) {  // 虚函数，支持多态
        cout << "基础检测：" << img_path << endl;
    }
};

class YOLODetector : public BaseDetector {
public:
    void detect(string img_path) override {
        cout << "YOLO检测：" << img_path << endl;
    }
};

int main() {
    // 容器与循环练习
    vector<string> img_paths = {"img_1.jpg", "img_2.jpg", "img_3.jpg"};
    for (auto& path : img_paths) {
        YOLODetector detector;
        detector.detect(path);
    }
    return 0;
}