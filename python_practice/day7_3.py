import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, 
                             QPushButton, QLabel, QVBoxLayout)

class LayoutWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("PyQt5布局与按钮示例")
        self.resize(640, 480)

        # 中心部件与布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 添加标签与按钮
        self.label = QLabel("未点击按钮", self)
        btn = QPushButton("启动检测", self)
        btn.clicked.connect(self.on_click)  # 信号绑定槽函数

        # 添加组件到布局
        layout.addWidget(self.label)
        layout.addWidget(btn)

    def on_click(self):
        self.label.setText("检测启动成功！")  # 按钮点击响应

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LayoutWindow()
    window.show()
    sys.exit(app.exec_())