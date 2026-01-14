import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

class SimpleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("PyQt5入门窗口")  
        self.resize(640, 480)  

        label = QLabel("行人检测与跟踪系统", self)
        label.move(250, 200)  

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SimpleWindow()
    window.show()
    sys.exit(app.exec_())
