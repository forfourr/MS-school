import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('창크기 조절 예제')
        self.resize(500,500)

        # button
        self.button = QPushButton("Click", self)
        self.button.clicked.connect(self.buttonClicked)

        # button position, size
        self.button.setGeometry(50, 50, 200, 50)

    def buttonClicked(self):
        print("CLICKED!")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    sys.exit((app.exec()))