import sys
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout,QMainWindow
from PySide6.QtWidgets import QWidget, QMessageBox

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Message box example')
        self.resize(500,200)

        layout = QVBoxLayout()

        info_button = QPushButton('info Message')
        info_button.clicked.connect(self.show_info_message)
        layout.addWidget(info_button)

        warning_button = QPushButton('Waring Message')
        warning_button.clicked.connect(self.show_warning_message)
        layout.addWidget(warning_button)

        question_button = QPushButton('Question Message')
        question_button.clicked.connect(self.show_question_message)
        layout.addWidget(question_button)

        self.setLayout(layout)

    def show_info_message(self):
        QMessageBox.information(self, "info", "This is information message.", QMessageBox.Ok)

    def show_warning_message(self):
        QMessageBox.warning(self, 'Warning', 'This is warning message.', QMessageBox.Ok)

    def show_question_message(self):
        result = QMessageBox.warning(self, 'Question', 'Keep going?', QMessageBox.Yes | QMessageBox.No)
        if result == QMessageBox.Yes:
            QMessageBox.information(self,'responese','you chose YES', QMessageBox.Ok)
        else:
            QMessageBox.information(self,'responese','you chose NO', QMessageBox.Ok)


if __name__ =='__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())