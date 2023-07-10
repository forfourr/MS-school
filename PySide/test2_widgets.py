import sys
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout,QMainWindow
from PySide6.QtWidgets import QLabel, QLineEdit, QPushButton, QCheckBox, QMessageBox


'''
app = QApplication([])
window= QWidget()

label = QLabel('Welcome, Enter your ID number')
line_edit = QLineEdit()
save_button = QPushButton('Enter')
checkbox = QCheckBox('Agree private info')

layout = QVBoxLayout()

layout.addWidget(label)σ
layout.addWidget(line_edit)
layout.addWidget(save_button)
layout.addWidget(checkbox)

window.setLayout(layout)
window.show()
app.exec()
'''

#Class로 한번에 표현하기
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Widgets example')
        self.resize(500,300)

        self.label = QLabel('Welcome, Enter your ID number')
        self.line_edit = QLineEdit()
        self.save_button = QPushButton('Enter')
        self.checkbox = QCheckBox('Agree private info')

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.line_edit)
        layout.addWidget(self.save_button)
        layout.addWidget(self.checkbox)

        self.setLayout(layout)

        self.save_button.clicked.connect(self.show_message)

        self.id_list =[]


        # button position, size

    def show_message(self):
        if self.checkbox.isChecked():
            message = self.line_edit.text()
            self.id_list.append(message)
            print(f"ID: {message}")
            print(f"list: {self.id_list}")
            self.line_edit.clear()
        else:
            error_message = 'Please check your button!'
            QMessageBox.critical(self,'Error',error_message)
            self.line_edit.clear()


        



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())