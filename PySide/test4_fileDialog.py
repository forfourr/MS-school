import sys
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout,QMainWindow, QFileDialog

def open_file_dialog():
    file_dialog = QFileDialog()
    file_dialog.setWindowTitle('file open')
    file_dialog.setFileMode(QFileDialog.ExistingFile)   #기존 파일 모드 선택택
    file_dialog.setViewMode(QFileDialog.Detail)         #상세보기

    if file_dialog.exec():
        selected_files = file_dialog.selectedFiles()
        print("selected files: ",selected_files)

    
app = QApplication([])
main_window = QMainWindow()
button  = QPushButton("open file", main_window)
button.clicked.connect(open_file_dialog)
main_window.show()
app.exec()



