import sys
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout,QMainWindow
from PySide6.QtWidgets import QFileDialog, QTreeWidget, QWidget, QTreeWidgetItem
import os
'''
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

'''

class FileExplorer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('File Explorer')
        self.resize(500, 400)

        self.folder_button = QPushButton('file open')
        self.folder_button.clicked.connect(self.open_file_dialog)

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(['File'])

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.folder_button)
        main_layout.addWidget(self.tree_widget)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)



    def open_file_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.Directory)   #기존 파일 모드 선택택
        file_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        file_dialog.directoryEntered.connect(self.set_folder_path)
        file_dialog.accepted.connect(self.display_file)
        file_dialog.exec_()

    def set_folder_path(self, folder_path):
        self.folder_path = folder_path

    def display_file(self):
        if self.folder_path:
                self.tree_widget.clear()

                root_item = QTreeWidgetItem(self.tree_widget, [self.folder_path])
                self.tree_widget.addTopLevelItem(root_item)

                for dir_path, _,file_names in os.walk(self.folder_path):
                     dir_item = QTreeWidgetItem(root_item, [os.path.basename(dir_path)])
                     root_item.addChild(dir_item)

                     for file_name in file_names:
                          file_item = QTreeWidgetItem(dir_item, [file_name])
                          dir_item.addChild(file_item)

                     root_item.setExpanded(True)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = FileExplorer()
    main_window.show()
    app.exit(app.exec())