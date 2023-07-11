import sys
import os
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout,QHBoxLayout,QMainWindow
from PySide6.QtWidgets import QTreeWidget, QFileDialog, QTreeWidgetItem, QLabel,QWidget, QListWidget,QBoxLayout
from PySide6 import QtCore, QtGui


class ImageViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Image viewer')
        self.resize(500,400)

        # button, image 자리 생성
        self.folder_button = QPushButton('Open folder')
        self.folder_button.clicked.connect(self.open_folder)

        self.back_button = QPushButton('Back')
        self.back_button.clicked.connect(self.back)

        self.forward_button = QPushButton('Forward')
        self.forward_button.clicked.connect(self.forward)

        self.image_label = QLabel()
        self.image_label.setAlignment(QtCore.Qt.AlignCenter)

        self.image_widget = QListWidget()
        self.image_widget.currentRowChanged.connect(self.display_image)

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabel(['File'])

        # Layout 설정
        left_layout = QBoxLayout()
        left_layout.addWidget(self.folder_button)
        left_layout.addWidget(self.image_widget)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.image_label)
        right_layout.addWidget(self.back_button)
        right_layout.addWidget(self.forward_button)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout,1)
        main_layout.addLayout(right_layout,2)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # 변수값 초기 설정
        self.current_folder = ''
        self.current_images = []
        self.current_idx = -1


        
    def open_folder(self):
        folder_dialog = QFileDialog(self)
        #폴더 선택값 설정
        folder_dialog.setFileMode(QFileDialog.directory)
        folder_dialog.setOption(QFileDialog.ShowDirsOnly, True)
        #파일 입력 되면
        folder_dialog.directoryEntered.connect(self.set_folder_path)
        #파일 받으면
        folder_dialog.accepted.connect(self.load_images)
        folder_dialog.exec_()

    def set_folder_path(self,folder_path):
        self.current_folder = folder_path

    def load_images(self):
        self.image_widget.clear()
        self.tree_widget.clear()

        if self.current_folder:
            self.current_images = []
            self.current_idx -1

            images_extensions = ('.jpg', '.jpeg','.png','.gif', 'bmp')

            root_item = QTreeWidgetItem(self.tree_widget, [self.current_folder])
            self.tree_widget.addTopLevelItem(root_item)

            for dir_path, _, file_names in os.walk(self.current_folder):
                dir_item = QTreeWidgetItem(root_item, [os.path.basename(dir_path)])
                root_item.addChild(dir_item)

                for file_name in file_names:
                    if file_name.lower().endswith(images_extensions):
                        file_item = QTreeWidgetItem(dir_item,[file_name])
                        dir_item.addChild(file_item)
                        file_path = os.path.join(dir_path, file_name)
                        self.current_images.append(file_path)
                        self.image_widget.addItem(file_name)

            if self.current_images:
                self.image_widget.setCurrentRow(0)

    def display_image(self, index):
        if 0<= index < len(self.current_images):
            self

