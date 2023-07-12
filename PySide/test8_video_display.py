import sys
import os
import cv2
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout,QStatusBar,QMainWindow
from PySide6.QtWidgets import QSizePolicy, QFileDialog, QTreeWidgetItem, QLabel,QWidget, QListWidget,QBoxLayout
from PySide6 import QtCore, QtGui
from PySide6.QtCore import Qt, QTimer

class VideoViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Video Viewer')
        self.resize(800,600)

        # 비디오 오픈 버튼
        self.video_file_button = QPushButton('Open video file')
        self.video_file_button.clicked.connect(self.open_video_file_dialog)
        # 비디오 재생 버튼
        self.play_button = QPushButton('Play video')
        self.play_button.clicked.connect(self.play_video)
        # 비디오 정지 버튼
        self.pause_button= QPushButton('Pause')
        self.pause_button.clicked.connect(self.pause_video)
        # 비디오 캡쳐 버튼
        self.capture_button = QPushButton('Capture')
        self.capture_button.clicked.connect('Capture')

        #???
        self.video_view_label = QLabel()
        self.video_view_label.setAlignment(Qt.AlignCenter)
        self.video_view_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        ##### Layout #####
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.video_file_button)
        main_layout.addWidget(self.play_button)
        main_layout.addWidget(self.pause_button)
        main_layout.addWidget(self.capture_button)
        main_layout.addWidget(self.video_view_label)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        ### 비디오 파라미터 설정
        self.video_path = ''
        self.video_width = 720
        self.video_height = 640

        self.video_capture = None
        self.video_timer = QTimer()
        self.video_timer.timeout.connect(self.display_next_frame)

        self.pause = False
        self.current_frame = 0
        self.capture_cnt = 0

        #?
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)




    def open_video_file_dialog(self):
        video_dialog = QFileDialog(self)
        #확장자 선택
        video_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mov *.mkv)")
        if video_dialog.exec():
            selected_video = video_dialog.selectedFiles()
            if selected_video:
                self.video_path = selected_video[0] #['video_path']
                self.status_bar.showMessage(f"Chosen Video: {self.video_path}")

    def display_next_frame(self):
        if self.video_path:
            ret, frame = self.video_capture.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def play_video(self):
        if self.video_path:
            if self.pause:
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                self.paused =  False
            else:
                self.video_capture = cv2.VideoCapture(self.video_path)
                self.current_frame = 0
            
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.capture_button.setEnabled(True)
            self.video_timer.start(30)

    def pause_video(self):
        self.video_timer.stop()
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.capture_button.setEnabled(not self.paused)
        self.pause = True

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoViewer()
    window.show()
    sys.exit(app.exec())