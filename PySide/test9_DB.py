import sys
import os
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, \
                                QLabel, QLineEdit, QPushButton, QMessageBox, QStackedWidget, QListWidget

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker
"""
QWidget는 작은 단일 위젯을 다루는 경우에 사용되고,
QApplication은 애플리케이션의 실행과 이벤트 루프를 처리하는 경우에 사용되며, 
QMainWindow는 주 창과 관련된 기능을 처리하는 경우에 사용

"""

# Set database
os.makedirs('db', exist_ok=True)
engine = create_engine('sqlite:///db/user.db', echo=True)   #db연결
Base = declarative_base()   #db model 정의= base 클래스 정의


# Set User model -> Table
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    password = Column(String)

    def __init__(self, username, password):
        self.username = username
        self.password = password

# sessionmaker
# ->트랜잭션을 관리하고, 변경 사항을 추적하며, 영구 저장소와의 상호 작용을 처리
Session = sessionmaker(bind=engine)
session = Session()

# #check
# Base.metadata.create_all(engine)

# Register Page
# QApp : 핵심 설정, 진입점
class RegisterPage(QWidget):
    def __init__(self, stacked_widget, main_window):
        super().__init__()

        self.stack_widget = stacked_widget
        self.main_window = main_window

        self.layout = QVBoxLayout()

        self.username_label = QLabel('User Name: ')
        self.username_input = QLineEdit()

        self.password_label = QLabel("Password: ")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)     #비밀번호 입력 안보이게

        self.register_button = QPushButton('Register')
        self.register_button.clicked.connect(self.register)

        self.login_button = QPushButton('Go back to Login')
        self.login_button.clicked.connect(self.login)

        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.register_button)
        self.layout.addWidget(self.login_button)
        self.setLayout(self.layout)


    
    def register(self):
        #입력받은 값 -> text
        username = self.username_input.text()
        password = self.password_input.text()
        
        #받지 않은 경우
        if not username or not password:
            QMessageBox.warning(self,'error','Please enter your username and password')
            return
        # Table 만들기
        user = User(username, password)

        # DB에 넣기
        session.add(user)
        session.commit()

        QMessageBox.information(self,'success','Register success!')
        # 로그인 페이지(1)로 넘어감
        # self.stack_widget.setCorrenetIndex(1)
        # self.main_window.show_Login_page()

    def login(self):
        self.main_window.show_login_page()


# stacked_widget 변수로 받는 이유: 페이지 이동
class Login_page(QWidget):
    def __init__(self, stacked_widget, main_window):
        super().__init__()

        self.stacked_widget = stacked_widget
        self.main_window = main_window

        self.layout = QVBoxLayout()

        self.username_label = QLabel('User Name: ')
        self.username_input = QLineEdit()

        self.password_label = QLabel("Password: ")
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.Password)     #비밀번호 입력 안보이게

        self.login_button = QPushButton('Login')
        self.login_button.clicked.connect(self.Login)

        self.register_button = QPushButton('If not have id, Register')
        self.register_button.clicked.connect(self.register)

        self.layout.addWidget(self.username_label)
        self.layout.addWidget(self.username_input)
        self.layout.addWidget(self.password_label)
        self.layout.addWidget(self.password_input)
        self.layout.addWidget(self.login_button)
        self.layout.addWidget(self.register_button)
        self.setLayout(self.layout)

    def Login(self):
        # 입력받은 값 -> txt
        username = self.username_input.text()
        password = self.password_input.text()

        # 입력 안됐을 때 warning
        if not username or not password:
            QMessageBox.warning(self,'error','Please enter your username and password')
            return
        
        #사용자 조회
        user = session.query(User).filter_by(username=username,
                                             password=password).first()
        if user:
            QMessageBox.information(self,'success','Login Sucessful')
            # admin 페이지 이동
            self.main_window.show_admin_page()
        else:
            QMessageBox.warning(self,'error','Invalid username or password')

    def register(self):
        #register page로 이동
        self.main_window.show_register_page()   # MainWindow의 내장 함수 실행
        



### Admin Page
class AdminPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        self.layout = QVBoxLayout()

        self.user_list = QListWidget()

        self.show_user_list_button = QPushButton('Show your List')
        self.show_user_list_button.clicked.connect(self.show_user_list)

        self.logout_button = QPushButton('Logout')
        self.logout_button.clicked.connect(self.logout)

        self.layout.addWidget(self.show_user_list_button)
        self.layout.addWidget(self.user_list)
        self.layout.addWidget(self.logout_button)
        self.setLayout(self.layout)


    def show_user_list(self):
        self.user_list.clear()

        #모든 사용자 조회
        users = session.query(User).all()
        for user in users:
            self.user_list.addItem(user.username)

    def logout(self):
        self.main_window.show_login_page()



# #### Main Page
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Main page')
        self.resize(600,400)

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # 각 페이지 정의
        self.register_page = RegisterPage(self.stacked_widget, self)
        self.login_page = Login_page(self.stacked_widget, self)
        self.admin_page = AdminPage(self)

        # 순서대로 인덱스 0,1,2
        self.stacked_widget.addWidget(self.login_page)      #초기 페이지 설정
        self.stacked_widget.addWidget(self.register_page)
        self.stacked_widget.addWidget(self.admin_page)

        #초기 페이지 보여줌
        self.show_login_page()

    def show_login_page(self):
        self.stacked_widget.setCurrentIndex(0)  #login_page로 이동
        self.login_page.username_input.clear()
        self.login_page.password_input.clear()

    def show_register_page(self):
        self.stacked_widget.setCurrentIndex(1)  #register_page 이동
        self.login_page.username_input.clear()
        self.login_page.password_input.clear()

    def show_admin_page(self):
        self.stacked_widget.setCurrentIndex(2)  #admin_page 이동
    
    
        





if __name__ =='__main__':
    app = QApplication(sys.argv)

    # database table 생성
    Base.metadata.create_all(engine)

    window =MainWindow()
    window.show()

    sys.exit(app.exec())