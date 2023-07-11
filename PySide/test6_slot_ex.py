import sys
import csv
from PySide6.QtWidgets import QApplication, QPushButton, QVBoxLayout,QMainWindow
from PySide6.QtWidgets import QWidget, QMessageBox, QLabel, QLineEdit, QDialog, QListWidget

class Slot_input(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Signal slot example')
        self.resize(500,300)

        self.label_age = QLabel('Age: ')
        self.input_age = QLineEdit()

        self.label_sex = QLabel('Sex: ')
        self.input_sex = QLineEdit()

        self.label_nation = QLabel('Nation: ')
        self.input_nation = QLineEdit()

        self.button = QPushButton('Enter')
        self.button.clicked.connect(self.save)

        layout = QVBoxLayout()
        layout.addWidget(self.label_age)
        layout.addWidget(self.input_age)
        layout.addWidget(self.label_sex)
        layout.addWidget(self.input_sex)
        layout.addWidget(self.label_nation)
        layout.addWidget(self.input_nation)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def save(self):
        age = self.input_age.text()
        sex = self.input_sex.text()
        nation = self.input_nation.text()
        
        infor_window = InfoWindow(age, sex, nation)
        infor_window.setModal(True)
        infor_window.exec()

class InfoWindow(QDialog):
    def __init__(self, age, sex, nation):
        super().__init__()
        self.setWindowTitle('Check info')
        self.resize(500,300)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"AGE: {age}"))
        layout.addWidget(QLabel(f"SEX: {sex}"))
        layout.addWidget(QLabel(f"Nation: {nation}"))

        save_button = QPushButton('SAVE')
        close_button = QPushButton('CLOSE')
        load_button = QPushButton('LOAD')

        layout.addWidget(save_button)
        layout.addWidget(close_button)
        layout.addWidget(load_button)

        self.setLayout(layout)

        save_button.clicked.connect(lambda: self.save(age, sex, nation))
        close_button.clicked.connect(self.close)
        load_button.clicked.connect(self.load)


    def save(self, age, sex, nation):
        data = [generate_id(), age, sex, nation]
        try:
            with open('info.csv','a', newline='') as csvfile:
                wirter = csv.writer(csvfile)
                wirter.writerow(data)

            QMessageBox.information(self, 'success','Complete Saving!')
        except Exception as e:
            QMessageBox.information(self,'fail','Error Enterrupted')

    def load(self):
        try:
            with open('info.csv','r') as csvfile:
                reader = csv.reader(csvfile)
                lines = [line for line in reader]

            if len(lines) >0:
                list_window = ListWindow(lines)
                list_window.exec()
            else:
                QMessageBox.information(self, 'laod',' No saved information')
        except Exception as e:
            QMessageBox.critical(self,'fail','Error uploading info')
       
class ListWindow(QDialog):
    def __init__(self, lines):
        super().__init__()
        self.setWindowTitle('Saved info')
        self.resize(300,400)

        list_widget = QListWidget()
        for line in lines:
            item = f"ID: {line[0]}, Age:{line[1]}, Sex:{line[2]}, Nation:{line[3]}"
            list_widget.addItem(item)

        layout = QVBoxLayout()
        layout.addWidget(list_widget)

        self.setLayout()


# ID생성 로직, 중복 방지를 위한 unique id
def generate_id():
    import time
    return str(int(time.time()))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Slot_input()
    window.show()
    app.exit(app.exec())