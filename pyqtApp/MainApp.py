import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication

class mainApp(QMainWindow):

    def __init__(self):
        super().__init__()
        uic.loadUi("mainUi.ui",self)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    GUI = mainApp()
    GUI.show()
    sys.exit(app.exec_())
