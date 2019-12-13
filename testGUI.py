from scoreGUI import App
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
from PyQt5.QtGui import QIcon, QImage, QPalette, QBrush,QColor,QFont
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import QSize
from PyQt5 import QtCore
import numpy as np
import pandas as pd

score=224
app = QApplication(sys.argv)
ex = App(curr_score=score)
ex.show()
sys.exit(app.exec_())
