import sys
import time
from PyQt5.QtWidgets import QDesktopWidget, QMainWindow, QApplication, QWidget, QAction, QTableWidget,QTableWidgetItem,QVBoxLayout
from PyQt5.QtGui import QIcon, QImage, QPalette, QBrush, QColor, QFont
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtCore import QSize
from PyQt5 import QtCore
import numpy as np
import pandas as pd

# df = pd.read_csv('score_list.txt')
# df=df.sort_values('score', ascending=False)
# df.reset_index(drop=True, inplace=True)
cs=0
class App(QWidget):

	def __init__(self,curr_score):
		super().__init__()        
		self.title = 'Leader Board'
		self.left = 0
		self.top = 0
		self.width = 1080 #
		self.height = 1920
		global cs
		cs= curr_score
		print(cs)
		#self.setGridStyle(QtCore.Qt.NoPen)
		#self.setStyleSheet("background-color: black")
		#self.setStyleSheet("border-image: url(nyoga-min.png)")
		self.setStyleSheet("background-color: rgb(20, 39, 42);")
		#self.setStyleSheet('font-size: 18pt; font-family: Courier;')
		#palette = QPalette()
		#palette.setBrush(QPalette.Window, QBrush(sImage))                        
		#self.setPalette(palette)
		self.initUI()
		
	def initUI(self):
		self.setWindowTitle(self.title)
		self.setGeometry(self.left, self.top, self.width, self.height)
		
		self.createTable()

		# Add box layout, add table to box layout and add box layout to widget
		self.layout = QVBoxLayout()
		self.layout.addWidget(self.tableWidget) 
		self.setLayout(self.layout) 

		# Show widget
		#self.move(1140,1460)
		self.move(1640,1640)
		self.show()
		self.close()

	def createTable(self):
		score_list=[1400,1255,1125,705,620,565,450,320,205]
		name_list=['Anshiqa ','Johnson ','Sean ','Cawin ','Cindy ','Sahad ','Lok Hen ','Yan Lin ','Arshad ']
		self.tableWidget = QTableWidget()
		self.tableWidget.setRowCount(10)
		#display_monitor = 0
		#monitor = .screenGeometry(display_monitor)
		#self.tableWidget.move(monitor.right(), monitor.top())
		self.tableWidget.setColumnCount(2)
		self.tableWidget.setColumnWidth(0,900)
		self.tableWidget.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
		self.tableWidget.setStyleSheet("font: 80pt Garamond")
		#self.tableWidget.setStyleSheet("background-image: url(nyoga.png)")
		#header = QTableWidgetItem("Leaderboard").setTextAlignment(Qt.AlignHCenter)
		#self.tableWidget.setItem(0,0, header)
		filled=0
		count=0

		while(count<9):
			
			if (cs==max(cs,score_list[count]) ):
				score_list.insert(count,cs)
				name_list.insert(count,'You')
				break
				# item1=QTableWidgetItem("You")
				# item1.setForeground(QBrush(QColor(255,0,0)))
				# item2=QTableWidgetItem(str(cs))
				# item2.setForeground(QBrush(QColor(255,0,0)))
				# self.tableWidget.setItem(count,0, item1)
				# self.tableWidget.setItem(count,1, item2)
				# item3=QTableWidgetItem(str(count+1)+".")
				# item3.setForeground(QBrush(QColor(255, 255, 255)))
			# else if(count):
			# 	item1=QTableWidgetItem(name_list[count])
			# 	item1.setForeground(QBrush(QColor(255,255,255)))
			# 	item2=QTableWidgetItem(str(score_list[count]))
			# 	item2.setForeground(QBrush(QColor(255,255,255)))
			# 	self.tableWidget.setItem(count,0, item1)
			# 	self.tableWidget.setItem(count,1, item2)
			# 	item3=QTableWidgetItem(str(count+1)+".")
			# 	item3.setForeground(QBrush(QColor(255, 255, 255)))
			# 	self.tableWidget.setVerticalHeaderItem(count, item3)
			count +=1
			if(count==9): 
				score_list.insert(count,cs)
				name_list.insert(count,'You')
		for i in range(10):
			if(i==count):
				item1=QTableWidgetItem(str(name_list[i]))
				item2=QTableWidgetItem(str(score_list[i]))
				item3=QTableWidgetItem(str(i+1)+".")
				item1.setForeground(QBrush(QColor(255, 0, 0)))
				item2.setForeground(QBrush(QColor(255, 0, 0)))
				item3.setForeground(QBrush(QColor(255, 0, 0)))
				self.tableWidget.setItem(i,0, item1)
				self.tableWidget.setItem(i,1, item2)
				self.tableWidget.setVerticalHeaderItem(i, item3)
				#self.tableWidget.setItem(i+1,0, item3)

			else:
				item1=QTableWidgetItem(str(name_list[i]))
				item2=QTableWidgetItem(str(score_list[i]))
				item3=QTableWidgetItem(str(i+1)+".")
				item1.setForeground(QBrush(QColor(255, 255, 255)))
				item2.setForeground(QBrush(QColor(255, 255, 255)))
				item3.setForeground(QBrush(QColor(255, 255, 255)))
				self.tableWidget.setItem(i,0, item1)
				self.tableWidget.setItem(i,1, item2)
				self.tableWidget.setVerticalHeaderItem(i,item3)
		#		self.tableWidget.setItem(i+1,0, item3)
			

		self.tableWidget.setGridStyle(QtCore.Qt.NoPen)
	#	self.tableWidget.setHorizontalHeaderLabels(['Leader','board'])
	#	self.tableWidget.horizontalHeader().setSectionResizeMode(0, QheaderView.Stretch)
		self.tableWidget.horizontalHeader().hide()
	#	self.tableWidget.verticalHeader().hide()
		self.tableWidget.move(0,0)
		self.tableWidget.resizeRowsToContents()
		self.tableWidget.resizeColumnsToContents()
		#self.tableWidget.setStyleSheet("background-color: rgba(0, 0, 0, 50);")
		


		# table selection change
		#self.tableWidget.doubleClicked.connect(self.on_click)

	@pyqtSlot()
	def on_click(self):
		print("\n")
		for currentQTableWidgetItem in self.tableWidget.selectedItems():
			print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())
 
if __name__ == '__main__':
	app = QApplication(sys.argv)
	ex = App()
	sys.exit(app.exec_())  
