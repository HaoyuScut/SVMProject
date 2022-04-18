from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel
from PyQt5.QtGui import (QPainter, QPen,QImage, QPixmap)
import cv2 as cv
from PyQt5.QtCore import Qt

class MyLabel(QLabel):
    pos_xy = []

    def mouseReleaseEvent(self, event):
        pos_test = (-1, -1)
        self.pos_xy.append(pos_test)
        # 鼠标移动事件

    def mouseMoveEvent(self, event):
        '''
                    按住鼠标移动事件：将当前点添加到pos_xy列表中
                    调用update()函数在这里相当于调用paintEvent()函数
                    每次update()时，之前调用的paintEvent()留下的痕迹都会清空
                '''
        # 中间变量pos_tmp提取当前点
        pos_tmp = (event.pos().x(), event.pos().y())
        # pos_tmp添加到self.pos_xy中
        self.pos_xy.append(pos_tmp)
        self.update()
        # 绘制事件

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.begin(self)

        pen = QPen(Qt.black, 20, Qt.SolidLine)
        painter.setPen(pen)

        if len(self.pos_xy) > 1:
            point_start = self.pos_xy[0]
            for pos_tmp in self.pos_xy:
                point_end = pos_tmp

                if point_end == (-1, -1):
                    point_start = (-1, -1)
                    continue
                if point_start == (-1, -1):
                    point_start = point_end
                    continue

                painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])

                point_start = point_end
        painter.end()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(900, 750)
        MainWindow.setMinimumSize(QtCore.QSize(900, 750))
        MainWindow.setMaximumSize(QtCore.QSize(900, 750))
        MainWindow.setStyleSheet("")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setEnabled(True)
        self.centralwidget.setMouseTracking(False)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_tittle = QtWidgets.QLabel(self.centralwidget)
        self.label_tittle.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label_tittle.setFont(font)
        self.label_tittle.setTextFormat(QtCore.Qt.AutoText)
        self.label_tittle.setAlignment(QtCore.Qt.AlignCenter)
        self.label_tittle.setObjectName("label_tittle")
        self.verticalLayout.addWidget(self.label_tittle)
        spacerItem = QtWidgets.QSpacerItem(877, 35, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setEnabled(True)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        spacerItem1 = QtWidgets.QSpacerItem(877, 13, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)

        self.label_draw = MyLabel(self)
        self.label_draw.setEnabled(True)
        self.label_draw.setMinimumSize(QtCore.QSize(400, 400))
        self.label_draw.setMaximumSize(QtCore.QSize(400, 400))
        self.label_draw.setMouseTracking(False)
        self.label_draw.setFrameShape(QtWidgets.QFrame.Box)
        self.label_draw.setFrameShadow(QtWidgets.QFrame.Plain)
        self.label_draw.setLineWidth(2)
        self.label_draw.setText("")
        self.label_draw.setAlignment(QtCore.Qt.AlignCenter)
        self.label_draw.setObjectName("label_draw")
        img = cv.imread('E:/SVM_Project/svmProject/white.jpg')
        height, width, bytesPerComponent = img.shape
        bytesPerLine = 3 * width
        cv.cvtColor(img, cv.COLOR_BGR2GRAY, img)
        QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(QImg)

        self.label_draw.setPixmap(pixmap)
        self.label_draw.setCursor(Qt.CrossCursor)
        self.show()

        self.horizontalLayout_2.addWidget(self.label_draw)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem3)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.verticalLayout.addWidget(self.label_3)
        spacerItem4 = QtWidgets.QSpacerItem(877, 13, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem4)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.lineEdit_result = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_result.setEnabled(True)
        self.lineEdit_result.setObjectName("lineEdit_result")
        self.horizontalLayout.addWidget(self.lineEdit_result)
        spacerItem5 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem5)
        self.btn_recognize = QtWidgets.QPushButton(self.centralwidget)
        self.btn_recognize.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_recognize.setFont(font)
        self.btn_recognize.setObjectName("btn_recognize")
        self.horizontalLayout.addWidget(self.btn_recognize)
        self.btn_clear = QtWidgets.QPushButton(self.centralwidget)
        self.btn_clear.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_clear.setFont(font)
        self.btn_clear.setObjectName("btn_clear")
        self.horizontalLayout.addWidget(self.btn_clear)
        self.btn_close = QtWidgets.QPushButton(self.centralwidget)
        self.btn_close.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.btn_close.setFont(font)
        self.btn_close.setObjectName("btn_close")
        self.horizontalLayout.addWidget(self.btn_close)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.btn_close.clicked.connect(MainWindow.close)
        self.btn_recognize.clicked.connect(MainWindow.btn_recognize_on_clicked)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_tittle.setText(_translate("MainWindow", "SVM手写数字识别"))
        self.label_2.setText(_translate("MainWindow", "作者：王浩宇  学号：202121018517"))
        self.label_3.setText(_translate("MainWindow", "用户手写板"))
        self.label.setText(_translate("MainWindow", "识别结果："))
        self.btn_recognize.setText(_translate("MainWindow", "识别"))
        self.btn_clear.setText(_translate("MainWindow", "清空"))
        self.btn_close.setText(_translate("MainWindow", "关闭"))
