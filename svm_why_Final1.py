from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QLabel,QFileDialog
from PyQt5.QtGui import (QPainter, QPen,QImage, QPixmap)
from PyQt5.QtCore import Qt,QRect
from PIL import ImageGrab, Image
import cv2 as cv
import numpy as np
from sklearn.decomposition import PCA
from keras.datasets import mnist
import matplotlib.pyplot as plt

# PCA前置
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 变换数据的形状并归一化
train_images = train_images.reshape(train_images.shape[0], -1)  # (60000, 784)
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape(test_images.shape[0], -1)
test_images = test_images.astype('float32') / 255

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
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setEnabled(True)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")

        self.verticalLayout.addWidget(self.label_2)
        spacerItem1 = QtWidgets.QSpacerItem(507, 13, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)

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

        self.horizontalLayout_3.addWidget(self.label_draw)


        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem3)
        self.verticalLayout.addLayout(self.horizontalLayout_3)

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

        spacerItem4 = QtWidgets.QSpacerItem(14, 13, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
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
        self.btn_clear.clicked.connect(MainWindow.btn_clear_on_clicked)
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

    def btn_clear_on_clicked(self):
        self.label_draw.pos_xy = []
        self.label_draw.setText('')
        self.update()

    def btn_recognize_on_clicked(self):
        # bbox = (240, 100, 500, 500)
        # im = ImageGrab.grab(bbox)  # 截屏，手写数字部分
        image_old = self.label_draw.grab()
        # image.save('6.jpg',"JPG")
        #image = self.label_draw.pixmap().toImage()
        image = image_old.toImage()
        # fdir, ftype = QFileDialog.getSaveFileName(self, "Save Image",
        #                                           "./", "Image Files (*.jpg)")
        # image.save(fdir)

        size = image.size()
        s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)  # format 0xffRRGGBB

        arr = np.fromstring(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))

        new_image = Image.fromarray(arr)
        # new_image.save('6.jpeg',"JPEG")


        # convert to gray
        new_image.convert("L")
        new_image.thumbnail((300, 300))
        # plt.imshow(new_image, cmap='gray')
        # plt.show()
        # new_image = new_image.resize((28, 28), Image.ANTIALIAS)  # 将截图转换成 28 * 28 像素
        #
        recognize_result = self.recognize_img(new_image)  # 调用识别函数

        self.lineEdit_result.setText(str(recognize_result))  # 显示识别结果
        self.update()



    def recognize_img(self, img):

        def pre_img(image):

            myimage = image.convert('L')  # 转换成灰度图
            myimage = np.array(myimage)

            # myimage = cv.bitwise_not(myimage)

            # cv.imshow('fan', myimage)

            ret, img1 = cv.threshold(myimage, 100, 255, cv.THRESH_BINARY_INV)

            # cv.namedWindow('img1',0)
            # cv.resizeWindow('img1',600,600)
            # cv.imshow('img1',img1)

            print(type(img1))
            print(img1.shape)
            print(img1.size)

            # cv.waitKey(2)
            kernel1 = np.ones((1, 1), np.uint8)  # 做一次膨胀
            img2 = cv.dilate(img1, kernel1)
            # cv.namedWindow('img2', 0)
            # cv.resizeWindow('img2', 600, 600)
            # cv.imshow('img2', img2)

            '剔除小连通域'
            contours, hierarchy = cv.findContours(img2, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
            # print(len(contours),hierarchy)
            for i in range(len(contours)):
                area = cv.contourArea(contours[i])
                if area < 200:  # '设定连通域最小阈值，小于该值被清理'
                    cv.drawContours(img2, [contours[i]], 0, 0, -1)
            # cv.namedWindow('2',0)
            # cv.resizeWindow('2',600,600)
            # cv.imshow('2',img2)
            # cv.waitKey(0)
            kernel2 = np.ones((3, 3), np.uint8)
            img3 = cv.dilate(img2, kernel2)
            # cv.namedWindow('img3',0)
            # cv.resizeWindow('img3',600,600)
            # cv.imshow('img3',img3)
            # cv.waitKey(0)


            img5 = cv.resize(img3, (28, 28))
            # cv.namedWindow('img5', 0)
            # cv.resizeWindow('img5', 600, 600)
            # cv.imshow('img5', img5)


            return img5

        img_pre = pre_img(img)
        # cv.imshow('img_pre', img_pre)
        # img_sw = img_pre.copy()

        # 将数据类型由uint8转为float32
        img = img_pre.astype(np.float32)
        # 图片形状由(28,28)转为(784,)
        img = img.reshape(-1, )
        # 增加一个维度变为(1,784)
        img = img.reshape(1, -1)

        pca = PCA(0.9)
        pca.fit(train_images)


        img1 = pca.transform(img)

        # print(pca.explained_variance_ratio_)
        print(img1.shape)
        #可尝试可视化

        # # 图片数据归一化
        # img1 = img1 / 255

        # 载入svm模型
        svm = cv.ml.SVM_load('E:\SVM_Project\svmProject\mnist_svm2.xml')
        # 进行预测
        img_pre = svm.predict(img1)
        print(img_pre[1])

        return int(img_pre[1])
