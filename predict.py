import cv2 as cv
import numpy as np
import glob
import os
from skimage import measure

def pre_img(image = '2.jpg'):
    # cv.imshow('1', image)
    # cv.waitKey(0)
    ret, img1 = cv.threshold(image, 100, 255, cv.THRESH_BINARY_INV)
    # print(type(img1))
    # print(img1.shape)
    # print(img1.size)
    img_fan = cv.bitwise_not(img1)

    cv.imshow('fan', img_fan)

    cv.waitKey(2)
    kernel1 = np.ones((3, 3), np.uint8)  # 做一次膨胀
    img2 = cv.dilate(img1, kernel1)
    cv.imshow("2",img2)
    cv.waitKey(0)

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
    kernel2 = np.ones((15, 15), np.uint8)
    img3 = cv.dilate(img2, kernel2)
    # cv.namedWindow('2',0)
    # cv.resizeWindow('2',600,600)
    # cv.imshow('2',img3)
    # cv.waitKey(0)

    'roi提取'
    contours, hierarchy = cv.findContours(img3, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv.boundingRect(contours[0])
    a = 100
    brcnt = np.array([[[x - a, y - a]], [[x + w + a, y - a]], [[x + w + a, y + h + a]], [[x - a, y + h + a]]])
    cv.namedWindow('result', 0)
    cv.drawContours(img3, [brcnt], -1, (255, 255, 255), 2)
    cv.imshow('result', img3)
    cv.waitKey(1)
    img4 = img3[y - a:y + h + a, x - a:x + w + a]  # img4就是提取roi后的图片
    cv.imshow('2', img4)
    cv.waitKey(1)
    img5 = cv.resize(img4, (28, 28))

    return img5




if __name__=='__main__':
    #读取图片
    img=cv.imread('4.png',0) #这里就是读取预处理好的图片了，当然你也可以把这个程序直接放在第二个程序后面，就不需要这一步了
    img_pre = pre_img(img)
    img_sw=img_pre.copy()

    #将数据类型由uint8转为float32
    img=img_pre.astype(np.float32)
    #图片形状由(28,28)转为(784,)
    img=img.reshape(-1,)
    #增加一个维度变为(1,784)
    img=img.reshape(1,-1)
    #图片数据归一化
    img=img/255

    #载入svm模型
    svm=cv.ml.SVM_load('mnist_svm.xml')
    #进行预测
    img_pre=svm.predict(img)
    print(int(img_pre[1]))

    cv.imshow('test',img_sw)
    cv.waitKey(0)

