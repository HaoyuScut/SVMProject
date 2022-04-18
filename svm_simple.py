import numpy as np
from sklearn import svm
from sklearn.datasets  import load_digits
from sklearn.model_selection  import train_test_split
import _pickle as pickle
from sklearn.decomposition import PCA
from keras.datasets import mnist

# mnist = load_digits()
# train_images,test_images,train_labels,test_labels = train_test_split(mnist.data,mnist.target,test_size=0.25,random_state=40)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 变换数据的形状并归一化
train_images = train_images.reshape(train_images.shape[0], -1)  # (60000, 784)
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape(test_images.shape[0], -1)
test_images = test_images.astype('float32') / 255

# 将标签数据转为int32 并且形状为(60000,1)
train_labels = train_labels.astype(np.int32)
test_labels = test_labels.astype(np.int32)
train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1)

pca = PCA(0.9)
pca.fit(train_images)
x_reduction = pca.transform(train_images)
print(x_reduction.shape)

x_test_reduction = pca.transform(test_images)
print(x_test_reduction.shape)

model = svm.SVC(kernel = "rbf")
model.fit(x_reduction, train_labels)
z = model.predict(x_test_reduction)
print('准确率:',np.sum(z==test_labels)/z.size)

with open('./model.pkl','wb') as file:
    pickle.dump(model,file)



#
# # -*- coding:utf-8 -*-
# import sys
# from sklearn.datasets import load_digits  # 加载手写数字识别数据
# import pylab as pl
# from sklearn.model_selection import train_test_split  # 训练测试数据分割
# from sklearn.preprocessing import StandardScaler  # 标准化工具
# from sklearn.svm import LinearSVC
# from sklearn.metrics import classification_report  # 预测结果分析工具
#
# digits = load_digits()
# # 数据维度，1797幅图，8*8
# print(digits.data.shape)
# # 长度为64的一维向量
# print(digits.data[0])
# # 分割数据
# X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=33)
#
# print(X_train.shape)  # (1347,64)
# print(Y_test.shape)  # (450,)
# print(Y_test)
#
# ss = StandardScaler()
# # fit是实例方法，必须由实例调用
# X_train = ss.fit_transform(X_train)
# X_test = ss.transform(X_test)
#
# lsvc = LinearSVC()
# lsvc.fit(X_train, Y_train)
#
# Y_predict = lsvc.predict(X_test)
#
# print(classification_report(Y_test, Y_predict, target_names=digits.target_names.astype(str)))
