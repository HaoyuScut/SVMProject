import cv2
import numpy as np
from keras.datasets import mnist
from keras import utils
from sklearn.decomposition import PCA

if __name__ == '__main__':
    # 直接使用Keras载入的训练数据(60000, 28, 28) (60000,)
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

    # 创建svm模型
    svm = cv2.ml.SVM_create()
    # 设置类型为SVM_C_SVC代表分类
    svm.setType(cv2.ml.SVM_C_SVC)
    # 设置核函数
    svm.setKernel(cv2.ml.SVM_RBF)
    # 设置其它属性
    svm.setGamma(0.02)
    svm.setC(1.0)
    # svm.setDegree(3)
    # 设置迭代终止条件
    svm.setTermCriteria((cv2.TermCriteria_MAX_ITER, 400, 1e-3))
    # 训练
    svm.train(x_reduction, cv2.ml.ROW_SAMPLE, train_labels)
    svm.save('mnist_svm_pac.xml')

    # 在测试数据上计算准确率
    # 进行模型准确率的测试 结果是一个元组 第一个值为数据1的结果
    test_pre = svm.predict(x_test_reduction)
    test_ret = test_pre[1]

    # 计算准确率
    test_ret = test_ret.reshape(-1, )
    test_labels = test_labels.reshape(-1, )
    test_sum = (test_ret == test_labels)
    acc = test_sum.mean()
    print(acc)
