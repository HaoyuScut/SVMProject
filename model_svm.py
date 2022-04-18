import cv2
import numpy as np
from keras.datasets import mnist
import tensorflow as tf
from sklearn.metrics import recall_score,accuracy_score
from sklearn.metrics import precision_score,f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# from itertools import cycle
# import time
# start_time = time.time()
# from sklearn.metrics import roc_curve
# from sklearn.metrics import auc
# from scipy.interpolate import interp1d as interp
# from sklearn.preprocessing import label_binarize


def show_confusion_matrix(confusion, classes, x_rot=-60):
    """
    绘制混淆矩阵
    :param confusion:
    :param classes:
    :param x_rot:
    :param figsize:
    :param save:
    :return:
    """
    # if figsize is not None:
    #     plt.rcParams['figure.figsize'] = figsize

    plt.imshow(confusion, cmap=plt.cm.Oranges)
    indices = range(len(confusion))
    plt.xticks(indices, classes, rotation=x_rot, fontsize=10)
    plt.yticks(indices, classes, fontsize=10)
    plt.colorbar()
    plt.title("Confusion_Matrix")
    plt.xlabel('y_pred')
    plt.ylabel('y_true')

    # 显示数据
    for first_index in range(len(confusion)):
        for second_index in range(len(confusion[first_index])):
            plt.text(first_index, second_index, confusion[first_index][second_index])

    # if save:
    #     plt.savefig("./confusion_matrix.png")
    plt.show()

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



    # 创建svm模型
    svm = cv2.ml.SVM_create()
    # 设置类型为SVM_C_SVC代表分类
    svm.setType(cv2.ml.SVM_C_SVC)
    # 设置核函数
    svm.setKernel(cv2.ml.SVM_RBF)
    # 设置其它属性
    svm.setGamma(0.02)
    svm.setC(10.0)
    # svm.setDegree(3)
    # 设置迭代终止条件
    svm.setTermCriteria((cv2.TermCriteria_MAX_ITER, 400, 1e-3))
    # 训练
    svm.train(train_images, cv2.ml.ROW_SAMPLE, train_labels)
    svm.save('mnist_svm_nopca.xml')
    svm = cv2.ml.SVM_load('mnist_svm_nopca.xml')

    # 在测试数据上计算准确率
    # 进行模型准确率的测试 结果是一个元组 第一个值为数据1的结果
    test_pre = svm.predict(test_images)
    test_ret = test_pre[1]
    #print(test_ret.shape)

    # 转为列向量
    test_ret = test_ret.reshape(-1, )
    test_labels = test_labels.reshape(-1, )

    confusion = confusion_matrix(test_labels, test_ret, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    show_confusion_matrix(confusion, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    print(confusion)

    # #计算测试集中各数字个数
    # test_num = np.zeros(10)
    # for n in range(0, 10):
    #     test_num[n] = np.sum(test_labels == n)
    # print(test_num)
    # print(np.sum(test_num))

    # #计算测试集中各数字正确的个数
    # pre_num = np.zeros(10)
    # for n in range(0, 10):
    #     for i in range(0, len(test_ret)):
    #         if test_ret[i] == test_labels[i] and test_ret[i] == n:
    #             pre_num[n] += 1
    # print(pre_num)
    #
    # #显示每一个数字的识别精度
    # for n in range(0, 10):
    #     print('数字 %d 的测试结果展示：'%(n))
    #     print('测试样本个数为：%d'%(test_num[n]))
    #     print('正确样本个数为：%d' % (pre_num[n]))
    #     print('数字 %d 的识别准确率为：%f'%(n,float(pre_num[n]/test_num[n])))
    #     print('........................................')
    #
    # sum = np.sum(pre_num)
    # print('总测试结果展示：')
    # print('总体样本数为：10000')
    # print('正确样本个数为：%d' % (sum))
    # print('总体准确率为：',sum/100)


    test_sum = (test_ret == test_labels)
    print(test_sum)
    acc = test_sum.mean()
    print(acc)

    # nb_classes = 10
    # Binarize the output
    # test_labels= label_binarize(test_labels, classes=[i for i in range(nb_classes)])
    # test_ret = label_binarize(test_ret, classes=[i for i in range(nb_classes)])

    # precision = precision_score(test_labels,test_ret,average='micro')
    # recall = recall_score(test_labels,test_ret, average='micro')
    # f1_score = f1_score(test_labels,test_ret, average='micro')
    # accuracy_score = accuracy_score(test_labels,test_ret)
    # print("Precision_score:", precision)
    # print("Recall_score:", recall)
    # print("F1_score:", f1_score)
    # print("Accuracy_score:", accuracy_score)
    #





    # roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）
    # 横坐标：假正率（False Positive Rate , FPR）

    # Compute ROC curve and ROC area for each class
    # fpr = dict()
    # tpr = dict()
    # roc_auc = dict()
    # for i in range(nb_classes):
    #     fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_ret[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #
    # # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(test_labels.ravel(), test_ret.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    #
    # # Compute macro-average ROC curve and ROC area
    #
    # # First aggregate all false positive rates
    # all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    #
    # # Then interpolate all ROC curves at this points
    # mean_tpr = np.zeros_like(all_fpr)
    # for i in range(nb_classes):
    #     mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    #
    # # Finally average it and compute AUC
    # mean_tpr /= nb_classes
    #
    # fpr["macro"] = all_fpr
    # tpr["macro"] = mean_tpr
    # roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    #
    # # 绘制所有类别平均的roc曲线
    # lw = 2
    # plt.figure()
    # plt.plot(fpr["micro"], tpr["micro"],
    #          label='micro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["micro"]),
    #          color='deeppink', linestyle=':', linewidth=4)
    #
    # plt.plot(fpr["macro"], tpr["macro"],
    #          label='macro-average ROC curve (area = {0:0.2f})'
    #                ''.format(roc_auc["macro"]),
    #          color='navy', linestyle=':', linewidth=4)
    #
    # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    # for i, color in zip(range(nb_classes), colors):
    #     plt.plot(fpr[i], tpr[i], color=color, lw=lw,
    #              label='ROC curve of class {0} (area = {1:0.2f})'
    #                    ''.format(i, roc_auc[i]))
    #
    # plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Some extension of Receiver operating characteristic to multi-class')
    # plt.legend(loc="lower right")
    # plt.savefig("../images/ROC/ROC_5分类.png")
    # plt.show()
    #
    # print("--- %s seconds ---" % (time.time() - start_time))


