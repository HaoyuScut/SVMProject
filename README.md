# SVMProject
SVM手写数字识别，MINST
手写数字识别系统的设计主要分为分类器模型搭建和训练、可视化界面设计两部分的内容。实现了用户自行手写数字，在界面进行识别、显示结果，并可一键清空手写板，继续进行数字的书写和识别的功能。
本设计采用了基于OpenCV的SVM方法进行手写数字识别。在Pycharm平台，使用Python语言，采用了OpenCV库、Numpy库、TensorFlow库进行SVM模型的搭建和训练。
分类器模型的搭建和训练流程如下图所示：
![image](https://user-images.githubusercontent.com/87626531/163778740-2b5224b7-d937-471c-af57-b315c0e1b51b.png)
 经过测试，各类数字的预测准确率如下图所示。
 ![image](https://user-images.githubusercontent.com/87626531/163779021-8cac894f-09a7-4a80-a883-743462bae3b9.png)
由上图可见，模型对数字0和1的预测效果最好，对数字8和9的预测效果相对较差，不过预测准确率都在97.5%以上，效果比较理想，满足手写数字识别系统的设计要求。
总体的测试结果如下图所示，可见总体的准确率、精确率、召回率、F1为98.54%
![image](https://user-images.githubusercontent.com/87626531/163779081-601fae88-b827-4db9-a842-c24478ebd360.png)

该模型预测的混淆矩阵如下图所示：
![image](https://user-images.githubusercontent.com/87626531/163779137-d6338cd6-4125-47c4-9b6e-87b8a4e2f61c.png)

由混淆矩阵可见，在预测时，数字0、1、3、5、6表现较好，数字2易和数字7混淆，数字4易和数字9混淆，数字7易和数字9、数字2混淆，数字8易和数字3、数字5混淆，数字9易和数字4、数字7混淆。

可视化界面：使用pyqt5搭建
![image](https://user-images.githubusercontent.com/87626531/163779200-9333a62f-fc67-472c-a4bf-6ffa29ad61d9.png)
