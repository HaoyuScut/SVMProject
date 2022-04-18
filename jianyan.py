import numpy as np
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], -1)  # (60000, 784)
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape(test_images.shape[0], -1)
test_images = test_images.astype('float32') / 255

# 将标签数据转为int32 并且形状为(60000,1)
train_labels = train_labels.astype(np.int32)
test_labels = test_labels.astype(np.int32)
train_labels = train_labels.reshape(-1, 1)
test_labels = test_labels.reshape(-1, 1) #(10000,1)


test_num = [0,0,0,0,0,0,0,0,0,0]
for n in range(0,10):
    test_num[n] = np.sum(test_labels == n)
print(test_num)
print(np.sum(test_num))