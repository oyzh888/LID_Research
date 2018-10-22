#!/usr/bin/env python
# coding: utf-8

"""Trains a ResNet on the CIFAR10 dataset.
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf

本程序调用已经训练好的cifar10网络(在文件夹saved_models中，可以手动更改)
生成较大batch下的全局LID分布，网络的输入是未经data augmentation的Cifar10数据集。
"""

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import keras
import math
import sys
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.datasets import cifar10
from keras.datasets import mnist
import numpy as np
import gc
import os
from sklearn.decomposition import PCA
import lid
from lid import LID
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
# plt.imshow
# Training parameters
batch_size = 5000  # orig paper trained all networks with batch_size=128x
num_classes = 10
# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True

# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# part=5000
# x_train=x_train[:part]
# y_train=y_train[:part]
# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

train_num,test_num = x_train.shape[0],x_test.shape[0]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)
# saved_models
model=load_model('./saved_models/preTrainedCIFAR10_augX10.h5')
# outputs = [layer.output for layer in model.layers]
outputs = []
layers_names=['average_pooling2d_1']
for layer in layers_names:
    outputs.append(model.get_layer(layer).output)
# print(outputs)

model_intermediate_layer = Model(inputs=model.input, outputs=outputs)
outputs_predict = model_intermediate_layer.predict(x_train, batch_size=1024)

import time
start_time=time.time()
outputs_predict_flatten = np.reshape(outputs_predict,newshape=(x_train.shape[0],-1))
outputs_predict_flatten = outputs_predict_flatten.astype('float32')
outputs_predict_lid=np.zeros(x_train.shape[0])
lid_k=int(math.sqrt(batch_size))
batch_num=int(train_num/batch_size)

train_idx=np.arange(train_num)
np.random.shuffle(train_idx)
from progressbar import *
pbar = ProgressBar()
for i in pbar(range(batch_num)):
    mask_batch=[]
    if((i+1)*batch_size<train_num):
        mask_batch = np.arange(i*batch_size,(i+1)*batch_size)  # 一个样本下标仅出现一次,顺序训练
    else:
        mask_batch = np.arange(i*batch_size,train_num)
    # import ipdb;ipdb.set_trace()
    dis = LID(outputs_predict_flatten[train_idx[mask_batch]], outputs_predict_flatten[train_idx[mask_batch]], lid_k)
    outputs_predict_lid[train_idx[mask_batch]] = dis

print("outputs_predic_lid time ",time.time()-start_time)
print("OUTPUT_PREDICT_LID FINISH!")
test_time = 3
exp_name='Cifar10_ground_truth_dataset%d_BS%d_lid_K%d_TEST%d'%(x_train.shape[0],batch_size,lid_k,test_time)


save_dir = os.path.join(os.getcwd(), 'picture')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
plt.xlabel('sample idx')
plt.ylabel('lid')
plt.title(exp_name+' bar plot')
plt.bar(range(len(outputs_predict_lid)),outputs_predict_lid)
plt.savefig('./picture/'+exp_name+'_bar_plot_average_pooling2d_1.jpg')  # bar图片的存储
plt.close()

print("SUCCESSFULLY SAVE FIG BAR PLOT!")

plt.title(exp_name+' box plot')
plt.boxplot(outputs_predict_lid)
plt.savefig('./picture/'+exp_name+'_box_plot_average_pooling2d_1.jpg')  # boxplot图片的存储
plt.close()

print("SUCCESSFULLY SAVE BOX PLOT! ")
# import ipdb; ipdb.set_trace()
# print(np.shape(outputs_predict_flatten))
save_dir = os.path.join(os.getcwd(), 'nparray')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
np.save('./nparray/'+exp_name+'_Lid.npy',outputs_predict_lid)

print("SUCESSFULLY SAVE LID ARRAY!")