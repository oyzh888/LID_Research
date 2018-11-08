#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""Trains a ResNet on the CIFAR10 dataset.
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
首先对网络进行训练，在训练完成后，分析LID值在各层feature map中的变化。LID值使用全局Batch（而非Class）
Batch Size为5000.
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model,load_model
from keras.datasets import cifar10
import numpy as np
import math
import os,gc
import psutil
import sys
from sklearn.decomposition import PCA
import util
from util import GPU_lid_eval
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import sys
from pathlib import *

if(len(sys.argv)!=1):
    order = int(sys.argv[1])
    batch_size = order
else:
    batch_size = 128  # orig paper trained all networks with batch_size=128

# Training parameters
# batch_size = 128  # orig paper trained all networks with batch_size=128
epochs = 20
data_augmentation = False
num_classes = 10

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
# exp_name = 'BaseLine_resNet_Cifar10_BS%d_epochs%d' % (batch_size, epochs)
exp_name = 'BaseLine_resNet_Cifar10_BS%d_epochs%d_Shuffle_LID_Depth_Analysis' % (batch_size, epochs)

n = 3

# Model version
# Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
version = 1

# Computed depth from supplied model parameter n
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
(x_train, y_train), (x_test, y_test) = (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))
# part=5000
# x_train = x_train[:part]
# y_train = y_train[:part]
work_path=Path('../../Cifar10_Layer_LID')
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

def lr_schedule(epoch):
    #Learning Rate Schedule
    lr = 1e-3
    if epoch >= epochs * 0.9:
        lr *= 0.5e-3
    elif epoch >= epochs * 0.8:
        lr *= 1e-3
    elif epoch >= epochs * 0.6:
        lr *= 1e-2
    elif epoch >= epochs * 0.4:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 name = None):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            activation-bn-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
x`
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Define Model

#ResNet:
model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy', 'top_k_categorical_accuracy'])
# model.summary()
# print(model_type)
# Prepare model model saving directory.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
from keras.callbacks import TensorBoard
callbacks = [lr_reducer, lr_scheduler,TensorBoard(
    log_dir= (work_path/'TB_Log'/exp_name).__str__())]

x_train_epoch = []
y_train_epoch = []
def renew_train_dataset():
    mask = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
    global x_train_epoch
    x_train_epoch = x_train[mask]
    global y_train_epoch
    y_train_epoch = y_train[mask]

def on_epoch_end(epoch, logs):
    print('End of epoch')
    renew_train_dataset()

# on_epoch_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)
# renew_train_dataset()
# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=epochs,
    #           validation_data=(x_test, y_test),
    #           shuffle=True,
    #           callbacks=callbacks)
    model = load_model(work_path/'saved_models/preTrainedCIFAR10_augX10.h5')
    # model.summary()
# 创建保存各层输出的变量outputs，该变量类似于占位符，不实际运算
layers_names=[];outputs = [];outputs_predict_lid=[]
# 创建保存各层LID值的文件夹
Path(work_path/'Layer_LID_nparray').mkdir(parents=True, exist_ok=True)
# 计算各层LID
layerIdx=0
import time
start_time=time.time()
output_predict_lid = np.zeros(train_num)
for layer in model.layers:
    # 过滤掉特殊层
    # if layer.__class__.__name__ in {"Conv2D","InputLayer","Activation","Add"}:
    if layer.__class__.__name__ not in {"Conv2D"}:
        continue
    threshold = 5000;cur_layer_shape=1
    # import ipdb;ipdb.set_trace()
    for i in layer.output_shape:
        if i is None:
            continue;
        cur_layer_shape *= i
    if cur_layer_shape > threshold:
        print("layer ",layer.name," size ",cur_layer_shape," skip")
        continue;
    print("Calculating LID in ",layer.name)
    layers_names.append(layer.name)
    model_intermediate_layer = Model(inputs=model.input, outputs=layer.output)
    output_predict = model_intermediate_layer.predict(x_train, batch_size=batch_size)
    output_predict_flatten = np.reshape(output_predict,newshape=(train_num,-1)).astype('float32')
    # 创建实际保存个层数出的outputs_predict，并计算各层输出。
    batch_num = int(train_num/batch_size);lid_k = int(math.sqrt(batch_size));
    train_idx = np.arange(train_num);np.random.shuffle(train_idx)
    from progressbar import *
    pbar = ProgressBar()
    for i in pbar(range(batch_num)):
        mask_batch = []
        if ((i + 1) * batch_size < train_num):
            mask_batch = np.arange(i * batch_size, (i + 1) * batch_size)  # 一个样本下标仅出现一次,顺序训练
        else:
            mask_batch = np.arange(i * batch_size, train_num)
        import tensorflow as tf
        with tf.device('/gpu:0'):
            # print("LID Batch Shape",output_predict_flatten[train_idx[mask_batch]].shape)
            dis = GPU_lid_eval(output_predict_flatten[train_idx[mask_batch]], lid_k)
        output_predict_lid[train_idx[mask_batch]] = dis
    # import ipdb;ipdb.set_trace()
    lidPath = '%d_%s_BS%d_LID.npy'%(layerIdx,layer.name,batch_size)
    np.save(Path(work_path/'Layer_LID_nparray'/lidPath).__str__(), output_predict_lid)
    layerIdx += 1
    pid = os.getpid()
    info = psutil.Process(pid).memory_full_info()
    memory = info.uss / 1024. / 1024.
    print("Currrent Memory Usage",memory)
    # import ipdb;ipdb.set_trace()
    # del output_predict_lid
    # gc.collect()
    # outputs_predict_lid.append(output_predict_lid)
print("Calculating All Layers LID Spent ",(time.time()-start_time)/60,"min")
np.save(Path(work_path/'Layer_LID_nparray'/'layers_ordial_name').__str__(),layers_names)