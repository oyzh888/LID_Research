from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, ReduceLROnPlateau,LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import torch
import time
import math
import os
from sklearn.decomposition import PCA
import lid
from lid import LID, low_speed_LID
import util
from util import GPU_lid_eval
from progressbar import *
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

# coding: utf-8


"""Trains a ResNet on the CIFAR10 dataset.

ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf

ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""


# Training parameters
batch_size = 64  # orig paper trained all networks with batch_size=128
epochs = 20
data_augmentation = False
num_classes = 10
exp_name = 'resNet_Cifar10_BS%d_epochs%d_' % (batch_size, epochs)
sample_name = "Bottom10PercentX0Weight_using_batch_dropout"
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

# Model name, depth and version
model_type = 'ResNet%dv%d' % (depth, version)

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# val_num = 5000
# x_val = x_train[:val_num]
# y_val = y_train[:val_num]
# x_train = x_train[val_num:]
# y_train = y_train[val_num:]


# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean


train_num,test_num = x_train.shape[0],x_test.shape[0]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
# print('x_val shape:', x_val.shape)
# print(x_val.shape[0], 'x_val samples')
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

# base_lr = 1e-6
# max_lr = 1e-3
# step_size = 5

# def lr_schedule_cycle(iterations):
#     cycle = np.floor(1+iterations/(2*step_size))
#     x = np.abs(iterations/step_size - 2*cycle + 1)
#     lr = base_lr + (max_lr-base_lr)*np.maximum(0, (1-x))/float(2**(cycle-1))
#     return lr

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



# Train PCA & Find LID

# 以validation各个类别作为pca分析的输入，将训练集中各个类别依照该pca进行降维，使得每个类别的降维方式一致。
# 每个类别降维之后，在类别内部根据LID值寻找异常点，将异常点进行记录。
cls_num = 10
# outlier_mask=np.zeros(train_num,dtype=int)
new_x_train = []
new_y_train = []

# Class Dropout
# for cls in range(cls_num):
#     print('Start class', cls)
#     selected_x_train = x_train[y_train.flatten()==cls]
#     selected_y_train = y_train[y_train.flatten()==cls]
#     x_train_temp = selected_x_train.reshape(selected_x_train.shape[0],-1)
#     k = int(batch_size/10)
#     for i in range(int(selected_x_train.shape[0]/batch_size)):
#         train_temp_batch_mask = np.random.choice(selected_x_train.shape[0],batch_size)
#         selected_x_train_batch = selected_x_train[train_temp_batch_mask]
#         selected_y_train_batch = selected_y_train[train_temp_batch_mask]
#         x_train_temp_batch = x_train_temp[train_temp_batch_mask]
#         y_train_temp_batch = selected_y_train[train_temp_batch_mask]
#         dis = LID(x_train_temp_batch, x_train_temp_batch, k)  # 也可以和validation计算
#         dis_idx = np.argwhere(dis < np.percentile(dis,99)).flatten()    # 将距离最far的1%的数据剔除掉
#         new_x_train.extend(selected_x_train_batch[dis_idx])
#         new_y_train.extend(selected_y_train_batch[dis_idx])

# Batch Dropout. PyTorch Version
# num_batch = int(train_num/batch_size)
# import math
# k = int(math.sqrt(batch_size))
# import time;start_time = time.time()
# x_train_pytorch = torch.from_numpy(x_train)
# train_idx = [];x_train_lid=np.ones(train_num);
# for i in range(num_batch):
#     train_temp_batch_mask = np.random.choice(train_num,batch_size)
#     # x_train_batch = x_train[train_temp_batch_mask]
#     # y_train_batch = y_train[train_temp_batch_mask]
#
#     x_train_batch = x_train_pytorch[train_temp_batch_mask]
#
#     dis = LID(x_train_batch, x_train_batch, k)  # 也可以和validation计算
#     dis_idx = np.argwhere(dis < np.percentile(dis,90)).flatten()    # 将距离最far的10%的数据剔除掉
#     x_train_lid[train_temp_batch_mask[dis_idx]] = dis;
#     # import ipdb;ipdb.set_trace()
#     train_idx.extend(train_temp_batch_mask[dis_idx])
# print("PyTorch LID time:",time.time()-start_time)

# x_train = x_train[train_idx]
# y_train = y_train[train_idx]

# np.save("../LID_Research_local/nparray/Cifar10_x_train_BS64_lid.npy",x_train_lid)
# np.save("../LID_Research_local/nparray/Cifar10_x_train_BS64_lid.npy",x_train)
# np.save("../LID_Research_local/nparray/Cifar10_y_train_BS64_lid.npy",y_train)
# print("as array")

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# y_val = keras.utils.to_categorical(y_val, num_classes)

print("Get rid of top Batch Top 10% Sample")

# Define Model

#inception_resnet_v2
# model = keras.applications.inception_resnet_v2.InceptionResNetV2(include_top=True,
#                                             weights=None,
#                                             input_tensor=None,
#                                             input_shape=None,
#                                             pooling=None,
#                                             classes=num_classes)

#DenseNet(This is an imagenet weight pretrain example, which may not work when dataset is cifar10)
# model = keras.applications.DenseNet121(include_top=True,
#                 weights='imagenet',
#                 input_tensor=None,
#                 input_shape=None,
#                 pooling=None,
#                 classes=1000)

#ResNet:
model = resnet_v1(input_shape=input_shape, depth=depth)
#Xception:
# model = keras.applications.xception.Xception(include_top=True,
#                                             weights=None,
#                                             input_tensor=None,
#                                             input_shape=None,
#                                             pooling=None,
#                                             classes=num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy'])
# model.summary()
# print(model_type)

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)
lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)


# Train
# average_pooling2d_1, conv2d_21,conv2d_17
layers_names = ['average_pooling2d_1']
def renew_train_dataset():
    # x_train_feature = model
    outputs = []
    for layer in layers_names:
        outputs.append(model.get_layer(layer).output)
    model_intermediate_layer = Model(inputs=model.input, outputs=outputs)
    outputs_predict = model_intermediate_layer.predict(x_train, batch_size=1024)
    # -----numpy转换为torch格式-----
    outputs_predict_pytorch = torch.from_numpy(outputs_predict)
    outputs_predict_lid = np.zeros(train_num)
    lid_k = int(math.log(batch_size))
    batch_num = int(train_num / batch_size)
    start_time = time.time()
    train_idx = np.arange(train_num)
    np.random.shuffle(train_idx)
    pbar = ProgressBar()
    low_lid_idx = []
    for i in pbar(range(batch_num)):
        mask_batch = []
        if ((i + 1) * batch_size < train_num):
            mask_batch = np.arange(i * batch_size, (i + 1) * batch_size)  # 一个样本下标仅出现一次,顺序训练
        else:
            mask_batch = np.arange(i * batch_size, train_num)
        # import ipdb;ipdb.set_trace()
        dis = LID(outputs_predict_pytorch[mask_batch],outputs_predict_pytorch[mask_batch], lid_k)
        dis_idx = np.argwhere(dis > np.percentile(dis, 10)).flatten()
        low_lid_idx.extend(train_idx[mask_batch[dis_idx]])
    print("outputs_predic_lid time ", time.time() - start_time)

    # mask = np.random.choice(x_train.shape[0],x_train.shape[0],replace=False)
    # print(type(low_lid_idx), len(low_lid_idx), low_lid_idx[0])
    global x_train_epoch
    # import ipdb;ipdb.set_trace()
    x_train_epoch = x_train[low_lid_idx]
    global y_train_epoch
    y_train_epoch = y_train[low_lid_idx]

def on_epoch_end(epoch, logs):
    print('End of epoch')
    global x_train_epoch
    print("dataset size ",x_train_epoch.shape[0])
    renew_train_dataset()

on_epoch_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)
callbacks = [lr_reducer, lr_scheduler,on_epoch_end_callback,TensorBoard(log_dir='../TB_logdir/LayerLIDNoAug/'+exp_name+sample_name+'_'+layers_names[0],write_images=False)]

print("Train!!!")
print("---"*10,"drop bottom 10% "+exp_name+sample_name+'_'+layers_names[0],"---"*10)
renew_train_dataset()
# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train_epoch, y_train_epoch,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])