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
from lid import *
import resNet_LID
from resNet_LID import *
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

# Model name, depth and version
model_type = 'ResNet%dv%d' % (3*6+2, 1)

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

# 以validation各个类别作为pca分析的输入，将训练集中各个类别依照该pca进行降维，使得每个类别的降维方式一致。
# 每个类别降维之后，在类别内部根据LID值寻找异常点，将异常点进行记录。
cls_num = 10
sorted_idx = np.argsort(y_train.flatten()).flatten()
x_train = x_train[sorted_idx]
y_train = y_train[sorted_idx]
selected_sample_idx = []
lid_k = int(np.sqrt(batch_size))
torch_x_train = torch.from_numpy(np.reshape(x_train,(len(x_train),-1)))
lid_train = get_lid_by_batch(torch_x_train, torch_x_train,
                                  lid_k, batch_size=batch_size)
global_dis_avg = get_dis_avg_by_batch(torch_x_train,torch_x_train,lid_k)
lid_selected_idx = np.argwhere(lid_train < np.percentile(lid_train, 99)).flatten()  # Drop highest 1%
top_LID_dis_avg = get_dis_avg(torch_x_train[lid_selected_idx],torch_x_train[lid_selected_idx],lid_k)
selected_sample_idx = lid_selected_idx.tolist()
selected_sample_idx.extend(np.random.choice(selected_sample_idx,train_num-len(selected_sample_idx)))
x_train = x_train[selected_sample_idx]
y_train = y_train[selected_sample_idx]

print("global_dis_avg median:",np.median(global_dis_avg))
print("top_LID_dis_avg median:",np.median(top_LID_dis_avg))
np.save("../LID_Research_local/nparray/global_dis_avg.npy",global_dis_avg)
np.save("../LID_Research_local/nparray/top_LID_dis_avg.npy",top_LID_dis_avg)

# np.save("../LID_Research_local/nparray/Cifar10_y_train_BS64_lid.npy",y_train)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
# y_val = keras.utils.to_categorical(y_val, num_classes)


#ResNet:
model = resnet_v1(input_shape=input_shape, depth=3*6+2)

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
# callbacks = [lr_reducer, lr_scheduler,on_epoch_end_callback,TensorBoard(log_dir='../TB_logdir/LayerLIDNoAug/'+exp_name+sample_name+'_'+layers_names[0],write_images=False)]
callbacks = [lr_reducer, lr_scheduler,TensorBoard(log_dir='../TB_logdir/LayerLIDNoAug/'+exp_name+sample_name,write_images=False)]

print("Train!!!")

# renew_train_dataset()
# Run training, with or without data augmentation.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])