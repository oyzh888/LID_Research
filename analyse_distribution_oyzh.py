#!/usr/bin/env python
# coding: utf-8

"""Trains a ResNet on the CIFAR10 dataset.
ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf
ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf

本程序调用已经训练好的mnist网络，查看样本在不同batch,不同卷积层中的lid值变化。
"""

from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import keras
import math
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
import os
from sklearn.decomposition import PCA
import lid
from lid import LID
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

outputs_predict_lid1 = np.load("./nparray/Cifar10_ground_truth_dataset50000_BS5000_lid_K70_Lid.npy")
outputs_predict_lid2 = np.load("./nparray/Cifar10_ground_truth_dataset50000_BS5000_lid_K70_test1_Lid.npy")

import numpy as np
from scipy import *
def asymmetricKL(P, Q):
    return sum(P * log(P / Q))  # calculate the kl divergence between P and Q
def symmetricalKL(P, Q):
    return (asymmetricKL(P, Q) + asymmetricKL(Q, P)) / 2.00
def get_l2_distance(P,Q):
    return np.sqrt(np.sum(np.square(P - Q)))

var1 = np.array(outputs_predict_lid1)
var2 = np.array(outputs_predict_lid2)

##Delete
var1 += 1e-7
var2 += 1e-7

#Copy var
import copy
var1_deep = copy.deepcopy(var1)
np.random.shuffle(var1_deep)




print('L2:')
print('var1 vs var2',get_l2_distance(var1,var2))
print('var1 vs var1',get_l2_distance(var1,var1))

# import ipdb; ipdb.set_trace()
print('> 1 LID',len(np.argwhere(var1>1)))
print('> 1e-3 LID',len(np.argwhere(var1>1e-3)))
# print(type(var1))
# ans =
# print(var1.shape)


# import ipdb; ipdb.set_trace()
# print('shape:', np.shape(var1_deep))
print('var1 vs random var1',get_l2_distance(var1,var1_deep))

print('KL:')
print('var1 vs var2',symmetricalKL(var1,var2))
print('var1 vs var1',symmetricalKL(var1,var1))
print('var1 vs random var1',symmetricalKL(var1,var1_deep))

print('KL Distance:')
print('var1 vs var2',asymmetricKL(var1,var2))
print('var1 vs var1',asymmetricKL(var1,var1))
print('var1 vs random var1',asymmetricKL(var1,var1_deep))

print('Average:')
print('var1 average',np.average(var1))
print('var2 average',np.average(var2))

print('Variance:')
print('var1 variance',np.var(var1))
print('var2 variance',np.var(var2))