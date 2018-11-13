import sys
from pathlib import *
import keras
from keras.datasets import cifar10,mnist,cifar100
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model,load_model
from keras.callbacks import LambdaCallback
from sklearn.decomposition import PCA
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, x_test) = (np.reshape(x_train, [x_train.shape[0],x_train.shape[1],x_train.shape[2],1]),
                         np.reshape(x_test, [x_test.shape[0], x_test.shape[1], x_test.shape[2], 1]))
y_train=y_train.flatten()
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# Subtracting pixel mean improves accuracy
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean

train_num,test_num = x_train.shape[0],x_test.shape[0]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

print("LOAD BS64 LID")
selected_sample_idx = []

# ground_truth_class_lid = np.load('../LID_Research_local/nparray/Cifar10_ground_truth_dataset50000_BS5000_Globale_Class_lid_K70.npy')
ground_truth_class_lid = np.load('../LID_Research_MNIST/nparray/ground_truthBS64_lid_K8_Lid.npy')
low_lid_idx = np.argwhere(ground_truth_class_lid < np.percentile(ground_truth_class_lid, 5)).flatten()
high_lid_idx = np.argwhere(ground_truth_class_lid > np.percentile(ground_truth_class_lid, 5)).flatten()
print('low_lid_idx total sample num:', len(low_lid_idx))
print('high_lid_idx  sample num:', len(high_lid_idx))
# lid_idx = np.argsort(lid_train[sorted_idx])

# print(np.random.choice(selected_sample_idx,train_num-len(selected_sample_idx)))
outputs=[]
# model = load_model('saved_models/preTrainedCIFAR10_augX10.h5')


# -----using PCA to decrease dimension-----
model = load_model('../LID_Research_MNIST/saved_models/preTrainedMNIST.h5')
layers_names=['average_pooling2d_1']
for layer in layers_names:
    outputs.append(model.get_layer(layer).output)
model_intermediate_layer = Model(inputs=model.input, outputs=outputs)
x_train_feature = model_intermediate_layer.predict(x_train, batch_size=1024)
pca = PCA(n_components=2)
pca.fit(x_train_feature.reshape(x_train_feature.shape[0],-1))
x_train_feature_PCA = pca.transform(x_train_feature.reshape(x_train_feature.shape[0],-1))


low_lid_x_train_feature_PCA = x_train_feature_PCA[low_lid_idx]
low_lid_y_train_feature_PCA = y_train[low_lid_idx]
high_lid_x_train_feature_PCA = x_train_feature_PCA[high_lid_idx]
high_lid_y_train_feature_PCA = y_train[high_lid_idx]
plt.xlim( (-10, 10) )
plt.ylim( (-10, 10) )
plt.scatter(low_lid_x_train_feature_PCA[:,0],low_lid_x_train_feature_PCA[:,1],c=low_lid_y_train_feature_PCA,marker='.',s=1)

plt.savefig('../LID_Research_local/picture/MNIST_BS64_low5percent_lid_x_train_2Dfeature_distribution.jpg')
plt.close()

# -----using network to decrease dimension-----
# model = load_model('../LID_Research_MNIST/saved_models/preTrainedMNIST_flatten2_dim_2.h5')
# layers_names=['dense_1']
# for layer in layers_names:
#     outputs.append(model.get_layer(layer).output)
# model_intermediate_layer = Model(inputs=model.input, outputs=outputs)
# x_train_feature = model_intermediate_layer.predict(x_train, batch_size=1024)
#
# low_lid_x_train_feature = x_train_feature[low_lid_idx]
# low_lid_y_train_feature = y_train[low_lid_idx]
# high_lid_x_train_feature = x_train_feature[high_lid_idx]
# high_lid_y_train_feature = y_train[high_lid_idx]
# plt.xlim( (-10, 10) )
# plt.ylim( (-10, 10) )
# plt.scatter(low_lid_x_train_feature[:,0],low_lid_x_train_feature[:,1],c=low_lid_y_train_feature,marker='.',s=1)
# plt.savefig('../LID_Research_local/picture/MNIST_global_low5percent_lid_x_train_2Dfeature_distribution.jpg')
# plt.close()