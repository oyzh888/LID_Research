from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from keras.datasets import cifar100, cifar10
import os
import numpy as np
from pathlib import *
from keras.datasets import cifar10,cifar100,mnist
# datagen=ImageDataGenerator(
#       rotation_range=40,
#       width_shift_range=0.1,
#       height_shift_range=0.1,
#       rescale=1./255,
#       shear_range=0.1,
#       zoom_range=0.1,
#       horizontal_flip=True,
#       fill_mode='nearest')
datagen = ImageDataGenerator(
        # set input mean to 0 over the dataset
        featurewise_center=False,
        # set each sample mean to 0
        samplewise_center=False,
        # divide inputs by std of dataset
        featurewise_std_normalization=False,
        # divide each input by its std
        samplewise_std_normalization=False,
        # apply ZCA whitening
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # randomly rotate images in the range (deg 0 to 180)
        rotation_range=0,
        # randomly shift images horizontally
        width_shift_range=0.1,
        # randomly shift images vertically
        height_shift_range=0.1,
        # set range for random shear
        shear_range=0.,
        # set range for random zoom
        zoom_range=0.,
        # set range for random channel shifts
        channel_shift_range=0.,
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # randomly flip images
        horizontal_flip=True,
        # randomly flip images
        vertical_flip=False,
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
# x_train = np.expand_dims(x_train, axis=3)
# x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
# part = 100
# (x_train, y_train) = (x_train[0:part], y_train[0:part])
# x=img_to_array(x_train)
# y=img_to_array(y_train)
augment_time = 10
print(x_train.shape)
print(y_train.shape)
print(type(x_train))
print(type(y_train))

root_path = Path('../Data_Aug/')
if not root_path.exists(): root_path.mkdir()
from progressbar import ProgressBar
pbar = ProgressBar()

aug_train_x = []
aug_train_y = []

for cls in pbar(range(100)):
    i = 0
    # print((y_train==cls).shape)
    # print(cls,y_train[1:10])
    # id = '%d' % (cls)
    # save_dir = root_path/id
    # if not(save_dir.exists()): save_dir.mkdir()
    x_cls=x_train[y_train.flatten()==cls]
    y_cls=y_train[y_train.flatten()==cls]
    # for batch in datagen.flow(x_cls, y_cls, batch_size=256,
    #                           save_to_dir=save_dir, save_prefix='cat', save_format='jpg'):
    # for batch in datagen.flow(x_cls, y_cls, batch_size=len(x_cls), save_to_dir=save_dir.__str__(),
    #                           save_prefix=id, save_format='jpg'):
    for batch in datagen.flow(x_cls, y_cls, batch_size=len(x_cls)):
        aug_train_x.extend(batch[0])
        aug_train_y.extend(batch[1])
        i += 1
        if i >= augment_time:
            break
# aug_train_x = np.reshape(aug_train_x,(x_train.shape[0]*augment_time,28,28))
# aug_train_y = np.sa
# save_dir = os.path.join(root_path)

print(np.shape(aug_train_x), np.shape(aug_train_y))
np.save(root_path/("cifar100_X{}_x_train.npy".format(augment_time)), aug_train_x)
np.save(root_path/("cifar100_X{}_y_train.npy".format(augment_time)), aug_train_y)
