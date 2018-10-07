from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from keras.datasets import cifar100, cifar10
import os
import numpy as np
datagen=ImageDataGenerator(
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      rescale=1./255,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
# img=cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x=img_to_array(x_train)
# y=img_to_array(y_train)
augment_time = 10
print(x_train.shape)
print(y_train.shape)
print(type(x_train))
print(type(y_train))

root_path = '../Cifar10_Aug'
if not (os.path.exists(root_path)): os.mkdir(root_path)
from progressbar import ProgressBar
pbar = ProgressBar()

aug_train_x = []
aug_train_y = []

for cls in pbar(range(10)):
    i = 0
    # print((y_train==cls).shape)
    # print(cls,y_train[1:10])
    # save_dir = os.path.join(root_path, '%d' % (cls))
    # if not(os.path.exists(save_dir)): os.mkdir(save_dir)
    x_cls=x_train[y_train.flatten()==cls]
    y_cls=y_train[y_train.flatten()==cls]
    # for batch in datagen.flow(x_cls, y_cls, batch_size=256,
    #                           save_to_dir=save_dir, save_prefix='cat', save_format='jpg'):
    for batch in datagen.flow(x_cls, y_cls, batch_size=len(x_cls),
                              save_prefix='cat', save_format='jpg'):
        aug_train_x.extend(batch[0])
        aug_train_y.extend(batch[1])
        i += 1
        if i >= augment_time:
            break


# aug_train_y = np.sa
# save_dir = os.path.join(root_path)

print(np.shape(aug_train_x),np.shape(aug_train_y))
np.save(root_path+"/aug_train_x.npy",aug_train_x)
np.save(root_path+"/aug_train_y.npy",aug_train_y)