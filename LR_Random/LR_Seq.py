import sys
from pathlib import *
from keras.callbacks import LambdaCallback
import LR_resNet
from LR_resNet import *
import time
from keras.optimizers import Adam,SGD,RMSprop,Adagrad
import random

if(len(sys.argv)>8):
    dataset_name = (sys.argv[1])
    batch_size = int(sys.argv[2])
    epochs = int(sys.argv[3])
    optimizer = (sys.argv[4])
    distribution_method = (sys.argv[5])
    dis_parameter1 = float(sys.argv[6])
    dis_parameter2 = float(sys.argv[7])
    work_path_name = (sys.argv[8])
    init_lr = float(sys.argv[9])
else:
    print('Wrong Params')
    # exit()
    # dataset_name = 'MNIST'
    dataset_name = 'CIFAR10'
    batch_size = 128  # orig paper trained all networks with batch_size=128
    epochs = 50
    optimizer = 'Adam'
    distribution_method = 'N'
    dis_parameter1 = 0.2
    dis_parameter2 = 0.8
    work_path_name = 'Default'
    init_lr = 1e-3

work_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/Tsinghua/Logs/RandomLR_12_18_Seq')
# work_path = Path('/unsullied/sharefs/ouyangzhihao/Share/LSTM/Text_Generation_Capacity/Logs/RandomLR_12_8_All')

work_path = work_path/work_path_name

# work_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/HTB/LID_Research/LR_Random')
max_acc_log_path = work_path/'res.txt'
convergence_epoch = 0

# Training parameters
exp_name = 'Sort%s_%d_%d_%s_%s_%.2f_%.2f_ResNet32' % (dataset_name,epochs,batch_size,optimizer,distribution_method,dis_parameter1,dis_parameter2)
if((work_path/'TB_Log'/exp_name).exists()):
    print('Already Finished!')
    exit()
##### Train

import keras
import torch
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.callbacks import TensorBoard
from keras.datasets import cifar10,mnist,cifar100
import numpy as np
import sys
from pathlib import *

# Load the dataset.
if dataset_name=='CIFAR10':
    print(dataset_name)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = len(set(y_train.flatten()))
if dataset_name=='CIFAR100':
    print(dataset_name)
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    num_classes = len(set(y_train.flatten()))

print("class num ",num_classes)

# Path(work_path/'Layer_LID_nparray').mkdir(parents=True, exist_ok=True)
input_shape = x_train.shape[1:]

x_train, x_test = preProcessData(x_train,x_test,subtract_pixel_mean=True)

train_num,test_num = x_train.shape[0],x_test.shape[0]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

def get_lr_seq(epochs, lr_type='Base', sort_lr = True):
    #In order to generate true random number, we have to generate multiple random numbers at the same time
    np.random.seed(int(time.time()))
    def U(tmp_lr_seq):
        factor = 1e1
        random_arr = np.random.random(len(tmp_lr_seq))
        tmp_lr_seq = tmp_lr_seq * factor * random_arr
        return tmp_lr_seq

    def N(tmp_lr_seq, mu=0, sigma=1):
        factor = 1e1
        random_arr = np.random.normal(mu, sigma, epochs)
        random_arr = abs(random_arr)
        tmp_lr_seq = tmp_lr_seq * random_arr * factor
        return tmp_lr_seq

    def Base(tmp_lr_seq): return tmp_lr_seq

    lr_seq = []
    for epoch in range(1,epochs+1):
        lr = init_lr
        if epoch >= epochs * 0.9:
            lr *= 0.5e-3
        elif epoch >= epochs * 0.8:
            lr *= 1e-3
        elif epoch >= epochs * 0.6:
            lr *= 1e-2
        elif epoch >= epochs * 0.4:
            lr *= 1e-1
        lr_seq.append(lr)
    # transfer to numpy in order to support element wise multiple
    lr_seq = np.array(lr_seq)
    if(lr_type == 'U'):
        lr_seq = U(lr_seq)
    elif(lr_type == 'N'):
        lr_seq = N(lr_seq)

    # sort in descending order
    if(sort_lr):
        lr_seq[::-1].sort()
    return lr_seq

lr_seq = get_lr_seq(epochs, lr_type=distribution_method, sort_lr=True)
def lr_schedule(epoch):
    lr = lr_seq[epoch-1]
    print('Learning rate: ', lr)
    return lr

#ResNet:
# model = keras.applications.resnet50.ResNet50(input_shape=None, include_top=True, weights=None)
model = resnet_v1(input_shape=input_shape, depth=5*6+2,num_classes = num_classes)
if optimizer=='Adam':
    opt = Adam(lr=lr_schedule(0))
elif optimizer=='SGD':
    opt = SGD(lr=lr_schedule(0))
elif optimizer =='RMSprop':
    opt = RMSprop(lr=lr_schedule(0))
elif optimizer == 'Adagrad':
    opt = Adagrad(lr=lr_schedule(0))

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy', 'top_k_categorical_accuracy'])
# model.summary()
print("-"*20+exp_name+'-'*20)

# Prepare model model saving directory.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


### Train end
last_acc = 0
convergence_epoch = 0
def on_epoch_end(epoch, logs):
    # from ipdb import set_trace as tr; tr()
    # print(logs)
    global last_acc
    if(logs['val_acc'] - last_acc > 0.01):
        global convergence_epoch
        convergence_epoch = epoch
    last_acc = logs['val_acc']
    print('End of epoch')
    # renew_train_dataset()
on_epoch_end_callback = LambdaCallback(on_epoch_end=on_epoch_end)


TB_log_path = work_path/'TB_Log'/exp_name
callbacks = [on_epoch_end_callback, lr_reducer, lr_scheduler, TensorBoard(log_dir= (TB_log_path.__str__()))]
# Run training, with or without data augmentation.
aug = True
if aug == False:
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
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
        rotation_range=10,
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

    # Compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1,batch_size=batch_size*4)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


### Final result output
final_accuracy = scores[1]
final_loss = scores[0]

print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("exp_name", 'final_accuracy', 'final_loss',
                                  'converage_epoch', 'distribution', 'par1', 'par2', 'dataset_name' ))
max_acc_log_line = "%s\t%f\t%f\t%d\t%s\t%d\t%s\t%s" % (exp_name, final_accuracy, final_loss, convergence_epoch, distribution_method, dis_parameter1, dis_parameter2, dataset_name)
print(max_acc_log_line)
# print("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("exp_name", 'final_accuracy', 'final_loss',
#                                   'converage_epoch', 'lid_method', 'drop_percent', 'model_name','dataset_name' ),file=open(max_acc_log_path.__str__(), 'a'))
print(max_acc_log_line, file=open(max_acc_log_path.__str__(), 'a'))
