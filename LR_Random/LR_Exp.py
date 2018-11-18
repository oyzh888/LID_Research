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
else:
    print('Wrong Params')
    # exit()
    # dataset_name = 'MNIST'
    dataset_name = 'CIFAR10'
    batch_size = 64  # orig paper trained all networks with batch_size=128
    epochs = 20
    optimizer = 'Adam'
    distribution_method = 'RL'
    dis_parameter1 = 0.2
    dis_parameter2 = 0.8
    work_path_name = 'Default'

work_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/Tsinghua/Logs/Exp_RandomLR/20epoch')
# work_path = Path('/home/ouyangzhihao/Share/LSTM/Text_Generation_Capacity/Code/Logs/Exp_RandomLR')
work_path = work_path/work_path_name

# work_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/HTB/LID_Research/LR_Random')
max_acc_log_path = work_path/'res.txt'
convergence_epoch = 0

# Training parameters
exp_name = '%s_%d_%d_%s_%s_%.2f_%.2f' % (dataset_name,epochs,batch_size,optimizer,distribution_method,dis_parameter1,dis_parameter2)

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

def lr_schedule(epoch):
    def U(tmp_lr):
        factor = 1e2
        np.random.seed(int(time.time()))
        tmp_lr = np.random.random() * factor / np.sqrt(factor) * tmp_lr
        return tmp_lr
    def N(tmp_lr,mu=0,sigma=1):
        np.random.seed(int(time.time()))
        tmp_lr_factor = np.random.normal(mu,sigma)
        tmp_lr_factor = abs(tmp_lr_factor)
        tmp_lr *= tmp_lr_factor
        return tmp_lr
    def RL(tmp_lr,explore_rate=0.2,exploit_rate=0.8):
        random.seed(int(time.time()))
        rand_float = random.random()
        if rand_float < explore_rate:
            factor = 1e2
            tmp_lr = np.random.random() * factor / np.sqrt(factor)
        return tmp_lr

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

    if distribution_method =='U':
        lr = U(lr)
    elif distribution_method =='N':
        lr = N(lr,dis_parameter1,dis_parameter2)
    elif distribution_method =='RL':
        lr = RL(lr,dis_parameter1,dis_parameter2)
    elif distribution_method =='Base':
        lr = lr
    print('Learning rate: ', lr)
    return lr


#ResNet:
# model = keras.applications.resnet50.ResNet50(input_shape=None, include_top=True, weights=None)
model = resnet_v1(input_shape=input_shape, depth=3*6+2,num_classes = num_classes)
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
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
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