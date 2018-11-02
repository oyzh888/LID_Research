import sys
from pathlib import *
from keras.callbacks import LambdaCallback

if(len(sys.argv)>4):
    dataset_name = (sys.argv[1])
    model_name = (sys.argv[2])
    lid_method = (sys.argv[3])
    drop_percent = int(sys.argv[4])
else:
    # print('Wrong Params')
    # exit()
    dataset_name = 'CIFAR10'
    # dataset_name = 'CIFAR10'
    model_name = 'resNet'
    batch_size = 64  # orig paper trained all networks with batch_size=128
    # lid_method = 'lid_high'
    lid_method = 'random'
    drop_percent = 1
    # exit()
# work_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/Tsinghua/Logs/Exp_LID_Data_Drop/')
work_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/HTB/LID_Research/LID_DataDrop/LID_Exps')
max_acc_log_path = work_path/'res.txt'
convergence_epoch = 0

# Training parameters
batch_size = 64
epochs = 20
exp_name = '%s_%s_%d_%s_%d_%d' % (dataset_name,model_name,batch_size,lid_method,drop_percent,epochs)


##### Train

import keras
import torch
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau, LambdaCallback
from keras.callbacks import TensorBoard
from keras.datasets import cifar10,mnist,cifar100
import numpy as np
from resNet_LID import *
from lid import LID, get_lid_by_batch
import sys
from pathlib import *

# Load the dataset.
if dataset_name=='MNIST':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    (x_train, x_test) = (np.reshape(x_train, [x_train.shape[0],x_train.shape[1],x_train.shape[2],1]),
                         np.reshape(x_test, [x_test.shape[0], x_test.shape[1], x_test.shape[2], 1]))
    num_classes = len(set(y_train.flatten()))
if dataset_name=='CIFAR10':
    print(dataset_name)
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    num_classes = len(set(y_train.flatten()))
if dataset_name=='CIFAR100':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    num_classes = len(set(y_train.flatten()))

print("class num ",num_classes)


# Path(work_path/'Layer_LID_nparray').mkdir(parents=True, exist_ok=True)
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


#ResNet:
if model_name=='resNet':
    # model = keras.applications.resnet50.ResNet50(input_shape=None, include_top=True, weights=None)
    model = resnet_v1(input_shape=input_shape, depth=3*6+2)
#Xception
if model_name=='Xception':
    # model = keras.applications.xception.Xception(include_top=True,
    #                                              weights=None,
    #                                              input_tensor=None,
    #                                              input_shape=None,
    #                                              pooling=None,
    #                                              classes=num_classes)
    model = Xception(include_top=True,
                     weights=None,
                     input_tensor=None,
                     input_shape=input_shape,
                     pooling=None,
                     classes=num_classes)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=lr_schedule(0)),
              metrics=['accuracy', 'top_k_categorical_accuracy'])
model.summary()
print("-"*20+exp_name+'-'*20)

# Prepare model model saving directory.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
# 按照Class计算LID
print("Class LID using batch")

sorted_idx = np.argsort(y_train.flatten()).flatten()
x_train = x_train[sorted_idx]
y_train = y_train[sorted_idx]
selected_sample_idx = []

if lid_method == 'random':
    for i in range(int(train_num/batch_size)):
        mask_batch=[]
        if ((i + 1) * batch_size < train_num):
            mask_batch = np.arange(i * batch_size, (i + 1) * batch_size)  # 一个样本下标仅出现一次,顺序训练
        else:
            mask_batch = np.arange(i * batch_size, train_num)
        selected_sample_idx.extend(np.random.choice(mask_batch,int(batch_size*(1-drop_percent/100)),replace=True))
        # selected_sample_idx.extend(
        #     np.random.choice(mask_batch, mask_batch, replace=False))
else:
    lid_k = int(np.log(batch_size))
    ###Sorted
    torch_x_train = torch.from_numpy(np.reshape(x_train,(len(x_train),-1)))
    lid_train = get_lid_by_batch(torch_x_train, torch_x_train,
                                  lid_k, batch_size=batch_size)
    if(lid_method == 'lid_low'):
        lid_selected_idx = np.argwhere(lid_train > np.percentile(lid_train, drop_percent)).flatten()  # Drop Low
    if (lid_method == 'lid_high'):
        lid_selected_idx = np.argwhere(lid_train < np.percentile(lid_train, 100 - drop_percent)).flatten()  # Drop Low
    selected_sample_idx = lid_selected_idx.tolist()

# import ipdb;ipdb.set_trace()
print("drop",len(selected_sample_idx))
# import ipdb; ipdb.set_trace()
# print(np.random.choice(selected_sample_idx,train_num-len(selected_sample_idx)))
selected_sample_idx.extend(np.random.choice(selected_sample_idx,train_num-len(selected_sample_idx)))
selected_x_train = x_train[selected_sample_idx]
selected_y_train = y_train[selected_sample_idx]
print("resample",len(selected_sample_idx))


# Convert class vectors to binary class matrices.
selected_y_train = keras.utils.to_categorical(selected_y_train, num_classes)
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


work_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/Tsinghua/Logs/Exp_LID_Data_Drop/')
work_path.mkdir(parents=True, exist_ok=True)
TB_log_path = work_path/'TB_Log'/exp_name
callbacks = [on_epoch_end_callback, lr_reducer, lr_scheduler, TensorBoard(log_dir= (TB_log_path.__str__()))]
# Run training, with or without data augmentation.
model.fit(selected_x_train, selected_y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])


### Final result output
final_accuracy = scores[1]
final_loss = scores[0]

print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("exp_name", 'final_accuracy', 'final_loss',
                                  'converage_epoch', 'lid_method', 'drop_percent', 'model_name' ))
max_acc_log_line = "%s\t%f\t%f\t%d\t%s\t%d\t%s" % (exp_name, final_accuracy, final_loss, convergence_epoch, lid_method, drop_percent, model_name)
print(max_acc_log_line)
# print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % ("exp_name", 'final_accuracy', 'final_loss',
#                                   'converage_epoch', 'lid_method', 'drop_percent', 'model_name' ),file=open(max_acc_log_path.__str__(), 'a'))
print(max_acc_log_line, file=open(max_acc_log_path.__str__(), 'a'))