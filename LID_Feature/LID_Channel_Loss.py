# LID on features map getting different features
import os
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input
import tensorflow as tf

# tf.enable_eager_execution()
# 全局变量
batch_size = 16
nb_classes = 10
epochs = 2
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)
k_LID = 10
# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 根据不同的backend定下不同的格式
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 转换为one_hot类型
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


## LID function
def LID(x, LID_extract):  # LID_extract:[batch,rows,cols,channels]
    ########change k=10?batch=128
    epsilon = 1e-12
    batch_size = 16
    # print('batch',batch_size,'XX:',LID_extract.shape)
    XX = LID_extract # XX:[batch,rows,cols,channels]
    XX = tf.transpose(XX, [0, 3, 1, 2])  # XX:[batch,channels,rows,cols]
    # print('XX:',XX.shape)

    ##这里有bug,只有BS=16才可以运行!
    # XX = tf.reshape(tensor=XX, shape=[batch_size, nb_filters, -1])  # [batch,features,channels]-----batch无法代入
    #import ipdb;
    #ipdb.set_trace()
    XX = tf.reshape(tensor=XX, shape=[batch_size, int(XX.shape[1]), -1])  # [batch,features,channels]-----batch无法代入
    x_LID = tf.constant([0], dtype="float32")
    # print('XX:',XX.shape)
    LID_k = 10
    for i_batch in range(batch_size):  # 第i个数据计算平均LID
        r = tf.reduce_sum(XX[i_batch] * XX[i_batch], 1) # 计算XX^2,同时去掉channel维,此时r:[batch,rows,cols]。此处*运算符按照元素进行计算(element-wise)
        # turn r into column vector
        r1 = tf.reshape(r, [-1, 1])
        D = r1 - 2 * tf.matmul(XX[i_batch], tf.transpose(XX[i_batch])) + tf.transpose(r1) + tf.ones([32, 32])
        # find the k nearest neighbor.00
        D1 = -tf.sqrt(D)
        D2, _ = tf.nn.top_k(D1, k=LID_k, sorted=True)
        D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]. 记录了每个点最近的k-1个距离

        m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))    # r_i/r_max, tf.multiply计算的是element-wise乘法。 为什么内部tf.transpose(D3)？
        # 上面两侧transpose应该可以去掉吧？
        v_log = tf.reduce_sum(tf.log(m + epsilon), axis=1)  # to avoid nan
        lids = -LID_k / v_log
        # print('lids:',lids.shape)
        # print('mean:',tf.reshape(tf.reduce_mean(input_tensor=lids,axis=0),[1,]).shape)
        #import ipdb;ipdb.set_trace()
        x_LID = tf.concat([x_LID, tf.reshape(tf.reduce_mean(input_tensor=lids, axis=0), [1, ])], axis=0)

    return x_LID[1:]  # 张量


# 构建模型

x_inputs = Input(shape=input_shape)
x = Conv2D(nb_filters, kernel_size=[kernel_size[0], kernel_size[1]], padding='same',
           activation='relu', input_shape=input_shape)(x_inputs)
x = Conv2D(nb_filters, kernel_size=[kernel_size[0], kernel_size[1]], padding='same', activation='relu')(x)
LID_extract = MaxPooling2D(pool_size=kernel_size, name='LID_extract')(x)

### x is normal loss , LID_extract is LID loss
x = Dropout(0.25)(LID_extract)
print('LID before shape', LID_extract.shape)
#  print('LID before shape type shape[0]', type(LID_extract.shape[0]))
LID_loss = LID(LID_extract, LID_extract)
print('LID after shape', LID_loss.shape)
# LID_loss = K.sum(K.reshape(LID_loss , shape=[batch_size, -1]))
LID_loss = K.sum(K.flatten(LID_loss))
###

x = Flatten()(x)
x = Dense(batch_size, activation='relu')(x)
x = Dropout(0.5)(x)
y_predictions = Dense(nb_classes, activation='softmax', name='y_predictions')(x)

y_inputs = Input(shape=(10,))


# =============================================================================
def myloss(y_true, y_pred):
    class_loss = K.categorical_crossentropy(y_true, y_pred)
    lid_loss = 0.01 * LID_loss
    #lid_loss=0
    print("my loss:",lid_loss," ",class_loss)
    return lid_loss + class_loss

# LID小好吗？二者loss的大小可能不是一个数量级
model = Model(inputs=x_inputs, outputs=[y_predictions])
model.compile(loss=myloss, optimizer='SGD', metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0, batch_size=batch_size)
print('Test score:', score[0])
print('Test accuracy:', score[1])
# =============================================================================