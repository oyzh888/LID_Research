from __future__ import absolute_import
from __future__ import print_function

import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from scipy.spatial.distance import pdist, cdist, squareform
from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import LearningRateScheduler
import tensorflow as tf

sess = tf.Session()
# Set random seed
np.random.seed(0)

def GPU_lid(logits, k=20):
    """
    Calculate LID for a minibatch of training samples based on the outputs of the network.
    
    :param logits:
    :param k: 
    :return: 
    """

    logits = tf

    epsilon = 1e-12
    batch_size = tf.shape(logits)[0]
    # n_samples = logits.get_shape().as_list()
    # calculate pairwise distance
    r = tf.reduce_sum(logits*logits, 1)
    # turn r into column vector
    r1 = tf.reshape(r, [-1, 1])
    D = r1 - 2 * tf.matmul(logits, tf.transpose(logits)) + tf.transpose(r1) + \
        tf.ones([batch_size, batch_size])

    # find the k nearest neighbor
    D1 = -tf.sqrt(D)
    D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
    D3 = -D2[:, 1:] # skip the x-to-x distance 0 by using [,1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + epsilon), axis=1)  # to avoid nan
    lids = -k/v_log
    ## l2 normalization
    # lids = tf.nn.l2_normalize(lids, dim=0, epsilon=epsilon)

    ## or max min scale
    # lids = tf.div(
    #     tf.subtract(
    #         lids,
    #         tf.reduce_min(lids)
    #     ),
    #     tf.maximum(
    #         tf.subtract(
    #         tf.reduce_max(lids),
    #         tf.reduce_min(lids)
    #         ),
    #         epsilon
    #     )
    # )
    return lids


def GPU_lid_eval(logits, k=20):
    import time
    start_time = time.time()
    """
    Calculate LID for a minibatch of training samples based on the outputs of the network.

    :param logits:
    :param k:
    :return:
    """
    print(logits.shape)
    logits = tf.constant(logits,dtype=tf.float32)

    epsilon = 1e-12
    batch_size = tf.shape(logits)[0]
    # n_samples = logits.get_shape().as_list()
    # calculate pairwise distance
    r = tf.reduce_sum(logits * logits, 1)
    # turn r into column vector
    r1 = tf.reshape(r, [-1, 1])
    D = r1 - 2 * tf.matmul(logits, tf.transpose(logits)) + tf.transpose(r1) + \
        tf.ones([batch_size, batch_size])

    # find the k nearest neighbor
    D1 = -tf.sqrt(D)
    D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
    D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + epsilon), axis=1)  # to avoid nan
    lids = -k / v_log
    ## l2 normalization
    # lids = tf.nn.l2_normalize(lids, dim=0, epsilon=epsilon)

    # import ipdb;
    # ipdb.set_trace()

    sess = tf.Session()
    with sess.as_default():
        lids = sess.run(lids)
    print('LID GPU Time:',time.time()-start_time)
    return lids

from keras import backend as K
def GPU_lid_eval_keras(logits, k=20):
    import time
    start_time = time.time()
    """
    Calculate LID for a minibatch of training samples based on the outputs of the network.

    :param logits:
    :param k:
    :return:
    """
    print(logits.shape)
    logits = K.constant(logits,dtype=tf.float32)

    epsilon = 1e-12
    batch_size = K.shape(logits)[0]
    # n_samples = logits.get_shape().as_list()
    # calculate pairwise distance
    r = K.reduce_sum(logits * logits, 1)
    # turn r into column vector
    r1 = K.reshape(r, [-1, 1])
    D = r1 - 2 * K.matmul(logits, K.transpose(logits)) + K.transpose(r1) + \
        K.ones([batch_size, batch_size])

    # find the k nearest neighbor
    D1 = -K.sqrt(D)
    D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
    D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]

    m = K.transpose(K.multiply(K.transpose(D3), 1.0 / D3[:, -1]))
    v_log = K.reduce_sum(tf.log(m + epsilon), axis=1)  # to avoid nan
    lids = -k / v_log
    ## l2 normalization
    # lids = tf.nn.l2_normalize(lids, dim=0, epsilon=epsilon)

    # import ipdb;
    # ipdb.set_trace()

    # sess = tf.Session()
    #     # with sess.as_default():
    #     #     lids = sess.run(lids)

    print('LID GPU Time:',time.time()-start_time)
    return lids.eval()

def GPU_lid_eval_opt(logits, k=20):
    import time
    start_time = time.time()
    """
    Calculate LID for a minibatch of training samples based on the outputs of the network.

    :param logits:
    :param k:
    :return:
    """
    logits = tf.constant(logits,dtype=tf.float32)

    epsilon = 1e-12
    batch_size = tf.shape(logits)[0]
    r = tf.reduce_sum(logits * logits, 1)
    # turn r into column vector
    r1 = tf.reshape(r, [-1, 1])
    ###Modified
    D = tf.square( r1 - tf.transpose(r1) ) + tf.ones([batch_size, batch_size])

    # find the k nearest neighbor
    D1 = -tf.sqrt(D)
    D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
    D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]

    m = tf.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(tf.log(m + epsilon), axis=1)  # to avoid nan
    lids = -k / v_log

    # est = -1 / (1 / k * tf.reduce_sum(tf.log(D3) , axis=1, keepdims=True) - np.log(r_max))

    with sess.as_default():
        lids = sess.run(lids)
    print('LID GPU Time:',time.time()-start_time)
    return lids


def mle_single(data, x, k):
    """
    lid of a single query point x.
    numpy implementation.
    
    :param data: 
    :param x: 
    :param k: 
    :return: 
    """
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    # dim = x.shape[1]

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1] + 1e-8))
    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]

def mle_batch(data, batch, k):
    """
    lid of a batch of query points X.
    numpy implementation.
    
    :param data: 
    :param batch: 
    :param k: 
    :return: 
    """
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data)-1)
    f = lambda v: - k / np.sum(np.log(v/v[-1] + 1e-8))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:,1:k+1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """
    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class

def get_lids_random_batch(model, X, k=20, batch_size=100):
    """
    Get the local intrinsic dimensionality of each Xi in X_adv
    estimated by k close neighbours in the random batch it lies in.
    :param model: if None: lid of raw inputs, otherwise LID of deep representations 
    :param X: normal images 
    :param k: the number of nearest neighbours for LID estimation  
    :param batch_size: default 100
    :return: lids: LID of normal images of shape (num_examples, lid_dim)
            lids_adv: LID of advs images of shape (num_examples, lid_dim)
    """
    if model is None:
        lids = []
        n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
        for i_batch in tqdm(range(n_batches)):
            start = i_batch * batch_size
            end = np.minimum(len(X), (i_batch + 1) * batch_size)
            X_batch = X[start:end].reshape((end-start, -1))

            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch = mle_batch(X_batch, X_batch, k=k)
            lids.extend(lid_batch)

        lids = np.asarray(lids, dtype=np.float32)
        return lids


    # get deep representations
    funcs = [K.function([model.layers[0].input, K.learning_phase()], [out])
                 for out in [model.layers[-3].output]]
    lid_dim = len(funcs)
    print("Number of layers to estimate: ", lid_dim)

    def estimate(i_batch):
        start = i_batch * batch_size
        end = np.minimum(len(X), (i_batch + 1) * batch_size)
        n_feed = end - start
        lid_batch = np.zeros(shape=(n_feed, lid_dim))
        for i, func in enumerate(funcs):
            X_act = func([X[start:end], 0])[0]
            X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))

            # Maximum likelihood estimation of local intrinsic dimensionality (LID)
            lid_batch[:, i] = mle_batch(X_act, X_act, k=k)

        return lid_batch

    lids = []
    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        lid_batch = estimate(i_batch)
        lids.extend(lid_batch)

    lids = np.asarray(lids, dtype=np.float32)

    return lids


def get_lr_scheduler(dataset):
    """
    customerized learning rate decay for training with clean labels.
     For efficientcy purpose we use large lr for noisy data.
    :param dataset: 
    :param noise_ratio:
    :return: 
    """
    if dataset in ['mnist', 'svhn']:
        def scheduler(epoch):
            if epoch > 40:
                return 0.001
            elif epoch > 20:
                return 0.01
            else:
                return 0.1
        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-10']:
        def scheduler(epoch):
            if epoch > 80:
                return 0.001
            elif epoch > 40:
                return 0.01
            else:
                return 0.1
        # def scheduler(epoch):
        #     if epoch > 20:
        #         return 0.001
        #     else:
        #         return 0.01
        return LearningRateScheduler(scheduler)
    elif dataset in ['cifar-100']:
        def scheduler(epoch):
            if epoch > 120:
                return 0.0001
            elif epoch > 80:
                return 0.001
            elif epoch > 40:
                return 0.01
            else:
                return 0.1
        return LearningRateScheduler(scheduler)

class LIDOnoffCallback(Callback):
    def __init__(self, lid_on):
        super().__init__()
        self.lid_on = lid_on

    def on_epoch_end(self, epoch, logs=None):
        # logs: acc, loss, val_acc, val_loss
        acc = logs.get('acc')
        val_acc = logs.get('val_acc')
        if acc > val_acc + 0.1:
            K.set_value(self.lid_on, True)
        else:
            K.set_value(self.lid_on, False)

def uniform_noise_model_P(num_classes, noise):
    """ The noise matrix flips any class to any other with probability
    noise / (num_classes - 1).
    """

    assert(noise >= 0.) and (noise <= 1.)

    P = noise / (num_classes - 1) * np.ones((num_classes, num_classes))
    np.fill_diagonal(P, (1 - noise) * np.ones(num_classes))

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P

def get_deep_representations(model, X, batch_size=128):
    """
    Get the deep representations before logits.
    :param model:
    :param X:
    :param batch_size:
    :return:
    """
    # last hidden layer is always at index -4
    output_dim = model.layers[-3].output.shape[-1].value
    get_encoding = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[-3].output]
    )

    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=(len(X), output_dim))
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]

    return output

