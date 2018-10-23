import numpy as np
import matplotlib.pyplot as plt
import torch

device = "gpu" if torch.cuda.is_available() else "cpu"


def distance(X, Y, sqrt):
    nX = X.size(0)
    nY = Y.size(0)
    X = X.view(nX, -1).to(device)
    X2 = (X*X).sum(1).resize_(nX, 1)
    Y = Y.view(nY, -1).to(device)
    Y2 = (Y*Y).sum(1).resize_(nY, 1)

    M = torch.zeros(nX, nY)
    M.copy_(X2.expand(nX, nY) + Y2.expand(nY, nX).transpose(0, 1) -
            2 * torch.mm(X, Y.transpose(0, 1)))

    del X, X2, Y, Y2

    if sqrt:
        M = ((M + M.abs()) / 2).sqrt()

    return M


def LID_elementwise(x, X, k):
    # only for X: [B, d]
    r = np.power(np.sum(np.power(X - x, 2), axis=-1), 0.5)
    r = np.sort(r, axis=0)
    r_max = r[k-1]
    return -1 / (np.mean(np.log(r[0:k] / r_max)))


def LID(X, Y, k):
    import time
    start_time = time.time()
    # X, Y: [B, h, w], [B, l], ...
    sum_axis = tuple([i for i in range(2, len(X.shape) + 1)])
    XX = X.reshape(X.shape[0], 1, *X.shape[1:]) # XX指的是数据集中的其余点
    YY = Y.reshape(1, Y.shape[0], *Y.shape[1:]) # YY指的是reference point
    dist_mat = np.sqrt(np.sum(np.square(XX - YY), axis=sum_axis))
    dist_mat = np.where(dist_mat < 1e-10, 1e10, dist_mat)

    sorted_mat = np.sort(dist_mat, axis=1)
    r_max = sorted_mat[:, k-1].reshape(-1, 1)
    # est = - 1 / np.mean(np.log(sorted_mat[:, 0:k] / (r_max + 1e-10)), axis=1)
    mask = (dist_mat <= r_max).astype(np.float)

    est = -1 / (1 / k * np.sum(np.log(dist_mat) * mask, axis=1, keepdims=True) - np.log(r_max))
    print('LID CPU Time:',time.time()-start_time)
    return est.reshape(-1)


def gaussian_dist(x, mean, std):
    return 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-np.power((x - mean), 2) / (2 * std ** 2))


def lid(Mxy, k):
    eps_mat = torch.where(Mxy > 1e-20, torch.zeros((1, 1)), torch.ones((1, 1)) * 1e-20).detach()
    Mxy = Mxy + eps_mat
    value, idx = Mxy.topk(k=k, largest=False)
    mask = torch.zeros(Mxy.size()).type(Mxy.type())
    mask.scatter_(1, idx, 1.0)
    r_max = value[:, -1].detach()

    # est = -1 / (1. / k * torch.sum(torch.log(Mxy + eps_mat) * mask, dim=-1) - torch.log(r_max))
    est = -1 / (torch.mean(torch.log(value), dim=-1) - torch.log(r_max))
    return est


from keras import backend as K
import tensorflow as tf
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
    r = tf.reduce_sum(logits * logits, 1)
    # turn r into column vector
    r1 = K.reshape(r, [-1, 1])
    D = r1 - 2 * tf.matmul(logits, K.transpose(logits)) + K.transpose(r1) + \
        K.ones([batch_size, batch_size])

    # find the k nearest neighbor
    D1 = -K.sqrt(D)
    D2, _ = tf.nn.top_k(D1, k=k, sorted=True)
    D3 = -D2[:, 1:]  # skip the x-to-x distance 0 by using [,1:]

    m = K.transpose(tf.multiply(tf.transpose(D3), 1.0 / D3[:, -1]))
    v_log = tf.reduce_sum(K.log(m + epsilon), axis=1)  # to avoid nan
    lids = -k / v_log

    print('LID GPU Time:',time.time()-start_time)

    return K.eval(lids)
