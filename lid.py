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

# original code
def low_speed_LID(X, Y, k):
    # X, Y: [B, h, w], [B, l], ...
    sum_axis = tuple([i for i in range(2, len(X.shape) + 1)])
    XX = X.reshape(X.shape[0], 1, *X.shape[1:]) # XX指的是数据集中的其余点
    YY = Y.reshape(1, Y.shape[0], *Y.shape[1:]) # YY指的是reference point
    dist_mat = np.power(np.sum(np.power(XX - YY, 2), axis=sum_axis), 0.5)
    dist_mat = np.where(dist_mat < 1e-10, 1e10, dist_mat)

    sorted_mat = np.sort(dist_mat, axis=1)
    r_max = sorted_mat[:, k-1].reshape(-1, 1)
    # est = - 1 / np.mean(np.log(sorted_mat[:, 0:k] / (r_max + 1e-10)), axis=1)
    mask = (dist_mat <= r_max).astype(np.float)

    est = -1 / (1 / k * np.sum(np.log(dist_mat) * mask, axis=1, keepdims=True) - np.log(r_max))
    return est.reshape(-1)

def low_speed_LID_fast(X, Y, k):
    # X, Y: [B, h, w], [B, l], ...
    sum_axis = tuple([i for i in range(2, len(X.shape) + 1)])
    XX = X.reshape(X.shape[0], 1, *X.shape[1:]) # XX指的是数据集中的其余点
    YY = Y.reshape(1, Y.shape[0], *Y.shape[1:]) # YY指的是reference point

    # import ipdb; ipdb.set_trace()

    abs_XX_YY = abs(XX - YY) + 1e-7
    dist_mat = np.sum(abs_XX_YY, axis=sum_axis)
    dist_mat = np.where(dist_mat < 1e-10, 1e10, dist_mat)
    sorted_mat_idx = np.argsort(dist_mat, axis=1).flatten()

    r_max = np.square(abs_XX_YY[sorted_mat_idx[k-1]])
    dist_mat = np.square(abs_XX_YY[sorted_mat_idx[:k-1]])
    # dist_mat += 1e-7

    # dist_mat = np.power(np.sum(np.power(XX - YY, 2), axis=sum_axis), 0.5)
    # dist_mat = np.where(dist_mat < 1e-10, 1e10, dist_mat)
    #
    # sorted_mat = np.sort(dist_mat, axis=1)
    # r_max = sorted_mat[:, k-1].reshape(-1, 1)
    # # est = - 1 / np.mean(np.log(sorted_mat[:, 0:k] / (r_max + 1e-10)), axis=1)
    # mask = (dist_mat <= r_max).astype(np.float)

    est = -1 / (1 / k * np.sum(np.log(dist_mat), axis=1, keepdims=True) - np.log(r_max))
    return est.reshape(-1)

# using pytorch to accelerate
def LID(X, Y, k):
    # import ipdb;ipdb.set_trace()
    # X, Y: [B, h, w], [B, l], ...
    sum_axis = tuple([i for i in range(2, len(X.shape) + 1)])
    XX = X.reshape(X.shape[0], 1, *X.shape[1:]) # XX指的是数据集中的其余点
    YY = Y.reshape(1, Y.shape[0], *Y.shape[1:]) # YY指的是reference point
    # XX = torch.from_numpy(XX)
    # YY = torch.from_numpy(YY)
    dist_mat = torch.pow(torch.sum(torch.pow(XX - YY, 2), dim=sum_axis), 0.5)
    dist_mat = torch.where(dist_mat < 1e-10, torch.full_like(dist_mat,1e10), dist_mat)
    sorted_mat = torch.sort(dist_mat, dim=1)
    r_max = sorted_mat[0][:, k-1].reshape(-1, 1)
    mask = (dist_mat <= r_max).float()

    est = -1 / (1 / k * torch.sum(torch.log(dist_mat) * mask, dim=1) - torch.log(r_max).reshape(-1))

    return est.reshape(-1)

def LID_fast_torch(X, Y, k):
    # import ipdb;ipdb.set_trace()
    # X, Y: [B, h, w], [B, l], ...
    sum_axis = tuple([i for i in range(2, len(X.shape) + 1)])
    XX = X.reshape(X.shape[0], 1, *X.shape[1:]) # XX指的是数据集中的其余点
    YY = Y.reshape(1, Y.shape[0], *Y.shape[1:]) # YY指的是reference point

    abs_X_Y = (XX-YY).abs()
    dist_mat = torch.sum(abs_X_Y,dim=sum_axis)
    dist_mat = torch.where(dist_mat < 1e-10, torch.full_like(dist_mat, 1e10), dist_mat)
    sorted_mat = torch.sort(dist_mat, dim=1)
    r_max = sorted_mat[0][:, k - 1].reshape(-1, 1)
    mask = (dist_mat <= r_max).float()

    # r_max =

    dist_mat = torch.pow(torch.sum(torch.pow(XX - YY, 2), dim=sum_axis), 0.5)
    dist_mat = torch.where(dist_mat < 1e-10, torch.full_like(dist_mat,1e10), dist_mat)
    sorted_mat = torch.sort(dist_mat, dim=1)
    r_max = sorted_mat[0][:, k-1].reshape(-1, 1)
    mask = (dist_mat <= r_max).float()

    est = -1 / (1 / k * torch.sum(torch.log(dist_mat) * mask, dim=1) - torch.log(r_max).reshape(-1))

    return est.reshape(-1)

from progressbar import *
import time
pbar = ProgressBar()
def get_lid_by_batch(X, Y, lid_k, batch_size):
    start_time = time.time()
    train_num = len(X)
    batch_num = int( train_num / batch_size )
    final_dis = []
    for i in pbar(range(batch_num)):
        mask_batch = []
        if ((i + 1) * batch_size < train_num):
            mask_batch = np.arange(i * batch_size, (i + 1) * batch_size)  # 一个样本下标仅出现一次,顺序训练
        else:
            mask_batch = np.arange(i * batch_size, train_num)
        # import ipdb;ipdb.set_trace()
        dis = LID(X[mask_batch],Y[mask_batch], lid_k)
        final_dis.extend(dis)
    print('Total Time Cost:', time.time() - start_time)
    return np.array(final_dis)

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


if __name__ == '__main__':
    N = 10
    k = 5
    # uniform distribution
    X = np.linspace(-1, 1, 3).reshape(-1, 1)
    Y = np.random.randn(N).reshape(-1, 1)
    est = LID(X, Y, k)

    Mxy = distance(X, Y)
    print(est)
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(X.reshape(-1), est)
    # plt.ylim([0, 4])
    # plt.title('Uniform Distribution: N = {}, k = {}'.format(N, k))
    #
    # plt.subplot(212)
    # plt.hist(Y, bins=100, normed=True)
    # plt.plot(X.reshape(-1), np.exp(-est))
    # plt.plot([0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 0.0])
    #
    # # gaussian distribution
    # plt.figure(2)
    # X = np.arange(-5, 5, 0.01).reshape(-1, 1)
    # Y = np.random.randn(N).reshape(-1, 1)
    # est = LID(X, Y, k)
    # plt.subplot(211)
    # plt.plot(X.reshape(-1), est)
    # plt.ylim([0, 4])
    # plt.title('Gaussian Distribution: N = {}, k = {}'.format(N, k))
    #
    # plt.subplot(212)
    # plt.plot(X.reshape(-1), np.exp(-est))
    # z = gaussian_dist(X.reshape(-1), 0, 1)
    # plt.hist(Y, bins=100, normed=True)
    # plt.plot(X.reshape(-1), z)
    # plt.ylim([0, 1])
    # plt.show()



