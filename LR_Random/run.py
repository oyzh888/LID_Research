import os

dataset_names = ['CIFAR10','CIFAR100']
# dataset_names = ['CIFAR10']
# batch_size = [32, 64, 128, 256, 512]  # orig paper trained all networks with batch_size=128
batch_size = [128]  # orig paper trained all networks with batch_size=128
epochs = 20
# optimizers = ['Adam','SGD','RMSprop', 'Adagrad']
optimizers = ['Adam']
# distribution_methods = ['RL','U','N','Base']
distribution_methods = ['Base','U']
dis_parameter1 = 0.2
dis_parameter2 = 0.8
work_path_name = 'datasets'

# rlaunch = 'rlaunch --cpu=1 --memory=4000 --gpu=1 --preemptible=no '
rlaunch = '' #With bash source
for dataset_name in dataset_names:
    for bs in batch_size:
        for optimizer in optimizers:
            for distribution_method in distribution_methods:
                cmd = rlaunch + 'python3 LR_Exp.py %s %d %d %s %s %f %f %s' \
                      % (dataset_name, bs, epochs, optimizer, distribution_method, dis_parameter1, dis_parameter2, work_path_name)
                os.system(cmd)