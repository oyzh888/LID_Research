import os

# dataset_names = ['CIFAR10','CIFAR100']
dataset_names = ['CIFAR10']
batch_size = [2048, 1024, 256, 128, 32 ]  # orig paper trained all networks with batch_size=128
# batch_size = [2048,256,32]  # orig paper trained all networks with batch_size=128
# batch_size = [1024]  # orig paper trained all networks with batch_size=128
epochs = 200
# optimizers = ['Adam','SGD','RMSprop', 'Adagrad']
optimizers = ['SGD']
# distribution_methods = ['RL','U','N','Base']
distribution_methods = ['Base']
dis_parameter1 = 0.2
dis_parameter2 = 0.8
init_lr = 1e-3
work_path_name = 'all_12_9'
# rlaunch = 'rlaunch --cpu=1 --memory=4000 --gpu=1 --preemptible=no bash'
# python3 run.py'
rlaunch = 'rlaunch --cpu=2 --memory=4096 --gpu=1 --preemptible=no '
# rlaunch = '' #With bash source
for dataset_name in dataset_names:
    for bs in batch_size:
        for optimizer in optimizers:
            for distribution_method in distribution_methods:
                cmd = rlaunch + 'python3 LR_Exp.py %s %d %d %s %s %f %f %s %f' \
                      % (dataset_name, bs, epochs, optimizer, distribution_method, dis_parameter1, dis_parameter2, work_path_name, init_lr)
                os.system(cmd)