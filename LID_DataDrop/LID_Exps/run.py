import os

dataset_name = ['MNIST', 'CIFAR10']
model_name = ['resNet', 'Xception']
lid_method = ['lid_high', 'lid_low', 'random']
drop_percent = [1,2,3,4,5,10,20,50]


dataset_name = 'CIFAR10'
model_name = 'resNet'
lid_method = 'random'
drop_percent = drop_percent[:1]

rlaunch = 'rlaunch --cpu=1 --memory=4000 --gpu=1 --preemptible=no '
for percent in drop_percent:
    cmd = rlaunch + 'python3 LID_Data_Drop_Exp.py %s %s %s %d' % (dataset_name, model_name, lid_method, percent)
    os.system(cmd)