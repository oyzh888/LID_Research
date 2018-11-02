import os

dataset_name = ['CIFAR10']
model_name = ['resNet']
drop_percent = [0,1,5,10]

model_name = 'resNet'
lid_method = 'lid_high'
drop_percent = drop_percent[:1]

rlaunch = 'rlaunch --cpu=1 --memory=4000 --gpu=1 --preemptible=no '
for percent in drop_percent:
    cmd = rlaunch + 'python3 LID_Data_Drop_Exp.py %s %s %s %d' % (dataset_name, model_name, lid_method, percent)
    os.system(cmd)
