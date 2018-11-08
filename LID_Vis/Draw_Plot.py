import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import *


exp_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/Tsinghua/Logs/Cifar10_epoch_Lid/LID_Epoch_NP/conv2d_17_LIDEpoch_resNet_Cifar10_BS128_epochs100')

train_np_path = exp_path/'train_lid.npy'
test_np_path = exp_path/'test_lid.npy'
train_lids_epoch = np.load(train_np_path)
test_lids_epoch = np.load(test_np_path)

x = np.arange(start=1, stop=len(train_lids_epoch)+1, step=1)
y_average_train = np.average(train_lids_epoch,axis=1)
y_average_test = np.average(test_lids_epoch,axis=1)
y_variance_train = np.var(train_lids_epoch,axis=1)
y_variance_test = np.var(test_lids_epoch,axis=1)

plt.subplot(211)
plt.title('Average LID')
# plt.ylim(ymin=0,ymax=max(y_average)*1.1)

plt.xlabel('epochs')
plt.ylabel('average')
plt.plot(x, y_average_train,label="train")
plt.plot(x, y_average_test,color="#DB7093",linestyle="--", label="test")
plt.legend(loc='upper right')

plt.subplots_adjust(wspace=0, hspace=1)  # 调整子图间距

plt.subplot(212)
plt.title('Variance LID')
plt.xlabel('epochs')
plt.ylabel('variance')
plt.plot(x, y_variance_train,label="train")
plt.plot(x, y_variance_test,color="#DB7093",linestyle="--",label="test")

plt.legend(loc='upper right')

pic_path = exp_path/'picture'
pic_path.mkdir(parents=True,exist_ok=True)

plt.savefig((pic_path/'test_train_average_variance.pdf').__str__())
# exit()

# import ipdb; ipdb.set_trace()

plt.close()
##Box Pic

# import ipdb; ipdb.set_trace()
# print(train_lids_epoch[0])
# print(train_lids_epoch[0][0])
# print(train_lids_epoch.shape)
train_lids_epoch = train_lids_epoch[0][:50000]
# for var in train_lids_epoch:
print(np.shape(x))
print(np.shape(train_lids_epoch))
# import ipdb; ipdb.set_trace()
# ans = map(str, x)
# print('ans:',ans)
ans = ['1','2','3']


plt.xticks(np.arange(len(x)),[i for i in x])
plt.boxplot(train_lids_epoch,patch_artist=True) #描点上色

# plt.boxplot(train_lids_epoch, labels=ans)
print("Finish Box")
plt.plot(x, y_average_train)
plt.savefig((pic_path/'box.pdf').__str__())

# plt.show()