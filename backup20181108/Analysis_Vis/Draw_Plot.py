import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import *


exp_path = Path('/unsullied/sharefs/ouyangzhihao/DataRoot/Exp/Tsinghua/LID_Research/Cifar10_epoch_Lid/LID_Epoch_NP/LIDEpoch_resNet_Cifar10_BS128_epochs20')

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
exit()

# import ipdb; ipdb.set_trace()
print(np.shape(train_lids_epoch))
plt.close()
##Box Pic
plt.boxplot(train_lids_epoch, patch_artist=True)
print("Finish Box")
plt.plot(x, y_average)
plt.savefig((pic_path/'box.pdf').__str__())

# plt.show()