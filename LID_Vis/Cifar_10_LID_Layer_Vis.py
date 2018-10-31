from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import numpy as np
import gc
import os
from pathlib import *
import matplotlib.pyplot as plt

'''
本程序使用Cifar_10_BaseLine_LID_Layer_Depth_Analysis.py生成的各层的LID值，绘制层数渐变时的LID分布箱型图。
'''
work_path=Path('../../Cifar10_Layer_LID/Layer_LID_nparray')
pys = work_path.glob('*LID.npy')
file_names=[]
for lid_file in pys:
    file_names.append(lid_file)
file_names.sort()
layers_lid = []
for file_name in file_names:
    lid = np.load(file_name.__str__())
    layers_lid.append(lid)
# import ipdb;
# ipdb.set_trace()
exp_name = 'Vis_Layer_Lid'
# print(file_names)

# outputs_predict_lid=np.load("./nparray/"+exp_name+".npy")
#
# save_dir = os.path.join(os.getcwd(), 'picture')
# if not os.path.isdir(save_dir):
#     os.makedirs(save_dir)
# plt.xlabel('sample idx')
# plt.ylabel('lid')
# plt.title(exp_name+' bar plot')
# plt.bar(range(len(outputs_predict_lid)),outputs_predict_lid)
# plt.savefig('./picture/'+exp_name+'_bar_plot_average_pooling2d_1.jpg')  # bar图片的存储
# plt.close()
#
# print("SUCCESSFULLY SAVE FIG BAR PLOT!")
#
# plt.title(exp_name+' box plot')
# plt.boxplot(outputs_predict_lid)
# plt.savefig('./picture/'+exp_name+'_box_plot_average_pooling2d_1.jpg')  # boxplot图片的存储
# plt.close()
#
# print("SUCCESSFULLY SAVE BOX PLOT! ")


print(np.shape(layers_lid))

x = np.arange(start=1, stop=len(layers_lid)+1, step=1)
y_average = np.average(layers_lid,axis=1)
y_variance = np.var(layers_lid,axis=1)
label = np.load(work_path/'layers_ordial_name.npy'.__str__())
label = label.tolist()
plt.subplot(211)
# import ipdb;ipdb.set_trace()
plt.title('Layer Average LID')
plt.ylabel('Average LID')
plt.xticks(x,[i for i in label])
# plt.xticks(label.tolist())   # x下标值自定义
plt.xlabel('Layer')
plt.plot(x, y_average)

plt.subplots_adjust(wspace=0, hspace=1)  # 调整子图间距

plt.subplot(212)
plt.title('Variance LID')
plt.xlabel('Layer')
# plt.xticks(label.tolist())

plt.xticks(x,[i for i in label])
plt.ylabel('variance')
plt.plot(x, y_variance)

picPath=Path('../../Cifar10_Layer_LID/picture')
Path(picPath).mkdir(parents=True, exist_ok=True)
plt.savefig((picPath/(exp_name+'.pdf')).__str__())  # boxplot图片的存储
plt.close()

# plt.setp(axes, xticks=[1,2,3],
#          xticklabels=['x1', 'x2', 'x3'])
import ipdb;ipdb.set_trace()
plt.xticks(np.arange(len(label)),[i for i in label])
plt.boxplot(layers_lid,patch_artist=True) #描点上色
plt.plot(x, y_average)
plt.savefig((picPath/(exp_name+'Boxplot'+'.pdf')).__str__())  # boxplot图片的存储