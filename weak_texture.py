from cProfile import label
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pylab import *
from pandas import DataFrame,Series
from scipy.interpolate import make_interp_spline
#file = open('C:/Users/Administrator/Desktop/语义定位手稿/gyr_synthetic_tran-2.txt')
from scipy.interpolate import spline

x_D2Net=[1,2,3,4,5,6,7,8,9,10]
y_D2Net=[0.15, 0.195, 0.269, 0.32, 0.362, 0.383, 0.403, 0.473, 0.515, 0.536]
#x_D2Net_new=np.linspace(min(x_D2Net),max(x_D2Net),300)
#y_smooth_D2Net=make_interp_spline(x_D2Net,y_D2Net,x_D2Net_new)

x_superglue=[1,2,3,4,5,6,7,8,9,10]
y_superglue= [0.22, 0.37, 0.539, 0.62, 0.683, 0.71, 0.74, 0.77, 0.79, 0.805]
#x_superglue_new=np.linspace(x_superglue.min(),x_superglue.max(),300)
#y_smooth_superglue=make_interp_spline(x_superglue,y_superglue,x_superglue_new)


x_COTR=[1,2,3,4,5,6,7,8,9,10]
y_COTR= [0.261, 0.42, 0.584, 0.669, 0.707, 0.741, 0.755, 0.777, 0.796, 0.8]
#x_COTR_new=np.linspace(x_COTR.min(),x_COTR.max(),300)
#y_smooth_COTR=make_interp_spline(x_COTR,y_COTR,x_COTR_new)


x_SparseNCNet=[1,2,3,4,5,6,7,8,9,10]
y_SparseNCNet= [0.196, 0.31, 0.385, 0.459, 0.542, 0.578, 0.616, 0.642, 0.665, 0.680]
#x_SparseNCNet_new=np.linspace(x_SparseNCNet.min(),x_SparseNCNet.max(),300)
#y_smooth_SparseNCNet=make_interp_spline(x_SparseNCNet,y_SparseNCNet,x_SparseNCNet_new)


x_LoFTR=[1,2,3,4,5,6,7,8,9,10]
y_LoFTR= [0.32, 0.47, 0.62, 0.688, 0.731, 0.755, 0.772, 0.8062, 0.823, 0.837]
#x_LoFTR_new=np.linspace(x_LoFTR.min(),x_LoFTR.max(),300)
#y_smooth_LoFTR=make_interp_spline(x_LoFTR,y_LoFTR,x_LoFTR_new)


x_MSFAT=[1,2,3,4,5,6,7,8,9,10]
y_MSFAT= [0.35, 0.541, 0.659, 0.71, 0.751, 0.775, 0.782, 0.804, 0.818, 0.826]
#x_MSFAT_new=np.linspace(x_MSFAT.min(),x_MSFAT.max(),300)
#y_smooth_MSFAT=make_interp_spline(x_MSFAT,y_MSFAT,x_MSFAT_new)


# for line in file:
#     if line != "\n":
#         data.append(line)
# print(len(data))
#data = file.readlines()
#画图
#figure(figsize=(16,8),dpi=200)
rcParams['font.sans-serif']=['SimHei'] #显示中文字符，SimHei黑体，FangSong仿宋
rcParams['axes.unicode_minus'] = False

config = {
    "font.family":'Time New',  # 设置字体类型
    #"font.size": ,
#     "mathtext.fontset":'stix',
}
rcParams.update(config)

plt.xlabel('Threshold (px)',size = 20)#横坐标命名
plt.ylabel('AUC',size = 20)
# x=np.arange(0.2,3)
plt.axis([1,10,0,1])

#plt.plot(data0_1000,'b',label='Gyroscope data')
#plt.plot(data1007_2025,'b',label='Gyroscope data')
plt.plot(x_D2Net,y_D2Net,'b',label='D2Net',linestyle='-.',marker='o')
plt.plot(x_superglue,y_superglue,'y',label='SuperGlue',linestyle='--',Marker='^')
plt.plot(x_COTR,y_COTR,'purple',label='COTR',linestyle='-',Marker='s')
plt.plot(x_LoFTR,y_LoFTR,'green',label='LoFTR',linestyle=':',Marker='v')
plt.plot(x_SparseNCNet,y_SparseNCNet,'m',label='Sparse-NCNet',linestyle='-.',Marker='^')
plt.plot(x_MSFAT,y_MSFAT,'r',label='MSFAT',Marker='*')

#plt.gcf().set_facecolor(np.ones(3)*240/255)   #生成画布网格大小
plt.grid()  #生成网格
#plt.title("weak texture",fontsize=16)#标题
plt.legend(loc='lower right',fontsize=12)#标签显示位置和大小   upper, lower
plt.subplots_adjust(top=0.976, bottom=0.126, right=0.878, left=0.127, hspace=0, wspace=0)
plt.savefig("AUC_weak texture.png", dpi = 500)
plt.show()

# markeredgecolor 或 mec 标记边缘颜色
# markeredgewidth 或 mew 标记边缘宽度
# markerfacecolor 或 mfc 标记面颜色
# markerfacecoloralt 或 mfcalt
# markersize 或 ms 标记大小