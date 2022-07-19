import numpy as np
#from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from cProfile import label
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
import random
from pylab import *
from pandas import DataFrame,Series


def smooth_xy(lx, ly):
    """数据平滑处理

    :param lx: x轴数据，数组
    :param ly: y轴数据，数组
    :return: 平滑后的x、y轴数据，数组 [slx, sly]
    """
    x = np.array(lx)
    y = np.array(ly)
    x_smooth = np.linspace(x.min(), x.max(), 1000)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]


if __name__ == '__main__':
    x_raw = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_D2Net = [0.103, 0.22, 0.356, 0.41, 0.44, 0.461, 0.473, 0.482, 0.49, 0.4998]
    xy_s_D2Net= smooth_xy(x_raw, y_D2Net)

    y_superglue = [0.22, 0.35, 0.47, 0.533, 0.578, 0.606, 0.633, 0.661, 0.6796, 0.686]
    xy_s_superglue = smooth_xy(x_raw, y_superglue)

    y_COTR = [0.31, 0.408, 0.49, 0.556, 0.597, 0.631, 0.663, 0.687, 0.706, 0.723]
    xy_s_COTR = smooth_xy(x_raw, y_COTR)

    y_SparseNCNet = [0.142, 0.27, 0.337, 0.38, 0.406, 0.43, 0.45, 0.463, 0.478, 0.492]
    xy_s_SparseNCNet = smooth_xy(x_raw, y_SparseNCNet)

    y_LoFTR = [0.37, 0.44, 0.535, 0.618, 0.683, 0.725, 0.755, 0.77, 0.782, 0.7937]
    xy_s_LoFTR = smooth_xy(x_raw, y_LoFTR)

    y_MSFAT = [0.34, 0.471, 0.6, 0.685, 0.721, 0.735, 0.748, 0.759, 0.765, 0.77]
    xy_s_MSFAT = smooth_xy(x_raw, y_MSFAT)


    # 原始折线图
    #plt.plot(x_raw, y_raw)
    #plt.show()
    config = {
        "font.family": 'Time New',  # 设置字体类型
        # "font.size": ,
        #     "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    plt.xlabel('Threshold (px)', size=16)  # 横坐标命名
    plt.ylabel('AUC', size=16)
    # x=np.arange(0.2,3)
    plt.axis([1, 10, 0, 1])


    # 处理后的平滑曲线
    plt.plot(xy_s_D2Net[0], xy_s_D2Net[1],'b',linewidth=2, label='D2-Net', linestyle='-.')#,marker='o'
    plt.plot(xy_s_superglue[0], xy_s_superglue[1], 'y', linewidth=2, label='SuperGlue', linestyle='--')#Marker='^'
    plt.plot(xy_s_COTR[0], xy_s_COTR[1], 'purple', linewidth=2, label='COTR', linestyle='-.') #, Marker='s'
    plt.plot(xy_s_LoFTR[0], xy_s_LoFTR[1], 'green', linewidth=2, label='LoFTR', linestyle='--') #, Marker='v'
    plt.plot(xy_s_SparseNCNet[0], xy_s_SparseNCNet[1], 'm', linewidth=2, label='Sparse-NCNet') #, Marker='*'
    plt.plot(xy_s_MSFAT[0], xy_s_MSFAT[1], 'r', linewidth=2, label='MSFA-T') #, Marker='*'



    plt.grid()  # 生成网格
    #plt.title("Wide baseline view", fontsize=16)  # 标题
    plt.legend(loc='lower right', fontsize=11)  # 标签显示位置和大小   upper, lower
    plt.subplots_adjust(top=0.976, bottom=0.126, right=0.878, left=0.127, hspace=0, wspace=0)
    plt.savefig("AUC_smooth_widebaseline.png", dpi=500)
    plt.show()
