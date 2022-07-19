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
    y_D2Net = [0.15, 0.195, 0.269, 0.32, 0.362, 0.396, 0.435, 0.473, 0.515, 0.536]
    xy_s_D2Net= smooth_xy(x_raw, y_D2Net)

    y_superglue = [0.22, 0.37, 0.539, 0.632, 0.683, 0.71, 0.74, 0.77, 0.79, 0.805]
    xy_s_superglue = smooth_xy(x_raw, y_superglue)

    y_COTR = [0.261, 0.42, 0.584, 0.669, 0.707, 0.7341, 0.759, 0.777, 0.791, 0.8]
    xy_s_COTR = smooth_xy(x_raw, y_COTR)

    y_SparseNCNet = [0.196, 0.31, 0.404, 0.478, 0.542, 0.590, 0.619, 0.642, 0.665, 0.680]
    xy_s_SparseNCNet = smooth_xy(x_raw, y_SparseNCNet)

    y_LoFTR = [0.32, 0.47, 0.62, 0.688, 0.731, 0.755, 0.782, 0.8062, 0.823, 0.837]
    xy_s_LoFTR = smooth_xy(x_raw, y_LoFTR)

    y_MSFAT = [0.35, 0.541, 0.659, 0.721, 0.751, 0.775, 0.792, 0.804, 0.818, 0.826]
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
    plt.plot(xy_s_COTR[0], xy_s_COTR[1], 'purple', linewidth=2, label='COTR') #, Marker='s'
    plt.plot(xy_s_LoFTR[0], xy_s_LoFTR[1], 'green', linewidth=2, label='LoFTR', linestyle='--') #, Marker='v'
    plt.plot(xy_s_SparseNCNet[0], xy_s_SparseNCNet[1], 'm', linewidth=2, label='Sparse-NCNet') #, Marker='*'
    plt.plot(xy_s_MSFAT[0], xy_s_MSFAT[1], 'r', linewidth=2, label='MSFA-T') #, Marker='*'



    plt.grid()  # 生成网格
    #plt.title("Weak texture", fontsize=16)  # 标题
    plt.legend(loc='lower right', fontsize=12)  # 标签显示位置和大小   upper, lower
    plt.subplots_adjust(top=0.976, bottom=0.126, right=0.878, left=0.127, hspace=0, wspace=0)
    plt.savefig("AUC_smooth_weak texture.png", dpi=500)
    plt.show()
