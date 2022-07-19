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
    x_smooth = np.linspace(x.min(), x.max(), 50)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return [x_smooth, y_smooth]


if __name__ == '__main__':
    x_raw = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y_D2Net = [0.07, 0.178, 0.285, 0.38, 0.452, 0.486, 0.499, 0.505, 0.509, 0.513]
    xy_s_D2Net= smooth_xy(x_raw, y_D2Net)

    y_superglue = [0.175, 0.345, 0.462, 0.546, 0.599, 0.642, 0.67, 0.69, 0.698, 0.705]
    xy_s_superglue = smooth_xy(x_raw, y_superglue)

    y_COTR = [0.2, 0.37, 0.49, 0.571, 0.625, 0.67, 0.696, 0.71, 0.721, 0.73]
    xy_s_COTR = smooth_xy(x_raw, y_COTR)

    y_SparseNCNet = [0.2036, 0.33, 0.439, 0.516, 0.57, 0.6, 0.620, 0.632, 0.642, 0.652]
    xy_s_SparseNCNet = smooth_xy(x_raw, y_SparseNCNet)

    y_LoFTR = [0.3, 0.45, 0.549, 0.618, 0.676, 0.713, 0.74, 0.759, 0.766, 0.773]
    xy_s_LoFTR = smooth_xy(x_raw, y_LoFTR)

    y_MSFAT = [0.316, 0.5, 0.592, 0.652, 0.6943, 0.724, 0.739, 0.745, 0.75, 0.753]
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
    #plt.title("Viewpoint", fontsize=16)  # 标题
    plt.legend(loc='lower right', fontsize=11)  # 标签显示位置和大小   upper, lower
    plt.subplots_adjust(top=0.976, bottom=0.126, right=0.878, left=0.127, hspace=0, wspace=0)
    plt.savefig("AUC_smooth_viewpoint.png", dpi=500)
    plt.show()
