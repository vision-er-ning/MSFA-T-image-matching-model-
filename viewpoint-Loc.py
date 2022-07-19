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
    x_raw = [0, 0.5, 1, 1.5, 2, 2.5, 3]

#             0      0.5   1      1.5     2     2.5    3
    y_IBL = [0.16, 0.376, 0.576, 0.621, 0.6396, 0.649, 0.652]
    xy_s_IBL = smooth_xy(x_raw, y_IBL)
    #             0      0.5   1      1.5     2     2.5    3
    y_IBL_MSFA = [0.23, 0.426, 0.676, 0.763, 0.798, 0.821, 0.831]
    xy_s_IBL_MSFA = smooth_xy(x_raw, y_IBL_MSFA)


#             0      0.5   1      1.5     2     2.5    3
    y_HAIL = [0.048, 0.21, 0.47, 0.561, 0.599, 0.633, 0.6650]
    xy_s_HAIL = smooth_xy(x_raw, y_HAIL)
       #             0      0.5    1      1.5     2     2.5       3
    y_HAIL_MSFA = [0.125, 0.274, 0.632, 0.721, 0.742, 0.762, 0.775]
    xy_s_HAIL_MSFA = smooth_xy(x_raw, y_HAIL_MSFA)


    #             0     0.5   1      1.5     2     2.5    3
    y_EfiLoc = [0.06, 0.446, 0.686, 0.742, 0.758, 0.767, 0.777]
    xy_s_EfiLoc = smooth_xy(x_raw, y_EfiLoc)
    #                  0      0.5   1      1.5     2     2.5    3
    y_EfiLoc_MSFAT = [0.22, 0.48, 0.695, 0.76, 0.783, 0.805, 0.820]
    xy_s_EfiLoc_MSFAT = smooth_xy(x_raw, y_EfiLoc_MSFAT)


    # 原始折线图
    #plt.plot(x_raw, y_raw)
    #plt.show()
    config = {
        "font.family": 'Time New',  # 设置字体类型
        # "font.size": ,
        #     "mathtext.fontset":'stix',
    }
    rcParams.update(config)

    plt.xlabel('Localization Error (m)', size=16)  # 横坐标命名
    plt.ylabel('CDF', size=16)
    # x=np.arange(0.2,3)
    plt.axis([0.3, 3, 0, 1])


    # 处理后的平滑曲线
    plt.plot(xy_s_IBL[0], xy_s_IBL[1],'green',linewidth=2, label='IBL',linestyle='-.')#,marker='o'
    plt.plot(xy_s_IBL_MSFA[0], xy_s_IBL_MSFA[1], 'green', linewidth=2, label='IBL-MSFA-T')  # , Marker='v'

    plt.plot(xy_s_HAIL[0], xy_s_HAIL[1], 'b', linewidth=2, label='HAIL', linestyle='-.')#Marker='^'
    plt.plot(xy_s_HAIL_MSFA[0], xy_s_HAIL_MSFA[1], 'b', linewidth=2, label='HAIL-MSFA-T')  # , Marker='*'

    plt.plot(xy_s_EfiLoc[0], xy_s_EfiLoc[1], 'r', linewidth=2, label='EfiLoc', linestyle='-.')  # , Marker='s'
    plt.plot(xy_s_EfiLoc_MSFAT[0], xy_s_EfiLoc_MSFAT[1], 'r', linewidth=2, label='EfiLoc-MSFA-T') #, Marker='*'



    plt.grid()  # 生成网格
    #plt.title("Viewpoint", fontsize=16)  # 标题
    plt.legend(loc='lower right', fontsize=11)  # 标签显示位置和大小   upper, lower

    plt.subplots_adjust(top=0.976, bottom=0.126, right=0.878, left=0.127, hspace=0, wspace=0)
    plt.savefig("viewpoint-LOC-no.png", dpi=500)
    plt.show()
