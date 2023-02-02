# -*- coding: utf-8 -*-

"""
Created on 06/20/2022
matplot_utils.
@author: AnonymousUser314156
"""
import torch
import numpy as np


# 卷积平滑
def conv_smooth(x, y, box_pts=5):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return x, y_smooth


# 差值平滑
def inter_smooth(x, y, t=300):
    from scipy.interpolate import make_interp_spline
    x_smooth = np.linspace(x.min(), x.max(), t)
    y_smooth = make_interp_spline(x, y)(x_smooth)
    return x_smooth, y_smooth


def is_outlier(points, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


# # 忽略异常值，限制坐标轴
# def ignore_outliers(x):
#     return x[~is_outlier(x)]


class PltClass(object):

    def __init__(self, xpixels=500, ypixels=500, dpi=None, nrows=1, ncols=1):
        import matplotlib.pyplot as plt
        plt.rcParams['axes.unicode_minus'] = False  # -
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rc('font', family='Times New Roman')
        if dpi is None:
            fig, ax = plt.subplots(nrows, ncols)
        else:
            xinch = xpixels / dpi
            yinch = ypixels / dpi
            fig, ax = plt.subplots(nrows, ncols, figsize=(xinch, yinch))
        self.plt = plt
        self.ax = ax
        self.color = ['red', 'orange', 'yellow', 'green', 'blue', 'cyan', 'purple', 'pink', 'coral', 'gold', 'lime', 'navy', 'teal', 'indigo']
        # import matplotlib.colors as mcolors
        # self.cs = list(mcolors.CSS4_COLORS.keys())
        # self.color = mcolors.CSS4_COLORS

        self.y_max = 0
        self.y_min = 1000

    def plt_score(self, score, layer=1, cnt=0, label=''):
        # for l, (m, s) in enumerate(score.items()):
        #     if l == layer:
        #         if cnt == 0:
        #             _s, _ind = torch.sort(torch.cat([torch.flatten(s)]), descending=True)
        #             _sam = np.arange(0, _s.shape[0], 100)
        #             np_s = _s[_sam].cpu().detach().numpy()
        #             self.plt.plot(range(1, len(np_s) + 1, 1), np_s, color=self.color[cnt], label=label)
        #             self.ind = _ind[_sam]
        #         else:
        #             _s = torch.cat([torch.flatten(s)])[self.ind]
        #             np_s = _s.cpu().detach().numpy()
        #             self.plt.scatter(range(1, len(np_s) + 1, 1), np_s, color=self.color[cnt], s=10, alpha=0.7, label=label)
        #         # _s, _ind = torch.sort(torch.cat([torch.flatten(s)]), descending=True)
        #         # np_s = _s.cpu().detach().numpy()
        #         # self.plt.plot(range(1, len(np_s) + 1, 1), np_s, color=self.color[cnt], label=label)
        #         # self.plt.plot(range(1, len(np_s) + 1, 1), np_s, color=self.color[self.cs[cnt]], label=label)
        #         # self.plt.axvline(np.argwhere(np_s > min(np_s))[-1], color=self.color[iter], linestyle=':')
        gain = 10e5
        if cnt == 0:
            _s, _ind = torch.sort(torch.cat([torch.flatten(x) for x in score.values()]), descending=True)
            _sam = np.arange(0, _s.shape[0], 500)
            np_s = _s[_sam].cpu().detach().numpy()*gain
            self.ax.plot(range(1, len(np_s) + 1, 1), np_s, color=self.color[cnt], linewidth=2, label=label)
            self.ind = _ind[_sam]
            # self.ignore_outliers(np_s)
        else:
            _s = torch.cat([torch.flatten(x) for x in score.values()])[self.ind]
            np_s = _s.cpu().detach().numpy()*gain
            self.ax.scatter(range(1, len(np_s) + 1, 1), np_s, color=self.color[cnt], s=10, alpha=0.7, label=label)
            # self.ignore_outliers(np_s)

    def plt_inset_axes(self, score, layer=1, cnt=0):
        # === 插图 ===
        if cnt == 0:
            from mpl_toolkits.axes_grid1.inset_locator import mark_inset
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            # 嵌入坐标系
            axins = inset_axes(self.ax, width="40%", height="40%", loc='lower left',
                               bbox_to_anchor=(0.55, 0.05, 1, 1),
                               bbox_transform=self.ax.transAxes)
            self.axins = axins

        # 在子坐标系中绘制原始数据
        gain = 10e6
        if cnt == 0:
            # _s, _ind = torch.sort(torch.cat([torch.flatten(x) for x in score.values()]), descending=True)
            # _sam = np.arange(0, _s.shape[0], 500)
            # np_s = _s[_sam].cpu().detach().numpy()*gain
            _s = torch.cat([torch.flatten(x) for x in score.values()])[self.ind]
            np_s = _s.cpu().detach().numpy()*gain
            self.axins.plot(range(1, len(np_s) + 1, 1), np_s, color=self.color[cnt])
        else:
            _s = torch.cat([torch.flatten(x) for x in score.values()])[self.ind]
            np_s = _s.cpu().detach().numpy()*gain
            self.axins.scatter(range(1, len(np_s) + 1, 1), np_s, color=self.color[cnt], s=10, alpha=0.7)

    def plt_end(self):
        fontsize = 20
        # self.plt.xscale('log')
        self.ax.axhline(0, color='gray', linestyle=':')
        self.ax.legend(fontsize=fontsize, loc='upper right')
        self.ax.tick_params(labelsize=fontsize)
        # self.ax.set_ylim([self.y_min, self.y_max])
        # self.ax.set_ylim([-0.5, 10])
        self.ax.set_ylabel('10e-5', fontsize=fontsize)

        self.axins.set_yscale('log')
        self.axins.tick_params(labelsize=fontsize * 0.5)
        self.axins.axes.xaxis.set_ticks([])
        # self.axins.set_xlim(xlim0, xlim1)
        # self.axins.set_ylim(ylim0, ylim1)
        # self.axins.set_xlabel('log')
        self.axins.set_title('log scale', fontsize=fontsize * 0.5)

        self.plt.show()

    # 忽略异常值，限制坐标轴
    def ignore_outliers(self, x):
        _y = x[~is_outlier(x, 1)]
        max = _y.max()
        min = _y.min()
        if max > self.y_max:
            self.y_max = max
        if min < self.y_min:
            self.y_min = min


class PlotPanning(PltClass):
    def __init__(self, xpixels=500, ypixels=500, dpi=None, nrows=1, ncols=1):
        super(PlotPanning, self).__init__(xpixels, ypixels, dpi, nrows, ncols)
        self.a1, self.a2, self.a3 = [], [], []
        self.s1, self.s2, self.s3 = [], [], []
        self.step = []

    def append(self, a, s):
        self.a1.append(a[0])
        self.a2.append(a[1])
        self.a3.append(a[2])
        self.s1.append(s[0])
        self.s2.append(s[1])
        self.s3.append(s[2])

    def plot(self):
        self.ax.plot(self.a1, color=self.color[0], linewidth=2, label='a1', alpha=0.7)
        self.ax.plot(self.a2, color=self.color[3], linewidth=2, label='a2', alpha=0.7)
        self.ax.plot(self.a3, color=self.color[4], linewidth=2, label='a3', alpha=0.7)

        ax2 = self.ax.twinx()
        ax2.plot(self.s1, color=self.color[0], linewidth=2, linestyle=':', label='s3', alpha=0.7)
        ax2.plot(self.s2, color=self.color[3], linewidth=2, linestyle=':', label='s4', alpha=0.7)
        ax2.plot(self.s3, color=self.color[4], linewidth=2, linestyle=':', label='s6', alpha=0.7)

        fontsize = 20
        # self.plt.xscale('log')
        # self.ax.axhline(0, color='gray', linestyle=':')
        # self.ax.legend(fontsize=fontsize, loc='upper right')

        self.ax.set_ylabel("Action", fontsize=fontsize)
        self.ax.tick_params(labelsize=fontsize)
        ax2.set_ylabel("Reward", fontsize=fontsize)
        ax2.tick_params(labelsize=fontsize)
        # self.plt.title('VGG19/CIFAR10', fontsize=fontsize)
        self.plt.title('LeNet5/MNIST', fontsize=fontsize)
        self.plt.grid()
        self.plt.show()

        # file_name = 'vgg_r9999'
        # import xlwt, xlrd
        # f = xlwt.Workbook()  # 创建一个工作簿
        # sheet1 = f.add_sheet('sheet1', cell_overwrite_ok=True)  # 创建sheet
        # row0 = [i+1 for i in range(100)]
        # column0 = ["a1", "a2", "a3", "s1", "s2", "s3"]
        # for i in range(0, len(row0)):
        #     sheet1.write(0, i+1, row0[i])
        # for i in range(0, len(column0)):
        #     sheet1.write(i+1, 0, column0[i])
        #
        # for i in range(0, len(self.a1)):
        #     sheet1.write(1, i+1, float(self.a1[i]))
        #     sheet1.write(2, i+1, float(self.a2[i]))
        #     sheet1.write(3, i+1, float(self.a3[i]))
        #     sheet1.write(4, i+1, float(self.s1[i]))
        #     sheet1.write(5, i+1, float(self.s2[i]))
        #     sheet1.write(6, i+1, float(self.s3[i]))

        # f.save(f'./runs/policy/{file_name}.xls')

    # def append_xls(self, file_name='vgg_r9999'):
    def append_xls(self, file_name='lenet_r9999'):

        import xlwt, xlrd
        data1 = xlrd.open_workbook(f'../runs/policy/{file_name}.xls')
        table = data1.sheets()[0]
        self.a1 = table.row_values(1, start_colx=1, end_colx=None)
        self.a2 = table.row_values(2, start_colx=1, end_colx=None)
        self.a3 = table.row_values(3, start_colx=1, end_colx=None)
        self.s1 = table.row_values(4, start_colx=1, end_colx=None)
        self.s2 = table.row_values(5, start_colx=1, end_colx=None)
        self.s3 = table.row_values(6, start_colx=1, end_colx=None)

    def end(self):
        pass


if __name__ == '__main__':
    fig = PlotPanning(600, 300, 60)

    fig.append_xls()
    fig.plot()

    # ratio = []
    # for i in range(100):
    #     keep_ratio = (1.0 - 0.9999) ** ((i + 1) / 100)
    #     ratio.append(keep_ratio)
    # fig.ax.plot(ratio, color=fig.color[6], linewidth=2, alpha=0.7)
    #
    # fontsize = 20
    #
    # title = ['$a_1$','$a_2$','$a_3$','$r_1$','$r_2$','$r_3$']
    # cl = [0, 3, 4]
    # mlines_method1 = [fig.plt.plot([], [], color=fig.color[cl[i]], linewidth=2, label=title[i])[0] for i in range(3)]
    # mlines_method2 = [fig.plt.plot([], [], color=fig.color[cl[i]], linewidth=2, linestyle=':', label=title[i+3])[0] for i in range(3)]
    # patches = mlines_method1 + mlines_method2
    # # fig.plt.legend(handles=patches, fontsize=fontsize)
    # fig.ax.legend(handles=mlines_method1, fontsize=fontsize, loc='lower left', title='Action', title_fontsize=fontsize)
    # from matplotlib.legend import Legend
    # leg = Legend(fig.ax, mlines_method2, title[3:6], loc='upper right', frameon=True, fontsize=fontsize, title='Reward', title_fontsize=fontsize)
    # fig.ax.add_artist(leg)
    # # fig.ax.legend(handles=mlines_method2, fontsize=fontsize, loc='upper right')
    #
    # fig.ax.set_yscale('log')
    # fig.ax.tick_params(labelsize=fontsize)
    # fig.plt.title('Remaining ratio', fontsize=fontsize)
    # fig.plt.grid()  # 生成网格
    # fig.plt.show()
