#!/usr/bin/env 
#coding:utf-8

""" 全体模块的导入 """
from PIL import Image
from numpy import *
from pylab import *

""" 纯化的，按照需求来导入所需的包，如 """
# import numpy as np
# import matplotlib.pyplot as plt

from scipy.ndimage import filters
def compute_harris_response(im, sigma=3):
    """ 在一幅灰度图像中，对没个像素计算Harris角点检测器响应函数 """

    # 计算导数
    imx = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imy)

    # 计算Harris矩阵的分量
    wxx = filters.gaussian_filter(imx**2, sigma)
    wxy = filters.gaussian_filter(imx*imy, sigma)
    wyy = filters.gaussian_filter(imy**2, sigma)

    # 计算特征值和迹
    Wdet = wxx*wxy - wxy**2
    Wtr = wxx + wxy

    return Wdet / Wtr

def get_harris_points(harrism, min_dist=10, threshold=0.1):
    """ 从一幅Harris响应图像中返回角点， min_dist为分割角点和图像边界的最少像素数目 """

    # 寻找高于阈值的候选角点
    corner_threshold = harrism.max() * threshold
    harrism_t = (harrism > corner_threshold)*1

    # 得到候选点的坐标
    coords = array(harrism_t.nonzero()).T

    # 以及他们的Harris响应值
    candidate_values = [harrism[c[0],c[1]] for c in coords]

    # 对候选点按照Harris响应值进行排序
    index = argsort(candidate_values)

    # 将可行点多位置保存到数组中
    allowws_locations = zeros(harrism.shape)
    allowws_locations[min_dist:-min_dist, min_dist:-min_dist] = 1

    # 按照 min_dist 原则，选择最佳 Harris 点
    filtered_coords = []
    for i in index:
        if allowws_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowws_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist), \
                    (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0

    return filtered_coords

def plot_harris_points(image, filtered_coords):
    """ 绘制图像中检测到的角点 """

    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    axis('off')
    show()




