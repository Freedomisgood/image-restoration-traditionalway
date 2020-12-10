# -*- coding: utf-8 -*-
# @Time    : 2020/12/9 15:11
# @Author  : Mrli
# @FileName: utils.py
# @Blog    : https://nymrli.top/
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def read_image(img_path):
    """
    读取图片，图片是以 np.array 类型存储
    :param img_path: 图片的路径以及名称
    :return: img np.array 类型存储
    """
    # 读取图片
    img = cv2.imread(img_path)

    # 如果图片是三通道，采用matplotlib展示图像时需要先转换通道
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_image(filename, image):
    """
    将np.ndarray 图像矩阵保存为一张 png 或 jpg 等格式的图片
    :param filename: 图片保存路径及图片名称和格式
    :param image: 图像矩阵，一般为np.array
    :return:
    """
    # np.copy() 函数创建一个副本。
    # 对副本数据进行修改，不会影响到原始数据，它们物理内存不在同一位置。
    img = np.copy(image)

    # 从给定数组的形状中删除一维的条目
    img = img.squeeze()

    # 将图片数据存储类型改为 np.uint8
    if img.dtype == np.double:
        # 若img数据存储类型是 np.double ,则转化为 np.uint8 形式
        img = img * np.iinfo(np.uint8).max

        # 转换图片数组数据类型
        img = img.astype(np.uint8)

    # 将 RGB 方式转换为 BGR 方式
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 生成图片
    cv2.imwrite(filename, img)


def plot_image(image, image_title, is_axis=False):
    """
    展示图像
    :param image: 展示的图像，一般是 np.array 类型
    :param image_title: 展示图像的名称
    :param is_axis: 是否需要关闭坐标轴，默认展示坐标轴
    :return:
    """
    # 解决图片中无法显示中文的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 展示图片
    plt.imshow(image)

    # 关闭坐标轴,默认关闭
    if not is_axis:
        plt.axis('off')

    # 展示受损图片的名称
    plt.title(image_title)

    # 展示图片
    plt.show()


class NoiseGenerator:
    @staticmethod
    def add_pulse_noise(im, noise_ratios=0.3):
        """
        给图像增加脉冲噪声
        脉冲噪声：噪声点只有两种情况，全黑or全白，因此又称为椒盐噪声
        :param im: 待处理的图像
        :param noise_ratios: 噪声比
        :return:
        """
        h, w = im.shape[0:2]                            # 获得图像的宽长
        p_size = h * w                                  # 计算出图像的平面大小
        for r in range(int(p_size * noise_ratios)):     # 噪声点数量
            # 获得噪声的随机位置
            rand_h = random.randint(0, h - 1)
            rand_w = random.randint(0, w - 1)
            # 两种处理: 1.像素点设置为全黑，2.像素点设置为全白
            im[rand_h, rand_w, :] = np.array([0, 0, 0]) if random.random() < 0.5 else np.array([255, 255, 255])
        return im

    @staticmethod
    def add_gaussian_noise(im, mean=0, var=0.005):
        """
        添加高斯噪声
        :param im:
        :param mean: 均值
        :param var: 方差
        :return:
        """
        # image = np.array(im / 255, dtype=float)   # 将像素值归一
        image = NoiseGenerator._normalization(im)   # 将像素值归一
        noise = np.random.normal(mean, var ** 0.5, image.shape)  # 产生高斯噪声
        noised_im = image + noise  # 直接将归一化的图片与噪声相加
        def clip_img(noised_im):
            """
            将值限制在(-1或0, 1)间，然后乘255恢复
            :param noised_im: 加了噪声的图片
            :return: clip过的图片
            """""
            low_clip = -1. if noised_im.min() < 0 else 0.
            out_im = np.clip(noised_im, low_clip, 1.0)
            out_im = np.uint8(out_im * 255)
            return out_im
        return clip_img(noised_im)


class Filter:
    def __init__(self, k = 3):
        self.k = k
        self.padding = None

    def get_median(self, imarray):
        """
        中值滤波
        :param imarray:
        :return:
        """
        height, width, channels = imarray.shape
        if not self.padding:
            edge = int((self.k - 1) / 2)
            if height - 1 - edge <= edge or width - 1 - edge <= edge:
                print("The parameter k is to large.")
                return None
            new_arr = np.zeros((height, width, 3), dtype="uint8")
            for i in range(height):
                for j in range(width):
                    for c in range(channels):  # 处理3个通道
                        if i <= edge - 1 or i >= height - 1 - edge \
                                or j <= edge - 1 or j >= width - edge - 1:
                            # 除了中心点以外其他边沿的点
                            new_arr[i, j, c] = imarray[i, j, c]
                        else:
                            # 中心点为排序后的中值
                            new_arr[i, j, c] = np.median(imarray[i-edge: i+edge+1, j-edge: j+edge+1, c])

            return new_arr

    def get_max(self, imarray):
        """
        最大值滤波
        :param imarray:
        :return:
        """
        height, width, channels = imarray.shape
        if not self.padding:
            edge = int((self.k - 1) / 2)
            if height - 1 - edge <= edge or width - 1 - edge <= edge:
                print("The parameter k is to large.")
                return None
            new_arr = np.zeros((height, width, 3), dtype="uint8")

            for i in range(height):
                for j in range(width):
                    for c in range(channels):  # 处理3个通道
                        if i <= edge - 1 or i >= height - 1 - edge \
                                or j <= edge - 1 or j >= width - edge - 1:
                            # 边界处理， 当i索引<=edge-1 -> 或者 i >= ((height-1)->图像边界-edge)->窗口在图像边界内的边界
                            new_arr[i, j, c] = imarray[i, j, c]
                        else:
                            # 中心点为排序后的中值
                            new_arr[i, j, c] = np.max(imarray[i-edge: i+edge+1, j-edge: j+edge+1, c])
            return new_arr

    def get_min(self, imarray):
        """
        最小值滤波
        :param imarray:
        :return:
        """
        height, width, channels = imarray.shape
        if not self.padding:
            edge = int((self.k - 1) / 2)
            if height - 1 - edge <= edge or width - 1 - edge <= edge:
                print("The parameter k is to large.")
                return None
            new_arr = np.zeros((height, width, 3), dtype="uint8")

            for i in range(height):
                for j in range(width):
                    for c in range(channels):  # 处理3个通道
                        if i <= edge - 1 or i >= height - 1 - edge \
                                or j <= edge - 1 or j >= width - edge - 1:
                            # 边界处理， 当i索引<=edge-1 -> 或者 i >= ((height-1)->图像边界-edge)->窗口在图像边界内的边界
                            new_arr[i, j, c] = imarray[i, j, c]
                        else:
                            # 中心点为排序后的中值
                            new_arr[i, j, c] = np.min(imarray[i-edge: i+edge+1, j-edge: j+edge+1, c])
            return new_arr

    def process(self, im):
        """
        先图像然后再选用滤波器组合
        :param im: 带噪声的图片, 值在[0, 255]
        :return: 修复好的图片
        """
        all_pixel_nums = im.size
        black_pixel_array = (im == 0)  # 为黑色的
        black_nums = np.count_nonzero(black_pixel_array)
        account = black_nums / all_pixel_nums       # 黑色所占比
        # print("黑色像素占比:", account)
        if account > 0.3:
            # 针对黑色较少的情况
            repaired_im = self.get_max(im)
            plot_image(repaired_im, "第一次滤波结果")
            plt.show()
            print("First Time: get_max修复成功!")
            repaired_im = self.get_median(repaired_im)
            plot_image(repaired_im, "第二次滤波结果")
            print("第二次使用get_median, 修复成功!")
        else:
            repaired_im = self.get_median(im)
            plot_image(repaired_im, "第一次滤波结果")
            plt.show()
        return repaired_im


def normalization(image):
    """
    将数据线性归一化
    :param image: 图片矩阵，一般是np.array 类型
    :return: 将归一化后的数据，在（0,1）之间
    """
    # 获取图片数据类型对象的最大值和最小值
    info = np.iinfo(image.dtype)
    # 图像数组数据放缩在 0-1 之间
    return image.astype(np.double) / info.max


def noise_mask_image(img, noise_ratio):
    """
    根据题目要求生成受损图片
    :param img: 图像矩阵，一般为 np.ndarray
    :param noise_ratio: 噪声比率，可能值是0.4/0.6/0.8
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None
    # -------------实现受损图像答题区域-----------------
    noise_img = deepcopy(img)
    h, w = img.shape[: 2]  # h为图片的长, w为图片的宽
    for dh in range(h):  # 遍历每行
        cols = range(w)
        mask_indexes = random.sample(cols, int(w * noise_ratio))
        pixel_list = [0 if i in mask_indexes else 1 for i in cols]
        for c in cols:
            noise_img[dh, c, :] = noise_img[dh, c, :] * pixel_list[c]
    # -----------------------------------------------
    noise_img = np.array(noise_img, dtype='double')
    return noise_img
