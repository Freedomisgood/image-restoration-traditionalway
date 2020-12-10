# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 10:16
# @Author  : Mrli
# @FileName: main.py
# @Blog    : https://nymrli.top/
import numpy as np
from copy import deepcopy
import random

from utils import (read_image,
                   plot_image,
                   Filter,
                   NoiseGenerator,
                   normalization)


def get_noise_mask(noise_img):
    """
    获取噪声图像，一般为 np.array
    :param noise_img: 带有噪声的图片
    :return: 噪声图像矩阵
    """
    # 将图片数据矩阵只包含 0和1,如果不能等于 0 则就是 1。
    return np.array(noise_img != 0, dtype='double')


def noise_mask_image(img, noise_ratio):
    """
    根据题目要求生成受损图片
    :param img: 图像矩阵，一般为 np.ndarray, 图像矩阵值 0-1 之间
    :param noise_ratio: 噪声比率，可能值是0.4/0.6/0.8
    :return: noise_img 受损图片, 图像矩阵值 0-1 之间，数据类型为 np.array,
             数据类型对象 (dtype): np.double, 图像形状:(height,width,channel),通道(channel) 顺序为RGB
    """
    # 受损图片初始化
    noise_img = None
    # -------------实现受损图像答题区域-----------------
    # TODO: 没过测试
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


def restore_image(noise_img, size=4):
    """
    使用 你最擅长的算法模型 进行图像恢复。
    :param noise_img: 一个受损的图像
    :param size: 输入区域半径，长宽是以 size*size 方形区域获取区域, 默认是 4
    :return: res_img 恢复后的图片，图像矩阵值 0-1 之间，数据类型为 np.array,
            数据类型对象 (dtype): np.double, 图像形状:(height,width,channel), 通道(channel) 顺序为RGB
    """
    # 恢复图片初始化，首先 copy 受损图片，然后预测噪声点的坐标后作为返回值。
    res_img = np.copy(noise_img)

    # 获取噪声图像
    noise_mask = get_noise_mask(noise_img)

    # -------------实现图像恢复代码答题区域----------------------------
    input_im = res_img * 255
    res_img = Filter(k=size-1).process(input_im)
    res_img = np.array(res_img / 255, dtype="double")
    # ---------------------------------------------------------------

    return res_img


if __name__ == '__main__':
    # 生成噪声文件
    # gray_girl = "A.png"
    # im = read_image(gray_girl)
    # norm_im = normalization(im)
    # noised_im = noise_mask_image(norm_im, noise_ratio=0.8)
    # plot_image(noised_im)
    # print(noised_im)
    # print("生成成功!")

    # 测试工程下所有图片
    # import os
    # f = Filter(k=3)
    # for root, dirs, files in os.walk(".", topdown=False):
    #     for name in files:
    #         filename = (os.path.join(root, name))
    #         if "noise" in filename:
    #             noised_im = read_image(filename)
    #             plot_image(noised_im, "噪声原图")
    #             f.process(noised_im)
    #             print(filename, "修复成功~\n")

    # 单独测试雪人文件
    img = read_image("A.png")        # 921600 40437
    nor_img = normalization(img)
    # im = read_image("A_noised.png")        # 921600 40437
    # im = read_image("A_noised_more.png")   # 921600 299076
    # im = read_image("./samples/xihu_random_noise.png")                          # 720000 432797
    # im = read_image("./samples/the_school_of_athens_random_noise.png")        # 838800 507671
    # im = read_image("./samples/potala_palace_random_noise.png")               # 321600 198858
    # im = read_image("./samples/mona_lisa_random_noise.png")        # 946680 569315
    # im = read_image("./samples/forest_random_noise.png")        # 810000 486582
    # 设置噪声生成率
    noise_ratio = 0.6
    # 生成受损图片
    noise_img = noise_mask_image(nor_img, noise_ratio)
    plot_image(noise_img, "噪声原图")
    # print(noise_img)
    repaired_im = restore_image(noise_img)
    plot_image(repaired_im, "滤波结果")
