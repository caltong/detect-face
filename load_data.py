import os
import numpy as np
from PIL import Image


def load_data():
    # 打开所有图片存入images 还未缩放大小
    data_folder_path = os.path.join('data', 'data_covered')  # 数据集路径
    files = os.listdir(data_folder_path)  # 遍历所有图片
    images = []
    for file in files:
        path = os.path.join(data_folder_path, file)
        image = Image.open(path)
        # pixel = np.asarray(image)
        images.append(image)

    # 标签 四个方向margin 顺时针顺序 左上右下
    y_train = []
    with open(os.path.join('data', 'margin.txt')) as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip('\n').strip('[').strip(']')  # 去掉前后[]和换行符
            line = line.split(',')  # str to list
            line = list(map(int, line))
            y_train.append(line)
    y_train = np.array(y_train)

    # 缩放每张图大小到224*224 VGG16输入尺寸 并且对应调整margin数值
    x_train = []
    for i in range(len(images)):
        image = images[i]
        # 调整margin
        width, height = image.size[0], image.size[1]
        width_ratio = 224 / width
        height_ratio = 224 / height
        y_train[i][0] = y_train[i][0] * width_ratio
        y_train[i][2] = y_train[i][2] * width_ratio
        y_train[i][1] = y_train[i][1] * height_ratio
        y_train[i][3] = y_train[i][3] * height_ratio
        # 缩放image
        image = image.resize((224, 224), Image.ANTIALIAS)
        pixel = np.asanyarray(image)
        x_train.append(pixel)
    x_train = np.array(x_train)

    # 使用YOLO_v1方式定义损失函数
    # 先处理margin数据 将四个margin转换为 x,y,w,h
    for i in range(y_train.shape[0]):
        [m0, m1, m2, m3] = y_train[i]
        y_train[i] = [(224 - m2 + m0) / 2, (224 - m3 + m1) / 2, 224 - m2 - m0, 224 - m3 - m1]

    # 归一化
    # x_train = x_train
    # y_train = y_train

    return x_train, y_train
