import os
import numpy as np
from PIL import Image

data_folder_path = os.path.join('data', 'data_covered')  # 数据集路径
files = os.listdir(data_folder_path)  # 遍历所有图片
x_train = []
for file in files:
    path = os.path.join(data_folder_path, file)
    image = Image.open(path)
    pixel = np.asarray(image)
    x_train.append(pixel)

y_train = []
with open(os.path.join('data', 'margin.txt')) as file:
    lines = file.readlines()
    for line in lines:
        line = line.strip('\n').strip('[').strip(']')  # 去掉前后[]和换行符
        line = line.split(',')  # str to list
        line = list(map(int, line))
        y_train.append(line)

y_train = np.array(y_train)
print(y_train.shape)