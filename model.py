import os
import numpy as np
from PIL import Image
import math
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, BatchNormalization
import keras
from keras import backend as K

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
    y_train[i][0] = int(math.ceil(y_train[i][0] * width_ratio))
    y_train[i][2] = int(math.ceil(y_train[i][2] * width_ratio))
    y_train[i][1] = int(math.ceil(y_train[i][1] * height_ratio))
    y_train[i][3] = int(math.ceil(y_train[i][3] * height_ratio))
    # 缩放image
    image = image.resize((224, 224), Image.ANTIALIAS)
    pixel = np.asanyarray(image)
    x_train.append(pixel)
x_train = np.array(x_train)

# 归一化
x_train = x_train / 255.0
y_train = y_train / 224.0

print(x_train.shape, y_train.shape)
# x_train (12000,224,224,3) y_train (12000,4)

model = Sequential()

model.add((Conv2D(filters=8, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(224, 224, 3))))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))

# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(224, 224, 3)))
# model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
# model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (112,112,64)

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(4, activation='sigmoid'))

Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(loss=keras.losses.mean_squared_error, optimizer=Adam, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=16, batch_size=32)
model.save('model_use_vgg16.h5')
