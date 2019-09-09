import cv2
import os
from PIL import Image
import random
import numpy as np

data_original_folder_path = os.path.join('data', 'data_original')
data_rename_folder_path = os.path.join('data', 'data_rename')
data_covered_folder_path = os.path.join('data', 'data_covered')
files = os.listdir(data_original_folder_path)
photos = os.listdir(data_rename_folder_path)


def rename_image():
    file_number_counter = 0
    for file in files:
        image = Image.open(os.path.join(data_original_folder_path, file))
        image = image.convert('RGB')
        if file_number_counter <= 9:
            file_name_str = '000' + str(file_number_counter)
        elif file_number_counter <= 99:
            file_name_str = '00' + str(file_number_counter)
        elif file_number_counter <= 999:
            file_name_str = '0' + str(file_number_counter)
        else:
            file_name_str = str(file_number_counter)
        file_name = os.path.join(data_rename_folder_path, file_name_str + '.jpg')
        image.save(file_name, 'jpeg')
        file_number_counter += 1


def create_pure_background(w, h):
    """
    生成随机宽高颜色的纯色背景图
    :param w: 宽
    :param h: 高
    :return: image: 图像
    """
    w_rate = random.random() / 2 + 1.0
    h_rate = random.random() / 2 + 1.0
    w = int(w_rate * w)
    h = int(h_rate * h)
    color = tuple(np.random.randint(256, size=[3]))
    image = Image.new('RGB', (w, h), color)

    return image


def cover_to_make_new_photo(photo_name):
    """
    合成随机照片
    :param photo_name: 照片名
    :return: 照片和各方向边距
    """
    photo = Image.open(os.path.join(data_rename_folder_path, photo_name))  # 读取photo
    background = create_pure_background(photo.size[0], photo.size[1])  # 生成背景

    left_margin = int(random.random() * (background.size[0] - photo.size[0]))  # 左边距
    up_margin = int(random.random() * (background.size[1] - photo.size[1]))  # 上边距

    background.paste(photo, (left_margin, up_margin))  # 合成
    margin = [left_margin,
              up_margin,
              background.size[0] - photo.size[0] - left_margin,
              background.size[1] - photo.size[1] - up_margin]  # 顺时针个方向margin 左上右下

    return background, margin


def make_data_set():
    file_name_counter = 0
    margin_list = []
    for photo in photos:
        for i in range(10):
            image, margin = cover_to_make_new_photo(photo)
            margin_list.append(margin)

            if file_name_counter <= 9:
                file_name_str = '0000' + str(file_name_counter)
            elif file_name_counter <= 99:
                file_name_str = '000' + str(file_name_counter)
            elif file_name_counter <= 999:
                file_name_str = '00' + str(file_name_counter)
            elif file_name_counter <= 9999:
                file_name_str = '0' + str(file_name_counter)
            else:
                file_name_str = str(file_name_counter)

            file_name = os.path.join(data_covered_folder_path, file_name_str + '.jpg')
            image.save(file_name, 'jpeg')
            file_name_counter += 1

    with open(os.path.join('data', 'margin.txt'), 'w') as f:
        for margin in margin_list:
            f.write(str(margin) + '\n')


# 重命名
# rename_image()
# 生成数据
# make_data_set()
