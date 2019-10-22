# -*- coding: utf-8 -*-
import cv2
import logging
import os
import numpy as np


class DetectFace(object):

    def __init__(self, file_path):
        self.file_path = file_path

    def cv_imread(self, img_path):
        """
        读取图像，解决imread不能读取中文路径的问题
        :return:
        """
        cv_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)  # 会出现黑白照片 没有RGB三个通道 与cv2.imread不同
        return cv_img  # 所以后面采用直接检测 不转灰度

    def detect_face(self, img_path):
        """
        检测人脸
        :param img_path: 图像路径
        :return: 是否是人脸
        """
        # 日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        logger = logging.getLogger(__name__)
        logger.info('Reading image...')
        image = self.cv_imread(img_path)  # 调用函数 以读取中文路径
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 不考虑灰度 直接判断
        logger.info('Detect faces...')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 在opencv官网中找对应版本的xml文件
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.15, minNeighbors=5, minSize=(3, 3))  # 参数调整
        search_info = "Find %d face." % len(faces) if len(faces) <= 1 else "Find %d faces." % len(faces)
        logger.info(search_info)
        # 有照片返回True 没有照片返回False
        if len(faces) == 0:
            return False
        else:
            return True

    def search_photo(self):
        """
        遍历需要检测的文件夹，找到人脸照片和非人脸照片并重命名
        """
        files = os.listdir(self.file_path)  # 检测文件夹的路径
        count_number = 0  # 计数 命名
        for file in files:
            path = os.path.join(self.file_path, file)
            if self.detect_face(img_path=path):
                os.rename(path, os.path.join(self.file_path, 'photo.png'))  # 若为人脸照片 则命名为photo.png
            else:
                os.rename(path,
                          os.path.join(self.file_path,
                                       'other' + str(count_number) + '.png'))  # 若不为人脸照片 则命名为other+数字.png
            count_number = count_number + 1


if __name__ == '__name__':
    search_photo = DetectFace('output_test')
    search_photo.search_photo()

# # 读取图像，解决imread不能读取中文路径的问题
# def cv_imread(file_path):
#     cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)  # 会出现黑白照片 没有RGB三个通道 与cv2.imread不同
#     return cv_img  # 所以后面采用直接检测 不转灰度
#
#
# def detect_face(img_path):
#     # 日志
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
#     logger = logging.getLogger(__name__)
#     logger.info('Reading image...')
#     image = cv_imread(img_path)  # 调用函数 以读取中文路径
#     # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 不考虑灰度 直接判断
#     logger.info('Detect faces...')
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 在opencv官网中找对应版本的xml文件
#     faces = face_cascade.detectMultiScale(image, scaleFactor=1.15, minNeighbors=5, minSize=(3, 3))  # 参数调整
#     search_info = "Find %d face." % len(faces) if len(faces) <= 1 else "Find %d faces." % len(faces)
#     logger.info(search_info)
#     # 有照片返回True 没有照片返回False
#     if len(faces) == 0:
#         return False
#     else:
#         return True
#
#
# def search_photo(dir_path):
#     files = os.listdir(dir_path)  # 检测文件夹的路径
#     count_number = 0  # 计数 命名
#     for file in files:
#         path = os.path.join(dir_path, file)
#         if detect_face(path):
#             os.rename(path, os.path.join(dir_path, 'photo.png'))  # 若为人脸照片 则命名为photo.png
#         else:
#             os.rename(path, os.path.join(dir_path, 'other' + str(count_number) + '.png'))  # 若不为人脸照片 则命名为other+数字.png
#         count_number = count_number + 1
#
#
# search_photo('output_test')
