import cv2
import logging
import numpy as np


class Resize(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def cv_imread(self):
        """
        读取图像，解决imread不能读取中文路径的问题
        :return: 返回照片
        """
        cv_img = cv2.imdecode(np.fromfile(self.file_path, dtype=np.uint8), -1)  # 会出现黑白照片 没有RGB三个通道 与cv2.imread不同
        return cv_img  # 所以后面采用直接检测 不转灰度

    def resize(self):
        image = self.cv_imread()
        eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        # eye_cascade.load('haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(image, 1.05, 3)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.imshow('img', image)
        cv2.waitKey(0)


photo = Resize('data/data_test/13.png-photo0.png')
photo.resize()
