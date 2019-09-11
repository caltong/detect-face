import cv2
import logging
import numpy as np


class CutPhoto(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def cv_imread(self):
        """
        读取图像，解决imread不能读取中文路径的问题
        :return: 返回照片
        """
        cv_img = cv2.imdecode(np.fromfile(self.file_path, dtype=np.uint8), -1)  # 会出现黑白照片 没有RGB三个通道 与cv2.imread不同
        return cv_img  # 所以后面采用直接检测 不转灰度

    def detect_face(self):
        """
        检测人脸
        :param img_path: 图像路径
        :return: image: 图像
                 faces: list 存储人脸位置和大小 [x,y,w,h]
        """
        # 日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
        logger = logging.getLogger(__name__)
        logger.info('Reading image...')
        image = self.cv_imread()  # 调用函数 以读取中文路径
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 不考虑灰度 直接判断
        logger.info('Detect faces...')
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 在opencv官网中找对应版本的xml文件
        faces = face_cascade.detectMultiScale(image, scaleFactor=1.15, minNeighbors=10, minSize=(3, 3))  # 参数调整
        search_info = "Find %d face." % len(faces) if len(faces) <= 1 else "Find %d faces." % len(faces)
        logger.info(search_info)

        return image, faces

    def cut_photo(self):
        """
        裁切出人脸照片
        :return:
        """
        image, faces = self.detect_face()
        file_name_list = []
        if len(faces) > 0:
            faces_number = 0
            for faceRect in faces:  # 若有多张人脸的情况
                x, y, w, h = faceRect
                x1, y1 = x - int(0.3 * w), y - int(0.5 * h)
                x2, y2 = x + int(1.3 * w), y + int(1.6 * h)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2, 8, 0)
                photo = image[y1:y2, x1:x2]
                status = cv2.imwrite(self.file_path + '-' + 'photo' + str(faces_number) + '.png', photo)
                file_name_list.append(self.file_path + '-' + 'photo' + str(faces_number) + '.png')
                if status:
                    print('Save succeed')
                # cv2.imshow("img", photo)
                # cv2.waitKey(0)

        else:
            print('Cut 0 photo')
        # print(faces)
        return file_name_list


cut_photo = CutPhoto('data/data_test/3.png')
cut_photo.cut_photo()

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
#
#     return image, faces
#
#
# image, faces = detect_face('5.png')
# if len(faces) > 0:
#     for faceRect in faces:
#         x, y, w, h = faceRect
#         x1, y1 = x - int(0.15 * w), y - int(0.3 * h)
#         x2, y2 = x + int(1.15 * w), y + int(1.3 * h)
#         # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2, 8, 0)
#         photo = image[y1:y2, x1:x2]
#
# cv2.imshow("img", photo)
# cv2.waitKey(0)
# print(faces)
