import cv2
import numpy as np
from PIL import Image


def cv_imread(path):
    """
    读取图像，解决imread不能读取中文路径的问题
    :return: 返回照片
    """
    cv_img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)  # 会出现黑白照片 没有RGB三个通道 与cv2.imread不同
    return cv_img  # 所以后面采用直接检测 不转灰度


def detect_face(path):
    """
    检测人脸
    :param img_path: 图像路径
    :return: image: 图像
             faces: list 存储人脸位置和大小 [x,y,w,h]
    """
    # 日志
    # logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    # logger = logging.getLogger(__name__)
    # logger.info('Reading image...')
    image = cv_imread(path)  # 调用函数 以读取中文路径
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 不考虑灰度 直接判断
    # logger.info('Detect faces...')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # 在opencv官网中找对应版本的xml文件
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.15, minNeighbors=10, minSize=(3, 3))  # 参数调整
    # search_info = "Find %d face." % len(faces) if len(faces) <= 1 else "Find %d faces." % len(faces)
    # logger.info(search_info)

    return image, faces


def cut_photo(path):
    """
    裁切出人脸照片
    :return:
    """
    global photo
    photo = 0
    image, faces = detect_face(path)
    file_name_list = []
    if len(faces) > 0:
        faces_number = 0
        for faceRect in faces:  # 若有多张人脸的情况
            x, y, w, h = faceRect
            x1, y1 = x - int(0.5 * w), y - int(0.5 * h)
            x2, y2 = x + int(1.5 * w), y + int(1.5 * h)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2, 8, 0)
            photo = image[y1:y2, x1:x2]
            photo = Image.fromarray(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))
            # status = cv2.imwrite(path + '-' + 'photo' + str(faces_number) + '.png', photo)
            # file_name_list.append(path + '-' + 'photo' + str(faces_number) + '.png')
            # if status:
            #     print('Save succeed')
            # # cv2.imshow("img", photo)
            # # cv2.waitKey(0)
            #
            # else:
            #     print('Cut 0 photo')
            #     # print(faces)
    if photo == 0:
        return None
    else:
        return photo


def cut_photo_use_model(path):
    import keras
    from PIL import Image, ImageDraw
    import numpy as np
    from keras import backend as K

    photo = cut_photo(path)

    # YOLO_v1 中x,y,w,h 损失函数
    def loss(y_true, y_pred):
        a = K.abs(y_pred[:, 0] - y_true[:, 0]) + K.abs(y_pred[:, 1] - y_true[:, 1])
        b = K.abs(K.sqrt(y_pred[:, 2]) - K.sqrt(y_true[:, 2])) + K.abs(K.sqrt(y_pred[:, 3]) - K.sqrt(y_true[:, 3]))
        value = a + b
        return value

    model = keras.models.load_model('model_by_2070.h5', custom_objects={'loss': loss})

    if photo is not None:
        image = photo.resize((224, 224), Image.ANTIALIAS)
        pixel = np.asarray(image)
        pixel = np.expand_dims(pixel, axis=0)
        pixel = pixel / 255.0
        predict = model.predict(pixel)
        predict = [predict[0][0], predict[0][1], predict[0][2] - 0.05, predict[0][3] - 0.05]  # 去除偏差
        x0, y0 = (predict[0] - predict[2] / 2) * photo.size[0], (predict[1] - predict[3] / 2) * photo.size[1]
        if x0 < 0:
            x0 = 0
        if y0 < 0:
            y0 = 0
        x1, y1 = (predict[0] + predict[2] / 2) * photo.size[0], (predict[1] + predict[3] / 2) * photo.size[1]
        if x1 > photo.size[0]:
            x1 = photo.size[0]
        if y1 > photo.size[1]:
            y1 = photo.size[1]
        croped_photo = photo.crop((x0, y0, x1, y1))

        croped_photo.save(path + '-' + 'photo' + '0' + '.png', 'PNG')

        return True
    else:
        return False


if __name__ == '__main__':
    for i in range(1, 16):
        path = 'data/data_test/' + str(i) + '.png'
        cut_photo_use_model(path)
