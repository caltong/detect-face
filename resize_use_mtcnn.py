from mtcnn.mtcnn import MTCNN
import cv2
from PIL import Image
import math


def load_mtcnn_data(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    result = detector.detect_faces(img)

    left_eye = result[0]['keypoints']['left_eye']
    right_eye = result[0]['keypoints']['right_eye']
    nose_tip = result[0]['keypoints']['nose']

    face_rectangle = {'left': result[0]['box'][0],
                      'top': result[0]['box'][1],
                      'width': result[0]['box'][2],
                      'height': result[0]['box'][3]}
    face_area = face_rectangle['width'] * face_rectangle['height']

    img = Image.fromarray(img)
    image_width, image_height = img.size
    image_area = image_width * image_height
    return img, left_eye, right_eye, nose_tip, face_area, image_area


def resize(filepath):
    image, left_eye, right_eye, nose_tip, face_area, image_area = load_mtcnn_data(filepath)
    center_coor = [((left_eye[0] + right_eye[0]) / 2 + nose_tip[0]) / 2,
                   ((left_eye[1] + right_eye[1]) / 2 + nose_tip[1]) / 2]
    # print(face_area, image_area)
    if image_area / face_area > 3.5:
        resize_area = face_area * 3.5
        width = math.sqrt(resize_area / 1.4)
        height = width * 1.4
        print(width, height)
        coor1 = [int(center_coor[0] - width / 2), int(center_coor[1] - height / 2)]
        coor2 = [int(center_coor[0] + width / 2), int(center_coor[1] + height / 2)]
        coor = coor1 + coor2
        for i in range(len(coor)):
            if coor[i] < 0:
                coor[i] = 0
        print(image.size, coor)
        image2 = image.crop(coor)
    else:
        image2 = image

    return image2


for i in range(2, 16):
    filepath = 'data/data_test/' + str(i) + '.png-photo0.png'
    image2 = resize(filepath)
    # image2.show()
    image2.save(filepath)
