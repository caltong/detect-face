import cv2
import numpy as np
from faceplusplus_api import faceplusplus_api
from PIL import Image, ImageDraw
import math


def load_api_data(filepath):
    data = faceplusplus_api(filepath)
    left_eye = [data['faces'][0]['landmark']['left_eye_center']['x'],
                data['faces'][0]['landmark']['left_eye_center']['y']]
    right_eye = [data['faces'][0]['landmark']['right_eye_center']['x'],
                 data['faces'][0]['landmark']['right_eye_center']['y']]
    nose_tip = [data['faces'][0]['landmark']['nose_tip']['x'], data['faces'][0]['landmark']['nose_tip']['y']]
    face_rectangle = {'width': data['faces'][0]['face_rectangle']['width'],
                      'top': data['faces'][0]['face_rectangle']['top'],
                      'left': data['faces'][0]['face_rectangle']['left'],
                      'height': data['faces'][0]['face_rectangle']['height']}
    face_area = face_rectangle['width'] * face_rectangle['height']

    image = Image.open(filepath)
    image_width, image_height = image.size
    image_area = image_width * image_height
    # draw = ImageDraw.Draw(image)
    # draw.ellipse([left_eye[0] - 3, left_eye[1] - 3, left_eye[0] + 3, left_eye[1] + 3], fill='red')
    # draw.ellipse([right_eye[0] - 3, right_eye[1] - 3, right_eye[0] + 3, right_eye[1] + 3], fill='red')
    # draw.ellipse([nose_tip[0] - 3, nose_tip[1] - 3, nose_tip[0] + 3, nose_tip[1] + 3], fill='red')
    # draw.rectangle([(face_rectangle['left'], face_rectangle['top']),
    #                 (face_rectangle['left'] + face_rectangle['width'],
    #                  face_rectangle['top'] + face_rectangle['height'])],
    #                outline='red')
    # image.show()
    return image, left_eye, right_eye, nose_tip, face_area, image_area


def resize(filepath):
    image, left_eye, right_eye, nose_tip, face_area, image_area = load_api_data(filepath)
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
        print(image.size, coor)
        image2 = image.crop(coor)

    return image2


for i in range(2, 16):
    filepath = 'data/data_test/' + str(i) + '.png-photo0.png'
    image2 = resize(filepath)
    # image2.show()
    image2.save(filepath)
