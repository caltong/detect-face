import cv2
import numpy as np
from cut_photo import CutPhoto

file_path = '5.png'
find_faces = CutPhoto(file_path)
cut_file_path = find_faces.cut_photo()[0]

image, faces = find_faces.detect_face()
print(faces)

thresh = cv2.Canny(image, 128, 256)  # 边缘处理
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度
gauss = cv2.GaussianBlur(gray, (3, 3), 0)
edges = cv2.Canny(gauss, 100, 200)
edges = np.array(edges)

bounding_box = [0, 0, 0, 0]

# 找bounding_box上边缘
first_flag = 0
for i in range(edges.shape[0]):
    column = int(edges.shape[1] / 2)
    pixel = edges[i][column]
    if pixel == 255 and first_flag == 0:
        bounding_box[1] = i
        first_flag = 1

# 找bounding_box左边缘
first_flag = 0
for i in range(edges.shape[1]):
    row = int(edges.shape[0] / 2)
    pixel = edges[row][i]
    if pixel == 255 and first_flag == 0:
        bounding_box[0] = i
        first_flag = 1

# 找bounding_box右边缘
for i in range(edges.shape[1]):
    row = int(edges.shape[0] / 2)
    pixel = edges[row][i]
    if pixel == 255:
        bounding_box[2] = i

bounding_box[3] = edges.shape[0]

cv2.rectangle(image, (bounding_box[0], bounding_box[1]), (bounding_box[2], bounding_box[3]), (255, 0, 0), 2)
cv2.imshow('1', image)
cv2.waitKey(0)
