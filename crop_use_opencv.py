import cv2
import numpy as np
from PIL import Image


def crop_use_opencv(img):
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = img[y:y + h, x:x + w]
    crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    return crop, x, y, w, h


# img = Image.open('data/data_test/4.png-photo0.png')
# img,x,y,w,h = crop_use_opencv(img)
# img.show()
