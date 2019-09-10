import keras
from PIL import Image
import numpy as np
from keras import backend as K
import keras.losses


# YOLO_v1 中x,y,w,h 损失函数
def loss(y_true, y_pred):
    a = K.square(y_pred[:, 0] - y_true[:, 0]) + K.square(y_pred[:, 1] - y_true[:, 1])
    b = K.square(K.sqrt(K.abs(y_pred[:, 2])) - K.sqrt(K.abs(y_true[:, 2]))) + K.square(
        K.sqrt(K.abs(y_pred[:, 3])) - K.sqrt(K.abs(y_true[:, 3])))
    value = a + b
    return value


keras.losses.custom_loss = loss

model = keras.models.load_model('model_use_vgg16.h5', custom_objects={'loss': loss})
output = []
while True:
    print('input image path:')
    path = input()
    image = Image.open(path)
    image = image.resize((224, 224), Image.ANTIALIAS)
    pixel = np.asarray(image)
    pixel = np.expand_dims(pixel, axis=0)
    pixel = pixel / 255.0
    predict = model.predict(pixel)
    output.append(predict)
    print(predict)
