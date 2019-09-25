import keras
from PIL import Image, ImageDraw
import numpy as np
from keras import backend as K

# YOLO_v1 中x,y,w,h 损失函数
def loss(y_true, y_pred):
    a = K.square(y_pred[:, 0] - y_true[:, 0]) + K.square(y_pred[:, 1] - y_true[:, 1])
    b = K.square(K.sqrt(y_pred[:, 2]) - K.sqrt(y_true[:, 2])) + K.square(
        K.sqrt(y_pred[:, 3]) - K.sqrt(y_true[:, 3]))
    value = a + b
    return value


model = keras.models.load_model('model_use_vgg16_fine_tuning.h5', custom_objects={'loss': loss})
# model = keras.models.load_model('model_20190918.h5')
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
    predict = [predict[0][0], predict[0][1], predict[0][2], predict[0][3]]
    print(predict)
    # image.show()
    draw = ImageDraw.Draw(image)
    for i in range(len(predict)):
        predict[i] = float(predict[i])

    x, y, w, h = predict[0] * 224, predict[1] * 224, predict[2] * 224, predict[3] * 224
    draw.rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], width=5, outline='blue')
    image.show()
