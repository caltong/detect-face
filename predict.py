import keras
from PIL import Image
import numpy as np
from keras.backend import backend as K


def loss(y_true, y_pred):
    return K.mean((K.square(y_true - y_pred)))


model = keras.models.load_model('model_use_vgg16.h5')

while True:
    print('input image path:')
    path = input()
    image = Image.open(path)
    image = image.resize((224, 224), Image.ANTIALIAS)
    pixel = np.asarray(image)
    pixel = np.expand_dims(pixel, axis=0)
    predict = model.predict(pixel)

    print(predict)