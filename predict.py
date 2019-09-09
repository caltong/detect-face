import keras
from PIL import Image
import numpy as np

model = keras.models.load_model('model_use_vgg16.h5')

image = Image.open('data/data_test/2.png-photo0.png')
image = image.resize((224, 224), Image.ANTIALIAS)
pixel = np.asarray(image)
pixel = np.expand_dims(pixel, axis=0)
predict = model.predict(pixel)

print(predict)
