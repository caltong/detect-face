from keras.applications.vgg16 import VGG16
from keras.layers import Dense, AveragePooling2D
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

base_model = VGG16(weights='imagenet', include_top=False)

x = base_model.output
x = AveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4)(x)


img_path = 'data/data_covered/0.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
print(features)
