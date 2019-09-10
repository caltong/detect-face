import keras
from load_data import load_data
from keras import backend as K

# K.set_floatx('float16')
x_train, y_train = load_data()

model = keras.models.load_model('model_use_vgg16.h5')

model.fit(x_train, y_train, epochs=64, batch_size=32)
model.save('model_use_vgg16.h5')
