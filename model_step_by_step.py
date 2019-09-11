import keras
from load_data import load_data
from keras import backend as K
from data_augmentation import make_data_set


# K.set_floatx('float16')


# YOLO_v1 中x,y,w,h 损失函数
def loss(y_true, y_pred):
    a = K.square(y_pred[:, 0] - y_true[:, 0]) + K.square(y_pred[:, 1] - y_true[:, 1])
    b = K.square(K.sqrt(K.abs(y_pred[:, 2])) - K.sqrt(K.abs(y_true[:, 2]))) + K.square(
        K.sqrt(K.abs(y_pred[:, 3])) - K.sqrt(K.abs(y_true[:, 3])))
    value = a + b
    return value


model = keras.models.load_model('model_use_vgg16.h5', custom_objects={'loss': loss})

for i in range(300):
    print('Round: ' + str(i))
    make_data_set()
    x_train, y_train = load_data()
    model.fit(x_train, y_train, epochs=2, batch_size=32)
    model.save('model_use_vgg16.h5')
