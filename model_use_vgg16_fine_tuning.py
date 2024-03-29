from keras.applications import VGG16, InceptionV3
from keras.utils import plot_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from load_data import load_data
from keras import backend as K
from data_augmentation import make_data_set
from keras.optimizers import SGD
from keras import losses

from keras import backend as K

# set GPU memory
if ('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

base_model = VGG16(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
prediction = Dense(4, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=prediction)

for layer in base_model.layers:
    layer.trainable = False


# YOLO_v1 中x,y,w,h 损失函数
def loss(y_true, y_pred):
    a = K.abs(y_pred[:, 0] - y_true[:, 0]) + K.abs(y_pred[:, 1] - y_true[:, 1])
    b = K.abs(K.sqrt(y_pred[:, 2]) - K.sqrt(y_true[:, 2])) + K.abs(K.sqrt(y_pred[:, 3]) - K.sqrt(y_true[:, 3]))
    value = K.mean(a + b, axis=-1)
    return value


model.compile(optimizer='rmsprop', loss=loss)

# fine tuning 全连接层
x_train, y_train = load_data()
print(x_train.shape, y_train.shape)
model.fit(x_train, y_train, epochs=2, batch_size=16, verbose=1)

# fine tuning 第二次
# make_data_set()
# x_train, y_train = load_data()
# for layer in model.layers[:11]:
#     layer.trainable = True
# for layer in model.layers[11:]:
#     layer.trainable = True
for layer in model.layers:
    layer.trainable = True
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss=loss)
model.fit(x_train, y_train, epochs=4, batch_size=16, verbose=1)

# fine tuning 第三次
# make_data_set()
# x_train, y_train = load_data()
# for layer in model.layers[:5]:
#     layer.trainable = False
# for layer in model.layers[5:]:
#     layer.trainable = True
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss=loss)
# model.fit(x_train, y_train, epochs=4, batch_size=16, verbose=1)

model.save('model_use_vgg16_fine_tuning.h5')
