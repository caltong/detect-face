from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten, BatchNormalization
from keras.layers import LeakyReLU, Activation
import keras
from keras import backend as K
from load_data import load_data

# K.set_floatx('float16')  # 设置半精度计算

x_train, y_train = load_data()

print(x_train.shape, y_train.shape)
# x_train (12000,224,224,3) y_train (12000,4)

model = Sequential()

model.add(
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', input_shape=(224, 224, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (112,112,64)

model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (56,56,128)

model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (28,28,256)

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (14,14,512)

model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(Conv2D(filters=512, kernel_size=(3, 3), padding='same'))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))  # (7,7,512)

model.add(Flatten())
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
# model.add(Dropout(0.5))

model.add(Dense(256))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
# model.add(Dropout(0.5))

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(4))
model.add(Activation('sigmoid'))

Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)


# YOLO_v1 中x,y,w,h 损失函数
def loss(y_true, y_pred):
    a = K.square(y_pred[:, 0] - y_true[:, 0]) + K.square(y_pred[:, 1] - y_true[:, 1])
    b = K.square(K.sqrt(y_pred[:, 2]) - K.sqrt(y_true[:, 2])) + K.square(
        K.sqrt(y_pred[:, 3]) - K.sqrt(y_true[:, 3]))
    value = a + b
    return value


model.compile(loss=loss, optimizer=Adam, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=4, batch_size=32)
model.save('model_use_vgg16.h5')
