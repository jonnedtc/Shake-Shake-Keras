import keras
import models
import numpy as np

from matplotlib import pyplot as plt
from keras_contrib.applications import resnet
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

# Parameters
epochs = 200
batch_size = 128

# Load and prepare the CIFAR10 data
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
x_train_mean = np.mean(x_train, axis=0)
x_train -= x_train_mean
x_test -= x_train_mean
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Data augmentation
augment = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# Learning rate
def schedule(epoch):
    if epoch >= 160: return 1e-5
    if epoch >= 120: return 1e-4
    return 1e-3
lr = LearningRateScheduler(schedule)

# Load and compile the model
model = models.create_shakeshake_cifar(n_classes=10)
model.compile('Adam', 'categorical_crossentropy', metrics=['accuracy'])

# Train the model
shake = model.fit_generator(augment.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs, validation_data=(x_test, y_test), callbacks=[lr])

# Load and compile the model
model = resnet.ResNet(input_shape=(32, 32, 3), classes=10, block='basic',
                      repetitions=[5, 5, 5], initial_filters=16, initial_strides=(1, 1),
                      initial_kernel_size=(3, 3), initial_pooling=None)
model.compile('Adam', 'categorical_crossentropy', metrics=['accuracy'])

# Train the model
res34 = model.fit_generator(augment.flow(x_train, y_train, batch_size=batch_size),
                            epochs=epochs, validation_data=(x_test, y_test), callbacks=[lr])

plt.figure(figsize=(16, 8))
plt.subplot(121)
plt.plot(shake.history['loss'], 'b-', label='Shake-Shake loss (train)')
plt.plot(shake.history['val_loss'], 'b--', label='Shake-Shake loss (test)')
plt.plot(res34.history['loss'], 'r-', label='ResidualNet loss (train)')
plt.plot(res34.history['val_loss'], 'r--', label='ResidualNet loss (test)')
plt.legend()
plt.subplot(122)
plt.plot(shake.history['acc'], 'b-', label='Shake-Shake accuracy (train)')
plt.plot(shake.history['val_acc'], 'b--', label='Shake-Shake accuracy (test)')
plt.plot(res34.history['acc'], 'r-', label='ResidualNet accuracy (train)')
plt.plot(res34.history['val_acc'], 'r--', label='ResidualNet accuracy (test)')
plt.legend()
plt.savefig('images/result.png')
plt.show()
