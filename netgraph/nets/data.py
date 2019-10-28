
from keras.datasets import mnist, cifar10

import keras

def data(dataset):

    if dataset == 'cifar10':
        num_classes = 10
        (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = cifar10.load_data()

        x_train = x_train_raw.astype('float32')
        x_test = x_test_raw.astype('float32')
        mean = [125.307, 122.95, 113.865]
        std  = [62.9932, 62.0887, 66.7048]
        for i in range(3):
            x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
            x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]


    if dataset == 'mnist':
        img_rows, img_cols = 28, 28
        num_classes = 10

        (x_train, y_train_raw), (x_test, y_test_raw) = mnist.load_data()

        x_train_raw = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test_raw = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

        x_train = x_train_raw.astype('float32')
        x_test = x_test_raw.astype('float32')
        x_train /= 255
        x_test /= 255

    y_train = keras.utils.to_categorical(y_train_raw, num_classes)
    y_test = keras.utils.to_categorical(y_test_raw, num_classes)

    return (x_train, x_train_raw, y_train, y_train_raw), (x_test, x_test_raw, y_test, y_test_raw)