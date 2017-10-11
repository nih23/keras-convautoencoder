import numpy as np
from keras import models
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten, Reshape
from autoencoder_layers import DependentDense, Deconvolution2D, DePool2D
from helpers import show_representations


def load_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)
    return (X_train, y_train), (X_test, y_test)


def build_model(nb_filters=32, nb_pool=2, nb_conv=2):
    model = models.Sequential()
    d = Dense(30)
    c = Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same', input_shape=(28, 28, 1))
    mp =MaxPooling2D(pool_size=(nb_pool, nb_pool))
    # =========      ENCODER     ========================
    model.add(c)
    model.add(Activation('relu'))
    model.add(mp)
    model.add(Dropout(0.25))
    # =========      BOTTLENECK     ======================
    model.add(Flatten())
    model.add(d)
    model.add(Activation('relu'))
    # =========      BOTTLENECK^-1   =====================
    model.add(DependentDense(units=nb_filters * 14 * 14, master_layer=d))
    model.add(Activation('relu'))
    model.add(Reshape((nb_filters, 14, 14)))
    # =========      DECODER     =========================
    model.add(DePool2D(mp, size=(nb_pool, nb_pool)))
    model.add(Deconvolution2D(binded_conv_layer=c, border_mode='same', filters = nb_filters, kernel_size = (nb_conv, nb_conv)))
    model.add(Activation('relu'))

    return model


if __name__ == '__main__':
    pModel = 'convae.h5'
    noEpochs = 10
    batchSz = 12000
    (X_train, y_train), (X_test, y_test) = load_data()
    X_train = np.swapaxes(X_train,1,3)
    X_test = np.swapaxes(X_test,1,3)
    model = build_model()
    if not False:
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.summary()
        model.fit(X_train, X_train, epochs=noEpochs, batch_size=batchSz, validation_split=0.2,
                  callbacks=[EarlyStopping(patience=3)])
        model.save(pModel, overwrite=True)
    else:
        #TODO: fix me.. broken
        model = models.load_model(pModel, custom_objects={"DependentDense": DependentDense, "Deconvolution2D": Deconvolution2D, "DePool2D": DePool2D})
        model.compile(optimizer='rmsprop', loss='mean_squared_error')

    show_representations(model, X_test)