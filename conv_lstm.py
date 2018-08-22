#!/usr/bin/env python2

from read_data import read_data

import numpy as np

from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint

def main():
    # Read data set
    n_cols = 25
    n_rows = 24
    dataset = read_data(n_rows, n_rows)
    dataset.read_file()
    dataset.split_data()

    X, y = dataset.X, dataset.y
    X = X.transpose(0, 2, 1)
    y = y.transpose(0, 2, 1)
    # train set = year 2015 + 2016 => 731 days
    # val set = year 2017 => 365 days
    train_size = 731*24/6
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    t = 1
    X_train_data = np.zeros((X_train.shape[0]-t, t,
                n_rows, n_cols, 1))
    y_train_data = np.zeros((y_train.shape[0]-t, t,
                n_rows, n_cols, 1))
    X_val_data = np.zeros((X_val.shape[0]-t, t,
                n_rows, n_cols, 1))
    y_val_data = np.zeros((y_val.shape[0]-t, t,
                n_rows, n_cols, 1))
    print(np.shape(X_train_data))
    print(np.shape(X_val_data))
    
    for i in range(t):
        X_train_data[:,i,:,:,0] = X_train[i:i+X_train.shape[0]-t]
        y_train_data[:,i,:,:,0] = y_train[i:i+y_train.shape[0]-t]
        X_val_data[:,i,:,:,0] = X_val[i:i+X_val.shape[0]-t]
        y_val_data[:,i,:,:,0] = y_val[i:i+y_val.shape[0]-t]
    
    n_filter = 64

    seq = Sequential()
    seq.add(ConvLSTM2D(filters=n_filter, kernel_size=(3, 3),
                       input_shape=(None, n_rows, n_cols, 1),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=n_filter, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    '''seq.add(ConvLSTM2D(filters=n_filter, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())

    seq.add(ConvLSTM2D(filters=n_filter, kernel_size=(3, 3),
                       padding='same', return_sequences=True))
    seq.add(BatchNormalization())'''

    seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
                   activation='sigmoid',
                   padding='same', data_format='channels_last'))

    seq.compile(loss='mean_absolute_error', optimizer='adadelta')

    # Train the network
    tensorboard = TensorBoard(log_dir='log/conv_lstm', histogram_freq=0,
                          write_graph=False, write_images=True)
    checkpoint = ModelCheckpoint(filepath='model/conv_lstm.{epoch:02d}.h5')

    seq.fit(X_train_data, y_train_data, batch_size=16,
            epochs=50, validation_data=(X_val_data, y_val_data), callbacks=[tensorboard, checkpoint])

    #save model
    seq.save('model/conv_lstm_final.h5')

if __name__ == '__main__':
    main()

