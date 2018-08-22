#!/usr/bin/env python2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.misc import imresize
import time
import csv

class read_data(object):

    def __init__(self, T_X, T_y):
        # Number of all stations in Seoul
        self.all_stations = 33
        # Number of used stations in Seoul data
        self.stations = 25
        # Raw all data
        self.all_data = None
        self.T_X = T_X # number of hours to be the historical hours
        self.T_y = T_y # number of hours to be the forecasting hours
        self.X = None
        self.y = None

    # read pm2_5 values
    def read_file(self):
        all_data = list()
        X = list()
        idx = 0
        for line in open('data/aqi2_5_seoul_stations_join_2015_2017'):
            idx += 1
            if idx == self.all_stations:
                idx = 0 # reset
            elif idx <= self.stations:
                values = line.split(",")
                row = list()
                for i in range(len(values)):
                    val = float(values[i])
                    row.append(val)
                X.append(row)
                # Process a batch of stations
                if idx == self.stations:
                    df = pd.DataFrame(data=X)
                    df = df.replace(-1.0, np.NaN)
                    df = df.fillna(df.mean())
                    df = df.replace(np.NaN, 1.0) # in case of fillna fall
                    # reindex stations follow new indexes
                    #df = df.reindex([9,8,2,16,22,23,24,10,15,5,1,17,21,13,12,3,20,19,18,11,6,4,14,0,7])
                    val = df.values
                    all_data.append(val.transpose())
                    X = list()

        all_data = np.array(all_data)
        all_data = all_data.reshape(all_data.shape[0]*all_data.shape[1], -1).transpose()
        print(all_data.shape)
        self.all_data = all_data

    def write_file(self):
        # File to write data
        outfile = "data/aqi2_5_seoul_by_stations_2015_2017.csv"
        f = open(outfile, 'w')
        w = csv.writer(f)
        # Write header
        w.writerow(['station_id', 'year', 'month', 'day', 'hour', 'aqi'])

        for i in range(self.all_data.shape[0]): # i = station_id
            t = time.strptime("15 01 01", "%y %m %d")
            t = time.mktime(t)
            for j in range(all_data.shape[1]): # j = hour
                # Get time info
                struct = time.localtime(t)
                year = struct[0]
                month = struct[1]
                day = struct[2]
                hour = j

                w.writerow([i, year, month, day, hour, all_data[i][j]])
                t += 3600
        f.flush()
        f.close()

    def split_data(self):
        X = list()
        y = list()
        all_data = self.all_data
        T_X, T_y = self.T_X, self.T_y
        for i in range(0, self.all_data.shape[1]-T_X-T_y, 6):
            X.append(all_data[:, i:i+T_X])
            y.append(all_data[:, i+T_X:i+T_X+T_y])
        X = np.array(X)
        print(X.shape)
        self.X = X

        y = np.array(y)
        # flip y
        for i in range(len(y)):
            tmp = np.zeros((y[i].shape[0], y[i].shape[1]))
            tmp[:] = y[i][:]
            for j in range(T_y):
                y[i][:,j] = tmp[:,T_y-1-j]
        print(y.shape)
        self.y = y

    def convert2D_to_3D(self):
        X = self.X
        T_X = self.T_X

        X = X.transpose(0, 2, 1)
        X = X.reshape(-1, T_X/3, 3, self.stations)
        X = X.transpose(0, 3, 1, 2)
        print(X.shape)
        self.X = X
        return self.X

    def rescale_data(self, X):
        from skimage.transform import rescale, resize

        X_new = [None] * X.shape[0]
        for i in range(X.shape[0]):
            X_new[i] = resize(X[i], (X[i].shape[0]*4, X[i].shape[1]*2))
        X = np.array(X_new)
        return X

    def visualize_data(self):
        from skimage.transform import rescale, resize

        #Visualizing as image
        X, y = self.X, self.y
        # resize X
        #X = self.rescale_data(X)

        for i in range(1):
            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.set_title('X[{}]'.format(i))
            X_i = X[i]
            ax.imshow(X_i, interpolation='bilinear')

            ax = fig.add_subplot(122)
            ax.set_title('y[{}]'.format(i))
            X_i1 = y[i]
            ax.imshow(X_i1, interpolation='bilinear')

            plt.show()
            #fig.savefig('figure_' + str(i) + '_' + str(i+1)+'.png', transparent=True)

if __name__ == '__main__':
    data = read_data(24, 24)
    data.read_file()
    data.split_data()
    #data.convert2D_to_3D()
    data.visualize_data()

