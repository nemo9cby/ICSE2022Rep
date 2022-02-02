# import tensorflow as tf
# from tensorflow import keras

# (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# print (x_train.shape())
import os
import sys
import time
import pickle
import random
import math
import numpy as np

class_num       = 10
image_size      = 32
img_channels    = 3

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def prepare_data():
    data_dir = '/root/repro/cifar-10-batches-py'
    image_dim = image_size * image_size * img_channels
    meta = unpickle( data_dir + '/batches.meta')

    label_names = meta[b'label_names']
    label_count = len(label_names)
    train_files = [ 'data_batch_%d' % d for d in range(1,6) ]
    train_data, train_labels = load_data(train_files, data_dir, label_count)
    test_data, test_labels = load_data([ 'test_batch' ], data_dir, label_count)

    print("Train data:",np.shape(train_data), np.shape(train_labels))
    print("Test data :",np.shape(test_data), np.shape(test_labels))
    print("======Load finished======")
    return (train_data, train_labels), (test_data, test_labels)

def load_data_one(file):
    batch  = unpickle(file)
    data   = batch[b'data']
    labels = batch[b'labels']
    print("Loading %s : %d." %(file, len(data)))
    return data, labels

def load_data(files, data_dir, label_count):
    global image_size, img_channels
    data, labels = load_data_one(data_dir + '/' + files[0])
    for f in files[1:]:
        data_n, labels_n = load_data_one(data_dir + '/' + f)
        data = np.append(data,data_n,axis=0)
        labels = np.append(labels,labels_n,axis=0)
    labels = np.array( [ [ float( i == label ) for i in range(label_count) ] for label in labels ] )
    data = data.reshape([-1,img_channels, image_size, image_size])
    data = data.transpose([0, 2, 3, 1])
    return data, labels

#(x_train, y_train), (x_test, y_test) = prepare_data()
#print (x_train.shape)
#print (y_train.shape)
#print (x_test.shape)
#print (y_test.shape)
#print(x_train[0])
