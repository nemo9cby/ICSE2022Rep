# import keras
import numpy as np
from tensorflow import keras
import tensorflow as tf


from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Input, add, Activation, Flatten, AveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
import os
import time

#os.environ['TF_DETERMINISTIC_OPS'] = 'true'


DEPTH              = 28
WIDE               = 10
IN_FILTERS         = 16

CLASS_NUM          = 100
IMG_ROWS, IMG_COLS = 32, 32
IMG_CHANNELS       = 3

BATCH_SIZE         = 128
EPOCHS             = 200
ITERATIONS         = 50000 // BATCH_SIZE + 1
WEIGHT_DECAY       = 0.0005
LOG_FILE_PATH      = './w_resnet/'

import random as python_random

#python_random.seed(123)
#np.random.seed(123)
#tf.random.set_seed(123)

import sys
sys.path.insert(1, './')

import cifar10_load
import cifar100_load

# from keras import backend as K
# set GPU memory 
# if('tensorflow' == K.backend()):

    #from tfdeterminism import patch
    #patch()
    #tf.set_random_seed(0)

    # from keras.backend.tensorflow_backend import set_session
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

def scheduler(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 120:
        return 0.02
    if epoch < 160:
        return 0.004
    return 0.0008

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.3, 123.0, 113.9]
    std  = [63.0,  62.1,  66.7]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    return x_train, x_test

def wide_residual_network(img_input,classes_num,depth,k):
    print('Wide-Resnet %dx%d' %(depth, k))
    n_filters  = [16, 16*k, 32*k, 64*k]
    n_stack    = (depth - 4) // 6

    def conv3x3(x,filters):
        return Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=False)(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def residual_block(x,out_filters,increase=False):
        global IN_FILTERS
        stride = (1,1)
        if increase:
            stride = (2,2)
            
        o1 = bn_relu(x)
        
        conv_1 = Conv2D(out_filters,
            kernel_size=(3,3),strides=stride,padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(WEIGHT_DECAY),
            use_bias=False)(o1)

        o2 = bn_relu(conv_1)
        
        conv_2 = Conv2D(out_filters, 
            kernel_size=(3,3), strides=(1,1), padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(WEIGHT_DECAY),
            use_bias=False)(o2)
        if increase or IN_FILTERS != out_filters:
            proj = Conv2D(out_filters,
                                kernel_size=(1,1),strides=stride,padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=l2(WEIGHT_DECAY),
                                use_bias=False)(o1)
            block = add([conv_2, proj])
        else:
            block = add([conv_2,x])
        return block

    def wide_residual_layer(x,out_filters,increase=False):
        global IN_FILTERS
        x = residual_block(x,out_filters,increase)
        IN_FILTERS = out_filters
        for _ in range(1,int(n_stack)):
            x = residual_block(x,out_filters)
        return x


    x = conv3x3(img_input,n_filters[0])
    x = wide_residual_layer(x,n_filters[1])
    x = wide_residual_layer(x,n_filters[2],increase=True)
    x = wide_residual_layer(x,n_filters[3],increase=True)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)
    x = Dense(classes_num,
        activation='softmax',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=False)(x)
    return x



class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        train_start_time = time.time()
        self.times = [train_start_time]
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time())

def write_overall_acc(evaluate_res, history, time_to_write, dest):
    # with open(dest, "a") as f:
    #     f.write("overall_acc," + str(evaluate_res[1])+'\n')
    #     f.write("loss,"+ str(history)+'\n')
    #     f.write("time,"+str(time_to_write) + '\n')
    print("overall_acc," + str(evaluate_res[1]))
    print("loss,"+ str(history))
    print("time,"+str(time_to_write))

def output_prediction_results(res, dest):
    # with open(dest, 'a') as f:
    #     for i in range(len(res)):
    #         f.write(str(np.argmax(res[i])) + '\n')
    for i in range(len(res)):
        print(str(np.argmax(res[i])))

def compute_per_class_accuracy(res, y_test, num_of_class, dest):
    #for i in range(len(res)):
    d = {}
    total_d = {}
    #for i in range(10):

    for i in range(len(res)):
        key = str(np.argmax(y_test[i]))
        if key not in total_d:
            total_d[key] = 1
            d[key] = 0
        else:
            total_d[key] += 1

    for i in range(len(res)):
        predict_class = np.argmax(res[i])
        if np.argmax(y_test[i]) == predict_class:
            key = str(predict_class)
            d[key] += 1
            
        #print(f"predict {predict_class} actual {y_test[i]}")
    #print (total_d)
    #print (d)
    total_correct = 0
    for i in range(num_of_class):
        key = str(i)
        total_correct += d[key]
        acc = d[key]/float(total_d[key])
        #print (f"accuracy for class {key}: {acc}")
        #with open(dest, 'a') as f:
            #f.write(f"{key},{acc}\n")
        print(f"{key},{acc}")
    total_acc = total_correct/float(len(y_test))
    #print (f"total acc {total_acc}")


def main():
    #args = sys.argv
    #output_path = args[1]
    output_path = ""
    # load data
    #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    (x_train, y_train), (x_test, y_test) = cifar100_load.prepare_data()
    #y_train = keras.utils.to_categorical(y_train, CLASS_NUM)
    #y_test = keras.utils.to_categorical(y_test, CLASS_NUM)
    
    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    # build network
    img_input = Input(shape=(IMG_ROWS,IMG_COLS,IMG_CHANNELS))
    output = wide_residual_network(img_input,CLASS_NUM,DEPTH,WIDE)
    resnet = Model(img_input, output)
    print(resnet.summary())
    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    #tb_cb = TensorBoard(log_dir=LOG_FILE_PATH, histogram_freq=0)
    time_cbk = TimeHistory()
    cbks = [time_cbk, LearningRateScheduler(scheduler)]

    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, time_cbk]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='reflect')

    datagen.fit(x_train)
    x_val = x_test[:7500]
    y_val = y_test[:7500]
    x_test = x_test[7500:]
    y_test = y_test[7500:]


    # start training
    history = resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=BATCH_SIZE),
                        steps_per_epoch=ITERATIONS,
                        epochs=EPOCHS,
                        callbacks=cbks,
                        validation_data=(x_val, y_val))

    eval_to_write = resnet.evaluate(x_test, y_test)
    history_to_write = history.history['loss']
    time_to_write = time_cbk.times
    res = resnet.predict(x_test)
    write_overall_acc(eval_to_write, history_to_write, time_to_write , output_path)
    compute_per_class_accuracy(res, y_test,CLASS_NUM, output_path)
    output_prediction_results(res, output_path)

    #resnet.save('wresnet.h5')

#main()
