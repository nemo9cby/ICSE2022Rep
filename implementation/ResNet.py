import tensorflow as tf
from tensorflow import keras
#import keras
import argparse
import numpy as np
from tensorflow.keras.datasets import cifar10, cifar100
#from keras.preprocessing.image import ImageDataGenerator
#from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers, regularizers
#from keras import backend as K

import random as python_random
import time

#python_random.seed(0)
#np.random.seed(0)


import sys
# import the path where you put cifar10_load.py or cifar100_load.py
sys.path.insert(1, '/path_to_change')

import cifar10_load




# set parameters via parser
parser = argparse.ArgumentParser()
parser.add_argument('-b','--batch_size', type=int, default=128, metavar='NUMBER',
                help='batch size(default: 128)')
parser.add_argument('-e','--epochs', type=int, default=200, metavar='NUMBER',
                help='epochs(default: 200)')
parser.add_argument('-n','--stack_n', type=int, default=5, metavar='NUMBER',
                help='stack number n, total layers = 6 * n + 2 (default: 5)')
parser.add_argument('-d','--dataset', type=str, default="cifar10", metavar='STRING',
                help='dataset. (default: cifar10)')
parser.add_argument('-o','--output_path', type=str, default="tmp.txt", metavar='STRING',
                help='output path. (default: tmp.txt)')
parser.add_argument('-s', '--seed', type=int, default=-1,metavar='NUMBER',
                help='random seed. (default:-1 indicating not setting seed')

args = parser.parse_args()

if args.seed != -1:
    python_random.seed(args.seed)
    np.random.seed(args.seed)
    #tf.set_random_seed(args.seed)
    tf.random.set_seed(args.seed)
# set GPU memory 
#if('tensorflow' == K.backend()):
#    import tensorflow as tf
    #from tfdeterminism import patch
    #patch()
    #from keras.backend.tensorflow_backend import set_session
    #config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)

stack_n            = args.stack_n
layers             = 6 * stack_n + 2
num_classes        = 10
img_rows, img_cols = 32, 32
img_channels       = 3
batch_size         = args.batch_size
epochs             = args.epochs
iterations         = 50000 // batch_size + 1
weight_decay       = 1e-4
output_path = args.output_path

def color_preprocessing(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.307, 122.95, 113.865]
    std  = [62.9932, 62.0887, 66.7048]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
    return x_train, x_test


def scheduler(epoch):
    if epoch < 81:
        return 0.1
    if epoch < 122:
        return 0.01
    return 0.001


def residual_network(img_input,classes_num=10,stack_n=5):
    
    def residual_block(x,o_filters,increase=False):
        stride = (1,1)
        if increase:
            stride = (2,2)

        o1 = Activation('relu')(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x))
        conv_1 = Conv2D(o_filters,kernel_size=(3,3),strides=stride,padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o1)
        o2  = Activation('relu')(tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(conv_1))
        conv_2 = Conv2D(o_filters,kernel_size=(3,3),strides=(1,1),padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(o2)
        if increase:
            projection = Conv2D(o_filters,kernel_size=(1,1),strides=(2,2),padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(o1)
            block = add([conv_2, projection])
        else:
            block = add([conv_2, x])
        return block

    # build model ( total layers = stack_n * 3 * 2 + 2 )
    # stack_n = 5 by default, total layers = 32
    # input: 32x32x3 output: 32x32x16
    x = Conv2D(filters=16,kernel_size=(3,3),strides=(1,1),padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(img_input)

    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x,16,False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x,32,True)
    for _ in range(1,stack_n):
        x = residual_block(x,32,False)
    
    # input: 16x16x32 output: 8x8x64
    x = residual_block(x,64,True)
    for _ in range(1,stack_n):
        x = residual_block(x,64,False)

    x = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(classes_num,activation='softmax',kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
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
    with open(dest, "a") as f:
        f.write("overall_acc," + str(evaluate_res[1])+'\n')
        f.write("loss,"+ str(history)+'\n')
        f.write("time,"+str(time_to_write) + '\n')

def output_prediction_results(res, dest):
    with open(dest, 'a') as f:
        for i in range(len(res)):
            f.write(str(np.argmax(res[i])) + '\n')

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
        print (f"accuracy for class {key}: {acc}")
        with open(dest, 'a') as f:
            f.write(f"{key},{acc}\n")
    total_acc = total_correct/float(len(y_test))
    print (f"total acc {total_acc}")



if __name__ == '__main__':

    print("========================================") 
    print("MODEL: Residual Network ({:2d} layers)".format(6*stack_n+2)) 
    print("BATCH SIZE: {:3d}".format(batch_size)) 
    print("WEIGHT DECAY: {:.4f}".format(weight_decay))
    print("EPOCHS: {:3d}".format(epochs))
    print("DATASET: {:}".format(args.dataset))


    print("== LOADING DATA... ==")
    # load data
    # global num_classes
    if args.dataset == "cifar100":
        num_classes = 100
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = cifar10_load.prepare_data()# cifar10.load_data()
        #(x_train, y_train), (x_test, y_test) = cifar10.load_data()
    #y_train = keras.utils.to_categorical(y_train, num_classes)
    #y_test = keras.utils.to_categorical(y_test, num_classes)
    

    print("== DONE! ==\n== COLOR PREPROCESSING... ==")
    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)


    print("== DONE! ==\n== BUILD MODEL... ==")
    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output    = residual_network(img_input,num_classes,stack_n)
    resnet    = Model(img_input, output)
    
    # print model architecture if you need.
    # print(resnet.summary())


    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    #adam = optimizers.Adam(lr=0.01)
    #resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    #cbks = [TensorBoard(log_dir='./resnet_{:d}_{}/'.format(layers,args.dataset), histogram_freq=0),
    #        LearningRateScheduler(scheduler)]
    time_cbk = TimeHistory()
    cbks = [time_cbk, LearningRateScheduler(scheduler)]
    # dump checkpoint if you need.(add it to cbks)
    # ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)

    # set data augmentation
    print("== USING REAL-TIME DATA AUGMENTATION, START TRAIN... ==")
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    x_val = x_test[:7500]
    y_val = y_test[:7500]
    x_test = x_test[7500:]
    y_test = y_test[7500:]

    # start training
    history = resnet.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                         steps_per_epoch=iterations,
                         epochs=epochs,
                         callbacks=cbks,
                         validation_data=(x_val, y_val))

    #output_path = "./tmp.txt"
    eval_to_write = resnet.evaluate(x_test, y_test)
    history_to_write = history.history['loss']
    time_to_write = time_cbk.times
    res = resnet.predict(x_test)
    write_overall_acc(eval_to_write, history_to_write, time_to_write, output_path)
    compute_per_class_accuracy(res, y_test,10, output_path)
    output_prediction_results(res, output_path)

    #resnet.save('resnet_{:d}_{}.h5'.format(layers,args.dataset))
    resnet.save('resnet_{}.h5'.format(output_path.split("/")[-1]))
