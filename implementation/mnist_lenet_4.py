import tensorflow as tf

if int(tf.__version__.split('.')[0]) > 1:
    import tensorflow.compat.v1 as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, AveragePooling2D


import random as python_random
import numpy as np
import time
import sys
#import cProfile

#python_random.seed(0)
#np.random.seed(0)
#tf.set_random_seed(0)

#from tfdeterminism import patch
#patch()

print("Do we use GPU:")
print(tf.test.is_gpu_available())

def load_data(path):
    with np.load(path, allow_pickle=True) as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        train_start_time = time.time()
        self.times = [train_start_time]
    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time())


def train(output_path):
    # mnist = keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = load_data("./mnist.npz")
    X_val, y_val = X_test[:7500, ..., np.newaxis], y_test[:7500]

    X_train, y_train = X_train[..., np.newaxis], y_train

    X_test, y_test =  X_test[7500:, ..., np.newaxis], y_test[7500:]


    X_train = np.pad(X_train, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_val = np.pad(X_val, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_test = np.pad(X_test, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_train, X_val, X_test = X_train/float(255), X_val/float(255), X_test/float(255)
    X_train -= np.mean(X_train)
    X_val -= np.mean(X_val)
    X_test -= np.mean(X_test)
    
    model = keras.models.Sequential()

    # C1: (None,32,32,1) -> (None,28,28,4).
    model.add(Conv2D(4, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=(32,32,1), padding='valid'))

    # P1: (None,28,28,4) -> (None,14,14,4).
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # C2: (None,14,14,4) -> (None,10,10,16).
    #model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='same'))
    model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid'))

    # P2: (None,10,10,16) -> (None,5,5,16).
    model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Flatten: (None,5,5,16) -> (None, 400).
    model.add(Flatten())

    # FC1: (None, 400) -> (None,120).
    model.add(Dense(120, activation='tanh'))


    # FC3: (None,120) -> (None,10).
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy']) 
    # reshape

    
    # model = keras.models.Sequential()
    # model.add(Conv2D(6, (5,5), activation="relu", padding='same', input_shape=[28,28,1]))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Conv2D(16, (5, 5), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Conv2D(120, (5, 5), activation='relu'))
    # model.add(Dropout(0.25))

    # model.add(Flatten())
    # model.add(Dense(84, activation="relu"))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation="softmax"))
    print(model.summary())


    #model.compile(loss="sparse_categorical_crossentropy",
    #                optimizer="sgd",
    #                metrics=["accuracy"])
    time_callback = TimeHistory()
    cbks = [time_callback]

    history = model.fit(X_train, y_train,batch_size=128, epochs=50, validation_data=(X_val, y_val), callbacks=cbks)
    #model.save("lenet5.h5")
    #print(history.history)
    #print(model.evaluate(X_test, y_test))
    eval_to_write = model.evaluate(X_test, y_test)
    history_to_write = history.history['loss']
    time_to_write = time_callback.times
    #print(time_callback.times)
    res = model.predict(X_test)
    #print(res[0])
    #print(np.argmax(res[0]))
    #print(y_test[0])
    write_overall_acc(eval_to_write, history_to_write, time_to_write, output_path)
    compute_per_class_accuracy(res, y_test,10, output_path)
    output_prediction_results(res, output_path)

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
        key = str(y_test[i])
        if key not in total_d:
            total_d[key] = 1
            d[key] = 0
        else:
            total_d[key] += 1

    for i in range(len(res)):
        predict_class = np.argmax(res[i])
        if y_test[i] == predict_class:
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
    args = sys.argv
    output_path = args[1]
    train(output_path)

	
#cProfile.run('train()')
