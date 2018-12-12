# -*- coding: utf-8 -*-
"""
3 Layer NN to detect handwritten digits 
"""

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# import the nn-models from keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

# import the class for general 3 layer mlps 

import mlp

# load the mnist dataset and split into test and training 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# print the final input shape ready for training
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_test = np_utils.to_categorical(y_test, n_classes)

# define the input shape for the input pictures
input_shape=(784,)
number_input_neurons = 800
number_hidden_neurons = 800 
activation_function_hidden = 'relu'
number_output_neurons = 10
activation_function_output = 'softmax'
dropout_hidden = 0.5
dropout_input = 0.2
loss_function = 'categorical_crossentropy'

# initiazize mpls I-IV with the corresponding properties
mlp_sgd = mlp.mlp(number_input_neurons,number_hidden_neurons,input_shape,activation_function_hidden,number_output_neurons,activation_function_output,dropout_hidden,dropout_input,loss_function,'none','sgd')
mlp_sgd_nesterov = mlp.mlp(number_input_neurons,number_hidden_neurons,input_shape,activation_function_hidden,number_output_neurons,activation_function_output,dropout_hidden,dropout_input,loss_function,'none','nesterov')
mlp_sgd_nesterov_l1 = mlp.mlp(number_input_neurons,number_hidden_neurons,input_shape,activation_function_hidden,number_output_neurons,activation_function_output,dropout_hidden,dropout_input,loss_function,'l1','nesterov')
mlp_sgd_nesterov_l2 = mlp.mlp(number_input_neurons,number_hidden_neurons,input_shape,activation_function_hidden,number_output_neurons,activation_function_output,dropout_hidden,dropout_input,loss_function,'l2','nesterov')

# define the training parameters for the mlps I-IV
batch_size = 6
epochs = 20

# train the mpls I-IV
#mlp_sgd.compile_mlp_model()
#mlp_sgd_nesterov.compile_mlp_model()
#mlp_sgd_nesterov_l1.compile_mlp_model()
mlp_sgd_nesterov_l2.compile_mlp_model()

print('Train MLP I: SGD without nesterov...')
#mlp_sgd.train_mlp_model(batch_size,epochs,X_test,X_train,Y_test,Y_train)
print('Train MLP II: SGD with nesterov...')
#mlp_sgd_nesterov.train_mlp_model(batch_size,epochs,X_test,X_train,Y_test,Y_train)
print('Train MLP III: SGD with nesterov and l1 regularization...')
#mlp_sgd_nesterov_l1.train_mlp_model(batch_size,epochs,X_test,X_train,Y_test,Y_train)
print('Train MLP IV: SGD without nesterov and l2 regularization...')
mlp_sgd_nesterov_l2.train_mlp_model(batch_size,epochs,X_test,X_train,Y_test,Y_train)

# plotting the metrics for the mlps I-IV
fig = plt.figure()
plt.subplot(2,1,1)
#plt.plot(mlp_sgd.history.history['acc'])
#plt.plot(mlp_sgd.history.history['val_acc'])
#plt.plot(mlp_sgd_nesterov.history.history['acc'])
#plt.plot(mlp_sgd_nesterov.history.history['val_acc'])
#plt.plot(mlp_sgd_nesterov_l1.history.history['acc'])
#plt.plot(mlp_sgd_nesterov_l1.history.history['val_acc'])
plt.plot(mlp_sgd_nesterov_l2.history.history['acc'])
plt.plot(mlp_sgd_nesterov_l2.history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train_sgd', 'test_sgd'])#, 'train_sgd_nesterov', 'test_sgd_nesterov','train_sgd_nesterov_l1', 'test_sgd_nesterov_l1','train_sgd_nesterov_l2', 'test_sgd_nesterov_l2'], loc='lower right')

plt.subplot(2,1,2)
#plt.plot(mlp_sgd.history.history['loss'])
#plt.plot(mlp_sgd.history.history['val_loss'])
#plt.plot(mlp_sgd_nesterov.history.history['loss'])
#plt.plot(mlp_sgd_nesterov.history.history['val_loss'])
#plt.plot(mlp_sgd_nesterov_l1.history.history['loss'])
#plt.plot(mlp_sgd_nesterov_l1.history.history['val_loss'])
plt.plot(mlp_sgd_nesterov_l2.history.history['loss'])
plt.plot(mlp_sgd_nesterov_l2.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train_sgd', 'test_sgd'])#,'train_sgd_nesterov', 'test_sgd_nesterov','train_sgd_nesterov_l1', 'test_sgd_nesterov_l1','train_sgd_nesterov_l2', 'test_sgd_nesterov_l2'], loc='upper right')

plt.tight_layout()

plt.savefig('results.png')

fig
