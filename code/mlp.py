# import the nn-models from keras
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras import regularizers
from keras import layers
from keras import optimizers
from keras.models import model_from_json
import numpy as np


class mlp(object):
    """A generic MLP class to create 3 layer mlps with different properties. 
    """
    def __init__(self,number_input_neurons,number_hidden_neurons,input_shape,activation_function_hidden,number_output_neurons,activation_function_output,dropout_hidden,dropout_input,loss_function,regularization,optimizer):
        """Return mlp object with some properties:"""
        self.model = Sequential()
        self.number_input_neurons = number_input_neurons
        self.number_hidden_neurons = number_hidden_neurons
        self.input_shape = input_shape
        self.activation_function_hidden = activation_function_hidden
        self.number_output_neurons = number_output_neurons
        self.activation_function_output = activation_function_output
        self.dropout_hidden = dropout_hidden 
        self.dropout_input = dropout_input
        self.loss_function = loss_function
        self.regularization = regularization
        self.optimizer = optimizer
        self.history = None
        self.file = None

    def compile_mlp_model(self):
        # add the input layer and the first hidden layer with input shape, activation and dropout
        if (self.regularization == 'l1') :
            self.model.add(Dense(self.number_hidden_neurons, input_shape=self.input_shape,kernel_regularizer=regularizers.l1(0.01)))
            self.model.add(layers.Dropout(self.dropout_input,input_shape=self.input_shape))
            self.model.add(Activation(self.activation_function_hidden))
            self.model.add(Dropout(self.dropout_hidden))
        if (self.regularization == 'l2') :
            self.model.add(Dense(self.number_hidden_neurons, input_shape=self.input_shape,kernel_regularizer=regularizers.l2(0.01)))
            self.model.add(layers.Dropout(self.dropout_input,input_shape=self.input_shape))
            self.model.add(Activation(self.activation_function_hidden))
            self.model.add(Dropout(self.dropout_hidden))
        else :
            self.model.add(Dense(self.number_hidden_neurons, input_shape=self.input_shape))
            self.model.add(layers.Dropout(self.dropout_input,input_shape=self.input_shape))
            self.model.add(Activation(self.activation_function_hidden))
            self.model.add(Dropout(self.dropout_hidden))

        # add second hidden layer with input shape, activation and dropout
        self.model.add(Dense(self.number_hidden_neurons))
        self.model.add(Activation(self.activation_function_hidden))
        self.model.add(Dropout(self.dropout_hidden))

        # define the output layer
        self.model.add(Dense(self.number_output_neurons))
        self.model.add(Activation(self.activation_function_output))
        
        # compiling the sequential model
        if (self.optimizer == 'nesterov'):
            self.model.compile(loss=self.loss_function, metrics=['accuracy'], optimizer=optimizers.SGD(lr=0.01, momentum=0.8, decay=0.0005, nesterov=True))
        else:
            self.model.compile(loss=self.loss_function, metrics=['accuracy'], optimizer=optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0005, nesterov=False))

    def train_mlp_model(self,batch_size,epochs,X_test,X_train,Y_test,Y_train):
        # training the model and saving metrics in history
        self.history = self.model.fit(X_train, Y_train,batch_size=batch_size, epochs=epochs,verbose=2,validation_data=(X_test, Y_test))
    
    def predict_classes(self,X_test,batch_size):
        # predict the model and output the classification accuracy on test data set
        y_hat = self.model.predict_classes(X_test,batch_size)
        return y_hat
    
    def classification_accuracy(self,X_test,y_test,batch_size):
        # return the classification accuracy of the model
        y_hat = self.predict_classes(X_test,batch_size)
        return np.mean(y_hat==y_test)*100

    def save_model(self,name):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open(name+'.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights(name+'.h5')
        print('Saved model to disk')
        self.file = name

    def load_model(self,name):
        # load json and create model
        json_file = open(name+'.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(name+'.h5')
        print('Loaded model from disk')