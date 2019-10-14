#!{sys.executable} -m pip install keras

import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense,Dropout
from keras.optimizers import SGD, Adam


#Design choices:
# 'relu' performs better on a Deep NN compared with others activation functions
# We are in the presence of a non exclusive Multi-Class classification problem so,
# the last layer will contain as many nodes as the number of classes,
# each of them applying activation = 'sigmoid' and loss = 'binary_crossentropy'
# In general adaptive learning rate optimzers converge faster, 
# so both 'adam' or 'rmsprop' optimzers are good options

#Network parameters 
input_size = 2 
output_activation = 'sigmoid'
activation = 'relu'     
output_size = 15

#Trainning parameters
loss = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['accuracy']
epochs = 50
verbose = 0
v_split= 0.1

# parameters for Bayesian inteligent search with hyperopt and hyperas
x_train , x_test, y_train, y_test = None, None, None, None

def get_NN_model():
    
    #when building several models is always a good idea to clear the model from memory     
    K.clear_session()
    
    model = Sequential()

    # neural network model, apply dropout to try to prevent overfitting
    model.add(Dense(128, input_shape = (input_size,), activation = activation))    
    model.add(Dropout(0.3))
    model.add(Dense(64, activation = activation))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation = activation))
    model.add(Dropout(0.1))    
    model.add(Dense(output_size, activation = output_activation))
    
    return model

