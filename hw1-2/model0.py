import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD, Adadelta
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
from sklearn.decomposition import PCA
import sys
import os


import numpy as np
import random
import math
import matplotlib.pyplot as plt


#Config
epochs = 1500
batch = 128
data_size = 10000

def trainingData():    
    def target(x):
    	assert( x >= 0 )
    	y = x * math.sin(5*math.pi *x)
    	return y
    
    x = []
    y = []
    for i in range(data_size):
        x.append(random.random())        
    y = [target(i) for i in x]
    
    return x,y

def createModel():
    model = Sequential()
    model.add(Dense(5, input_shape=(1,), activation='relu'))
    model.add(Dense(10,  activation='relu'))
    model.add(Dense(10,  activation='relu'))
    model.add(Dense(10,  activation='relu'))
    model.add(Dense(10,  activation='relu'))    
    model.add(Dense(10,  activation='relu'))
    model.add(Dense(5,  activation='relu'))
    model.add(Dense(1,  activation='linear'))
    
    optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error',
                  )
    return model

def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad



def compressed_weights(model):
    model_weights = np.array(model.get_weights())
    #print(DNN_weights)
    comp = []
    for i in range(len(model_weights)):
        model_weights[i] = model_weights[i].reshape(len(model_weights[i]),-1)
        model_weights[i] = model_weights[i].flatten()
        model_weights[i] = model_weights[i].reshape(len(model_weights[i]),-1)
        #print(model_weights[i].shape)
        for j in range(len(model_weights[i])):
            comp.append(model_weights[i][j])
    comp = np.array(comp).flatten()
    return comp
    

def get_gradients_norm(model, inputs, outputs):
    #from https://stackoverflow.com/questions/51140950/how-to-obtain-the-gradients-in-keras
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    grad_sum = 0
    for i in range(len(output_grad)):
        grad_sum += np.sum(output_grad[i])**2
    grad_norm = grad_sum ** 0.5
    return grad_norm


#Ref:
#https://stackoverflow.com/questions/46473823/how-to-get-weight-matrix-of-one-layer-at-every-epoch-in-lstm-model-based-on-kera
#https://github.com/keras-team/keras/issues/2231
class CollectWeightCallback(Callback):
    def __init__(self, layer_index):
        super(CollectWeightCallback, self).__init__()
        self.layer_index = layer_index
        self.weights = []
        
    
    def on_train_begin(self, logs=None):
        self.epochs = 0
        self.train_batch_loss = []

    def on_epoch_end(self, epoch, logs=None):        
        self.train_batch_loss.append(logs.get('loss'))
        if self.epochs % 3 ==0:
            layer = self.model.layers[self.layer_index]
            self.weights.append(layer.get_weights())
            
        self.epochs +=1   
        

if __name__ =='__main__':
    
    x_train, y_train = trainingData()
    model = createModel()   
    
    collect_interval = 3
    weights = list()
    grads = list()
    
    
    for cnt in range(epochs):
        history = model.fit(x_train, y_train, batch_size=batch, verbose = 0)
        print("epoch:%d ,loss:%s"%(cnt, history.history['loss']))
        
        grads.append (get_gradients_norm(model, x_train, y_train) )
        
        if(cnt%collect_interval == 0):
            w = compressed_weights(model)
            weights.append(w)
            
    
            
    pca = PCA(n_components=2)
    pca.fit(np.transpose(weights))
    x = []
    y = []
    a = pca.components_
    for i in range(pca.components_.shape[1]):
        x.append(pca.components_[0][i])
        y.append(pca.components_[1][i])
    plt.scatter(x,y)
    
    base_dir = "./PCA_deisgned"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        
    np.save("./PCA_deisgned/model0_x_{}".format(sys.argv[1]), x)
    np.save("./PCA_deisgned/model0_y_{}".format(sys.argv[1]), y)
        
        
        
        
        
