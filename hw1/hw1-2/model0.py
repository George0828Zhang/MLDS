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
epochs = 300
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
    weights_whole = list()
    weights_first = list()
    grads = list()
    loss = list()    
    
    for cnt in range(epochs):
        history = model.fit(x_train, y_train, batch_size=batch, verbose = 0)
        print("epoch:%d ,loss:%s"%(cnt, history.history['loss']))
        
        grads.append (get_gradients_norm(model, x_train, y_train) )
        loss.append(history.history['loss'])
        
        if(cnt%collect_interval == 0):
            w = compressed_weights(model)
            weights_whole.append(w)
            weights_first.append(w[:10])
            
    
    ###pca_whole
    pca_whole = PCA(n_components=2)
    pca_whole.fit(np.transpose(weights_whole))
    x_whole = []
    y_whole = []
    for i in range(pca_whole.components_.shape[1]):
        x_whole.append(pca_whole.components_[0][i])
        y_whole.append(pca_whole.components_[1][i])
    #plt.scatter(x_whole,y_whole)
    
    
    
    ###pca_first
    pca_first = PCA(n_components=2)
    pca_first.fit(np.transpose(weights_first))
    x_first = []
    y_first = []
    for i in range(pca_first.components_.shape[1]):
        x_first.append(pca_first.components_[0][i])
        y_first.append(pca_first.components_[1][i])
    #plt.scatter(x_first,y_first)
    
    
    ###save data
    base_dir = "./designed_pca_loss_grads"
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    model_base_dir = "./designed_models"
    if not os.path.exists(model_base_dir):
        os.mkdir(model_base_dir)
        
    np.save(base_dir + "/model0_x_w_{}".format(sys.argv[1]), x_whole)
    np.save(base_dir + "/model0_y_w_{}".format(sys.argv[1]), y_whole)
    np.save(base_dir + "/model0_x_f_{}".format(sys.argv[1]), x_first)
    np.save(base_dir + "/model0_y_f_{}".format(sys.argv[1]), y_first)

    
    np.save(base_dir + "/model0_loss_{}".format(sys.argv[1]), loss)
    np.save(base_dir + "/model0_grads_{}".format(sys.argv[1]), grads)
    model.save(model_base_dir + "/model0_{}.h5".format(sys.argv[1]))
        
        
        
        
