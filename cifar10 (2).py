#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adadelta
from keras.layers import Input, Flatten, Dense, Dropout
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import keras
from keras import backend as K


cifar10 = tf.keras.datasets.cifar10
epoch_range = 500


# In[2]:


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
    


# In[3]:


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


# In[4]:

def minimal_ratio(model, sample_amount, x, y, has_metrics=False):
	max_len = 1e-5
	weights = np.array(model.get_weights())
	config = model.get_config()
	pmodel = Model.from_config(config)
	if has_metrics:
		base_loss, metrics = model.evaluate(x=x, y=y)
	else:
		base_loss = model.evaluate(x=x, y=y)

	num_greater = 0

	for i in range(sample_amount):
		noise = np.random.rand(weights.size)
		noise_len = np.random.uniform(0.0, max_len)
		noise *=  (noise_len / np.linalg.norm(noise))
		noise.reshape(weights.shape)

		sample = weights + noise
		sample = [ a for a in sample ]

		pmodel.set_weight(sample)
		if has_metrics:
			loss, metrics = pmodel.evaluate(x=x, y=y)
		else:
			loss = pmodel.evaluate(x=x, y=y)
		if loss > base_loss:
			num_greater += 1
	return num_greater / sample_amount




class MY_DNN1(object):
    def __init__(self, width = 32, height = 32, channels = 3):
        self.width = width
        self.height = height
        self.channels = channels
        
        self.shape = (self.width, self.height, self.channels)
        self.optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False)
        
        self.model = self.__model()
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
    def __model(self):
        inputs = Input(shape = self.shape)
        x = Flatten(input_shape = self.shape)(inputs)
        x = Dense((self.width * self.height * self.channels), activation='relu')(x)
        x = Dense(512)(x)
        y = Dropout(0.2)(x)
        outputs = Dense(10,activation='softmax')(y)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        
        return model
    
    
    def train(self, x_train, y_train, epochs = 500, batch = 256, collect_interval = 3):
        loss = []
        accuracy = []
        weights = []
        grads = []
        
        for cnt in range(epochs):
            history = self.model.fit(x_train, y_train, batch_size=batch, verbose = 0)
            print("epoch:%d ,loss:%s, accuracy:%s "%(cnt, history.history['loss'], history.history['acc']))
            loss.append(history.history['loss'])
            accuracy.append(history.history['acc'])
            grads.append(get_gradients_norm(self.model, x_train, y_train))
            
            if(cnt%collect_interval == 0):
                w = compressed_weights(self.model)
                weights.append(w)
            
        self.loss = loss
        self.accuracy = accuracy
        self.weights = weights
        self.grads = grads

    
                


# In[ ]:


class MY_DNN2(object):
    def __init__(self, width = 32, height = 32, channels = 3):
        self.width = width
        self.height = height
        self.channels = channels
        
        self.shape = (self.width, self.height, self.channels)
        self.optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False)
        
        self.model = self.__model()
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
    def __model(self):
        inputs = Input(shape = self.shape)
        x = Flatten(input_shape = self.shape)(inputs)
        x = Dense((self.width * self.height * self.channels), activation='relu')(x)
        x = Dense(400)(x)
        x = Dense(850)(x)
        y = Dropout(0.2)(x)
        outputs = Dense(10,activation='softmax')(y)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        
        return model
        
    def train(self, x_train, y_train, epochs = 500, batch = 256, collect_interval = 3):
        loss = []
        accuracy = []
        weights = []
        grads = []
        
        for cnt in range(epochs):
            history = self.model.fit(x_train, y_train, batch_size=batch, verbose = 0)
            print("epoch:%d ,loss:%s, accuracy:%s "%(cnt, history.history['loss'], history.history['acc']))
            loss.append(history.history['loss'])
            accuracy.append(history.history['acc'])
            grads.append(get_gradients_norm(self.model, x_train, y_train))
            
            if(cnt%collect_interval == 0):
                w = compressed_weights(self.model)
                weights.append(w)

            
        self.loss = loss
        self.accuracy = accuracy
        self.weights = weights
        self.grads = grads
    
        


# In[ ]:


class MY_SHALLOW1(object):
    def __init__(self, width = 32, height = 32, channels = 3):
        self.width = width
        self.height = height
        self.channels = channels
        
        self.shape = (self.width, self.height, self.channels)
        self.optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False)
        
        self.model = self.__model()
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        
    def __model(self):
        inputs = Input(shape = self.shape)
        x = Flatten(input_shape = self.shape)(inputs)
        x = Dense((self.width * self.height * self.channels + 502), activation='relu')(x)
        y = Dropout(0.2)(x)
        outputs = Dense(10,activation='softmax')(y)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        
        return model
        
    def train(self, x_train, y_train, epochs = 500, batch = 256, collect_interval = 3):
        loss = []
        accuracy = []
        weights = []
        grads = []
        
        for cnt in range(epochs):
            history = self.model.fit(x_train, y_train, batch_size=batch, verbose = 0)
            print("epoch:%d ,loss:%s, accuracy:%s "%(cnt, history.history['loss'], history.history['acc']))
            loss.append(history.history['loss'])
            accuracy.append(history.history['acc'])
            grads.append(get_gradients_norm(self.model, x_train, y_train))
            
            if(cnt%collect_interval == 0):
                w = compressed_weights(self.model)
                weights.append(w)
            
        self.loss = loss
        self.accuracy = accuracy
        self.weights = weights
        self.grads = grads
                


# In[ ]:


if __name__ == '__main__':
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")
    
    (x_train, y_train),(x_test, y_test) = cifar10.load_data()
    #normalization for not getting to big output for 'relu' and other activation
    x_train = x_train/255
    x_test = x_test/255
    y_train = np_utils.to_categorical(y_train, num_classes=10)
    y_test = np_utils.to_categorical(y_test, num_classes=10)

    
    #for dense layer, the parameters = (input_shape + 1) * size;
    DNN1 = MY_DNN1()
    DNN1.train(x_train, y_train)
    DNN1.model.save("CIFAR10_DNN1.h5")
    np.save('CIFAR10_DNN1_LOSS', DNN1.loss)
    np.save('CIFAR10_DNN1_ACC', DNN1.accuracy)
    np.save('CIFAR10_DNN1_WEIGHTS',DNN1.weights)
    np.save('CIFAR10_DNN1_GRADS',DNN1.grads)
    
    
    


# In[ ]:


plt.plot(range(epoch_range), DNN1.loss)
plt.plot(range(epoch_range), DNN1.accuracy)
plt.plot(range(epoch_range), DNN1.grads)
plt.show()
#     pca = PCA(n_components=2)
#     pca.fit(DNN1.weights)
#     x = []
#     y = []
#     for i in range(len(pca.components_)):
#         x.append(pca.components_[i][0])
#         y.append(pca.components_[i][1])
#     plt.scatter(x,y)


# In[ ]:


DNN2 = MY_DNN2()
DNN2.train(x_train, y_train)
DNN2.model.save("CIFAR10_DNN2.h5")
np.save('CIFAR10_DNN2_LOSS', DNN2.loss)
np.save('CIFAR10_DNN2_ACC', DNN2.accuracy)
np.save('CIFAR10_DNN1_WEIGHTS',DNN2.weights)
np.save('CIFAR10_DNN1_GRADS',DNN2.grads)


# In[ ]:


plt.plot(range(epoch_range), DNN2.loss)
plt.plot(range(epoch_range), DNN2.accuracy)
plt.plot(range(epoch_range), DNN2.grads)
plt.show()
#     pca = PCA(n_components=2)
#     pca.fit(DNN2.weights)
#     x = []
#     y = []
#     for i in range(len(pca.components_)):
#         x.append(pca.components_[i][0])
#         y.append(pca.components_[i][1])
#     plt.scatter(x,y)


# In[ ]:



SHALLOW1 = MY_SHALLOW1()
SHALLOW1.train(x_train, y_train)
SHALLOW1.model.save("CIFAR10_SHALLOW1.h5")
np.save('CIFAR10_SHALLOW1_LOSS', SHALLOW1.loss)
np.save('CIFAR10_SHALLOW1_ACC', SHALLOW1.accuracy)
np.save('CIFAR10_DNN1_WEIGHTS',SHALLOW1.weights)
np.save('CIFAR10_DNN1_GRADS',SHALLOW1.grads)


# In[ ]:



plt.plot(range(epoch_range), SHALLOW1.loss)
plt.plot(range(epoch_range), SHALLOW1.accuracy)
plt.plot(range(epoch_range), SHALLOW1.grads)
#     pca = PCA(n_components=2)
#     pca.fit(SHALLOW1.weights)
#     x = []
#     y = []
#     for i in range(len(pca.components_)):
#         x.append(pca.components_[i][0])
#         y.append(pca.components_[i][1])
#     plt.scatter(x,y)

