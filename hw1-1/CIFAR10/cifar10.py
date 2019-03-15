
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adadelta
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
cifar10 = tf.keras.datasets.cifar10


# In[2]:


def compressed_weights(model):
    model_weights = np.array(model.get_weights())
    #print(DNN_weights)
    comp = []
    for i in range(len(model_weights)):
        model_weights[i] = model_weights[i].reshape(len(model_weights[i]),-1)
        model_weights[i] = model_weights[i].flatten()
        model_weights[i] = model_weights[i].reshape(len(model_weights[i]),-1)
        print(model_weights[i].shape)
        for j in range(len(model_weights[i])):
            comp.append(model_weights[i][j])
    comp = np.array(comp).flatten()
    return comp


# In[3]:


class MY_DNN1(object):
    def __init__(self, width = 32, height = 32, channels = 3):
        self.width = width
        self.height = height
        self.channels = channels
        
        self.shape = (self.width, self.height, self.channels)
        self.optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False)
        
        self.model = self.__model()
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])
        
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
        
    def train(self, x_train, y_train, epochs = 1000, batch = 256, collect_interval = 3):
        loss = []
        accuracy = []
        weights = []
        for cnt in range(epochs):
            history = self.model.fit(x_train, y_train, batch_size=batch, verbose = 0)
            print("epoch:%d ,loss:%s, accuracy:%s "%(cnt, history.history['loss'], history.history['acc']))
            loss.append(history.history['loss'])
            accuracy.append(history.history['acc'])
            if(epochs%collect_interval == 0):
                w = compressed_weights(self.model)
                weights.append(w)
        self.loss = loss
        self.accuracy = accuracy
        self.weights = weights


    
                


# In[4]:


class MY_DNN2(object):
    def __init__(self, width = 32, height = 32, channels = 3):
        self.width = width
        self.height = height
        self.channels = channels
        
        self.shape = (self.width, self.height, self.channels)
        self.optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False)
        
        self.model = self.__model()
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])
        
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
        
    def train(self, x_train, y_train, epochs = 1000, batch = 256, collect_interval = 3):
        loss = []
        accuracy = []
        weights = []
        for cnt in range(epochs):
            history = self.model.fit(x_train, y_train, batch_size=batch, verbose = 0)
            print("epoch:%d ,loss:%s, accuracy:%s "%(cnt, history.history['loss'], history.history['acc']))
            loss.append(history.history['loss'])
            accuracy.append(history.history['acc'])
            if(epochs%collect_interval == 0):
                w = compressed_weights(self.model)
                weights.append(w)
        self.loss = loss
        self.accuracy = accuracy
        self.weights = weights
    
        


# In[5]:


class MY_SHALLOW1(object):
    def __init__(self, width = 32, height = 32, channels = 3):
        self.width = width
        self.height = height
        self.channels = channels
        
        self.shape = (self.width, self.height, self.channels)
        self.optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False)
        
        self.model = self.__model()
        self.model.compile(loss='mean_squared_error', optimizer=self.optimizer, metrics=['accuracy'])
        
    def __model(self):
        inputs = Input(shape = self.shape)
        x = Flatten(input_shape = self.shape)(inputs)
        x = Dense((self.width * self.height * self.channels + 502), activation='relu')(x)
        y = Dropout(0.2)(x)
        outputs = Dense(10,activation='softmax')(y)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        
        return model
        
    def train(self, x_train, y_train, epochs = 1000, batch = 256, collect_interval = 3):
        loss = []
        accuracy = []
        weights = []
        for cnt in range(epochs):
            history = self.model.fit(x_train, y_train, batch_size=batch, verbose = 0)
            print("epoch:%d ,loss:%s, accuracy:%s "%(cnt, history.history['loss'], history.history['acc']))
            loss.append(history.history['loss'])
            accuracy.append(history.history['acc'])
            if(epochs%collect_interval == 0):
                w = compressed_weights(self.model)
                weights.append(w)
        self.loss = loss
        self.accuracy = accuracy  
        self.weights = weights
                


# In[6]:


if __name__ == '__main__':
    #handle remote jupyter notebook 
    if 'session' in locals() and session is not None:
        print('Close interactive session')
        session.close()
    
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
    
    
    


# In[ ]:


plt.plot(range(1000), DNN1.loss)
plt.plot(range(1000), DNN1.accuracy)
pca = PCA(n_components=2)
pca.fit(DNN1.weights)
x = []
y = []
for i in range(len(pca.components_)):
    x.append(pca.components_[i][0])
    y.append(pca.components_[i][1])
plt.scatter(x,y)


# In[ ]:


DNN2 = MY_DNN2()
DNN2.train(x_train, y_train)
DNN2.model.save("CIFAR10_DNN2.h5")


# In[ ]:


plt.plot(range(1000), DNN2.loss)
plt.plot(range(1000), DNN2.accuracy)
pca = PCA(n_components=2)
pca.fit(DNN2.weights)
x = []
y = []
for i in range(len(pca.components_)):
    x.append(pca.components_[i][0])
    y.append(pca.components_[i][1])
plt.scatter(x,y)


# In[ ]:



SHALLOW1 = MY_SHALLOW1()
SHALLOW1.train(x_train, y_train)
SHALLOW1.model.save("CIFAR10_SHALLOW1.h5")


# In[ ]:



plt.plot(range(1000), SHALLOW1.loss)
plt.plot(range(1000), SHALLOW1.accuracy)
pca = PCA(n_components=2)
pca.fit(SHALLOW1.weights)
x = []
y = []
for i in range(len(pca.components_)):
    x.append(pca.components_[i][0])
    y.append(pca.components_[i][1])
plt.scatter(x,y)


# In[ ]:


np.save('CIFAR10_DNN1_LOSS', DNN1.loss)
np.save('CIFAR10_DNN1_ACC', DNN1.accuracy)

np.save('CIFAR10_DNN2_LOSS', DNN2.loss)
np.save('CIFAR10_DNN2_ACC', DNN2.accuracy)

np.save('CIFAR10_SHALLOW1_LOSS', SHALLOW1.loss)
np.save('CIFAR10_SHALLOW1_ACC', SHALLOW1.accuracy)

