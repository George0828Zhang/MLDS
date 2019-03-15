
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model,load_model


# In[2]:


if __name__ == '__main__':
    
    
    ###comment this code, if you want to use your default gpu. I change to CPU becuase
    ###I get Resource exhausted on my GPU = =.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    ###
    if tf.test.gpu_device_name():
        print('GPU found')
    else:
        print("No GPU found")

    
    DNN1 = load_model("CIFAR10_DNN1.h5")
    DNN2 = load_model("CIFAR10_DNN2.h5")
    SHALLOW1 = load_model("CIFAR10_SHALLOW1.h5")
    
    print("DNN1")
    DNN1.summary()
    print()
    
    print("DNN2")
    DNN2.summary()
    print()
    
    print("SHALLOW")
    SHALLOW1.summary()
    print()
    
    DNN1_loss = np.load('CIFAR10_DNN1_LOSS.npy')
    DNN2_loss = np.load('CIFAR10_DNN2_LOSS.npy')
    SHALLOW1_loss = np.load('CIFAR10_SHALLOW1_LOSS.npy')
    plt.plot(range(1000), DNN1_loss)
    plt.plot(range(1000), DNN2_loss)
    plt.plot(range(1000), SHALLOW1_loss)
    plt.title('CIFAR10 models loss')
    plt.legend(['DNN1 loss', 'DNN2 loss','SHALLOW1 loss','ground truth'], loc='upper left')
    plt.show()
    
    DNN1_acc = np.load('CIFAR10_DNN1_ACC.npy')
    DNN2_acc = np.load('CIFAR10_DNN2_ACC.npy')
    SHALLOW1_acc = np.load('CIFAR10_SHALLOW1_ACC.npy')
    plt.plot(range(1000), DNN1_acc)
    plt.plot(range(1000), DNN2_acc)
    plt.plot(range(1000), SHALLOW1_acc)
    plt.title('CIFAR10 models accuracy')
    plt.legend(['DNN1 loss', 'DNN2 loss','SHALLOW1 loss','ground truth'], loc='upper left')
    plt.show()

