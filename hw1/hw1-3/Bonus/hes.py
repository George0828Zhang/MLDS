TRAIN_LOSS = []
TEST_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
BATCH_SIZE = [4,8,16,32,64,128,256,512,1024,2048][::-1]
HES_NORM = []


from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Sequential
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import os

    
for b in BATCH_SIZE:
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')
    
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    
   
    
    
    x = tf.placeholder(tf.float32, [None, 28, 28, 1])
    y_true = tf.placeholder(tf.float32, [None, 10, ])
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        model = Sequential()
        model.add(Conv2D(22, (3, 3), input_shape=(28, 28, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(22, (2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.1))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(10, activation='softmax'))
        
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=b, epochs=5)
        
        TRAIN = model.evaluate(x_train, y_train)
        TEST =  model.evaluate(x_test, y_test)
        TRAIN_LOSS.append(TRAIN[0])
        TRAIN_ACC.append(TRAIN[1])
        
        TEST_LOSS.append(TEST[0])
        TEST_ACC.append(TEST[1])






        sess.run(init)
        y_pred = model.apply(x)
        loss = keras.losses.categorical_crossentropy(y_true, y_pred)
        hessian = tf.hessians(loss, model.trainable_variables)
        hes = sess.run(hessian, feed_dict={
                       x: x_train[:500], y_true: y_train[:500]})
    print("checkpoint")
    
    
    for i in range(len(hes)):
        dim = len(hes[i].shape)
        
        m_size = 1
        for j in hes[i].shape[:int(dim/2)]:
            m_size = m_size * j
        
        hes[i] = hes[i].reshape((m_size, m_size))
        
    norm = 0
    for i in hes[:4]:
        try:
            norm += np.linalg.norm(i, 2)
        except:
            norm += 0
            
    HES_NORM.append(norm)
    del(hes)



'''
TRAIN_LOSS = []
TEST_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
BATCH_SIZE = [4,8,16,32,64,128,256,512,1024,2048]
HES_NORM = []
'''
np.save("TRAIN_LOSS.npy", TRAIN_LOSS)
np.save("TEST_LOSS.npy", TEST_LOSS)
np.save("TRAIN_ACC.npy", TRAIN_ACC)
np.save("TEST_ACC.npy", TEST_ACC ) 
np.save("HES_NORM.npy", HES_NORM)