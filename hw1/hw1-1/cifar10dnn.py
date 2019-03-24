#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.models import Model #,Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Convolution2D, MaxPooling2D
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD

import numpy as np
import math
import random
import matplotlib.pyplot as plt

epoch = 50
batch_size = 256

def unpickle(file):
	import pickle
	with open(file, 'rb') as fo:
		d = pickle.load(fo, encoding='bytes')
	return d

if __name__ == "__main__":

	data = []
	labels = []

	#for i in range(1,6):
	#	dic = unpickle("cifar10/data_batch_{}".format(i))
	#	for d in dic[b'data']:
	#		r = d[:1024].reshape((32,32))
	#		g = d[1024:2048].reshape((32,32))
	#		b = d[2048:].reshape((32,32))
	#		# data.append([r,g,b])
	#		# np.expand_dims(x, axis=1)
	#		data.append(np.expand_dims(0.299*r+0.587*g+0.114*b, axis=2))

	#	for l in dic[b'labels']:
	#		labels.append([(1 if l == x else 0) for x in range(10)])
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	y_train = tf.keras.utils.to_categorical(y_train, 10)
	y_test = tf.keras.utils.to_categorical(y_test, 10)
	inputs = Input(shape=(32,32,1))

	# x = Convolution2D(filters=1,kernel_size=3,strides=4,data_format="channels_first",activation='relu',padding='same')(inputs)
	# x = MaxPooling2D(pool_size=3,strides=2,padding='same')(x)
	# x = Flatten()(x)
	# x = Dense(512, activation='relu')(x)
	# x = Dense(512, activation='relu')(x)
	# x = Dense(128, activation='relu')(x)
	# x = Dense(128, activation='relu')(x)
	# x = Dense(128, activation='relu')(x)
	# x = Dense(64, activation='relu')(x)
	# predictions = Dense(10, activation='softmax')(x)

	x = Convolution2D(filters=6,kernel_size=3,strides=1,activation='relu',padding='same')(inputs)
	x = MaxPooling2D(pool_size=2,strides=2,padding='same')(x)
	x = Convolution2D(filters=16,kernel_size=3,strides=1,activation='relu',padding='same')(x)
	x = MaxPooling2D(pool_size=2,strides=2,padding='same')(x)
	x = Flatten()(x)
	x = Dense(120,activation='relu')(x)
	x = Dense(84,activation='relu')(x)
	predictions = Dense(10,activation='softmax')(x)

	model = Model(inputs=inputs, outputs=predictions)
	optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False)
	# optimizer = tf.keras.optimizers.RMSprop(lr=0.01, decay=1e-6)
	model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])

	history = model.fit(x=np.asarray(data),y=np.asarray(labels), epochs=epoch,batch_size=batch_size)#validation_split=0.20

	plt.plot(range(epoch), history.history['loss'])
	plt.plot(range(epoch), history.history['accuracy'])
	model.save("deepModel_cifar10.h5")



	plt.title('Model loss & accuracy')
	plt.ylabel('Value')
	plt.xlabel('Epoch')
	plt.legend(['loss', 'accuracy'], loc='upper right')
	plt.show()



	# v_data = []
	# v_labels = []

	# for i in range(test_size):
	# 	x = random.uniform(domain[0], domain[1])
	# 	v_data.append(x)
	# 	v_labels.append(target(x))

	# loss = model.evaluate(np.asarray(v_data), np.asarray(v_labels), batch_size=128)
	# print ("Loss={:.5f}".format(loss))
