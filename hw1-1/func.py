import tensorflow as tf
from tensorflow.keras.models import Model #,Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD, Adadelta

import numpy as np
import math
import random
import matplotlib.pyplot as plt

epoch = 5000
data_size = 10000
batch_size = 128
domain = (0, 1)

def target(x):
	assert( x >= 0 )
	# y = math.sin(math.sqrt(x))
	# y = math.sin(5*math.pi*x)/(5*math.pi*x)
	y = x * math.sin(5*math.pi *x)
	return y

if __name__ == "__main__":

	data = []
	labels = []

	random.seed()

	for i in range(data_size):
		x = random.uniform(domain[0], domain[1])
		data.append(x)
		labels.append(target(x))


	inputs = Input(shape=(1,))

	# a layer instance is callable on a tensor, and returns a tensor

	x = Dense(5, activation='relu')(inputs)
	x = Dense(10, activation='relu')(x)
	x = Dense(10, activation='relu')(x)
	x = Dense(10, activation='relu')(x)
	x = Dense(10, activation='relu')(x)
	x = Dense(10, activation='relu')(x)
	x = Dense(5, activation='relu')(x)
	predictions = Dense(1, activation='linear')(x)

	model = Model(inputs=inputs, outputs=predictions)
	optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False)
	model.compile(loss='mean_squared_error',optimizer=optimizer)
	history = model.fit(x=np.asarray(data),y=np.asarray(labels), epochs=epoch,batch_size=batch_size)#validation_split=0.20

	plt.plot(range(epoch), history.history['loss'])
	model.save("Model0.h5")




	x = Dense(10, activation='relu')(inputs)
	x = Dense(18, activation='relu')(x)
	x = Dense(15, activation='relu')(x)
	x = Dense(4, activation='relu')(x)
	predictions = Dense(1, activation='linear')(x)

	model = Model(inputs=inputs, outputs=predictions)
	optimizer = SGD(lr=0.01, momentum=0.9, decay=1e-8, nesterov=False)
	model.compile(loss='mean_squared_error',optimizer=optimizer)
	history = model.fit(x=np.asarray(data),y=np.asarray(labels), epochs=epoch,batch_size=batch_size)#validation_split=0.20

	plt.plot(range(epoch), history.history['loss'])
	model.save("Model1.h5")






	inputs = Input(shape=(1,))
	x = Dense(190, activation='relu')(inputs)
	predictions = Dense(1, activation='linear')(x)
	model = Model(inputs=inputs, outputs=predictions)
	optimizer = Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
	model.compile(loss='mean_squared_error',optimizer=optimizer)
	history = model.fit(x=np.asarray(data),y=np.asarray(labels), epochs=epoch,batch_size=batch_size)

	plt.plot(range(epoch), history.history['loss'])
	model.save("Model2.h5")


	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Model0','Model1','Model2'], loc='upper right')
	plt.show()



	# v_data = []
	# v_labels = []

	# for i in range(test_size):
	# 	x = random.uniform(domain[0], domain[1])
	# 	v_data.append(x)
	# 	v_labels.append(target(x))

	# loss = model.evaluate(np.asarray(v_data), np.asarray(v_labels), batch_size=128)
	# print ("Loss={:.5f}".format(loss))
