import tensorflow as tf
from tensorflow.keras.models import Model #,Sequential
from tensorflow.keras.layers import Input, Dense

import numpy as np
import math
import random
import matplotlib.pyplot as plt

epoch = 6000
data_size = 10000
batch_size = 128
domain = (0, 1)
# test_size = 2000

def target(x):
	assert( x >= 0 )
	y = math.atan(math.sqrt(x))
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
	x = Dense(190, activation='relu')(inputs)
	predictions = Dense(1, activation='linear')(x)

	model = Model(inputs=inputs, outputs=predictions)
	model.compile(loss='mean_squared_error',optimizer='sgd')
	history = model.fit(x=np.asarray(data),y=np.asarray(labels),epochs=epoch,batch_size=batch_size)

	print(history.history['loss'])

	plt.plot(range(epoch), history.history['loss'])
	plt.show()


	# v_data = []
	# v_labels = []

	# for i in range(test_size):
	# 	x = random.uniform(domain[0], domain[1])
	# 	v_data.append(x)
	# 	v_labels.append(target(x))

	# loss = model.evaluate(np.asarray(v_data), np.asarray(v_labels), batch_size=128)
	# print ("Loss={:.5f}".format(loss))