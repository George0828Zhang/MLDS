import tensorflow as tf
from tensorflow.keras.models import Model,load_model #,Sequential
from tensorflow.keras.layers import Input, Dense

import numpy as np
import math
import random
import matplotlib.pyplot as plt

sample = 10000

from func import target

if __name__ == "__main__":

	x = []
	ground = []

	model0 = load_model("Model0.h5")
	model1 = load_model("Model1.h5")
	model2 = load_model("Model2.h5")

	# from tensorflow.keras.utils import plot_model
	# plot_model(model, show_shapes=True, to_file='model.png')

	for i in range(sample):
		x.append(i/sample)
		ground.append(target(i/sample))
	model1_predict = model1.predict(np.asarray(x))
	model0_predict = model0.predict(np.asarray(x))
	model1_predict = model1.predict(np.asarray(x))
	model2_predict = model2.predict(np.asarray(x))


	plt.plot(x, model0_predict)
	plt.plot(x, model1_predict)
	plt.plot(x, model2_predict)
	plt.plot(x, ground)
	


	plt.title('Function')
	plt.legend(['model0', 'model1','model2','ground truth'], loc='upper left')
	plt.show()



	# v_data = []
	# v_labels = []

	# for i in range(test_size):
	# 	x = random.uniform(domain[0], domain[1])
	# 	v_data.append(x)
	# 	v_labels.append(target(x))

	# loss = model.evaluate(np.asarray(v_data), np.asarray(v_labels), batch_size=128)
	# print ("Loss={:.5f}".format(loss))