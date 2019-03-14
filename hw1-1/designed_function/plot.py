import tensorflow as tf
from tensorflow.keras.models import Model,load_model #,Sequential
from tensorflow.keras.layers import Input, Dense

import numpy as np
import math
import random
import matplotlib.pyplot as plt

sample = 10000

def target(x):
	assert( x >= 0 )
	y = x * math.sin(5*math.pi *x)
	return y


def functionPlot():
	x = []
	ground = []

	model0 = load_model("model0.h5")
	model1 = load_model("model1.h5")
	model2 = load_model("model2.h5")

	# from tensorflow.keras.utils import plot_model
	# plot_model(model, show_shapes=True, to_file='model.png')

	for i in range(sample):
		x.append(i/sample)
		ground.append(target(i/sample))


	model0_predict = model0.predict(np.asarray(x))
	model1_predict = model1.predict(np.asarray(x))
	model2_predict = model2.predict(np.asarray(x))


	plt.plot(x, model0_predict)
	plt.plot(x, model1_predict)
	plt.plot(x, model2_predict)
	plt.plot(x, ground)
	


	plt.title('Function')
	plt.legend(['model0', 'model1','model2','ground truth'], loc='upper left')
	plt.savefig("functionPlot.png")
	plt.close()

def lossPlot():
	epoch = 6000
	loss0 = np.load("model0_loss.npy")	
	loss1 = np.load("model1_loss.npy")
	loss2 = np.load("model2_loss.npy")
	plt.plot(range(epoch), loss0)
	plt.plot(range(epoch), loss1)
	plt.plot(range(epoch), loss2)

	plt.title('Model loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Model0','Model1','Model2'], loc='upper right')
	plt.savefig("lossPlot.png")
	plt.close()

if __name__ == "__main__":
	functionPlot()
	lossPlot()

