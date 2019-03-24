import tensorflow as tf
import numpy as np
#mnist = tf.keras.datasets.mnist
mnist = tf.keras.datasets.cifar10
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = np.array([[1 if t == lb else 0 for t in range(10)] for lb in y_train])

y_test = np.array([[1 if t == lb else 0 for t in range(10)] for lb in y_test])

from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model

inputs = Input(shape=x_train[0].shape)
x = Flatten()(inputs)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
prediction = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=prediction)
optimizer = tf.keras.optimizers.SGD(lr=0.05, momentum=0.9)
#optimizer = tf.keras.optimizers.Adam(lr=0.5)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=128, epochs=200)
loss, metrics = model.evaluate(x_test, y_test)
print("loss={}, accu={}".format(loss, metrics))
