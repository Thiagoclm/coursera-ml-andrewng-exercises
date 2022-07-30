import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# X_train = np.array([[1],[2]])
# Y_train = np.array([[300],[500]])

# fig, ax = plt.subplots(1,1)
# ax.scatter(X_train, Y_train, marker='x', c='r')
# # plt.show()

# linear_layer = tf.keras.layers.Dense(units=1, activation='linear')

# a1 = linear_layer(X_train[0].reshape(1,1))
# print(a1)

# w,b = linear_layer.get_weights()
# print(f"w = {w}, b = {b}")

# set_w = np.array([[200]])
# set_b = np.array([100])

# # set_weights takes a list of numpy arrays
# linear_layer.set_weights([set_w, set_b])
# print(linear_layer.get_weights())

# a1 = linear_layer(X_train[0].reshape(1,1))
# print(a1)
# alin = np.dot(set_w,X_train[0].reshape(1,1)) + set_b
# print(alin)

# prediction_tf = linear_layer(X_train)
# prediction_np = np.dot( X_train, set_w) + set_b

# Sigmoid

X_train = np.array([0,1,2,3,4,5]).reshape(-1,1)
# reshape(-1,1): set to column=1 but rows unknown -> 1 column array
Y_train = np.array([0,0,0,1,1,1]).reshape(-1,1)

pos = Y_train == 1
neg = Y_train == 0

fig, ax = plt.subplots(1,1)
ax.scatter(X_train[pos], Y_train[pos], marker='x', c='r', label='y=1')
ax.scatter(X_train[neg], Y_train[neg], marker='o', c='b', label='y=0')
ax.legend(fontsize=12)
# plt.show()

# Logistic Neuron
model = tf.keras.Sequential([tf.keras.layers.Dense(1, 
                                            input_dim=1,
                                            activation='sigmoid',
                                            name='L1')])
# model.summary()

logistic_layer = model.get_layer('L1')
w,b = logistic_layer.get_weights()
print(w,b)
print(w.shape,b.shape)

set_w = np.array([[2]])
set_b = np.array([-4.5])

logistic_layer.set_weights([set_w, set_b])
a1 = model.predict(X_train[0].reshape(1,1))
print(a1)
alog = 1/(1+np.exp(-(np.dot(set_w,X_train[0].reshape(1,1)) + set_b)))
print(alog)





