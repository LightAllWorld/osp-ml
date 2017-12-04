import numpy as np
import tensorflow as tf

xy = np.loadtxt("dataset.csv", delimiter = ",", dtype = np.float32)

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

parm_list = [W]
saver = tf.train.Saver(parm_list)

h = tf.matmul(W, X)
hypothesis = tf.div(1. ,1. + tf.exp(-h)) # sigmoid

cost = -tf.reduce_mean(Y*tf.log(hypothesis) +(1-Y)*tf.log(1-hypothesis))

rate = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(rate)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)

for step in range(2001): # number of training data
	sess.run(train, feed_dict={X : x_data, Y : y_data})

# Save model


save_path = saver.save(sess, "/tmp/Training_model.ckpt")
print("Success", save_path)