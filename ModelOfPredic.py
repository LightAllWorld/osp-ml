import tensorflow as tf
import numpy as np

xy = np.loadtxt("dataset.csv", delimiter = ",", dtype = np.float32) #if using others data( Test or user data ), chnage file name.

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

rate = 0.03
optimizer = tf.train.GradientDescentOptimizer(rate)
train = tf.train.GradientDescentOptimizer(rate).minimize(cost)

comp_pred = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(comp_pred, dtype=tf.float32))



with tf.Session() as sess :
    saver.restore(sess, "/tmp/Training_model.ckpt")

    for step in range(500 + 1) : #if using user data, comment out this for-loop.
        _, loss,acc = sess.run(
        	[train, cost, accuracy], feed_dict={X : x_data, Y : y_data})
        if step % 10 == 0 :
        	print("step :", step)
        	print("loss :", loss)
        	print("acc :", acc)
    #sess.run(train, feed_dict = {X : x_data, Y : y_data}) # if using user data, using this statement.