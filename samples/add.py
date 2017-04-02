# add.py

# remove tf.Session warning
# refs. https://github.com/tensorflow/tensorflow/issues/8037
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import tensorflow as tf

sess = tf.Session()

# define eqation
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
c = tf.constant(4)
#equation = a + b
equation = tf.add(tf.multiply(c, a), b)

# calculate
res = sess.run(equation, feed_dict={a: 3, b: 5})
res = sess.run(equation, feed_dict={a: 3, b: 5})

# print result
print res
