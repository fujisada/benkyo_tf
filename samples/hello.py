# hello.py

# remove tf.Session warning
# refs. https://github.com/tensorflow/tensorflow/issues/8037
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


import tensorflow as tf

sess = tf.Session()

hello = tf.constant('hello world')
print sess.run(hello)

