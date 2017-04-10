# coding:utf-8

# remove tf.Session warning
# refs. https://github.com/tensorflow/tensorflow/issues/8037
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# input-output設定
input_x = [
	[0., 0., 0., 1., 0.],
	[0., 0., 1., 1., 1.],
	[0., 1., 1., 0., 0.],
	[0., 1., 1., 1., 1.],
	[1., 0., 0., 0., 1.],
	[1., 0., 1., 1., 0.],
	[1., 1., 0., 1., 1.]]
input_y = [
	[0.],
	[1.],
	[0.],
	[1.],
	[0.],
	[1.],
	[1.]]

# placeholder宣言
x = tf.placeholder("float", [None, 5])
y_ = tf.placeholder("float", [None, 1])

# モデル式定義
w = tf.Variable([[0.1],[0.1],[0.1],[0.1],[0.1]], name="weight")
b = tf.Variable([0.1], name="bias")
y = tf.sigmoid(tf.matmul(x, w)+b)

# 実際に最適化を行う対象：実際の値と現在のパラメータのモデルで求めた値の二乗誤差
loss = tf.reduce_sum(tf.square(y_ - y))

# 最適化アルゴリズムは最急降下法
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)


sess = tf.Session()
# 初期値設定
sess.run(tf.global_variables_initializer())
print 'step数\t二乗誤差'
print '0\t' + str(sess.run(loss, feed_dict={x: input_x, y_: input_y}))

for step in range(50000):
    sess.run(train_step, feed_dict={x: input_x, y_: input_y})
    if (step+1) % 1000 == 0:
        print str(step+1) + '\t' + str(sess.run(loss, feed_dict={x: input_x, y_: input_y}))

print "predict values:"
print sess.run(y, feed_dict={x: input_x})

print "別のデータ使って計算"
next = [
	[1., 0., 0., 0., 0.],
	[1., 1., 1., 1., 1.],
	[0., 0., 1., 0., 1.]]
print sess.run(y, feed_dict={x: next})

