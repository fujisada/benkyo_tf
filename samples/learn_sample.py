# coding:utf-8
# test.py

# remove tf.Session warning
# refs. https://github.com/tensorflow/tensorflow/issues/8037
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

# 気温データ設定
input_x = [[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.], [9.], [10.], [11.], [12.]]
input_y = [[6.1], [7.2], [10.1], [15.4], [20.2], [22.4], [25.4], [27.1], [24.4], [18.7], [11.4], [8.9]]

# placeholder宣言
x = tf.placeholder("float", [None, 1])
y_ = tf.placeholder("float", [None, 1])

# モデル式定義
a = tf.Variable([11.], name="amp")
w = tf.Variable([0.3], name="freq")
p = tf.Variable([6.], name="phase")
c = tf.Variable([13.], name="const")
y = a * tf.sin((w*x)+p) + c

# 実際に最適化を行う対象：実際の値と現在のパラメータのモデルで求めた値の二乗誤差
loss = tf.reduce_sum(tf.square(y_ - y))

# 最適化アルゴリズムは最急降下法
train_step = tf.train.GradientDescentOptimizer(0.00002).minimize(loss)


sess = tf.Session()
# 初期値設定
sess.run(tf.global_variables_initializer())
print 'step数\t二乗誤差'
print '0\t' + str(sess.run(loss, feed_dict={x: input_x, y_: input_y}))

for step in range(20000):
    sess.run(train_step, feed_dict={x: input_x, y_: input_y})
    if (step+1) % 1000 == 0:
        print str(step+1) + '\t' + str(sess.run(loss, feed_dict={x: input_x, y_: input_y}))

print "\ntrained model: y = %f sin( %f * x + %f)  + %f" % (sess.run(a), sess.run(w), sess.run(p), sess.run(c))
print "教師データを学習済みモデルで計算"
print sess.run(y, feed_dict={x: input_x})

