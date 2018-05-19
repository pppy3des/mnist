from PIL import Image
import csv
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

yy = tf.nn.softmax(tf.matmul(x, W) + b)

saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver.restore(sess, "tmp/mnist/mnist.ckpt")

ans = tf.argmax(yy, 1)

with open('test.csv', 'r') as f:
  reader = csv.reader(f)
  for row in reader:
    img = Image.open(row[0]).convert('L')
    img.thumbnail((28, 28))
    img = np.array(img, dtype=np.float32)

    img = 1 - np.array(img / 255)
    img = img.reshape(1, 784)

    print("識別結果:%s 正解ラベル:%s"%(sess.run(ans, feed_dict={x:img})[0], row[1]))

