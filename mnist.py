import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

yy = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = -tf.reduce_sum(y * tf.log(yy))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(yy, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar("accuracy", accuracy)

saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init)

  merged = tf.summary.merge_all()
  writer = tf.summary.FileWriter("./tmp/logs/mnist", sess.graph)

  for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys})
      print("step %d, training accuray %g"%(i, train_accuracy))

    summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y: batch_ys})
    writer.add_summary(summary, global_step=i)

  saver.save(sess, "tmp/mnist/mnist.ckpt")
